"""Execute task node for executor graph.

Phase 1.2: Token streaming support - stream LLM tokens through executor.
Phase 1.1: Model routing support - select model based on task complexity.
Phase 2.3.2: Dry run mode support.
Phase 2.3.3: Pause before destructive operations.
Phase 2.3: Shell tool integration - track results and handle failures.
Phase 7.2: Subagent routing support - route tasks to specialized subagents.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Phase 7.2: Subagent routing imports
from ai_infra.executor.agents.base import ExecutionContext, SubAgentResult
from ai_infra.executor.agents.spawner import spawn_for_task
from ai_infra.executor.shell.persistence import (
    ensure_snapshot_dir,
    generate_snapshot_filename,
    save_snapshot,
)
from ai_infra.executor.shell.restoration import diff_snapshots
from ai_infra.executor.shell.snapshot import ShellSnapshot, capture_shell_state
from ai_infra.executor.state import (
    ExecutorGraphState,
    NodeTimeouts,
    NonRetryableErrors,
    ShellError,
)
from ai_infra.executor.streaming import (
    ExecutorStreamEvent,
    StreamingConfig,
    create_llm_done_event,
    create_llm_error_event,
    create_llm_thinking_event,
    create_llm_token_event,
    create_llm_tool_end_event,
    create_llm_tool_start_event,
)
from ai_infra.executor.utils.safety import (
    check_agent_result_for_destructive_ops,
    check_files_for_destructive_ops,
    format_destructive_warning,
)
from ai_infra.llm.shell.tool import get_current_session
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.agent import Agent
    from ai_infra.executor.routing import ModelRouter, RoutingMetrics

logger = get_logger("executor.nodes.execute")


# =============================================================================
# Phase 7.2: Subagent Routing Helpers
# =============================================================================


def _detect_project_type(workspace: Path) -> str:
    """Detect project type from workspace files.

    Phase 16.5.5.6: Detect project type from configuration files.

    Args:
        workspace: Path to workspace root.

    Returns:
        Project type string (python, node, rust, go, unknown).
    """
    # Check for Python project indicators
    if (workspace / "pyproject.toml").exists():
        return "python"
    if (workspace / "setup.py").exists():
        return "python"
    if (workspace / "requirements.txt").exists():
        return "python"

    # Check for Node.js project indicators
    if (workspace / "package.json").exists():
        return "node"

    # Check for Rust project
    if (workspace / "Cargo.toml").exists():
        return "rust"

    # Check for Go project
    if (workspace / "go.mod").exists():
        return "go"

    return "unknown"


def _get_file_preview(path: Path, max_lines: int = 50) -> str | None:
    """Get a preview of file contents.

    Phase 16.5.5.8: Read first N lines of a file for context.

    Args:
        path: Path to the file.
        max_lines: Maximum number of lines to read.

    Returns:
        File contents preview or None if unreadable.
    """
    try:
        if not path.exists() or not path.is_file():
            return None
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")[:max_lines]
        if len(content.split("\n")) > max_lines:
            lines.append(f"... ({len(content.split(chr(10))) - max_lines} more lines)")
        return "\n".join(lines)
    except Exception:
        return None


def _extract_code_patterns(workspace: Path, files: list[str]) -> dict[str, str]:
    """Extract coding patterns from existing files.

    Phase 16.5.5.9: Detect code conventions from first created files.

    Args:
        workspace: Path to workspace root.
        files: List of file paths to analyze.

    Returns:
        Dictionary of pattern names to values.
    """
    patterns: dict[str, str] = {}

    for f in files[:3]:  # Only check first 3 files
        path = workspace / f
        if path.suffix == ".py" and path.exists():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                first_500 = content[:500]

                # Detect docstring style
                if '"""' in first_500 or "'''" in first_500:
                    if "Args:" in content:
                        patterns["docstring_style"] = "google"
                    elif ":param" in content:
                        patterns["docstring_style"] = "sphinx"
                    elif "Parameters" in content and "----------" in content:
                        patterns["docstring_style"] = "numpy"

                # Detect future imports
                if "from __future__ import annotations" in content:
                    patterns["future_annotations"] = "yes"

                # Detect type hints
                if "->" in content[:2000]:
                    patterns["type_hints"] = "yes"

                # Detect logging pattern
                if "get_logger(" in content or "logging.getLogger(" in content:
                    patterns["logging"] = "structured"

                # If we found patterns, stop checking more files
                if patterns:
                    break

            except Exception:
                continue

    return patterns


def _build_task_summaries(run_memory: dict[str, Any]) -> list[str]:
    """Build task summaries from run memory.

    Phase 16.5.5.5: Extract summaries of completed tasks.

    Args:
        run_memory: Dictionary of task memories.

    Returns:
        List of task summary strings.
    """
    summaries: list[str] = []

    for task_id, memory in run_memory.items():
        if isinstance(memory, dict):
            summary = memory.get("summary", "")
            files = memory.get("files_created", [])
            status = memory.get("status", "")

            if summary or files:
                parts = [f"Task {task_id}"]
                if summary:
                    # Truncate long summaries
                    short_summary = summary[:100] + "..." if len(summary) > 100 else summary
                    parts.append(f": {short_summary}")
                if files:
                    parts.append(f" (files: {', '.join(files[:3])})")
                    if len(files) > 3:
                        parts.append(f" +{len(files) - 3} more")
                summaries.append("".join(parts))

    return summaries


def _build_session_brief(
    tasks_completed: int,
    files_created: list[str],
    project_type: str,
) -> str:
    """Build a brief summary of the session so far.

    Phase 16.5.5.7: Create session context summary.

    Args:
        tasks_completed: Number of tasks completed.
        files_created: List of files created.
        project_type: Detected project type.

    Returns:
        Session brief string.
    """
    files_summary = ", ".join(files_created[:5]) if files_created else "none yet"
    if len(files_created) > 5:
        files_summary += f" (+{len(files_created) - 5} more)"

    return f"""Session Progress:
- Tasks completed: {tasks_completed}
- Files created: {files_summary}
- Project type: {project_type}

Continue following the established patterns from existing files."""


def _get_relevant_files(
    workspace: Path,
    files_modified: list[str],
    max_files: int = 10,
) -> list[str]:
    """Get list of relevant files for context.

    Phase 16.5.5.4: Populate relevant_files from state.

    Args:
        workspace: Path to workspace root.
        files_modified: Files already modified.
        max_files: Maximum number of files to return.

    Returns:
        List of relevant file paths.
    """
    relevant: list[str] = []

    # Include recently modified files (most relevant)
    relevant.extend(files_modified[-max_files:])

    # Also check for key project files
    key_files = [
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "package.json",
        "README.md",
        "Makefile",
    ]

    for key_file in key_files:
        if (workspace / key_file).exists() and key_file not in relevant:
            relevant.append(key_file)
            if len(relevant) >= max_files:
                break

    return relevant[:max_files]


def _build_file_contents_cache(
    workspace: Path,
    files: list[str],
    max_files: int = 5,
    max_lines: int = 50,
) -> dict[str, str]:
    """Build cache of file contents for context.

    Phase 16.5.5.3: Cache recently created file contents.

    Args:
        workspace: Path to workspace root.
        files: Files to cache.
        max_files: Maximum number of files to cache.
        max_lines: Maximum lines per file.

    Returns:
        Dictionary mapping file paths to contents.
    """
    cache: dict[str, str] = {}

    for f in files[-max_files:]:  # Get most recent files
        path = workspace / f
        preview = _get_file_preview(path, max_lines)
        if preview:
            cache[f] = preview

    return cache


def _build_execution_context(state: ExecutorGraphState) -> ExecutionContext:
    """Build ExecutionContext from graph state for subagent execution.

    Phase 7.2.2: Creates the context object needed by spawn_for_task().
    Phase 16.5.5: Enhanced with rich context for better subagent code quality.
    Phase 16.5.11.5: Additional context for quality improvements.

    Args:
        state: Current executor graph state.

    Returns:
        ExecutionContext populated from state with enriched context.
    """
    roadmap_path = state.get("roadmap_path")
    workspace = Path(roadmap_path).parent if roadmap_path else Path.cwd()

    # Get all files modified
    files_modified = state.get("files_modified", [])

    # Phase 16.5.5.6: Detect project type
    project_type = _detect_project_type(workspace)

    # Phase 16.5.5.4: Get relevant files
    relevant_files = _get_relevant_files(workspace, files_modified)

    # Phase 16.5.5.5: Build task summaries from run_memory
    run_memory = state.get("run_memory", {})
    previous_summaries = _build_task_summaries(run_memory)

    # Phase 16.5.5.9: Extract code patterns
    patterns = _extract_code_patterns(workspace, files_modified)

    # Phase 16.5.5.7: Build session brief
    session_brief = _build_session_brief(
        tasks_completed=len(run_memory),
        files_created=files_modified,
        project_type=project_type,
    )

    # Phase 16.5.5.3: Build file contents cache
    file_cache = _build_file_contents_cache(workspace, files_modified)

    # Phase 16.5.11.5.1: Extract completed tasks from todos
    completed_tasks: list[str] = []
    todos = state.get("todos", [])
    for todo in todos:
        if isinstance(todo, dict) and todo.get("status") == "completed":
            completed_tasks.append(todo.get("title", "Unknown task"))

    # Phase 16.5.11.5.2: Collect existing source files
    existing_files: list[str] = []
    try:
        for ext in ("*.py", "*.ts", "*.js", "*.tsx", "*.jsx"):
            for f in workspace.rglob(ext):
                rel_path = str(f.relative_to(workspace))
                if ".git" not in rel_path and "node_modules" not in rel_path:
                    existing_files.append(rel_path)
                if len(existing_files) >= 50:
                    break
            if len(existing_files) >= 50:
                break
    except (OSError, ValueError):
        pass

    # Phase 16.5.11.5.3: Project-wide patterns for CoderAgent
    project_patterns: dict[str, Any] = {}
    if patterns:
        project_patterns["code_style"] = patterns
    # Add common project conventions
    pyproject = workspace / "pyproject.toml"
    if pyproject.exists():
        project_patterns["build_tool"] = "poetry"
    if (workspace / "pytest.ini").exists() or (workspace / "conftest.py").exists():
        project_patterns["test_framework"] = "pytest"

    return ExecutionContext(
        workspace=workspace,
        files_modified=files_modified,
        relevant_files=relevant_files,
        project_type=project_type,
        summary=session_brief,
        run_memory=run_memory,
        dependencies=state.get("dependencies", []),
        # Phase 16.5.5: Enriched context
        previous_task_summaries=previous_summaries,
        established_patterns=patterns,
        file_contents_cache=file_cache,
        # Phase 16.5.11.5: Quality improvements context
        completed_tasks=completed_tasks,
        existing_files=existing_files,
        project_patterns=project_patterns,
        source_file_previews={},  # Populated by spawn_for_task for test tasks
    )


def _sanitize_task_id(task_id: str) -> str:
    """Normalize task id for snapshot filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_")
    return cleaned or "unknown"


async def _capture_task_snapshot(
    *,
    task_id: str,
    workspace_dir: Path,
    phase: str,
) -> tuple[ShellSnapshot | None, Path | None]:
    """Capture and persist a shell snapshot for a task phase."""
    try:
        snapshot = await capture_shell_state()
    except Exception as e:
        logger.warning(f"Shell snapshot capture failed ({phase}) for task [{task_id}]: {e}")
        return None, None

    try:
        snapshot_dir = ensure_snapshot_dir(workspace_dir)
        safe_task_id = _sanitize_task_id(task_id)
        suffix = f"-task-{safe_task_id}-{phase}"
        filename = generate_snapshot_filename(
            snapshot.shell_type,
            timestamp=snapshot.captured_at,
            suffix=suffix,
        )
        path = snapshot_dir / filename
        saved_path = save_snapshot(snapshot, path=path)
        logger.info(f"Saved {phase} shell snapshot for task [{task_id}] to {saved_path}")
        return snapshot, saved_path
    except Exception as e:
        logger.warning(f"Shell snapshot save failed ({phase}) for task [{task_id}]: {e}")
        return snapshot, None


async def _finalize_task_snapshot(
    *,
    task_id: str,
    workspace_dir: Path,
    pre_snapshot: ShellSnapshot | None,
    pre_snapshot_path: Path | None,
    enabled: bool,
) -> dict[str, Any]:
    """Capture post-task snapshot and compute diff."""
    if not enabled or pre_snapshot is None:
        return {}

    post_snapshot, post_snapshot_path = await _capture_task_snapshot(
        task_id=task_id,
        workspace_dir=workspace_dir,
        phase="post",
    )
    if post_snapshot is None:
        return {
            "shell_snapshot_pre_path": str(pre_snapshot_path) if pre_snapshot_path else None,
            "shell_snapshot_post_path": None,
            "shell_snapshot_diff": None,
        }

    diff = diff_snapshots(pre_snapshot, post_snapshot)
    if diff.has_changes:
        logger.info(f"Shell snapshot diff for task [{task_id}]: {diff.summary()}")
    else:
        logger.debug(f"Shell snapshot diff for task [{task_id}]: no changes")

    return {
        "shell_snapshot_pre_path": str(pre_snapshot_path) if pre_snapshot_path else None,
        "shell_snapshot_post_path": str(post_snapshot_path) if post_snapshot_path else None,
        "shell_snapshot_diff": diff.to_dict(),
    }


# =============================================================================
# Phase 16.5.11.4: Routing Accuracy Metrics
# =============================================================================


def _update_routing_metrics(
    state: ExecutorGraphState,
    task_id: str,
    task_title: str,
    orchestrator_choice: str,
    keyword_choice: str,
    confidence: float,
    used_fallback: bool,
    reasoning: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Update routing accuracy metrics comparing orchestrator vs keyword routing.

    Phase 16.5.11.4: Track and compare routing decisions for analysis.

    Args:
        state: Current graph state.
        task_id: ID of the task being routed.
        task_title: Title of the task.
        orchestrator_choice: Agent type chosen by orchestrator.
        keyword_choice: Agent type that keyword routing would choose.
        confidence: Confidence score from orchestrator.
        used_fallback: Whether fallback was used.
        reasoning: Reasoning from orchestrator.

    Returns:
        Tuple of (updated_metrics dict, updated_comparison_log list).
    """
    # Get existing metrics or create new
    existing_metrics = state.get("routing_accuracy_metrics") or {
        "orchestrator_matches": 0,
        "orchestrator_differs": 0,
        "fallback_used": 0,
        "confidence_histogram": {"0.0-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0},
        "agent_distribution": {},
    }

    # Get existing comparison log
    existing_log = state.get("routing_comparison_log") or []

    # Determine if choices match
    matched = orchestrator_choice == keyword_choice

    # Update metrics
    if matched:
        existing_metrics["orchestrator_matches"] += 1
    else:
        existing_metrics["orchestrator_differs"] += 1
        logger.info(
            f"Routing divergence for task [{task_id}]: "
            f"orchestrator={orchestrator_choice}, keyword={keyword_choice}"
        )

    if used_fallback:
        existing_metrics["fallback_used"] += 1

    # Update confidence histogram
    if confidence < 0.5:
        existing_metrics["confidence_histogram"]["0.0-0.5"] += 1
    elif confidence < 0.7:
        existing_metrics["confidence_histogram"]["0.5-0.7"] += 1
    elif confidence < 0.9:
        existing_metrics["confidence_histogram"]["0.7-0.9"] += 1
    else:
        existing_metrics["confidence_histogram"]["0.9-1.0"] += 1

    # Update agent distribution
    existing_metrics["agent_distribution"][orchestrator_choice] = (
        existing_metrics["agent_distribution"].get(orchestrator_choice, 0) + 1
    )

    # Add to comparison log
    comparison_entry = {
        "task_id": task_id,
        "task_title": task_title,
        "orchestrator_choice": orchestrator_choice,
        "keyword_choice": keyword_choice,
        "matched": matched,
        "confidence": confidence,
        "used_fallback": used_fallback,
        "reasoning": reasoning[:100] if len(reasoning) > 100 else reasoning,
    }
    updated_log = existing_log + [comparison_entry]

    return existing_metrics, updated_log


async def execute_task_node(
    state: ExecutorGraphState,
    *,
    agent: Agent | None = None,
    dry_run: bool = False,
    pause_destructive: bool = True,
    model_router: ModelRouter | None = None,
    routing_metrics: RoutingMetrics | None = None,
    use_subagents: bool = False,
    subagent_config: Any = None,  # SubAgentConfig | None (Phase 7.4)
    # Orchestrator configuration
    orchestrator_model: str = "gpt-4o-mini",
    orchestrator_confidence_threshold: float = 0.7,
) -> ExecutorGraphState:
    """Execute the current task using the Agent or specialized subagent.

    This node:
    1. Gets prompt from state
    2. Routes to subagent if use_subagents=True (Phase 7.2)
    3. Uses orchestrator for intelligent routing
    4. Otherwise routes to model based on task complexity (Phase 1.1)
    5. Calls agent.arun() with timeout (unless dry_run)
    6. Checks result for destructive operations (if pause_destructive)
    7. Extracts files_modified from agent result
    8. Updates state with execution results

    Phase 2.3.2: Supports dry_run mode to preview actions without executing.
    Phase 2.3.3: Pauses execution on destructive operations.
    Phase 7.2: Routes to specialized subagents (Coder, Tester, Debugger, etc.).
    Phase 7.4: Supports subagent model configuration.

    Args:
        state: Current graph state with prompt.
        agent: The Agent instance to use for execution.
        dry_run: If True, log planned actions without executing.
        pause_destructive: If True, interrupt on destructive operations.
        model_router: Optional model router for complexity-based routing.
        routing_metrics: Optional metrics collector for routing decisions.
        use_subagents: If True, route tasks to specialized subagents.
        subagent_config: Optional model config for subagents (Phase 7.4).
        orchestrator_model: Model to use for orchestrator routing.
        orchestrator_confidence_threshold: Minimum confidence for routing.

    Returns:
        Updated state with agent_result and files_modified.
    """
    current_task = state.get("current_task")
    prompt = state.get("prompt", "")

    # Phase 2.3.2: Handle dry run mode
    if dry_run or state.get("dry_run", False):
        task_id = str(current_task.id) if current_task else "unknown"
        task_title = current_task.title if current_task else "Unknown task"

        logger.info(f"[DRY RUN] Would execute task: {task_title}")
        logger.info(f"[DRY RUN] Task ID: {task_id}")
        logger.info(f"[DRY RUN] Prompt preview ({len(prompt)} chars): {prompt[:500]}...")

        return {
            **state,
            "agent_result": {
                "dry_run": True,
                "task_id": task_id,
                "task_title": task_title,
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:1000],
            },
            "files_modified": [],
            "error": None,
            "verified": True,  # Skip verification in dry run
        }

    if not prompt:
        logger.error("No prompt available for execution")
        return {
            **state,
            "agent_result": None,
            "files_modified": [],
            "error": {
                "error_type": "execution",
                "message": "No prompt available for execution",
                "node": "execute_task",
                "task_id": str(current_task.id) if current_task else None,
                "recoverable": False,
                "stack_trace": None,
            },
        }

    # Check for subagent routing first (Phase 7.2)
    effective_use_subagents = use_subagents or state.get("use_subagents", False)

    # Only require agent if NOT using subagents
    if agent is None and model_router is None and not effective_use_subagents:
        logger.error("No agent or router provided for execution")
        return {
            **state,
            "agent_result": None,
            "files_modified": [],
            "error": {
                "error_type": "execution",
                "message": "No agent or router provided for execution",
                "node": "execute_task",
                "task_id": str(current_task.id) if current_task else None,
                "recoverable": False,
                "stack_trace": None,
            },
        }

    task_id = str(current_task.id) if current_task else "unknown"

    # Phase 16.4: Pre-task shell snapshot
    enable_shell_snapshots = state.get("enable_shell_snapshots", False)
    roadmap_path = state.get("roadmap_path")
    workspace_dir = Path(roadmap_path).parent if roadmap_path else Path.cwd()
    pre_snapshot: ShellSnapshot | None = None
    pre_snapshot_path: Path | None = None
    if enable_shell_snapshots and current_task is not None:
        pre_snapshot, pre_snapshot_path = await _capture_task_snapshot(
            task_id=task_id,
            workspace_dir=workspace_dir,
            phase="pre",
        )

    # -------------------------------------------------------------------------
    # Phase 7.2: Subagent Routing with Orchestrator
    # -------------------------------------------------------------------------
    # If use_subagents is enabled (via parameter or state), route to specialized
    # subagents using LLM-based orchestrator for intelligent routing.

    if effective_use_subagents and current_task:
        from ai_infra.executor.agents.registry import SubAgentType

        routing_decision_dict: dict[str, Any] | None = None
        orchestrator_tokens = 0

        # Use LLM-based orchestrator for intelligent routing
        from ai_infra.executor.agents.orchestrator import (
            OrchestratorAgent,
            RoutingContext,
        )

        try:
            # Build routing context
            todos = state.get("todos", [])
            completed_titles = [
                t.title for t in todos if hasattr(t, "status") and t.status == "completed"
            ]
            existing_files = state.get("files_modified", [])

            routing_context = RoutingContext(
                workspace=workspace_dir,
                completed_tasks=completed_titles,
                existing_files=existing_files,
                project_type=_detect_project_type(workspace_dir),
                previous_agent=state.get("subagent_used"),
            )

            # Create orchestrator and route
            orchestrator = OrchestratorAgent(
                model=orchestrator_model,
                confidence_threshold=orchestrator_confidence_threshold,
            )
            routing_decision = await orchestrator.route(current_task, routing_context)

            # Extract tokens from orchestrator metrics
            if hasattr(orchestrator, "_last_metrics"):
                orchestrator_tokens = orchestrator._last_metrics.get("total_tokens", 0)

            agent_type = routing_decision.agent_type
            agent_name = agent_type.value

            # Build routing decision dict for state
            routing_decision_dict = {
                "agent_type": agent_name,
                "confidence": routing_decision.confidence,
                "reasoning": routing_decision.reasoning,
                "used_fallback": routing_decision.used_fallback,
                "task_id": task_id,
                "task_title": current_task.title if current_task else "",
            }

            logger.info(
                f"Orchestrator routed task [{task_id}] to {agent_name} "
                f"(confidence={routing_decision.confidence:.2f}, "
                f"fallback={routing_decision.used_fallback}, "
                f"reason={routing_decision.reasoning[:50]}...)"
            )

        except Exception as e:
            # Fall back to CODER on orchestrator failure
            logger.warning(
                f"Orchestrator routing failed for task [{task_id}], defaulting to CODER: {e}"
            )
            agent_type = SubAgentType.CODER
            agent_name = agent_type.value
            routing_decision_dict = {
                "agent_type": agent_name,
                "confidence": 0.0,
                "reasoning": f"Orchestrator failed: {e}",
                "used_fallback": True,
                "task_id": task_id,
                "task_title": current_task.title if current_task else "",
            }

        logger.info(
            f"Routing task [{task_id}] to {agent_name} agent",
            extra={"task_id": task_id, "agent_type": agent_name},
        )

        try:
            # Build execution context from state
            context = _build_execution_context(state)

            # Pass pre-determined agent_type from orchestrator
            subagent_result: SubAgentResult = await spawn_for_task(
                current_task, context, subagent_config, agent_type=agent_type
            )

            # Phase 16.5.1: Extract token metrics from subagent result
            subagent_metrics = subagent_result.metrics or {}
            subagent_tokens = subagent_metrics.get("total_tokens", 0)

            logger.info(
                f"Task [{task_id}] executed by {agent_name}: "
                f"success={subagent_result.success}, "
                f"files_modified={len(subagent_result.files_modified)}, "
                f"tokens={subagent_tokens}"
            )

            # Phase 7.3.3: Track subagent usage metrics
            existing_usage = state.get("subagent_usage", {})
            updated_usage = {
                **existing_usage,
                agent_name: existing_usage.get(agent_name, 0) + 1,
            }

            # Phase 16.5.1: Accumulate subagent tokens in state
            existing_subagent_tokens = state.get("subagent_tokens_total", 0)
            updated_subagent_tokens = existing_subagent_tokens + subagent_tokens

            # Accumulate orchestrator tokens and routing history
            existing_orchestrator_tokens = state.get("orchestrator_tokens_total", 0)
            updated_orchestrator_tokens = existing_orchestrator_tokens + orchestrator_tokens
            existing_routing_history = state.get("orchestrator_routing_history", [])
            updated_routing_history = (
                existing_routing_history + [routing_decision_dict]
                if routing_decision_dict
                else existing_routing_history
            )

            # Phase 16.5.11.4: Update routing accuracy metrics
            keyword_choice = (
                routing_decision_dict.get("keyword_would_choose", agent_name)
                if routing_decision_dict
                else agent_name
            )
            updated_metrics, updated_comparison_log = _update_routing_metrics(
                state=state,
                task_id=task_id,
                task_title=current_task.title if current_task else "",
                orchestrator_choice=agent_name,
                keyword_choice=keyword_choice,
                confidence=routing_decision_dict.get("confidence", 1.0)
                if routing_decision_dict
                else 1.0,
                used_fallback=routing_decision_dict.get("used_fallback", True)
                if routing_decision_dict
                else True,
                reasoning=routing_decision_dict.get("reasoning", "")
                if routing_decision_dict
                else "",
            )

            # Return state with subagent results
            if subagent_result.success:
                snapshot_update = await _finalize_task_snapshot(
                    task_id=task_id,
                    workspace_dir=workspace_dir,
                    pre_snapshot=pre_snapshot,
                    pre_snapshot_path=pre_snapshot_path,
                    enabled=enable_shell_snapshots,
                )
                return {
                    **state,
                    "agent_result": subagent_result.to_dict(),
                    "files_modified": subagent_result.files_modified,
                    "subagent_used": agent_name,
                    "subagent_usage": updated_usage,
                    "subagent_tokens_total": updated_subagent_tokens,
                    "subagent_tokens_task": subagent_tokens,
                    # Phase 16.5.11.3: Orchestrator tracking
                    "orchestrator_routing_decision": routing_decision_dict,
                    "orchestrator_tokens_total": updated_orchestrator_tokens,
                    "orchestrator_routing_history": updated_routing_history,
                    # Phase 16.5.11.4: Routing accuracy metrics
                    "routing_accuracy_metrics": updated_metrics,
                    "routing_comparison_log": updated_comparison_log,
                    "error": None,
                    **snapshot_update,
                }
            else:
                snapshot_update = await _finalize_task_snapshot(
                    task_id=task_id,
                    workspace_dir=workspace_dir,
                    pre_snapshot=pre_snapshot,
                    pre_snapshot_path=pre_snapshot_path,
                    enabled=enable_shell_snapshots,
                )
                return {
                    **state,
                    "agent_result": subagent_result.to_dict(),
                    "files_modified": [],
                    "subagent_used": agent_name,
                    "subagent_usage": updated_usage,
                    "subagent_tokens_total": updated_subagent_tokens,
                    "subagent_tokens_task": subagent_tokens,
                    # Phase 16.5.11.3: Orchestrator tracking
                    "orchestrator_routing_decision": routing_decision_dict,
                    "orchestrator_tokens_total": updated_orchestrator_tokens,
                    "orchestrator_routing_history": updated_routing_history,
                    # Phase 16.5.11.4: Routing accuracy metrics
                    "routing_accuracy_metrics": updated_metrics,
                    "routing_comparison_log": updated_comparison_log,
                    "error": {
                        "error_type": "subagent_execution",
                        "message": subagent_result.error or "Subagent execution failed",
                        "node": "execute_task",
                        "task_id": task_id,
                        "recoverable": True,
                        "stack_trace": None,
                    },
                    **snapshot_update,
                }

        except Exception as e:
            logger.exception(f"Subagent routing failed for task [{task_id}]: {e}")
            snapshot_update = await _finalize_task_snapshot(
                task_id=task_id,
                workspace_dir=workspace_dir,
                pre_snapshot=pre_snapshot,
                pre_snapshot_path=pre_snapshot_path,
                enabled=enable_shell_snapshots,
            )
            # Phase 16.5.11.3: Include routing decision even on failure
            existing_routing_history = state.get("orchestrator_routing_history", [])
            updated_routing_history = (
                existing_routing_history + [routing_decision_dict]
                if routing_decision_dict
                else existing_routing_history
            )
            return {
                **state,
                "agent_result": None,
                "files_modified": [],
                "subagent_used": None,
                # Phase 16.5.11.3: Orchestrator tracking
                "orchestrator_routing_decision": routing_decision_dict,
                "orchestrator_routing_history": updated_routing_history,
                "error": {
                    "error_type": "subagent_routing",
                    "message": f"Failed to route to subagent: {e}",
                    "node": "execute_task",
                    "task_id": task_id,
                    "recoverable": True,
                    "stack_trace": str(e),
                },
                **snapshot_update,
            }

    # -------------------------------------------------------------------------
    # Phase 1.1: Model routing - select optimal model based on task complexity
    # -------------------------------------------------------------------------
    execution_agent = agent
    routed_model = None
    if model_router is not None:
        from ai_infra.agent import Agent
        from ai_infra.executor.routing import TaskContext

        # Build context for routing
        task_plan = state.get("task_plan", {})
        task_context = TaskContext(
            previous_failures=state.get("retry_count", 0),
            similar_task_failed=False,  # TODO: Check project memory
            file_count_estimate=len(task_plan.get("likely_files", [])) or 1,
            dependency_count=len(task_plan.get("dependencies", [])),
            complexity_from_plan=task_plan.get("complexity"),
        )

        # Get routed model configuration
        model_config = model_router.select_model(current_task, task_context)
        routed_model = model_config.model_name

        # Record routing decision
        if routing_metrics is not None:
            complexity_score = model_router._compute_complexity_score(current_task, task_context)
            routing_decision = routing_metrics.record_decision(
                current_task, model_config, complexity_score
            )
        else:
            routing_decision = None

        # Create agent with routed model
        # Preserve tools from default agent if available
        if agent is not None:
            execution_agent = Agent(
                model=model_config.model_name,
                max_tokens=model_config.max_tokens,
                tools=getattr(agent, "_tools", None),
                system_prompt=getattr(agent, "_system_prompt", None),
            )
        else:
            execution_agent = Agent(
                model=model_config.model_name,
                max_tokens=model_config.max_tokens,
            )

        logger.info(
            f"Routed task [{task_id}] to model={routed_model} (tier={model_config.tier.value})"
        )
    else:
        routing_decision = None

    logger.info(f"Executing task [{task_id}]" + (f" with {routed_model}" if routed_model else ""))

    # Determine if we should check for destructive ops (parameter or state)
    should_pause_destructive = pause_destructive or state.get("pause_destructive", True)

    # Get workspace root for file path resolution
    workspace_root = state.get("roadmap_path")
    if workspace_root:
        workspace_root = str(Path(workspace_root).parent)

    try:
        # Execute with timeout
        import time

        execution_start = time.time()
        result = await asyncio.wait_for(
            execution_agent.arun(prompt),
            timeout=NodeTimeouts.EXECUTE_TASK,
        )
        execution_latency_ms = (time.time() - execution_start) * 1000

        # Phase 1.1: Record routing outcome
        if routing_decision is not None and routing_metrics is not None:
            routing_metrics.record_outcome(
                routing_decision,
                actual_latency_ms=execution_latency_ms,
                success=True,
            )
            logger.debug(
                f"Routing outcome: task={task_id}, "
                f"latency={execution_latency_ms:.1f}ms, success=True"
            )

        # Extract files modified (implementation-specific)
        files_modified = _extract_files_modified(result, execution_agent)

        # Phase 2.3.3: Check for destructive operations
        # Check BOTH agent result AND actual file contents on disk
        if should_pause_destructive:
            # Check agent response
            destructive_ops = check_agent_result_for_destructive_ops(result)

            # Also check actual file contents written to disk
            # This catches cases where the pattern is in the file but not in the response
            if files_modified:
                file_ops = check_files_for_destructive_ops(files_modified, workspace_root)
                destructive_ops.extend(file_ops)

            # Additionally, scan workspace for recently modified files
            # (agent may create files not tracked in files_modified)
            # Only do this if workspace_root looks like a valid execution scenario
            # (has .executor directory) to avoid scanning source code directories
            if workspace_root and _is_valid_execution_workspace(workspace_root):
                recent_files = _find_recently_modified_files(workspace_root)
                if recent_files:
                    recent_ops = check_files_for_destructive_ops(recent_files, workspace_root)
                    # Deduplicate by match text
                    existing_matches = {op.match for op in destructive_ops}
                    for op in recent_ops:
                        if op.match not in existing_matches:
                            destructive_ops.append(op)

            if destructive_ops:
                op_descriptions = [op.description for op in destructive_ops]
                warning_message = format_destructive_warning(destructive_ops)

                logger.warning(
                    f"Task [{task_id}] contains destructive operations: {op_descriptions}"
                )

                snapshot_update = await _finalize_task_snapshot(
                    task_id=task_id,
                    workspace_dir=workspace_dir,
                    pre_snapshot=pre_snapshot,
                    pre_snapshot_path=pre_snapshot_path,
                    enabled=enable_shell_snapshots,
                )

                return {
                    **state,
                    "agent_result": None,  # Don't apply result yet
                    "files_modified": [],
                    "interrupt_requested": True,
                    "pause_reason": warning_message,
                    "detected_destructive_ops": op_descriptions,
                    "pending_result": {
                        "result": result,
                        "files_modified": files_modified,
                    },
                    "error": None,
                    **snapshot_update,
                }

        # Phase 2.3.2: Extract shell results from session history
        shell_results, shell_error = _extract_shell_results()

        # Phase 2.3.3: Log shell errors but don't fail the task
        # The agent can interpret shell output and decide how to proceed
        if shell_error:
            logger.warning(
                f"Task [{task_id}] had shell command failure: {shell_error.get('command', 'unknown')} "
                f"(exit_code={shell_error.get('exit_code', -1)})"
            )

        logger.info(
            f"Task [{task_id}] executed successfully. Files modified: {len(files_modified)}, "
            f"Shell commands: {len(shell_results)}"
        )

        # Phase 2.3.1: Check if shell session is active
        session = get_current_session()
        shell_session_active = session is not None and getattr(session, "is_running", False)

        # Merge new shell results with existing ones
        existing_shell_results = state.get("shell_results", [])

        snapshot_update = await _finalize_task_snapshot(
            task_id=task_id,
            workspace_dir=workspace_dir,
            pre_snapshot=pre_snapshot,
            pre_snapshot_path=pre_snapshot_path,
            enabled=enable_shell_snapshots,
        )

        return {
            **state,
            "agent_result": result,
            "files_modified": files_modified,
            "error": None,
            # Clear any previous pause state
            "pause_reason": None,
            "detected_destructive_ops": None,
            "pending_result": None,
            # Phase 2.3: Shell tool integration
            "shell_session_active": shell_session_active,
            "shell_results": existing_shell_results + shell_results,
            "shell_error": shell_error,
            **snapshot_update,
        }

    except TimeoutError:
        logger.error(f"Task [{task_id}] execution timed out")
        # Phase 1.1: Record routing failure
        if routing_decision is not None and routing_metrics is not None:
            routing_metrics.record_outcome(
                routing_decision,
                actual_latency_ms=NodeTimeouts.EXECUTE_TASK * 1000,
                success=False,
            )
        snapshot_update = await _finalize_task_snapshot(
            task_id=task_id,
            workspace_dir=workspace_dir,
            pre_snapshot=pre_snapshot,
            pre_snapshot_path=pre_snapshot_path,
            enabled=enable_shell_snapshots,
        )
        return {
            **state,
            "agent_result": None,
            "files_modified": [],
            "error": {
                "error_type": "timeout",
                "message": f"Execution timed out after {NodeTimeouts.EXECUTE_TASK}s",
                "node": "execute_task",
                "task_id": task_id,
                "recoverable": True,
                "stack_trace": None,
            },
            **snapshot_update,
        }

    except asyncio.CancelledError:
        # Handle cancellation (interrupt)
        logger.warning(f"Task [{task_id}] execution was cancelled")
        # Phase 1.1: Record routing failure
        if routing_decision is not None and routing_metrics is not None:
            routing_metrics.record_outcome(
                routing_decision,
                actual_latency_ms=0,
                success=False,
            )
        snapshot_update = await _finalize_task_snapshot(
            task_id=task_id,
            workspace_dir=workspace_dir,
            pre_snapshot=pre_snapshot,
            pre_snapshot_path=pre_snapshot_path,
            enabled=enable_shell_snapshots,
        )
        return {
            **state,
            "agent_result": None,
            "files_modified": [],
            "interrupt_requested": True,
            "error": {
                "error_type": "cancelled",
                "message": "Execution was cancelled",
                "node": "execute_task",
                "task_id": task_id,
                "recoverable": False,
                "stack_trace": None,
            },
            **snapshot_update,
        }

    except Exception as e:
        error_type = type(e).__name__
        is_recoverable = not NonRetryableErrors.is_non_retryable(str(e))

        logger.exception(f"Task [{task_id}] execution failed: {e}")

        # Phase 1.1: Record routing failure
        if routing_decision is not None and routing_metrics is not None:
            routing_metrics.record_outcome(
                routing_decision,
                actual_latency_ms=0,
                success=False,
            )

        import traceback

        snapshot_update = await _finalize_task_snapshot(
            task_id=task_id,
            workspace_dir=workspace_dir,
            pre_snapshot=pre_snapshot,
            pre_snapshot_path=pre_snapshot_path,
            enabled=enable_shell_snapshots,
        )

        return {
            **state,
            "agent_result": None,
            "files_modified": [],
            "error": {
                "error_type": error_type,
                "message": str(e),
                "node": "execute_task",
                "task_id": task_id,
                "recoverable": is_recoverable,
                "stack_trace": traceback.format_exc(),
            },
            **snapshot_update,
        }


# =============================================================================
# Phase 1.2: Token Streaming
# =============================================================================


async def execute_task_streaming(
    state: ExecutorGraphState,
    *,
    agent: Agent | None = None,
    streaming_config: StreamingConfig | None = None,
    model_router: ModelRouter | None = None,
    routing_metrics: RoutingMetrics | None = None,
) -> AsyncIterator[ExecutorStreamEvent]:
    """Execute task with token streaming.

    Phase 1.2: Streams LLM tokens through the executor for real-time output.

    This generator yields ExecutorStreamEvents for each LLM token, tool call,
    and completion. The final state update is NOT yielded - use
    execute_task_node() for the state update.

    Args:
        state: Current graph state with prompt.
        agent: The Agent instance to use for execution.
        streaming_config: Streaming configuration.
        model_router: Optional model router for task-based model selection.
        routing_metrics: Optional routing metrics tracker.

    Yields:
        ExecutorStreamEvent for each LLM token, tool, thinking, done, or error.

    Example:
        ```python
        async for event in execute_task_streaming(state, agent=agent):
            if event.event_type == StreamEventType.LLM_TOKEN:
                print(event.data["content"], end="", flush=True)
        ```
    """
    config = streaming_config or StreamingConfig()
    current_task = state.get("current_task")
    prompt = state.get("prompt", "")
    task_id = str(current_task.id) if current_task else "unknown"

    if not prompt:
        yield create_llm_error_event(
            error="No prompt available for execution",
            node_name="execute_task",
        )
        return

    if agent is None and model_router is None:
        yield create_llm_error_event(
            error="No agent or router provided for execution",
            node_name="execute_task",
        )
        return

    # Phase 1.1: Model routing - select optimal model based on task complexity
    execution_agent = agent
    routed_model = None
    if model_router is not None:
        from ai_infra.agent import Agent
        from ai_infra.executor.routing import TaskContext

        # Build context for routing
        task_plan = state.get("task_plan", {})
        task_context = TaskContext(
            previous_failures=state.get("retry_count", 0),
            similar_task_failed=False,
            file_count_estimate=len(task_plan.get("likely_files", [])) or 1,
            dependency_count=len(task_plan.get("dependencies", [])),
            complexity_from_plan=task_plan.get("complexity"),
        )

        # Get routed model configuration
        model_config = model_router.select_model(current_task, task_context)
        routed_model = model_config.model_name

        # Create agent with routed model
        if agent is not None:
            execution_agent = Agent(
                model=model_config.model_name,
                max_tokens=model_config.max_tokens,
                tools=getattr(agent, "_tools", None),
                system_prompt=getattr(agent, "_system_prompt", None),
            )
        else:
            execution_agent = Agent(
                model=model_config.model_name,
                max_tokens=model_config.max_tokens,
            )

        logger.info(
            f"Streaming task [{task_id}] with routed model={routed_model} "
            f"(tier={model_config.tier.value})"
        )
    else:
        logger.info(f"Streaming task [{task_id}]")

    try:
        # Stream tokens from agent
        visibility = config.token_visibility
        tools_called = 0

        async for event in execution_agent.astream(prompt, visibility=visibility):
            # Map StreamEvent types to ExecutorStreamEvent types
            if event.type == "thinking":
                if config.show_llm_thinking:
                    yield create_llm_thinking_event(
                        model=event.model,
                        node_name="execute_task",
                    )

            elif event.type == "token":
                if config.stream_tokens and event.content:
                    yield create_llm_token_event(
                        content=event.content,
                        node_name="execute_task",
                    )

            elif event.type == "tool_start":
                if config.show_llm_tools:
                    tools_called += 1
                    yield create_llm_tool_start_event(
                        tool=event.tool or "unknown",
                        tool_id=event.tool_id,
                        arguments=event.arguments if visibility in ("detailed", "debug") else None,
                        node_name="execute_task",
                    )

            elif event.type == "tool_end":
                if config.show_llm_tools:
                    yield create_llm_tool_end_event(
                        tool=event.tool or "unknown",
                        tool_id=event.tool_id,
                        latency_ms=event.latency_ms,
                        result=event.result if visibility in ("detailed", "debug") else None,
                        preview=event.preview if visibility == "debug" else None,
                        node_name="execute_task",
                    )

            elif event.type == "done":
                yield create_llm_done_event(
                    tools_called=event.tools_called or tools_called,
                    node_name="execute_task",
                )

            elif event.type == "error":
                yield create_llm_error_event(
                    error=event.error or "Unknown streaming error",
                    node_name="execute_task",
                )

    except asyncio.CancelledError:
        yield create_llm_error_event(
            error="Streaming was cancelled",
            node_name="execute_task",
        )
        raise

    except Exception as e:
        logger.exception(f"Task [{task_id}] streaming failed: {e}")
        yield create_llm_error_event(
            error=str(e),
            node_name="execute_task",
        )


def _extract_shell_results() -> tuple[list[dict[str, Any]], ShellError | None]:
    """Extract shell results from the current session.

    Phase 2.3.2: Retrieves command history from the active shell session
    and formats it for state storage.

    Phase 2.3.3: Also extracts the last shell error (if any) for graceful
    failure handling. Non-zero exit codes are captured but don't fail the task.

    Returns:
        Tuple of (shell_results list, shell_error or None).
        - shell_results: List of dicts with command, exit_code, stdout, stderr, duration_ms, timed_out
        - shell_error: ShellError if the last command failed, None otherwise
    """
    session = get_current_session()
    if session is None:
        return [], None

    results: list[dict[str, Any]] = []
    last_error: ShellError | None = None

    # Get command history from session
    history = getattr(session, "command_history", [])
    for shell_result in history:
        result_dict: dict[str, Any] = {
            "command": shell_result.command,
            "exit_code": shell_result.exit_code,
            "stdout": shell_result.stdout,
            "stderr": shell_result.stderr,
            "duration_ms": shell_result.duration_ms,
            "timed_out": shell_result.timed_out,
        }
        results.append(result_dict)

        # Phase 2.3.3: Track failures but don't fail the task
        # Non-zero exit code or timeout is captured as shell_error
        if not shell_result.success:
            last_error = ShellError(
                command=shell_result.command,
                exit_code=shell_result.exit_code,
                stderr=shell_result.stderr,
                stdout=shell_result.stdout,
                cwd=None,  # Session doesn't expose cwd per-command
                timed_out=shell_result.timed_out,
            )

    return results, last_error


def _extract_files_modified(result: Any, agent: Any) -> list[str]:
    """Extract list of files modified from agent execution result.

    This inspects the agent's internal state to determine what files
    were modified during execution.

    Args:
        result: The result from agent.arun().
        agent: The Agent instance.

    Returns:
        List of file paths that were modified.
    """
    files: list[str] = []

    # Try to get files from agent's tool usage history
    if hasattr(agent, "tool_calls"):
        for call in agent.tool_calls:
            if call.get("tool") in ("create_file", "edit_file", "replace_string_in_file"):
                file_path = call.get("args", {}).get("file_path")
                if file_path and file_path not in files:
                    files.append(file_path)

    # Try to get files from result if it has file info
    if hasattr(result, "files_modified"):
        for f in result.files_modified:
            if f not in files:
                files.append(f)

    # Try to get from result if it's a dict
    if isinstance(result, dict):
        result_files = result.get("files_modified", [])
        for f in result_files:
            if f not in files:
                files.append(f)

    return files


def _find_recently_modified_files(
    workspace_root: str,
    max_age_seconds: float = 120.0,
    extensions: tuple[str, ...] = (".py", ".sh", ".sql", ".yaml", ".yml"),
) -> list[str]:
    """Find files recently modified in the workspace.

    This helps catch files created/modified by the agent that aren't
    tracked in the files_modified list (e.g., due to agent implementation).

    Args:
        workspace_root: Root directory to scan.
        max_age_seconds: Only include files modified within this time window.
        extensions: File extensions to check.

    Returns:
        List of recently modified file paths.
    """
    import time

    recent_files: list[str] = []
    now = time.time()
    cutoff = now - max_age_seconds

    try:
        root_path = Path(workspace_root)
        if not root_path.exists():
            return []

        # Scan for recently modified files with relevant extensions
        for ext in extensions:
            for file_path in root_path.rglob(f"*{ext}"):
                try:
                    # Skip hidden directories and common non-source paths
                    if any(
                        part.startswith(".")
                        or part in ("node_modules", "__pycache__", "venv", ".venv")
                        for part in file_path.parts
                    ):
                        continue

                    mtime = file_path.stat().st_mtime
                    if mtime >= cutoff:
                        recent_files.append(str(file_path))
                except (OSError, PermissionError):
                    continue

    except Exception as e:
        logger.debug(f"Error scanning for recent files: {e}")

    return recent_files


def _is_valid_execution_workspace(workspace_root: str) -> bool:
    """Check if the workspace root is a valid execution scenario.

    This prevents scanning source code directories (like ai-infra itself)
    for destructive patterns. A valid execution workspace should be
    in a scenarios directory or temporary directory, NOT a source code directory.

    Args:
        workspace_root: Root directory to validate.

    Returns:
        True if workspace looks like a valid execution scenario.
    """
    try:
        root_path = Path(workspace_root).resolve()

        # FIRST: Skip known source code directories (these take priority)
        # This prevents scanning the ai-infra source which contains pattern examples
        source_indicators = [
            "pyproject.toml",
            "src/ai_infra",
            "src/svc_infra",
            "src/fin_infra",
        ]
        for indicator in source_indicators:
            if (root_path / indicator).exists():
                # Exception: if this is inside execution-testor/scenarios, allow it
                if "execution-testor" in str(root_path) and "scenarios" in str(root_path):
                    return True
                return False

        # Check if path contains "scenarios" (execution-testor scenarios)
        if "scenarios" in root_path.parts or "execution-testor" in str(root_path):
            return True

        # Check if path is in /tmp or temp directories (often used for tests)
        root_str = str(root_path)
        if root_str.startswith("/tmp") or root_str.startswith("/var/tmp"):  # nosec B108
            return True

        # Check for .executor directory (created by executor)
        # This is last because it might exist in source directories from previous runs
        if (root_path / ".executor").exists():
            # But only if this doesn't look like a source directory
            if not (root_path / ".git").exists():
                return True

        # Default: don't scan (fail closed for safety in unknown directories)
        return False

    except Exception:
        return False
