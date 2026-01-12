"""Execute task node for executor graph.

Phase 1.2.2: Executes the current task using the Agent.
Phase 2.3.2: Dry run mode support.
Phase 2.3.3: Pause before destructive operations.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.executor.state import ExecutorGraphState, NodeTimeouts, NonRetryableErrors
from ai_infra.executor.utils.safety import (
    check_agent_result_for_destructive_ops,
    check_files_for_destructive_ops,
    format_destructive_warning,
)
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.agent import Agent

logger = get_logger("executor.nodes.execute")


async def execute_task_node(
    state: ExecutorGraphState,
    *,
    agent: Agent | None = None,
    dry_run: bool = False,
    pause_destructive: bool = True,
) -> ExecutorGraphState:
    """Execute the current task using the Agent.

    This node:
    1. Gets prompt from state
    2. Calls agent.arun() with timeout (unless dry_run)
    3. Checks result for destructive operations (if pause_destructive)
    4. Extracts files_modified from agent result
    5. Updates state with execution results

    Phase 2.3.2: Supports dry_run mode to preview actions without executing.
    Phase 2.3.3: Pauses execution if destructive operations are detected.

    Args:
        state: Current graph state with prompt.
        agent: The Agent instance to use for execution.
        dry_run: If True, log planned actions without executing.
        pause_destructive: If True, interrupt on destructive operations.

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

    if agent is None:
        logger.error("No agent provided for execution")
        return {
            **state,
            "agent_result": None,
            "files_modified": [],
            "error": {
                "error_type": "execution",
                "message": "No agent provided for execution",
                "node": "execute_task",
                "task_id": str(current_task.id) if current_task else None,
                "recoverable": False,
                "stack_trace": None,
            },
        }

    task_id = str(current_task.id) if current_task else "unknown"
    logger.info(f"Executing task [{task_id}]")

    # Determine if we should check for destructive ops (parameter or state)
    should_pause_destructive = pause_destructive or state.get("pause_destructive", True)

    # Get workspace root for file path resolution
    workspace_root = state.get("roadmap_path")
    if workspace_root:
        workspace_root = str(Path(workspace_root).parent)

    try:
        # Execute with timeout
        result = await asyncio.wait_for(
            agent.arun(prompt),
            timeout=NodeTimeouts.EXECUTE_TASK,
        )

        # Extract files modified (implementation-specific)
        files_modified = _extract_files_modified(result, agent)

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
                }

        logger.info(
            f"Task [{task_id}] executed successfully. Files modified: {len(files_modified)}"
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
        }

    except TimeoutError:
        logger.error(f"Task [{task_id}] execution timed out")
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
        }

    except asyncio.CancelledError:
        # Handle cancellation (interrupt)
        logger.warning(f"Task [{task_id}] execution was cancelled")
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
        }

    except Exception as e:
        error_type = type(e).__name__
        is_recoverable = not NonRetryableErrors.is_non_retryable(str(e))

        logger.exception(f"Task [{task_id}] execution failed: {e}")

        import traceback

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
        }


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
        if root_str.startswith("/tmp") or root_str.startswith("/var/tmp"):
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
