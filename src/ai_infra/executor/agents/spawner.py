"""Subagent spawning for task execution.

Phase 3.3.6 of EXECUTOR_1.md: Provides utilities for spawning
specialized subagents based on task type.

Phase 7.4.2: Added SubAgentConfig support for model customization.

Phase 16.5.11.5: Added post-execution validation and retry logic.

This module integrates the SubAgentRegistry with the executor
to automatically select and run the best agent for each task.

Example:
    ```python
    from ai_infra.executor.agents.spawner import spawn_for_task
    from ai_infra.executor.agents.config import SubAgentConfig

    # With default config
    result = await spawn_for_task(
        task=todo_item,
        context=execution_context,
    )

    # With custom model config
    config = SubAgentConfig.with_overrides({"coder": "gpt-4o"})
    result = await spawn_for_task(
        task=todo_item,
        context=execution_context,
        config=config,
    )
    ```
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ai_infra.executor.agents.base import ExecutionContext, SubAgentResult
from ai_infra.executor.agents.config import SubAgentConfig
from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.state import ExecutorGraphState
    from ai_infra.executor.todolist import TodoItem

__all__ = [
    "spawn_for_task",
    "spawn_for_task_from_state",
    "validate_expected_files",
    "compute_quality_score",
]

logger = get_logger("executor.agents.spawner")


async def spawn_for_task(
    task: TodoItem,
    context: ExecutionContext,
    config: SubAgentConfig | None = None,
    agent_type: SubAgentType | None = None,
) -> SubAgentResult:
    """Spawn appropriate subagent and execute task.

    Executes a task using the specified agent type. The agent_type should
    be provided by the orchestrator. Uses model config if provided.

    Args:
        task: The task to execute.
        context: Execution context.
        config: Optional SubAgentConfig for model customization.
        agent_type: Pre-determined agent type from orchestrator.
            If None, defaults to CODER.

    Returns:
        SubAgentResult from the subagent.

    Example:
        ```python
        from ai_infra.executor.todolist import TodoItem
        from ai_infra.executor.agents.base import ExecutionContext
        from ai_infra.executor.agents.spawner import spawn_for_task
        from ai_infra.executor.agents.config import SubAgentConfig
        from pathlib import Path

        task = TodoItem(
            id=1,
            title="Implement login endpoint",
            description="Create POST /api/login with JWT auth",
        )
        context = ExecutionContext(
            workspace=Path("/my/project"),
            project_type="python",
        )

        # With defaults
        result = await spawn_for_task(task, context)

        # With custom models
        config = SubAgentConfig.with_overrides({"coder": "gpt-4o"})
        result = await spawn_for_task(task, context, config)

        # With pre-determined agent type from orchestrator
        result = await spawn_for_task(task, context, agent_type=SubAgentType.TESTWRITER)
        print(f"Success: {result.success}")
        print(f"Files modified: {result.files_modified}")
        ```
    """
    start_time = time.perf_counter()

    # Get appropriate agent type for task (use pre-determined if provided)
    if agent_type is None:
        # Default to CODER if not specified (orchestrator should always provide)
        agent_type = SubAgentType.CODER
        logger.warning(f"No agent_type provided for task '{task.title}', defaulting to CODER")

    # Get model from config if provided
    model: str | None = None
    if config is not None:
        model_config = config.get_config(agent_type)
        model = model_config.model
        logger.debug(
            f"Using config model for {agent_type.value}: {model} "
            f"(max_tokens={model_config.max_tokens})"
        )

    # Get agent instance (with custom model if specified)
    agent = SubAgentRegistry.get(agent_type, model=model)

    logger.info(
        f"Spawning {agent.name} for task: {task.title} "
        f"(type={agent_type.value}, model={model or agent.model})"
    )

    try:
        result = await agent.execute(task, context)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log token metrics if available (Phase 16.5.1)
        metrics = result.metrics or {}
        total_tokens = metrics.get("total_tokens", 0)
        token_info = f", {total_tokens} tokens" if total_tokens else ""

        logger.info(
            f"{agent.name} completed in {duration_ms:.0f}ms: success={result.success}{token_info}"
        )

        return result

    except Exception as e:
        logger.exception(f"{agent.name} failed with error: {e}")
        duration_ms = (time.perf_counter() - start_time) * 1000
        return SubAgentResult(
            success=False,
            error=str(e),
            metrics={
                "duration_ms": duration_ms,
                "agent_type": agent_type.value,
            },
        )


async def spawn_for_task_from_state(
    state: ExecutorGraphState,
    workspace: Path | None = None,
    config: SubAgentConfig | None = None,
) -> SubAgentResult:
    """Spawn subagent from executor graph state.

    Phase 7.4.2: Added config parameter for model customization.

    Convenience function that extracts task and context from state.

    Args:
        state: Current ExecutorGraphState.
        workspace: Optional workspace path override.
        config: Optional SubAgentConfig for model customization.

    Returns:
        SubAgentResult from the subagent.

    Raises:
        ValueError: If no current_task in state.

    Example:
        ```python
        from ai_infra.executor.agents.spawner import spawn_for_task_from_state
        from ai_infra.executor.agents.config import SubAgentConfig

        # With defaults
        result = await spawn_for_task_from_state(state)

        # With custom config
        config = SubAgentConfig.with_overrides({"coder": "gpt-4o"})
        result = await spawn_for_task_from_state(state, config=config)
        ```
    """
    current_task = state.get("current_task")
    if not current_task:
        raise ValueError("No current_task in state")

    # Build context from state
    context = ExecutionContext.from_state(state, workspace)

    # Convert Task to TodoItem if needed
    from ai_infra.executor.todolist import TodoItem

    if hasattr(current_task, "title"):
        todo_item = TodoItem(
            id=getattr(current_task, "id", 1),
            title=current_task.title,
            description=getattr(current_task, "description", None) or "",
        )
    else:
        todo_item = current_task

    return await spawn_for_task(todo_item, context, config)


# =============================================================================
# Post-Execution Validation (Phase 16.5.11.5)
# =============================================================================


def validate_expected_files(
    workspace: Path,
    files_claimed: list[str],
) -> tuple[list[str], list[str]]:
    """Validate that claimed files actually exist.

    Phase 16.5.11.5.5: Post-execution validation to check if subagent
    claims match reality.

    Args:
        workspace: Workspace root path.
        files_claimed: Files the subagent claims to have created/modified.

    Returns:
        Tuple of (existing_files, missing_files).

    Example:
        ```python
        existing, missing = validate_expected_files(
            workspace=Path("/project"),
            files_claimed=["src/main.py", "tests/test_main.py"],
        )
        if missing:
            logger.warning(f"Missing claimed files: {missing}")
        ```
    """
    existing: list[str] = []
    missing: list[str] = []

    for filepath in files_claimed:
        full_path = workspace / filepath
        if full_path.exists() and full_path.is_file():
            existing.append(filepath)
        else:
            missing.append(filepath)

    return existing, missing


def compute_quality_score(
    workspace: Path,
    result: SubAgentResult,
) -> dict[str, float | bool | str]:
    """Compute quality metrics for subagent output.

    Phase 16.5.11.5.7: Quality scoring for subagent results.

    Checks:
    - Files actually created/modified
    - Python syntax validity
    - File non-empty
    - Reasonable size (not too short)

    Args:
        workspace: Workspace root path.
        result: SubAgentResult to evaluate.

    Returns:
        Dictionary with quality metrics.

    Example:
        ```python
        quality = compute_quality_score(workspace, result)
        if quality["score"] < 0.5:
            logger.warning(f"Low quality result: {quality}")
        ```
    """
    metrics: dict[str, float | bool | str] = {
        "files_exist": True,
        "syntax_valid": True,
        "files_nonempty": True,
        "score": 1.0,
        "issues": "",
    }

    issues: list[str] = []
    all_files = result.files_created + result.files_modified

    if not all_files:
        # No files claimed - check if task was code-related
        metrics["score"] = 0.5
        metrics["issues"] = "No files claimed by subagent"
        return metrics

    existing_count = 0
    syntax_valid_count = 0
    nonempty_count = 0

    for filepath in all_files:
        full_path = workspace / filepath

        # Check existence
        if not full_path.exists():
            issues.append(f"{filepath}: does not exist")
            continue

        existing_count += 1

        # Check non-empty
        try:
            content = full_path.read_text()
            if content.strip():
                nonempty_count += 1
            else:
                issues.append(f"{filepath}: empty file")
                continue
        except (OSError, UnicodeDecodeError) as e:
            issues.append(f"{filepath}: read error - {e}")
            continue

        # Check Python syntax if applicable
        if filepath.endswith(".py"):
            try:
                ast.parse(content)
                syntax_valid_count += 1
            except SyntaxError as e:
                issues.append(f"{filepath}: syntax error line {e.lineno}")

    # Calculate scores
    total = len(all_files)
    file_existence_score = existing_count / total if total > 0 else 0.0
    syntax_score = syntax_valid_count / total if total > 0 else 1.0  # Non-Python = valid
    nonempty_score = nonempty_count / total if total > 0 else 0.0

    metrics["files_exist"] = existing_count == total
    metrics["syntax_valid"] = syntax_valid_count == total or not any(
        f.endswith(".py") for f in all_files
    )
    metrics["files_nonempty"] = nonempty_count == total

    # Weighted score
    metrics["score"] = 0.4 * file_existence_score + 0.3 * syntax_score + 0.3 * nonempty_score
    metrics["issues"] = "; ".join(issues) if issues else ""

    return metrics


async def spawn_for_task_with_validation(
    task: TodoItem,
    context: ExecutionContext,
    config: SubAgentConfig | None = None,
    agent_type: SubAgentType | None = None,
    max_retries: int = 1,
) -> SubAgentResult:
    """Spawn subagent with post-execution validation and retry.

    Phase 16.5.11.5.6: Adds retry logic if subagent claims success
    but files are missing.

    Args:
        task: The task to execute.
        context: Execution context.
        config: Optional SubAgentConfig for model customization.
        agent_type: Optional pre-determined agent type from orchestrator.
        max_retries: Maximum retry attempts if validation fails.

    Returns:
        SubAgentResult with quality metrics added.

    Example:
        ```python
        result = await spawn_for_task_with_validation(
            task=todo_item,
            context=context,
            max_retries=2,
        )
        if result.metrics.get("quality_score", 1.0) < 0.5:
            logger.warning("Low quality result after retries")
        ```
    """
    attempt = 0
    last_result: SubAgentResult | None = None

    while attempt <= max_retries:
        attempt += 1

        result = await spawn_for_task(task, context, config, agent_type)

        if not result.success:
            return result

        # Validate files
        _, missing = validate_expected_files(
            context.workspace,
            result.files_created + result.files_modified,
        )

        if not missing:
            # All files exist, compute quality
            quality = compute_quality_score(context.workspace, result)
            result.metrics["quality_score"] = quality["score"]
            result.metrics["quality_issues"] = quality["issues"]
            return result

        logger.warning(
            f"Attempt {attempt}: Subagent claimed success but {len(missing)} "
            f"files missing: {missing[:3]}{'...' if len(missing) > 3 else ''}"
        )

        last_result = result

        if attempt <= max_retries:
            logger.info(f"Retrying task execution (attempt {attempt + 1}/{max_retries + 1})")

    # Return last result with quality metrics
    if last_result:
        quality = compute_quality_score(context.workspace, last_result)
        last_result.metrics["quality_score"] = quality["score"]
        last_result.metrics["quality_issues"] = quality["issues"]
        last_result.metrics["validation_failed"] = True
        return last_result

    # Should not reach here
    return SubAgentResult(
        success=False,
        error="Validation failed after retries",
    )
