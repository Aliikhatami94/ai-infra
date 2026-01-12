"""Checkpoint node for executor graph.

Phase 1.2.2: Creates git checkpoints after successful task verification.
Phase 1.4.1: Integrates git checkpointer with graph.
Phase 1.4.2: Integrates TodoListManager for ROADMAP sync.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from ai_infra.executor.state import ExecutorGraphState, NodeTimeouts
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.checkpoint import Checkpointer
    from ai_infra.executor.project_memory import ProjectMemory
    from ai_infra.executor.run_memory import RunMemory
    from ai_infra.executor.todolist import TodoListManager

logger = get_logger("executor.nodes.checkpoint")


async def checkpoint_node(
    state: ExecutorGraphState,
    *,
    checkpointer: Checkpointer | None = None,
    todo_manager: TodoListManager | None = None,
    run_memory: RunMemory | None = None,
    project_memory: ProjectMemory | None = None,
    sync_roadmap: bool = True,
) -> ExecutorGraphState:
    """Create a git checkpoint after successful task verification.

    Phase 1.4.1: Git checkpointer integration.
    Phase 1.4.2: TodoListManager integration for ROADMAP sync.
    Phase 2.1.1: Records task outcome to RunMemory for context injection.
    Phase 2.1.2: Updates ProjectMemory with run outcomes for cross-run insights.

    This node:
    1. Gets current_task and files_modified from state
    2. Creates a git commit with descriptive message
    3. Updates last_checkpoint_sha in state
    4. Marks todo as completed and syncs to ROADMAP
    5. Records task outcome to RunMemory for subsequent tasks
    6. Updates ProjectMemory after task completion (Phase 2.1.2)

    Note: Graph state persistence is handled by decide_next node (after todos updated).

    Args:
        state: Current graph state with current_task and files_modified.
        checkpointer: The Checkpointer instance for git operations.
        todo_manager: The TodoListManager for ROADMAP sync.
        run_memory: RunMemory instance for recording task outcomes.
        project_memory: ProjectMemory instance for cross-run insights (Phase 2.1.2).
        sync_roadmap: Whether to sync completion to ROADMAP.md.

    Returns:
        Updated state with last_checkpoint_sha.
    """
    current_task = state.get("current_task")
    files_modified = state.get("files_modified", [])
    task_id = str(current_task.id) if current_task else "unknown"

    if current_task is None:
        logger.error("No current task for checkpoint")
        return {
            **state,
            "error": {
                "error_type": "checkpoint",
                "message": "No current task for checkpoint",
                "node": "checkpoint",
                "task_id": None,
                "recoverable": False,
                "stack_trace": None,
            },
        }

    # Phase 1.4.2: Always sync todo completion, regardless of checkpointer
    # This ensures ROADMAP checkboxes are updated even without git
    if todo_manager is not None:
        _sync_todo_completion(
            todo_manager=todo_manager,
            current_task=current_task,
            files_modified=files_modified,
            sync_roadmap=sync_roadmap,
        )

    # Phase 2.1.1: Record task outcome to RunMemory for context injection
    # This happens regardless of git checkpointing
    if run_memory is not None:
        _record_task_outcome(
            run_memory=run_memory,
            current_task=current_task,
            files_modified=files_modified,
            status="completed",
        )

        # Phase 2.1.2: Update ProjectMemory with run outcomes for cross-run insights
        # This happens after task outcome is recorded to run_memory
        if project_memory is not None:
            _update_project_memory(
                project_memory=project_memory,
                run_memory=run_memory,
                current_task=current_task,
            )

    # If no checkpointer, skip git checkpoint (development mode)
    if checkpointer is None:
        logger.warning(f"No checkpointer provided, skipping git checkpoint for [{task_id}]")
        return {**state, "error": None}

    # If no files modified, nothing to git checkpoint
    if not files_modified:
        logger.info(f"No files modified for task [{task_id}], skipping git checkpoint")
        return {**state, "error": None}

    logger.info(f"Creating checkpoint for task [{task_id}] ({len(files_modified)} files)")

    try:
        # Build commit message
        commit_message = _build_commit_message(current_task, files_modified)

        # Create checkpoint with timeout
        # Run synchronous checkpoint in executor to avoid blocking
        loop = asyncio.get_event_loop()
        commit_info = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: checkpointer.checkpoint(
                    message=commit_message,
                    files=files_modified,
                ),
            ),
            timeout=NodeTimeouts.CHECKPOINT,
        )

        sha = commit_info.sha if commit_info else None

        if sha:
            logger.info(f"Checkpoint created for task [{task_id}]: {sha[:8]}")
            return {**state, "last_checkpoint_sha": sha, "error": None}
        else:
            # Checkpoint returned None (nothing to commit)
            logger.info(f"No changes to checkpoint for task [{task_id}]")
            return {**state, "error": None}

    except TimeoutError:
        logger.error(f"Checkpoint timed out for task [{task_id}]")
        return {
            **state,
            "error": {
                "error_type": "timeout",
                "message": f"Checkpoint timed out after {NodeTimeouts.CHECKPOINT}s",
                "node": "checkpoint",
                "task_id": task_id,
                "recoverable": True,
                "stack_trace": None,
            },
        }

    except Exception as e:
        logger.exception(f"Checkpoint failed for task [{task_id}]: {e}")

        import traceback

        return {
            **state,
            "error": {
                "error_type": "checkpoint",
                "message": str(e),
                "node": "checkpoint",
                "task_id": task_id,
                "recoverable": True,  # Can retry
                "stack_trace": traceback.format_exc(),
            },
        }


def _sync_todo_completion(
    todo_manager: TodoListManager | None,
    current_task: object,
    files_modified: list[str],
    sync_roadmap: bool,
) -> None:
    """Sync todo completion to TodoListManager and optionally ROADMAP.

    Phase 1.4.2: TodoListManager integration.
    Phase 1.3.1: Also saves to .executor/todos.json.

    Args:
        todo_manager: The TodoListManager instance.
        current_task: The completed task.
        files_modified: List of modified file paths.
        sync_roadmap: Whether to sync to ROADMAP.md.
    """
    if todo_manager is None:
        return

    todo_id = getattr(current_task, "id", None)
    if todo_id is None:
        return

    try:
        # Mark todo as completed with files created
        todo_manager.mark_completed(
            todo_id=todo_id,
            files_created=files_modified,
            sync_roadmap=sync_roadmap,
        )
        logger.info(f"Todo [{todo_id}] marked completed, sync_roadmap={sync_roadmap}")

        # Phase 1.3.1: Also save to .executor/todos.json
        saved_path = todo_manager.save_to_json(create_if_missing=True)
        if saved_path:
            logger.debug(f"Todo state saved to {saved_path}")
    except Exception as e:
        # Log but don't fail - git checkpoint already succeeded
        logger.warning(f"Failed to sync todo completion: {e}")


def _build_commit_message(task: object, files_modified: list[str]) -> str:
    """Build a descriptive commit message for the checkpoint.

    Args:
        task: The completed task.
        files_modified: List of modified file paths.

    Returns:
        Formatted commit message.
    """
    task_id = getattr(task, "id", "?")
    title = getattr(task, "title", "Task completed")

    # Truncate title if too long
    if len(title) > 50:
        title = title[:47] + "..."

    lines = [
        f"[executor] {title}",
        "",
        f"Task ID: {task_id}",
        "",
        "Files modified:",
    ]

    # Add files (limit to 10)
    for f in files_modified[:10]:
        lines.append(f"  - {f}")

    if len(files_modified) > 10:
        lines.append(f"  ... and {len(files_modified) - 10} more")

    return "\n".join(lines)


def _record_task_outcome(
    run_memory: RunMemory,
    current_task: object,
    files_modified: list[str],
    status: str = "completed",
) -> None:
    """Record task outcome to RunMemory for context injection.

    Phase 2.1.1: Records completed tasks to RunMemory so subsequent tasks
    can see what was done before.

    Args:
        run_memory: The RunMemory instance to record to.
        current_task: The completed task.
        files_modified: List of modified file paths.
        status: Task completion status ("completed", "failed", "skipped").
    """
    from ai_infra.executor.run_memory import FileAction, TaskOutcome

    task_id = getattr(current_task, "id", "?")
    title = getattr(current_task, "title", "Unknown task")
    description = getattr(current_task, "description", "")

    try:
        # Build files dict with actions (assume all are created/modified)
        files: dict[Path, FileAction] = {}
        for f in files_modified:
            path = Path(f)
            # Simple heuristic: if it's a new file we created, mark as CREATED
            # In practice, both CREATED and MODIFIED are useful context
            files[path] = FileAction.MODIFIED

        # Create outcome
        outcome = TaskOutcome(
            task_id=str(task_id),
            title=title,
            status=status,
            files=files,
            key_decisions=[],  # Could be extracted from agent response in future
            summary=description if description else f"Completed task: {title}",
            duration_seconds=0.0,  # TODO: Track in Phase 2.2.1
            tokens_used=0,  # TODO: Track in Phase 2.2.1
        )

        # Record to run memory
        run_memory.add_outcome(outcome)

        logger.info(
            f"Recorded outcome for task [{task_id}] to run memory "
            f"({len(files_modified)} files, status={status})"
        )

    except Exception as e:
        # Log but don't fail - this is supplementary context
        logger.warning(f"Failed to record task outcome to run memory: {e}")


def _update_project_memory(
    project_memory: ProjectMemory,
    run_memory: RunMemory,
    current_task: object,
) -> None:
    """Update ProjectMemory with run outcomes for cross-run insights.

    Phase 2.1.2: Updates project memory after each task completion so that
    subsequent runs benefit from accumulated knowledge.

    Args:
        project_memory: The ProjectMemory instance to update.
        run_memory: The current RunMemory with task outcomes.
        current_task: The completed task.
    """
    task_id = getattr(current_task, "id", "?")

    try:
        # Update project memory from the current run
        # This incrementally saves key files, lessons learned, and run history
        project_memory.update_from_run(run_memory)

        logger.info(
            f"Updated project memory after task [{task_id}]: "
            f"{len(project_memory.key_files)} key files tracked"
        )

    except Exception as e:
        # Log but don't fail - this is supplementary context
        logger.warning(f"Failed to update project memory: {e}")
