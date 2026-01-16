"""Executor state manager for CLI utilities.

This module provides a lightweight ExecutorManager class for CLI commands
that need to interact with executor state without running tasks.

Supports:
- Status queries (state, roadmap)
- Checkpoint operations (rollback)
- Review/resume workflow
- State reset

This is separate from ExecutorGraph which handles actual task execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.executor.checkpoint import Checkpointer
from ai_infra.executor.context import ProjectContext
from ai_infra.executor.parser import RoadmapParser
from ai_infra.executor.roadmap import Roadmap
from ai_infra.executor.state import ExecutorState, TaskStatus
from ai_infra.executor.todolist import TodoListManager
from ai_infra.executor.types import ReviewInfo
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.recovery import RollbackResult

logger = get_logger("executor.manager")


# =============================================================================
# Executor Manager
# =============================================================================


class ExecutorManager:
    """Lightweight state manager for executor CLI utilities.

    Provides access to executor state, roadmap, and checkpointer without
    the overhead of full task execution components.

    This class is for CLI commands like:
    - status: Query task status
    - reset: Clear executor state
    - rollback: Revert commits
    - review: Show pending changes
    - resume: Approve/reject changes
    - sync: Sync state to roadmap

    For running tasks, use ExecutorGraph directly.
    """

    def __init__(
        self,
        roadmap: str | Path,
        *,
        checkpointer: Checkpointer | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            roadmap: Path to the ROADMAP.md file.
            checkpointer: Git checkpointer (auto-created if in a git repo).
        """
        self._roadmap_path = Path(roadmap).resolve()
        self._checkpointer = checkpointer
        self._checkpointer_initialized = checkpointer is not None

        # Lazily initialized
        self._state: ExecutorState | None = None
        self._roadmap: Roadmap | None = None
        self._context: ProjectContext | None = None
        self._todo_manager: TodoListManager | None = None
        self._last_run_results: list[Any] = []  # For review workflow

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def roadmap_path(self) -> Path:
        """Get the roadmap file path."""
        return self._roadmap_path

    @property
    def state(self) -> ExecutorState:
        """Get the execution state."""
        if self._state is None:
            self._state = ExecutorState.load(self._roadmap_path)
        return self._state

    @property
    def roadmap(self) -> Roadmap:
        """Get the parsed roadmap."""
        if self._roadmap is None:
            parser = RoadmapParser()
            self._roadmap = parser.parse(self._roadmap_path)
        return self._roadmap

    @property
    def checkpointer(self) -> Checkpointer | None:
        """Get the git checkpointer."""
        if not self._checkpointer_initialized:
            self._checkpointer_initialized = True
            try:
                self._checkpointer = Checkpointer(self._roadmap_path.parent)
            except Exception:
                self._checkpointer = None
        return self._checkpointer

    @property
    def todo_manager(self) -> TodoListManager | None:
        """Get the todo list manager if initialized."""
        return self._todo_manager

    # =========================================================================
    # State Management
    # =========================================================================

    def reset(self) -> None:
        """Reset the executor state completely.

        Clears all task status and re-parses the roadmap.
        """
        logger.info("resetting_executor_state")
        self._state = None
        self._roadmap = None
        # Re-initialize state
        _ = self.state
        _ = self.roadmap
        logger.info("executor_state_reset", run_id=self.state.run_id)

    def sync_roadmap(self) -> int:
        """Sync completed task status back to ROADMAP.md.

        Returns:
            Number of checkboxes updated.
        """
        from ai_infra.executor.todolist import NormalizedTodoFile

        todos_json = self._roadmap_path.parent / ".executor" / "todos.json"
        if not todos_json.exists():
            return 0

        todo_file = NormalizedTodoFile.load(todos_json)
        if not self._todo_manager:
            self._todo_manager = TodoListManager(
                roadmap_path=self._roadmap_path,
                use_llm=False,
            )

        updated = self._todo_manager.sync_to_roadmap(todo_file.todos)
        return updated

    # =========================================================================
    # Review/Resume Workflow
    # =========================================================================

    def get_changes_for_review(self) -> ReviewInfo:
        """Get information about changes pending review.

        Returns:
            ReviewInfo with task IDs, files changed, and commit info.
        """
        # Find tasks that are IN_PROGRESS (paused for review)
        in_progress_tasks = [
            tid for tid in self.state._tasks if self.state.get_status(tid) == TaskStatus.IN_PROGRESS
        ]

        files_modified: list[str] = []
        files_created: list[str] = []
        files_deleted: list[str] = []
        commits: list[Any] = []

        # Get files from last run results
        for result in self._last_run_results:
            if hasattr(result, "files_modified"):
                files_modified.extend(result.files_modified)
            if hasattr(result, "files_created"):
                files_created.extend(result.files_created)
            if hasattr(result, "files_deleted"):
                files_deleted.extend(result.files_deleted)

        # Get commit info from checkpointer
        if self.checkpointer and in_progress_tasks:
            for task_id in in_progress_tasks:
                commit_info = self.checkpointer.get_commit_for_task(task_id)
                if commit_info:
                    commits.append(commit_info)

        return ReviewInfo(
            task_ids=in_progress_tasks,
            files_modified=list(set(files_modified)),
            files_created=list(set(files_created)),
            files_deleted=list(set(files_deleted)),
            commits=commits,
        )

    def resume(
        self,
        approved: bool = True,
        rollback: bool = False,
    ) -> RollbackResult | None:
        """Resume execution after review.

        Args:
            approved: Whether changes are approved.
            rollback: Whether to rollback changes if rejected.

        Returns:
            RollbackResult if rollback was requested and performed.
        """
        in_progress_tasks = [
            tid for tid in self.state._tasks if self.state.get_status(tid) == TaskStatus.IN_PROGRESS
        ]

        if approved:
            # Mark in-progress tasks as completed
            for task_id in in_progress_tasks:
                self.state.set_status(task_id, TaskStatus.COMPLETED)
            self.state.save()
            return None
        else:
            # Mark in-progress tasks as pending (reset)
            for task_id in in_progress_tasks:
                self.state.set_status(task_id, TaskStatus.PENDING)

            # Rollback if requested
            result = None
            if rollback and self.checkpointer and in_progress_tasks:
                # Rollback to before the first in-progress task
                result = self.checkpointer.rollback(in_progress_tasks[0])

            self.state.save()
            return result

    # =========================================================================
    # Backwards Compatibility
    # =========================================================================

    async def ensure_todo_manager(self) -> None:
        """Ensure todo manager is initialized.

        This is a no-op for the manager since ExecutorGraph handles
        todo management during actual execution.
        """
        pass
