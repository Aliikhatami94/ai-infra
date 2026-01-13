"""State management for the Executor module.

This module provides persistent state tracking for task execution:
- ExecutorState for managing task states
- JSON sidecar file storage
- Crash recovery
- ROADMAP.md synchronization
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from ai_infra.executor.failure import FailureCategory
from ai_infra.executor.models import TaskStatus
from ai_infra.executor.parser import RoadmapParser
from ai_infra.executor.todolist import TodoItem
from ai_infra.logging import get_logger

logger = get_logger("executor.state")


# =============================================================================
# Constants
# =============================================================================

STATE_VERSION = 1
STATE_DIR_NAME = ".executor"
STATE_FILE_NAME = "state.json"


# =============================================================================
# Task State
# =============================================================================


@dataclass
class TaskState:
    """State of a single task.

    Attributes:
        status: Current task status.
        started_at: When the task was started.
        completed_at: When the task was completed (if completed).
        failed_at: When the task failed (if failed).
        files_modified: List of files modified by this task.
        error: Error message if the task failed.
        failure_category: Category of failure if failed.
        token_usage: Token usage for this task.
        attempts: Number of execution attempts.
        agent_run_id: ID of the agent run that executed this task.
    """

    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    files_modified: list[str] = field(default_factory=list)
    error: str | None = None
    failure_category: FailureCategory | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    attempts: int = 0
    agent_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "status": self.status.value,
            "attempts": self.attempts,
        }

        if self.started_at:
            result["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            result["completed_at"] = self.completed_at.isoformat()
        if self.failed_at:
            result["failed_at"] = self.failed_at.isoformat()
        if self.files_modified:
            result["files_modified"] = self.files_modified
        if self.error:
            result["error"] = self.error
        if self.failure_category:
            result["failure_category"] = self.failure_category.value
        if self.token_usage:
            result["token_usage"] = self.token_usage
        if self.agent_run_id:
            result["agent_run_id"] = self.agent_run_id

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskState:
        """Create from dictionary."""
        return cls(
            status=TaskStatus(data.get("status", "pending")),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            failed_at=datetime.fromisoformat(data["failed_at"]) if data.get("failed_at") else None,
            files_modified=data.get("files_modified", []),
            error=data.get("error"),
            failure_category=FailureCategory(data["failure_category"])
            if data.get("failure_category")
            else None,
            token_usage=data.get("token_usage", {}),
            attempts=data.get("attempts", 0),
            agent_run_id=data.get("agent_run_id"),
        )


# =============================================================================
# State Summary
# =============================================================================


@dataclass
class StateSummary:
    """Summary statistics for execution state.

    Attributes:
        pending: Number of pending tasks.
        in_progress: Number of in-progress tasks.
        completed: Number of completed tasks.
        failed: Number of failed tasks.
        total_tokens: Total tokens used across all tasks.
        total_attempts: Total number of execution attempts.
    """

    pending: int = 0
    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    total_tokens: int = 0
    total_attempts: int = 0

    @property
    def total(self) -> int:
        """Get total number of tasks."""
        return self.pending + self.in_progress + self.completed + self.failed

    @property
    def progress(self) -> float:
        """Get completion progress (0.0 to 1.0)."""
        if self.total == 0:
            return 1.0
        return self.completed / self.total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pending": self.pending,
            "in_progress": self.in_progress,
            "completed": self.completed,
            "failed": self.failed,
            "total_tokens": self.total_tokens,
            "total_attempts": self.total_attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateSummary:
        """Create from dictionary."""
        return cls(
            pending=data.get("pending", 0),
            in_progress=data.get("in_progress", 0),
            completed=data.get("completed", 0),
            failed=data.get("failed", 0),
            total_tokens=data.get("total_tokens", 0),
            total_attempts=data.get("total_attempts", 0),
        )


# =============================================================================
# Executor State
# =============================================================================


class ExecutorState:
    """Manages execution state for a ROADMAP.md file.

    State is persisted in a JSON sidecar file at `.executor/state.json`
    relative to the ROADMAP.md file.

    Example:
        >>> state = ExecutorState.load("./ROADMAP.md")
        >>> state.mark_started("1.1.1")
        >>> state.mark_completed("1.1.1", files_modified=["src/foo.py"])
        >>> state.save()

        >>> # Crash recovery
        >>> state.recover()  # Reset IN_PROGRESS -> PENDING

        >>> # Sync back to ROADMAP
        >>> state.sync_to_roadmap()

    Attributes:
        roadmap_path: Path to the ROADMAP.md file.
        run_id: Current run identifier.
        tasks: Dictionary of task states by task ID.
    """

    def __init__(
        self,
        roadmap_path: str | Path,
        *,
        run_id: str | None = None,
        roadmap_hash: str | None = None,
    ) -> None:
        """Initialize executor state.

        Args:
            roadmap_path: Path to the ROADMAP.md file.
            run_id: Run identifier (auto-generated if not provided).
            roadmap_hash: Hash of the roadmap content (auto-computed if not provided).
        """
        self._roadmap_path = Path(roadmap_path).resolve()
        self._run_id = run_id or self._generate_run_id()
        self._roadmap_hash = roadmap_hash or self._compute_roadmap_hash()
        self._tasks: dict[str, TaskState] = {}
        self._created_at = datetime.now(UTC)
        self._last_updated = self._created_at

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def roadmap_path(self) -> Path:
        """Get the roadmap file path."""
        return self._roadmap_path

    @property
    def run_id(self) -> str:
        """Get the current run identifier."""
        return self._run_id

    @property
    def roadmap_hash(self) -> str:
        """Get the roadmap content hash."""
        return self._roadmap_hash

    @property
    def state_dir(self) -> Path:
        """Get the state directory path."""
        return self._roadmap_path.parent / STATE_DIR_NAME

    @property
    def state_file(self) -> Path:
        """Get the state file path."""
        return self.state_dir / STATE_FILE_NAME

    @property
    def tasks(self) -> dict[str, TaskState]:
        """Get all task states."""
        return self._tasks.copy()

    @property
    def last_updated(self) -> datetime:
        """Get the last update timestamp."""
        return self._last_updated

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def load(cls, roadmap_path: str | Path) -> ExecutorState:
        """Load state from the sidecar file.

        If no state file exists, returns a fresh state initialized
        from the roadmap.

        Args:
            roadmap_path: Path to the ROADMAP.md file.

        Returns:
            Loaded or fresh ExecutorState.
        """
        roadmap_path = Path(roadmap_path).resolve()
        state_file = roadmap_path.parent / STATE_DIR_NAME / STATE_FILE_NAME

        if state_file.exists():
            try:
                with open(state_file, encoding="utf-8") as f:
                    data = json.load(f)
                return cls._from_dict(data, roadmap_path)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load state file: {e}. Creating fresh state.")

        return cls.from_roadmap(roadmap_path)

    @classmethod
    def from_roadmap(cls, roadmap_path: str | Path) -> ExecutorState:
        """Create fresh state from a ROADMAP.md file.

        Parses the roadmap and initializes task states based on
        checkbox status in the markdown.

        Args:
            roadmap_path: Path to the ROADMAP.md file.

        Returns:
            Fresh ExecutorState with tasks from roadmap.
        """
        roadmap_path = Path(roadmap_path).resolve()
        state = cls(roadmap_path)

        # Parse roadmap to get initial task states
        if roadmap_path.exists():
            parser = RoadmapParser()
            roadmap = parser.parse(roadmap_path)

            for task in roadmap.all_tasks():
                initial_status = task.status
                state._tasks[task.id] = TaskState(status=initial_status)

        return state

    @classmethod
    def _from_dict(cls, data: dict[str, Any], roadmap_path: Path) -> ExecutorState:
        """Create from dictionary (internal)."""
        state = cls(
            roadmap_path,
            run_id=data.get("run_id"),
            roadmap_hash=data.get("roadmap_hash"),
        )

        # Load task states
        for task_id, task_data in data.get("tasks", {}).items():
            state._tasks[task_id] = TaskState.from_dict(task_data)

        # Load timestamps
        if data.get("created_at"):
            state._created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_updated"):
            state._last_updated = datetime.fromisoformat(data["last_updated"])

        return state

    # =========================================================================
    # State Updates
    # =========================================================================

    def mark_started(
        self,
        task_id: str,
        *,
        agent_run_id: str | None = None,
    ) -> None:
        """Mark a task as started.

        Args:
            task_id: Task identifier.
            agent_run_id: ID of the agent run executing this task.
        """
        task_state = self._get_or_create_task(task_id)
        task_state.status = TaskStatus.IN_PROGRESS
        task_state.started_at = datetime.now(UTC)
        task_state.attempts += 1
        task_state.agent_run_id = agent_run_id
        self._touch()
        logger.info(f"Task {task_id} started (attempt {task_state.attempts})")

    def mark_completed(
        self,
        task_id: str,
        *,
        files_modified: list[str] | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """Mark a task as completed.

        Args:
            task_id: Task identifier.
            files_modified: List of files modified by this task.
            token_usage: Token usage for this task.
        """
        task_state = self._get_or_create_task(task_id)
        task_state.status = TaskStatus.COMPLETED
        task_state.completed_at = datetime.now(UTC)
        task_state.error = None
        task_state.failure_category = None

        if files_modified:
            task_state.files_modified = files_modified
        if token_usage:
            task_state.token_usage = token_usage

        self._touch()
        logger.info(f"Task {task_id} completed")

    def mark_failed(
        self,
        task_id: str,
        *,
        error: str,
        category: FailureCategory | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """Mark a task as failed.

        Args:
            task_id: Task identifier.
            error: Error message describing the failure.
            category: Failure category for analysis.
            token_usage: Token usage for this task.
        """
        task_state = self._get_or_create_task(task_id)
        task_state.status = TaskStatus.FAILED
        task_state.failed_at = datetime.now(UTC)
        task_state.error = error
        task_state.failure_category = category

        if token_usage:
            task_state.token_usage = token_usage

        self._touch()
        logger.warning(f"Task {task_id} failed: {error}")

    def reset_task(self, task_id: str) -> None:
        """Reset a task to pending status.

        Args:
            task_id: Task identifier.
        """
        task_state = self._get_or_create_task(task_id)
        task_state.status = TaskStatus.PENDING
        task_state.started_at = None
        task_state.completed_at = None
        task_state.failed_at = None
        task_state.error = None
        task_state.failure_category = None
        # Keep attempts count for history
        self._touch()
        logger.info(f"Task {task_id} reset to pending")

    def _get_or_create_task(self, task_id: str) -> TaskState:
        """Get or create a task state."""
        if task_id not in self._tasks:
            self._tasks[task_id] = TaskState()
        return self._tasks[task_id]

    def _touch(self) -> None:
        """Update the last_updated timestamp."""
        self._last_updated = datetime.now(UTC)

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_task_state(self, task_id: str) -> TaskState | None:
        """Get state for a specific task.

        Args:
            task_id: Task identifier.

        Returns:
            Task state or None if not found.
        """
        return self._tasks.get(task_id)

    def get_status(self, task_id: str) -> TaskStatus:
        """Get status for a specific task.

        Args:
            task_id: Task identifier.

        Returns:
            Task status (PENDING if not found).
        """
        task_state = self._tasks.get(task_id)
        return task_state.status if task_state else TaskStatus.PENDING

    def get_summary(self) -> StateSummary:
        """Get summary statistics.

        Returns:
            Summary of task states.
        """
        summary = StateSummary()

        for task_state in self._tasks.values():
            if task_state.status == TaskStatus.PENDING:
                summary.pending += 1
            elif task_state.status == TaskStatus.IN_PROGRESS:
                summary.in_progress += 1
            elif task_state.status == TaskStatus.COMPLETED:
                summary.completed += 1
            elif task_state.status == TaskStatus.FAILED:
                summary.failed += 1

            summary.total_attempts += task_state.attempts
            if task_state.token_usage:
                summary.total_tokens += sum(task_state.token_usage.values())

        return summary

    def get_in_progress_tasks(self) -> list[str]:
        """Get IDs of tasks currently in progress.

        Returns:
            List of in-progress task IDs.
        """
        return [
            task_id
            for task_id, state in self._tasks.items()
            if state.status == TaskStatus.IN_PROGRESS
        ]

    def get_failed_tasks(self) -> list[str]:
        """Get IDs of failed tasks.

        Returns:
            List of failed task IDs.
        """
        return [
            task_id for task_id, state in self._tasks.items() if state.status == TaskStatus.FAILED
        ]

    def get_completed_tasks(self) -> list[str]:
        """Get IDs of completed tasks.

        Returns:
            List of completed task IDs.
        """
        return [
            task_id
            for task_id, state in self._tasks.items()
            if state.status == TaskStatus.COMPLETED
        ]

    # =========================================================================
    # Recovery
    # =========================================================================

    def recover(self) -> list[str]:
        """Recover from crash by resetting in-progress tasks.

        Tasks that were IN_PROGRESS when the executor crashed are
        reset to PENDING so they can be retried.

        Returns:
            List of task IDs that were recovered.
        """
        recovered = []

        for task_id, state in self._tasks.items():
            if state.status == TaskStatus.IN_PROGRESS:
                state.status = TaskStatus.PENDING
                state.started_at = None
                recovered.append(task_id)
                logger.info(f"Recovered task {task_id} from IN_PROGRESS to PENDING")

        if recovered:
            self._touch()

        return recovered

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self) -> Path:
        """Save state to the sidecar file.

        Creates the state directory if it doesn't exist.

        Returns:
            Path to the saved state file.
        """
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Add .gitignore if not present
        gitignore = self.state_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("# Executor state files\n*\n!.gitignore\n")

        # Write state file
        data = self.to_dict()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug(f"State saved to {self.state_file}")
        return self.state_file

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": STATE_VERSION,
            "roadmap_path": str(self._roadmap_path),
            "roadmap_hash": self._roadmap_hash,
            "run_id": self._run_id,
            "created_at": self._created_at.isoformat(),
            "last_updated": self._last_updated.isoformat(),
            "tasks": {task_id: state.to_dict() for task_id, state in self._tasks.items()},
            "summary": self.get_summary().to_dict(),
        }

    # =========================================================================
    # ROADMAP Synchronization
    # =========================================================================

    def sync_to_roadmap(self) -> int:
        """Sync completed tasks back to ROADMAP.md.

        Updates checkboxes in the ROADMAP file:
        - `- [ ]` -> `- [x]` for completed tasks

        Returns:
            Number of checkboxes updated.

        Raises:
            FileNotFoundError: If ROADMAP file doesn't exist.
        """
        if not self._roadmap_path.exists():
            raise FileNotFoundError(f"ROADMAP file not found: {self._roadmap_path}")

        content = self._roadmap_path.read_text(encoding="utf-8")
        original_content = content
        updated_count = 0

        # Parse the roadmap to get task positions
        parser = RoadmapParser()
        roadmap = parser.parse(self._roadmap_path)

        # Build a map of task titles to their completed status from our state
        completed_titles: set[str] = set()
        for task_id, state in self._tasks.items():
            if state.status == TaskStatus.COMPLETED:
                task = roadmap.get_task(task_id)
                if task:
                    completed_titles.add(task.title)

        # Replace unchecked boxes with checked for completed tasks
        lines = content.split("\n")
        new_lines = []

        for line in lines:
            new_line = line
            # Match checkbox patterns - both bold (**title**) and plain text
            # Pattern 1: Bold title like "- [ ] **Create file**"
            bold_match = re.match(r"^(\s*[-*+]\s*)\[ \](\s+\*\*(.+?)\*\*.*)", line)
            # Pattern 2: Plain text like "- [ ] Create file"
            plain_match = re.match(r"^(\s*[-*+]\s*)\[ \](\s+(.+))$", line)

            if bold_match:
                prefix = bold_match.group(1)
                rest = bold_match.group(2)
                title = bold_match.group(3)

                if title in completed_titles:
                    new_line = f"{prefix}[x]{rest}"
                    updated_count += 1
            elif plain_match:
                prefix = plain_match.group(1)
                rest = plain_match.group(2)
                title = plain_match.group(3).strip()

                if title in completed_titles:
                    new_line = f"{prefix}[x]{rest}"
                    updated_count += 1

            new_lines.append(new_line)

        # Write back if changed
        new_content = "\n".join(new_lines)
        if new_content != original_content:
            self._roadmap_path.write_text(new_content, encoding="utf-8")
            # Update hash
            self._roadmap_hash = self._compute_roadmap_hash()
            self._touch()
            logger.info(f"Updated {updated_count} checkboxes in {self._roadmap_path}")

        return updated_count

    def check_roadmap_changed(self) -> bool:
        """Check if the ROADMAP has changed since state was created.

        Returns:
            True if the roadmap content hash differs.
        """
        current_hash = self._compute_roadmap_hash()
        return current_hash != self._roadmap_hash

    def refresh_from_roadmap(self) -> int:
        """Refresh state from ROADMAP.md.

        Adds new tasks from the roadmap that aren't in state,
        and updates status for tasks based on checkbox state.

        Returns:
            Number of tasks added or updated.
        """
        if not self._roadmap_path.exists():
            return 0

        parser = RoadmapParser()
        roadmap = parser.parse(self._roadmap_path)
        updated = 0

        for task in roadmap.all_tasks():
            if task.id not in self._tasks:
                # New task
                self._tasks[task.id] = TaskState(status=task.status)
                updated += 1
            else:
                # Check if checkbox status changed in ROADMAP
                current_state = self._tasks[task.id]
                if task.status == TaskStatus.COMPLETED:
                    if current_state.status != TaskStatus.COMPLETED:
                        # Task was marked complete in ROADMAP directly
                        current_state.status = TaskStatus.COMPLETED
                        current_state.completed_at = datetime.now(UTC)
                        updated += 1

        if updated:
            self._roadmap_hash = self._compute_roadmap_hash()
            self._touch()

        return updated

    # =========================================================================
    # Utilities
    # =========================================================================

    def _generate_run_id(self) -> str:
        """Generate a unique run identifier."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    def _compute_roadmap_hash(self) -> str:
        """Compute hash of roadmap content."""
        if not self._roadmap_path.exists():
            return ""
        content = self._roadmap_path.read_text(encoding="utf-8")
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        """Get string representation."""
        summary = self.get_summary()
        return (
            f"ExecutorState(roadmap={self._roadmap_path.name}, "
            f"tasks={summary.total}, completed={summary.completed}, "
            f"failed={summary.failed})"
        )


# =============================================================================
# Graph-Based Executor State (Phase 1.1)
# =============================================================================

# TypedDict and TodoItem imported at module top for LangGraph StateGraph
# to resolve type hints at runtime


class ExecutorErrorType:
    """Error types for executor graph state."""

    EXECUTION = "execution"
    VERIFICATION = "verification"
    VALIDATION = "validation"  # Phase 1.2: Pre-write validation errors
    TIMEOUT = "timeout"
    ROLLBACK = "rollback"
    PARSE = "parse"
    CONTEXT = "context"
    SHELL = "shell"  # Phase 2.2: Shell command errors


class ExecutorError(TypedDict, total=False):
    """Error state for the executor graph.

    Phase 1.1.2b: Structured error tracking for graph-based execution.
    Phase 1.2: Added validation error type.
    Phase 2.2: Added shell error type.

    Attributes:
        error_type: Category of error (execution, verification, validation, timeout, rollback, shell).
        message: Human-readable error description.
        node: Which graph node produced the error.
        task_id: ID of the task that failed (if applicable).
        recoverable: Whether the error can be retried.
        stack_trace: Full stack trace for debugging (optional).
    """

    error_type: Literal[
        "execution",
        "verification",
        "validation",
        "timeout",
        "rollback",
        "parse",
        "context",
        "shell",
    ]
    message: str
    node: str
    task_id: str | None
    recoverable: bool
    stack_trace: str | None


class ShellError(TypedDict, total=False):
    """Shell-specific error details for executor graph state.

    Phase 2.2.2: Detailed shell command error tracking.
    Used when a shell command fails during task execution.

    Attributes:
        command: The shell command that failed.
        exit_code: The command's exit code (non-zero indicates failure).
        stderr: Standard error output from the command.
        stdout: Standard output from the command (may contain error details).
        cwd: Working directory where the command was executed.
        timed_out: Whether the command timed out.

    Example:
        >>> error: ShellError = {
        ...     "command": "npm install",
        ...     "exit_code": 1,
        ...     "stderr": "npm ERR! code ERESOLVE",
        ...     "stdout": "",
        ...     "cwd": "/project",
        ...     "timed_out": False,
        ... }
    """

    command: str
    exit_code: int
    stderr: str
    stdout: str
    cwd: str | None
    timed_out: bool


class ExecutorGraphState(TypedDict, total=False):
    """State for the graph-based executor.

    Phase 1.1.2: Complete state schema for ExecutorGraph using ai_infra.graph.Graph.

    This TypedDict defines all state variables passed between graph nodes.
    The graph checkpointer persists this state for recovery and HITL workflows.

    State Categories:
        1. Input Configuration: roadmap_path, max_retries
        2. Task Tracking: todos, current_task, completed_todos, failed_todos
        3. Execution Context: run_memory, context, prompt
        4. Error Handling: error, retry_count
        5. Control Flow: should_continue, interrupt_requested
        6. Checkpointing: last_checkpoint_sha, files_modified

    Usage:
        ```python
        from ai_infra.graph import Graph
        from ai_infra.executor.state import ExecutorGraphState

        graph = Graph(
            state_schema=ExecutorGraphState,
            nodes={...},
            edges=[...],
        )
        ```
    """

    # -------------------------------------------------------------------------
    # Input Configuration
    # -------------------------------------------------------------------------
    roadmap_path: str
    """Absolute path to the ROADMAP.md file."""

    max_retries: int
    """Maximum retries per task (default: 3). Configurable via ExecutorConfig."""

    max_tasks: int
    """Maximum tasks to execute (0 = unlimited)."""

    dry_run: bool
    """Dry run mode - preview actions without executing (Phase 2.3.2)."""

    adaptive_mode: str | None
    """DEPRECATED (Phase 2.4): Adaptive mode for replanning.
    Adaptive replanning has been removed in favor of targeted repair.
    Kept for backward compatibility."""

    # -------------------------------------------------------------------------
    # Task Tracking
    # -------------------------------------------------------------------------
    todos: list[TodoItem]
    """All parsed todos from ROADMAP. Populated by parse_roadmap_node."""

    current_task: TodoItem | None
    """Currently executing task. Set by pick_task_node, cleared after checkpoint."""

    completed_todos: list[str]
    """Task IDs that completed successfully. Append-only during execution."""

    failed_todos: list[str]
    """Task IDs that failed permanently (after max retries). Append-only."""

    tasks_completed_count: int
    """Total number of tasks completed this run. For max_tasks limit."""

    # -------------------------------------------------------------------------
    # Execution Context
    # -------------------------------------------------------------------------
    run_memory: dict[str, Any]
    """Current run context. Passed between tasks for continuity."""

    context: str
    """Built context for current task (files, memory, etc.)."""

    prompt: str
    """Final prompt to send to agent for current task."""

    agent_result: dict[str, Any] | None
    """Result from agent execution. Contains tool calls, messages, etc."""

    files_modified: list[str]
    """Files modified by current task. For verification and checkpointing."""

    # -------------------------------------------------------------------------
    # Phase 1.1: Pre-Write Validation
    # -------------------------------------------------------------------------
    generated_code: dict[str, str]
    """Generated code from execute_task, keyed by file path.
    Populated by execute_task_node, consumed by validate_code_node.
    Example: {"src/app.py": "def main(): ..."}"""

    validated: bool
    """Whether generated_code passed pre-write validation.
    Set by validate_code_node. If True, proceed to write_files."""

    needs_repair: bool
    """Whether generated_code needs repair due to validation errors.
    Set by validate_code_node. If True, route to repair_code_node."""

    validation_errors: dict[str, dict]
    """Validation errors by file path.
    Each value contains: error_type, error_message, error_line, repair_prompt.
    Example: {"src/app.py": {"error_type": "syntax", "error_line": 5, ...}}"""

    repair_count: int
    """Number of validation repair attempts for current task.
    Incremented by repair_code_node. Max 2 before escalation.
    Reset to 0 by pick_task_node when starting new task."""

    repair_results: dict[str, dict]
    """Repair results by file path from repair_code_node.
    Phase 1.2: Tracks what was repaired and status.
    Each value contains: status ('repaired' or 'failed'), original_error, error (if failed).
    Example: {"src/app.py": {"status": "repaired", "original_error": "missing colon"}}"""

    # -------------------------------------------------------------------------
    # Phase 1.3: Separated File Writing
    # -------------------------------------------------------------------------
    files_written: bool
    """Whether validated code was successfully written to disk.
    Set by write_files_node. If True, proceed to verify_task."""

    write_errors: list[dict] | None
    """Write errors if some files failed to write.
    Phase 1.3: Each entry contains: file (path), error (message).
    Example: [{"file": "src/app.py", "error": "Permission denied"}]"""

    # -------------------------------------------------------------------------
    # Error Handling
    # -------------------------------------------------------------------------
    error: ExecutorError | None
    """Current error state. Set by nodes on failure, cleared on success."""

    retry_count: int
    """DEPRECATED (Phase 2.2): Use repair_count and test_repair_count instead.
    Kept for backward compatibility. Reset when moving to new task."""

    test_repair_count: int
    """Phase 2.2: Number of test failure repair attempts for current task.
    Incremented by repair_test_node (future). Max 2 before escalation.
    Reset to 0 by pick_task_node when starting new task."""

    # -------------------------------------------------------------------------
    # Phase 2.3.1: Adaptive Replanning (DEPRECATED in Phase 2.5)
    # -------------------------------------------------------------------------
    failure_classification: str | None
    """DEPRECATED (Phase 2.5): Classification of failure (TRANSIENT, WRONG_APPROACH, FATAL).
    Replaced by failure_category for more granular categorization.
    Kept for backward compatibility."""

    failure_reason: str | None
    """DEPRECATED (Phase 2.5): Human-readable explanation of why the task failed.
    This was used by analyze_failure_node for adaptive replanning.
    Kept for backward compatibility."""

    suggested_fix: str | None
    """DEPRECATED (Phase 2.5): Suggested fix for WRONG_APPROACH failures.
    This was used by analyze_failure_node for adaptive replanning.
    Kept for backward compatibility."""

    # -------------------------------------------------------------------------
    # Phase 2.4.1: Detailed Failure Category
    # -------------------------------------------------------------------------
    failure_category: str | None
    """Detailed failure category from FailureCategory enum (e.g., SYNTAX_ERROR,
    IMPORT_ERROR, TYPE_ERROR, TEST_FAILURE). More granular than classification."""

    execution_plan: str | None
    """DEPRECATED (Phase 2.4): Revised execution plan generated by replan_task_node.
    Kept for backward compatibility."""

    replan_count: int
    """DEPRECATED (Phase 2.4): Number of replanning attempts for current task.
    Adaptive replanning has been removed in favor of targeted repair.
    Kept for backward compatibility."""

    # -------------------------------------------------------------------------
    # Phase 2.4.2: Pre-Execution Planning
    # -------------------------------------------------------------------------
    task_plan: dict[str, Any] | None
    """Pre-execution plan generated by plan_task_node. Contains:
    - likely_files: Files to be modified/created
    - dependencies: New dependencies needed
    - risks: Potential edge cases
    - approach: Implementation strategy
    - complexity: LOW/MEDIUM/HIGH estimate"""

    enable_planning: bool
    """Whether pre-execution planning is enabled (Phase 2.4.2, default: False)."""

    # -------------------------------------------------------------------------
    # Phase 2.4.3: Per-Node Cost Tracking
    # -------------------------------------------------------------------------
    node_metrics: dict[str, Any] | None
    """Per-node metrics dict. Each key is a node name, value is NodeMetrics.to_dict().
    Tracks tokens_in, tokens_out, duration_ms, llm_calls per node."""

    enable_node_metrics: bool
    """Whether per-node metrics collection is enabled (Phase 2.4.3, default: True)."""

    # -------------------------------------------------------------------------
    # Phase 2.3.3: Pause Before Destructive Operations
    # -------------------------------------------------------------------------
    pause_destructive: bool
    """Whether to pause on destructive operations (default: True)."""

    pause_reason: str | None
    """Reason for pause if destructive operations detected."""

    detected_destructive_ops: list[str] | None
    """List of detected destructive operation descriptions."""

    pending_result: dict | None
    """Agent result pending approval after destructive ops detected."""

    # -------------------------------------------------------------------------
    # Control Flow
    # -------------------------------------------------------------------------
    should_continue: bool
    """Whether to continue to next task. Set False to end execution."""

    interrupt_requested: bool
    """HITL pause flag. Set True to interrupt before next action."""

    verified: bool
    """Whether current task passed verification. Set by verify_task_node."""

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    last_checkpoint_sha: str | None
    """DEPRECATED (Phase 2.5): Git SHA of last successful checkpoint.
    Previously used for rollback, but rollback was removed in Phase 2.1.
    Kept for backward compatibility."""

    thread_id: str | None
    """Graph thread ID for resuming interrupted runs."""

    # -------------------------------------------------------------------------
    # Phase 2.3: Repair Test Results
    # -------------------------------------------------------------------------
    test_repair_results: dict[str, dict]
    """Phase 2.3: Repair results by file path from repair_test_node.
    Tracks what was repaired and status for test failures.
    Each value contains: status ('repaired' or 'failed'), test_name, line_number, error_type, traceback.
    Example: {"src/app.py": {"status": "repaired", "test_name": "test_main", "line_number": 42, "error_type": "AssertionError", "traceback": "..."}}
    """

    # -------------------------------------------------------------------------
    # Phase 2.1 & 2.2: Shell Tool Integration
    # -------------------------------------------------------------------------
    enable_shell: bool
    """Whether shell tool is enabled for this run (Phase 2.1, default: True)."""

    shell_session_active: bool
    """Whether a shell session is currently active (Phase 2.2).
    True when arun/astream has started a session that hasn't been closed yet.
    The actual ShellSession object is managed by ExecutorGraph, not stored in state."""

    shell_results: list[dict[str, Any]]
    """History of shell commands executed during this run (Phase 2.1).
    Each entry contains: command, exit_code, stdout, stderr, duration_ms, timed_out.
    Useful for debugging and audit trails.
    Example: [{"command": "npm install", "exit_code": 0, "stdout": "...", "stderr": "", "duration_ms": 1234}]
    """

    shell_error: ShellError | None
    """Details of the most recent shell command failure (Phase 2.2).
    Set when a shell command fails with non-zero exit code or times out.
    Cleared when the next task starts or when the error is handled."""

    # -------------------------------------------------------------------------
    # Phase 3.2: Autonomous Verification
    # -------------------------------------------------------------------------
    enable_autonomous_verify: bool
    """Whether autonomous verification is enabled (Phase 3.2, default: False).
    When True, the VerificationAgent will run tests autonomously after each task.
    When False, only fast syntax checks are performed."""

    verify_timeout: float
    """Timeout for autonomous verification in seconds (Phase 3.2, default: 300.0).
    Maximum time allowed for the VerificationAgent to complete verification."""

    autonomous_verify_result: dict[str, Any] | None
    """Result from the last autonomous verification (Phase 3.2).
    Contains: passed, checks_run, failures, suggestions, duration_ms.
    Set by verify_task_node when enable_autonomous_verify is True."""


# =============================================================================
# Timeout and Retry Configuration (Phase 1.1.5)
# =============================================================================


class NodeTimeouts:
    """Timeout configuration for each graph node.

    Phase 1.1.5: Configurable per-node timeouts.
    All values in seconds.
    """

    PARSE_ROADMAP: float = 30.0
    """Timeout for parsing ROADMAP.md (file I/O)."""

    PICK_TASK: float = 5.0
    """Timeout for selecting next task (in-memory operation)."""

    BUILD_CONTEXT: float = 60.0
    """Timeout for building task context (may scan codebase)."""

    EXECUTE_TASK: float = 300.0
    """Timeout for agent execution (LLM + tool calls)."""

    VERIFY_TASK: float = 120.0
    """Timeout for verification (may run tests)."""

    CHECKPOINT: float = 30.0
    """Timeout for git checkpoint operations."""

    ROLLBACK: float = 30.0
    """Timeout for git rollback operations."""

    HANDLE_FAILURE: float = 5.0
    """Timeout for failure handling (in-memory operation)."""

    DECIDE_NEXT: float = 5.0
    """Timeout for deciding next action (in-memory operation)."""


class RetryPolicy:
    """Retry policy with exponential backoff.

    Phase 1.1.5: Configurable retry strategy.
    """

    MAX_RETRIES: int = 3
    """Maximum retry attempts per task."""

    BASE_DELAY: float = 1.0
    """Base delay in seconds (first retry)."""

    MULTIPLIER: float = 2.0
    """Exponential multiplier for subsequent retries."""

    JITTER_PERCENT: float = 0.1
    """Random jitter as percentage of delay (±10%)."""

    @classmethod
    def get_delay(cls, attempt: int) -> float:
        """Calculate delay for a given retry attempt.

        Args:
            attempt: Retry attempt number (1-based).

        Returns:
            Delay in seconds with jitter applied.
        """
        import random

        base = cls.BASE_DELAY * (cls.MULTIPLIER ** (attempt - 1))
        jitter = base * cls.JITTER_PERCENT * (2 * random.random() - 1)
        return max(0, base + jitter)


class NonRetryableErrors:
    """Errors that should not be retried.

    Phase 1.1.5: Fail immediately on these errors.
    """

    PATTERNS: list[str] = [
        "authentication failed",
        "invalid api key",
        "rate limit exceeded",
        "invalid roadmap format",
        "git conflict",
        "merge conflict",
        "permission denied",
        "roadmap not found",
    ]

    @classmethod
    def is_non_retryable(cls, error_message: str) -> bool:
        """Check if an error message indicates a non-retryable error.

        Args:
            error_message: The error message to check.

        Returns:
            True if the error should not be retried.
        """
        lower_message = error_message.lower()
        return any(pattern in lower_message for pattern in cls.PATTERNS)


# =============================================================================
# Graph Node Names (Phase 1.1.1)
# =============================================================================


class ExecutorNodes:
    """Node names for the executor graph.

    Phase 1.1.1: Standard node identifiers for consistent wiring.
    """

    PARSE_ROADMAP = "parse_roadmap"
    PICK_TASK = "pick_task"
    BUILD_CONTEXT = "build_context"
    EXECUTE_TASK = "execute_task"
    VERIFY_TASK = "verify_task"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"
    HANDLE_FAILURE = "handle_failure"
    DECIDE_NEXT = "decide_next"

    @classmethod
    def all(cls) -> list[str]:
        """Get all node names."""
        return [
            cls.PARSE_ROADMAP,
            cls.PICK_TASK,
            cls.BUILD_CONTEXT,
            cls.EXECUTE_TASK,
            cls.VERIFY_TASK,
            cls.CHECKPOINT,
            cls.ROLLBACK,
            cls.HANDLE_FAILURE,
            cls.DECIDE_NEXT,
        ]


# =============================================================================
# Edge Routing Targets (Phase 1.1.3)
# =============================================================================


class EdgeTargets:
    """Edge routing target constants.

    Phase 1.1.3: Consistent target identifiers for conditional edges.
    """

    # After pick_task
    CONTEXT = "build_context"
    END = "__end__"

    # After execute_task
    VERIFY = "verify_task"
    FAILURE = "handle_failure"

    # After verify_task
    CHECKPOINT = "checkpoint"

    # After handle_failure
    ROLLBACK = "rollback"
    DECIDE = "decide_next"

    # After rollback
    EXECUTE = "execute_task"

    # After decide_next
    PICK = "pick_task"
