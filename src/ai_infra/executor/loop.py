"""Executor orchestration loop for autonomous task execution.

This module provides the main Executor class that:
- Parses tasks from ROADMAP.md
- Manages execution state
- Runs tasks through an agent
- Verifies task completion
- Handles failures and recovery
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

# Import adaptive planning types (avoid circular import)
# Type checking only - actual import in methods that need it
from typing import TYPE_CHECKING, Any, Protocol

from ai_infra.executor.checkpoint import (
    Checkpointer,
    CheckpointResult,
    CommitInfo,
    RollbackResult,
)
from ai_infra.executor.context import ProjectContext
from ai_infra.executor.dependencies import (
    ChangeAnalysis,
    DependencyTracker,
    DependencyWarning,
    ParallelGroup,
    TaskDependencyGraph,
)
from ai_infra.executor.failure import FailureAnalyzer, FailureCategory
from ai_infra.executor.models import FileWriteRecord, FileWriteSummary, Task, TaskStatus
from ai_infra.executor.observability import (
    ExecutorCallbacks,
    ExecutorMetrics,
    log_task_context,
    log_verification_result,
)
from ai_infra.executor.outcome_extractor import extract_outcome
from ai_infra.executor.parser import RoadmapParser
from ai_infra.executor.project_memory import ProjectMemory
from ai_infra.executor.recovery import (
    CheckpointMetadata,
    RecoveryManager,
    RecoveryResult,
    RecoveryStrategy,
    RollbackPreview,
    SelectiveRollbackResult,
)
from ai_infra.executor.run_memory import RunMemory, TaskOutcome

if TYPE_CHECKING:
    from ai_infra.executor.adaptive import PlanAnalyzer, PlanSuggestion, SuggestionResult
from ai_infra.executor.roadmap import ParsedTask, Roadmap
from ai_infra.executor.state import ExecutorState
from ai_infra.executor.todolist import TodoItem, TodoListManager
from ai_infra.executor.verifier import CheckLevel, TaskVerifier, VerificationResult
from ai_infra.logging import get_logger
from ai_infra.tracing import Span, Tracer, get_tracer

logger = get_logger("executor.loop")


# =============================================================================
# Execution Result
# =============================================================================


class VerifyMode(str, Enum):
    """Verification mode for task completion (Phase 5.9.2).

    Attributes:
        AUTO: Auto-detect project type and run appropriate test runner (default).
        AGENT: Agent writes and runs its own verification as part of the task.
        SKIP: Skip verification entirely.
        PYTEST: Force pytest (legacy behavior for Python-only projects).
    """

    AUTO = "auto"
    AGENT = "agent"
    SKIP = "skip"
    PYTEST = "pytest"


class ExecutionStatus(str, Enum):
    """Status of a task execution."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result of executing a single task.

    Attributes:
        task_id: The task that was executed.
        title: Human-readable title for the task/todo.
        status: Execution status.
        files_modified: List of files modified by the task.
        files_created: List of files created by the task.
        files_deleted: List of files deleted by the task.
        agent_output: Raw output from the agent.
        verification: Verification results if verification was run.
        error: Error message if execution failed.
        token_usage: Token usage for this execution.
        duration_ms: Execution duration in milliseconds.
        started_at: When execution started.
        completed_at: When execution completed.
        dependency_warnings: Warnings about files that may be affected.
        suggestions: Plan suggestions from adaptive planning (Phase 5.5).
    """

    task_id: str
    status: ExecutionStatus
    title: str = ""
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    agent_output: str = ""
    verification: VerificationResult | None = None
    error: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    dependency_warnings: list[DependencyWarning] = field(default_factory=list)
    suggestions: list[Any] = field(default_factory=list)  # Phase 5.5: PlanSuggestion list

    @property
    def success(self) -> bool:
        """Whether the execution succeeded."""
        return self.status == ExecutionStatus.SUCCESS

    @property
    def all_files(self) -> list[str]:
        """Get all files affected by this execution."""
        return self.files_modified + self.files_created

    @property
    def has_dependency_warnings(self) -> bool:
        """Whether there are dependency warnings."""
        return len(self.dependency_warnings) > 0

    @property
    def has_suggestions(self) -> bool:
        """Whether there are plan suggestions (Phase 5.5)."""
        return len(self.suggestions) > 0

    @property
    def error_message(self) -> str:
        """Get the error message (alias for adaptive planning compatibility)."""
        return self.error or ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status.value,
            "files_modified": self.files_modified,
            "files_created": self.files_created,
            "files_deleted": self.files_deleted,
            "error": self.error,
            "token_usage": self.token_usage,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "dependency_warnings": [w.to_dict() for w in self.dependency_warnings],
            "suggestions": [s.to_dict() if hasattr(s, "to_dict") else s for s in self.suggestions],
        }


# =============================================================================
# Run Summary
# =============================================================================


class RunStatus(str, Enum):
    """Status of an executor run."""

    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"
    STOPPED = "stopped"
    NO_TASKS = "no_tasks"


@dataclass
class RunSummary:
    """Summary of an executor run.

    Attributes:
        status: Overall run status.
        tasks_completed: Number of tasks completed successfully.
        tasks_failed: Number of tasks that failed.
        tasks_skipped: Number of tasks skipped.
        tasks_remaining: Number of pending tasks.
        total_tasks: Total tasks in the roadmap.
        results: Individual execution results.
        paused: Whether the run was paused for human review.
        pending_review: Tasks pending human review.
        duration_ms: Total run duration.
        total_tokens: Total tokens used.
        started_at: When the run started.
        completed_at: When the run completed.
        run_id: Unique run identifier.
    """

    status: RunStatus
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_skipped: int = 0
    tasks_remaining: int = 0
    total_tasks: int = 0
    results: list[ExecutionResult] = field(default_factory=list)
    paused: bool = False
    pending_review: list[str] = field(default_factory=list)
    pause_reason: str = ""
    duration_ms: float = 0.0
    total_tokens: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    run_id: str = ""

    @property
    def success_rate(self) -> float:
        """Get success rate (0.0 to 1.0)."""
        executed = self.tasks_completed + self.tasks_failed
        if executed == 0:
            return 1.0
        return self.tasks_completed / executed

    @property
    def progress(self) -> float:
        """Get overall progress (0.0 to 1.0)."""
        if self.total_tasks == 0:
            return 1.0
        return self.tasks_completed / self.total_tasks

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_skipped": self.tasks_skipped,
            "tasks_remaining": self.tasks_remaining,
            "total_tasks": self.total_tasks,
            "paused": self.paused,
            "pending_review": self.pending_review,
            "pause_reason": self.pause_reason,
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
            "success_rate": self.success_rate,
            "progress": self.progress,
            "run_id": self.run_id,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Run: {self.run_id}",
            f"Status: {self.status.value}",
            f"Progress: {self.tasks_completed}/{self.total_tasks} completed",
            f"  - Completed: {self.tasks_completed}",
            f"  - Failed: {self.tasks_failed}",
            f"  - Remaining: {self.tasks_remaining}",
            f"Success Rate: {self.success_rate:.0%}",
            f"Total Tokens: {self.total_tokens:,}",
            f"Duration: {self.duration_ms / 1000:.1f}s",
        ]

        if self.paused:
            lines.append(f"Paused for review: {len(self.pending_review)} tasks")
            if self.pause_reason:
                lines.append(f"Reason: {self.pause_reason}")

        return "\n".join(lines)


# =============================================================================
# Review Info
# =============================================================================


@dataclass
class ReviewInfo:
    """Information about changes pending human review.

    Provides a summary of what changed during task execution,
    including files modified, commits created, and whether
    destructive operations occurred.

    Attributes:
        task_ids: Task IDs that were executed.
        files_modified: Files that were modified.
        files_created: Files that were created.
        files_deleted: Files that were deleted.
        commits: Git commits created (if checkpointing enabled).
        has_destructive: Whether destructive operations occurred.
        pause_reason: Why execution was paused.
    """

    task_ids: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    commits: list[CommitInfo] = field(default_factory=list)
    has_destructive: bool = False
    pause_reason: str = ""

    @property
    def total_files_affected(self) -> int:
        """Total number of files affected."""
        all_files = set(self.files_modified + self.files_created + self.files_deleted)
        return len(all_files)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_ids": self.task_ids,
            "files_modified": self.files_modified,
            "files_created": self.files_created,
            "files_deleted": self.files_deleted,
            "commits": [c.to_dict() for c in self.commits],
            "has_destructive": self.has_destructive,
            "pause_reason": self.pause_reason,
            "total_files_affected": self.total_files_affected,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Pending Review: {len(self.task_ids)} task(s)",
            f"  - Files Modified: {len(self.files_modified)}",
            f"  - Files Created: {len(self.files_created)}",
            f"  - Files Deleted: {len(self.files_deleted)}",
            f"  - Commits: {len(self.commits)}",
        ]
        if self.has_destructive:
            lines.append("  - WARNING: Destructive operations detected")
        if self.pause_reason:
            lines.append(f"  - Reason: {self.pause_reason}")
        return "\n".join(lines)


# =============================================================================
# Agent Protocol
# =============================================================================


class AgentProtocol(Protocol):
    """Protocol for agent implementations."""

    async def arun(self, prompt: str) -> str:
        """Run the agent with a prompt and return output."""
        ...


# =============================================================================
# Executor Configuration
# =============================================================================


@dataclass
class ExecutorConfig:
    """Configuration for the Executor.

    Attributes:
        model: LLM model to use for agent.
        max_tasks: Maximum tasks to execute per run (0 = unlimited).
        stop_on_failure: Stop execution after first failure.
        checkpoint_every: Create git checkpoint after N tasks (0 = disabled).
        require_human_approval_after: Pause for review after N tasks (0 = disabled).
        pause_before_destructive: Pause before file deletions.
        verification_level: Maximum verification level to run.
        skip_verification: Skip verification entirely.
        dry_run: Show what would be done without executing.
        retry_failed: Retry failed tasks up to N times.
        context_max_tokens: Maximum tokens for context.
        save_state_every: Save state after N tasks.
        agent_timeout: Timeout for agent execution in seconds.
        recovery_strategy: Default strategy for recovering from failures.
        checkpoint_tags: Default tags to apply to checkpoints.
        sync_roadmap: Sync completed tasks to ROADMAP.md after each task.
        parallel_tasks: Max tasks to run in parallel (1 = sequential, 0 = auto).
        parallel_file_overlap: Consider tasks with shared files as dependent.
        parallel_import_analysis: Use import graph for dependency detection.
        adaptive_mode: Planning mode for ROADMAP modification (Phase 5.5).
        verify_mode: Verification mode (Phase 5.9.2).
        enable_run_memory: Enable run memory for task-to-task context (Phase 5.8).
        enable_project_memory: Enable project memory for cross-run persistence (Phase 5.8).
        memory_token_budget: Token budget shared across all memory layers (Phase 5.8).
        extract_outcomes_with_llm: Use LLM for outcome extraction (Phase 5.8).
        verify_writes: Enable strict file write verification with checksum validation (Phase 5.11.4).
    """

    model: str = "claude-sonnet-4-20250514"
    max_tasks: int = 0
    stop_on_failure: bool = True
    checkpoint_every: int = 1
    require_human_approval_after: int = 0
    pause_before_destructive: bool = True
    verification_level: CheckLevel = CheckLevel.TESTS
    skip_verification: bool = False
    dry_run: bool = False
    retry_failed: int = 1
    context_max_tokens: int = 50000
    save_state_every: int = 1
    agent_timeout: float = 300.0
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.ROLLBACK_ALL
    checkpoint_tags: list[str] = field(default_factory=list)
    sync_roadmap: bool = True  # Sync checkboxes after each task
    parallel_tasks: int = 1  # Phase 5.1: Max parallel tasks (1 = sequential)
    parallel_file_overlap: bool = True  # Phase 5.1: Use file overlap detection
    parallel_import_analysis: bool = True  # Phase 5.1: Use import graph analysis
    adaptive_mode: str = "suggest"  # Phase 5.5: "no_adapt", "suggest", or "auto_fix"
    verify_mode: VerifyMode = VerifyMode.AUTO  # Phase 5.9.2: Verification mode
    # Phase 5.8: Memory configuration
    enable_run_memory: bool = True  # Task-to-task context within a run
    enable_project_memory: bool = True  # Cross-run persistence
    memory_token_budget: int = 6000  # Shared token budget for memory context
    extract_outcomes_with_llm: bool = False  # Use LLM for extraction (slower but better)
    # Phase 5.11.4: File write verification
    verify_writes: bool = False  # Enable strict file write verification with checksum
    # Phase 5.13: LLM-based ROADMAP normalization
    normalize_with_llm: bool = False  # Use LLM to normalize non-checkbox formats

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "max_tasks": self.max_tasks,
            "stop_on_failure": self.stop_on_failure,
            "checkpoint_every": self.checkpoint_every,
            "require_human_approval_after": self.require_human_approval_after,
            "pause_before_destructive": self.pause_before_destructive,
            "verification_level": self.verification_level.value,
            "skip_verification": self.skip_verification,
            "dry_run": self.dry_run,
            "retry_failed": self.retry_failed,
            "context_max_tokens": self.context_max_tokens,
            "save_state_every": self.save_state_every,
            "agent_timeout": self.agent_timeout,
            "recovery_strategy": self.recovery_strategy.value,
            "checkpoint_tags": self.checkpoint_tags,
            "sync_roadmap": self.sync_roadmap,
            "parallel_tasks": self.parallel_tasks,
            "parallel_file_overlap": self.parallel_file_overlap,
            "parallel_import_analysis": self.parallel_import_analysis,
            "adaptive_mode": self.adaptive_mode,
            "verify_mode": self.verify_mode.value,
            # Phase 5.8: Memory config
            "enable_run_memory": self.enable_run_memory,
            "enable_project_memory": self.enable_project_memory,
            "memory_token_budget": self.memory_token_budget,
            "extract_outcomes_with_llm": self.extract_outcomes_with_llm,
            # Phase 5.11.4: File write verification
            "verify_writes": self.verify_writes,
            # Phase 5.13: LLM normalization
            "normalize_with_llm": self.normalize_with_llm,
        }


# =============================================================================
# Task Complete Callback
# =============================================================================

TaskCompleteCallback = Callable[[ParsedTask, ExecutionResult], Any]


# =============================================================================
# Executor
# =============================================================================


class Executor:
    """Autonomous task executor that runs tasks from ROADMAP.md.

    The Executor orchestrates the full task execution lifecycle:
    1. Parse tasks from ROADMAP.md
    2. Build project context
    3. Pick next pending task
    4. Execute task via agent
    5. Verify task completion
    6. Update state and checkpoint
    7. Repeat until done or limits reached

    Example:
        >>> executor = Executor(
        ...     roadmap="./ROADMAP.md",
        ...     config=ExecutorConfig(max_tasks=5),
        ... )
        >>>
        >>> async def on_complete(task, result):
        ...     print(f"Completed: {task.title}")
        >>>
        >>> summary = await executor.run(on_complete=on_complete)
        >>> print(summary.summary())

    Attributes:
        roadmap_path: Path to the ROADMAP.md file.
        config: Executor configuration.
        state: Execution state manager.
        roadmap: Parsed roadmap.
    """

    def __init__(
        self,
        roadmap: str | Path,
        *,
        config: ExecutorConfig | None = None,
        agent: AgentProtocol | None = None,
        verifier: TaskVerifier | None = None,
        failure_analyzer: FailureAnalyzer | None = None,
        checkpointer: Checkpointer | None = None,
        callbacks: ExecutorCallbacks | None = None,
    ) -> None:
        """Initialize the Executor.

        Args:
            roadmap: Path to the ROADMAP.md file.
            config: Executor configuration.
            agent: Agent for task execution (will be created if not provided).
            verifier: Task verifier (will be created if not provided).
            failure_analyzer: Failure analyzer for tracking patterns.
            checkpointer: Git checkpointer (auto-created if in a git repo).
            callbacks: Executor callbacks for observability (metrics, tracing).
        """
        self._roadmap_path = Path(roadmap).resolve()
        self._config = config or ExecutorConfig()
        self._agent = agent
        self._verifier = verifier
        self._failure_analyzer = failure_analyzer
        self._checkpointer = checkpointer
        self._checkpointer_initialized = checkpointer is not None
        self._callbacks = callbacks

        # Tracing support
        self._tracer: Tracer | None = None

        # Will be initialized on first run
        self._state: ExecutorState | None = None
        self._roadmap: Roadmap | None = None
        self._context: ProjectContext | None = None
        self._dependency_tracker: DependencyTracker | None = None
        self._dependency_graph_built = False
        self._recovery_manager: RecoveryManager | None = None
        self._recovery_manager_initialized = False
        self._task_dependency_graph: TaskDependencyGraph | None = None  # Phase 5.1

        # Phase 5.8: Memory layers
        self._run_memory: RunMemory | None = None
        self._project_memory: ProjectMemory | None = None
        self._llm: Any = None  # LLM for outcome extraction (set via set_llm())

        # Initialize project memory if enabled (persists across runs)
        if self._config.enable_project_memory:
            self._project_memory = ProjectMemory.load(self._roadmap_path.parent)

        # Run tracking
        self._tasks_this_run = 0
        self._run_started: datetime | None = None
        self._processed_task_ids: set[str] = set()  # Track processed tasks to avoid infinite loops
        self._last_run_results: list[ExecutionResult] = []  # Results from last run (for review)

        # Phase 5.11.4: File write verification tracking
        self._file_write_tracker: dict[str, FileWriteRecord] = {}  # path -> record
        self._expected_files_per_task: dict[str, list[str]] = {}  # task_id -> [paths]

        # TodoList manager for normalized task tracking
        self._todo_manager: TodoListManager | None = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def roadmap_path(self) -> Path:
        """Get the roadmap file path."""
        return self._roadmap_path

    @property
    def config(self) -> ExecutorConfig:
        """Get the configuration."""
        return self._config

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
    def todo_manager(self) -> TodoListManager:
        """Get the todo list manager.

        Creates a normalized todo list from the roadmap with smart grouping
        of related tasks. This allows the executor to work with logical units
        of work rather than individual bullet points.

        Note: When normalize_with_llm is enabled, call ensure_todo_manager()
        before accessing this property to ensure async LLM normalization is done.
        """
        if self._todo_manager is None:
            if self._config.normalize_with_llm:
                raise RuntimeError(
                    "LLM normalization requires async initialization. "
                    "Call await executor.ensure_todo_manager() first."
                )
            self._todo_manager = TodoListManager.from_roadmap(
                roadmap=self.roadmap,
                roadmap_path=self._roadmap_path,
                group_strategy="smart",
            )
            logger.info(
                "todo_list_created",
                total_todos=self._todo_manager.total_count,
                source_tasks=self.roadmap.total_tasks,
            )
        return self._todo_manager

    async def ensure_todo_manager(self) -> TodoListManager:
        """Ensure the todo manager is initialized, using LLM if configured.

        This method must be called before accessing todo_manager when
        normalize_with_llm is enabled. It handles the async LLM normalization.

        Returns:
            The initialized TodoListManager.
        """
        if self._todo_manager is not None:
            return self._todo_manager

        if self._config.normalize_with_llm:
            if self._agent is None:
                raise RuntimeError(
                    "LLM normalization requires an agent. Create Executor with an agent parameter."
                )
            logger.info("llm_normalization_starting", roadmap=str(self._roadmap_path))
            self._todo_manager = await TodoListManager.from_roadmap_llm(
                roadmap_path=self._roadmap_path,
                agent=self._agent,
            )
            logger.info(
                "llm_normalization_complete",
                total_todos=self._todo_manager.total_count,
                uses_json_only=self._todo_manager.uses_json_only,
            )
        else:
            self._todo_manager = TodoListManager.from_roadmap(
                roadmap=self.roadmap,
                roadmap_path=self._roadmap_path,
                group_strategy="smart",
            )
            logger.info(
                "todo_list_created",
                total_todos=self._todo_manager.total_count,
                source_tasks=self.roadmap.total_tasks,
            )
        return self._todo_manager

    @property
    def verifier(self) -> TaskVerifier:
        """Get the task verifier."""
        if self._verifier is None:
            self._verifier = TaskVerifier(self._roadmap_path.parent)
        return self._verifier

    @property
    def failure_analyzer(self) -> FailureAnalyzer:
        """Get the failure analyzer."""
        if self._failure_analyzer is None:
            self._failure_analyzer = FailureAnalyzer()
        return self._failure_analyzer

    @property
    def checkpointer(self) -> Checkpointer | None:
        """Get the git checkpointer.

        Returns None if checkpointing is disabled or not in a git repo.
        """
        if self._config.checkpoint_every == 0:
            return None
        if not self._checkpointer_initialized:
            self._checkpointer = Checkpointer.for_workspace(self._roadmap_path.parent)
            self._checkpointer_initialized = True
        return self._checkpointer

    @property
    def recovery_manager(self) -> RecoveryManager | None:
        """Get the recovery manager for enhanced rollback.

        Returns None if checkpointing is disabled or not in a git repo.

        The recovery manager provides:
        - Enhanced checkpoint metadata (file hashes, verification status)
        - Selective rollback (by file, by tag, preview mode)
        - Recovery strategies (retry with context, partial rollback)
        """
        if self._config.checkpoint_every == 0:
            return None
        if not self._recovery_manager_initialized:
            checkpointer = self.checkpointer
            if checkpointer is not None:
                self._recovery_manager = RecoveryManager(checkpointer)
            self._recovery_manager_initialized = True
        return self._recovery_manager

    @property
    def callbacks(self) -> ExecutorCallbacks | None:
        """Get the executor callbacks for observability.

        Returns None if callbacks were not configured.
        """
        return self._callbacks

    @property
    def tracer(self) -> Tracer:
        """Get the tracer for distributed tracing.

        Lazily initialized on first access.
        """
        if self._tracer is None:
            self._tracer = get_tracer()
        return self._tracer

    @property
    def run_memory(self) -> RunMemory | None:
        """Get the run memory for task-to-task context (Phase 5.8).

        Returns None if run memory is disabled or run hasn't started.
        """
        return self._run_memory

    @property
    def project_memory(self) -> ProjectMemory | None:
        """Get the project memory for cross-run persistence (Phase 5.8).

        Returns None if project memory is disabled.
        """
        return self._project_memory

    def set_llm(self, llm: Any) -> None:
        """Set the LLM for outcome extraction (Phase 5.8).

        When an LLM is provided and extract_outcomes_with_llm is True,
        the executor uses the LLM to extract richer summaries and
        decisions from agent responses.

        Args:
            llm: An LLM instance with chat() or achat() methods.
        """
        self._llm = llm

    @property
    def plan_analyzer(self) -> PlanAnalyzer:
        """Get the plan analyzer for adaptive planning (Phase 5.5).

        Analyzes failures and suggests plan fixes based on the configured
        adaptive mode (no_adapt, suggest, auto_fix).
        """
        # Import here to avoid circular import
        from ai_infra.executor.adaptive import AdaptiveMode, PlanAnalyzer

        if not hasattr(self, "_plan_analyzer") or self._plan_analyzer is None:
            mode = AdaptiveMode(self._config.adaptive_mode)
            self._plan_analyzer = PlanAnalyzer(
                roadmap=self.roadmap,
                mode=mode,
                project_root=self._roadmap_path.parent,
            )
        return self._plan_analyzer

    def get_metrics(self) -> ExecutorMetrics | None:
        """Get aggregated metrics from the callbacks.

        Returns:
            ExecutorMetrics if callbacks are configured, None otherwise.
        """
        if self._callbacks is not None:
            return self._callbacks.get_metrics()
        return None

    @property
    def dependency_tracker(self) -> DependencyTracker:
        """Get the dependency tracker for multi-file awareness.

        Tracks import relationships between files to:
        - Warn when changes might break dependents
        - Identify files that need updates
        - Support impact analysis
        """
        if self._dependency_tracker is None:
            self._dependency_tracker = DependencyTracker(self._roadmap_path.parent)
        return self._dependency_tracker

    async def build_dependency_graph(self, *, force: bool = False) -> None:
        """Build the dependency graph for the project.

        This should be called before analyzing change impacts.
        It's automatically called on first run.

        Args:
            force: Rebuild even if already built.
        """
        if self._dependency_graph_built and not force:
            return
        await self.dependency_tracker.build_graph(force=force)
        self._dependency_graph_built = True
        logger.info(
            f"Dependency graph built: {self.dependency_tracker.file_count} files, "
            f"{self.dependency_tracker.dependency_count} dependencies"
        )

    def build_task_dependency_graph(self, *, force: bool = False) -> TaskDependencyGraph:
        """Build the task dependency graph for parallel execution.

        This analyzes task file_hints and explicit dependencies to determine
        which tasks can safely run in parallel.

        Args:
            force: Rebuild even if already built.

        Returns:
            TaskDependencyGraph for the current pending tasks.
        """
        if self._task_dependency_graph is not None and not force:
            return self._task_dependency_graph

        # Create task dependency graph with configured options
        graph = TaskDependencyGraph(
            dependency_tracker=self._dependency_tracker if self._dependency_graph_built else None,
            use_file_overlap=self._config.parallel_file_overlap,
            use_import_analysis=self._config.parallel_import_analysis,
        )

        # Add all pending tasks to the graph
        for task in self.roadmap.pending_tasks():
            if self._is_task_pending(task):
                graph.add_task(
                    task.id,
                    file_hints=task.file_hints,
                    dependencies=task.dependencies,
                )

        # Build the graph
        graph.build()

        self._task_dependency_graph = graph
        logger.info(f"Task dependency graph built: {graph.task_count} tasks")

        return graph

    def get_parallel_groups(self) -> list[ParallelGroup]:
        """Get groups of tasks that can execute in parallel.

        Returns:
            List of ParallelGroup objects, ordered by execution level.
        """
        graph = self.build_task_dependency_graph()
        pending_ids = [
            task.id for task in self.roadmap.pending_tasks() if self._is_task_pending(task)
        ]
        return graph.get_parallel_groups(pending_only=pending_ids)

    def analyze_change_impact(
        self,
        changed_files: list[str | Path],
        *,
        check_transitive: bool = False,
    ) -> ChangeAnalysis:
        """Analyze the impact of file changes on the codebase.

        Args:
            changed_files: Files that were modified.
            check_transitive: Also check transitive dependencies.

        Returns:
            ChangeAnalysis with warnings about potentially affected files.

        Example:
            >>> analysis = executor.analyze_change_impact(["src/core.py"])
            >>> if analysis.has_warnings:
            ...     for warning in analysis.warnings:
            ...         print(f"WARNING: {warning.dependent_file} may be affected")
        """
        if not self._dependency_graph_built:
            logger.warning("Dependency graph not built - run build_dependency_graph() first")
            return ChangeAnalysis(changed_files=[Path(f) for f in changed_files])

        paths = [Path(f).resolve() for f in changed_files]
        return self.dependency_tracker.analyze_changes(
            paths,
            check_transitive=check_transitive,
        )

    def get_dependents(self, file_path: str | Path) -> list[Path]:
        """Get files that depend on a given file.

        Args:
            file_path: Path to the file.

        Returns:
            List of files that import from this file.
        """
        if not self._dependency_graph_built:
            return []
        return self.dependency_tracker.get_dependent_files(Path(file_path).resolve())

    # =========================================================================
    # Main Run Loop
    # =========================================================================

    async def run(
        self,
        on_complete: TaskCompleteCallback | None = None,
    ) -> RunSummary:
        """Run the executor loop.

        Executes pending tasks from the roadmap until:
        - All tasks are complete
        - max_tasks limit is reached
        - A failure occurs (if stop_on_failure=True)
        - Human approval is required

        Args:
            on_complete: Callback invoked after each task completes.

        Returns:
            Summary of the run including results and statistics.
        """
        self._run_started = datetime.now(UTC)
        self._tasks_this_run = 0
        self._processed_task_ids = set()  # Reset for new run
        results: list[ExecutionResult] = []

        # Phase 5.11.4: Reset file write tracking for new run
        self._file_write_tracker = {}
        self._expected_files_per_task = {}

        # Phase 5.8: Initialize run memory for this run
        if self._config.enable_run_memory:
            self._run_memory = RunMemory(
                run_id=self.state.run_id,
                started_at=self._run_started.isoformat(),
            )

        # Initialize observability
        if self._callbacks is not None:
            self._callbacks.set_run_context(run_id=self.state.run_id)

        # Start tracing span for the entire run
        run_span = self.tracer.start_span(
            "executor.run",
            attributes={
                "roadmap": str(self._roadmap_path),
                "max_tasks": self._config.max_tasks,
                "dry_run": self._config.dry_run,
            },
        )

        try:
            # Handle crash recovery
            recovered = self.state.recover()
            if recovered:
                logger.info(f"Recovered {len(recovered)} in-progress tasks")

            # Check for pending tasks
            if not self._has_pending_tasks():
                self._finalize_run(run_span, RunStatus.NO_TASKS, results)
                return self._build_summary(
                    results=results,
                    status=RunStatus.NO_TASKS,
                )

            # Build project context if needed (Phase 5.1: with token budget)
            if self._context is None:
                self._context = ProjectContext(
                    self._roadmap_path.parent,
                    max_tokens=self._config.context_max_tokens,
                )
                await self._context.build_structure()

            # Build dependency graph for multi-file awareness
            await self.build_dependency_graph()

            # Phase 5.1: Check if parallel execution is enabled
            if self._config.parallel_tasks != 1:
                # Use parallel execution loop
                logger.info(
                    f"Using parallel execution mode (max {self._config.parallel_tasks} concurrent)"
                )
                parallel_results = await self._run_parallel_loop(on_complete=on_complete)
                results.extend(parallel_results)

                # Check final status
                has_failure = any(
                    r.status in (ExecutionStatus.FAILED, ExecutionStatus.ERROR) for r in results
                )
                if has_failure and self._config.stop_on_failure:
                    self._finalize_run(run_span, RunStatus.FAILED, results)
                    return self._build_summary(
                        results=results,
                        status=RunStatus.FAILED,
                    )

                # Save final state
                self.state.save()
                self._finalize_run(run_span, RunStatus.COMPLETED, results)

                return self._build_summary(
                    results=results,
                    status=RunStatus.COMPLETED,
                )

            # Sequential execution loop (original behavior)
            while self._has_pending_tasks():
                # Check limits
                if self._reached_task_limit():
                    logger.info("Reached max_tasks limit")
                    break

                # Get next task
                task = self._get_next_pending_task()
                if task is None:
                    logger.info("No more pending tasks")
                    break

                # Mark as processed to prevent infinite loops
                self._processed_task_ids.add(task.id)

                # Notify callbacks of task start
                if self._callbacks is not None:
                    self._callbacks.on_task_start(task.id, task.title)

                # Execute task with retry support (Phase 5.6)
                result = await self._execute_task_with_retry(task)
                results.append(result)
                self._tasks_this_run += 1

                # Notify callbacks of task completion
                if self._callbacks is not None:
                    self._callbacks.on_task_end(
                        task.id,
                        success=result.success,
                        files_modified=len(result.all_files),
                        error=result.error,
                    )

                # Analyze dependency impact for successful tasks
                if result.success and result.all_files:
                    impact = self.analyze_change_impact(result.all_files)
                    if impact.has_warnings:
                        for warning in impact.warnings:
                            logger.warning(
                                f"Dependency warning: {warning.dependent_file} "
                                f"may be affected by changes to {warning.changed_file} "
                                f"(impact: {warning.impact_level.value})"
                            )
                            # Notify callbacks of dependency warning
                            if self._callbacks is not None:
                                self._callbacks.on_dependency_warning(
                                    file_path=str(warning.changed_file),
                                    impact=warning.impact_level.value,
                                    affected_files=[str(warning.dependent_file)],
                                )
                        # Store warnings in result for review
                        result.dependency_warnings = impact.warnings

                # Check for destructive operations pause (BEFORE continuing)
                if (
                    self._config.pause_before_destructive
                    and result.success
                    and self._has_destructive_changes(result)
                ):
                    logger.info("Pausing for human approval: destructive changes detected")
                    self.state.save()
                    self._last_run_results = results  # Save for review
                    self._finalize_run(run_span, RunStatus.PAUSED, results)
                    return self._build_summary(
                        results=results,
                        status=RunStatus.PAUSED,
                        paused=True,
                        pause_reason="Destructive operations detected (files deleted)",
                    )

                # Check for human approval gate (AFTER N tasks)
                if self._should_pause_for_human():
                    logger.info("Pausing for human approval: task limit reached")
                    self.state.save()
                    self._last_run_results = results  # Save for review
                    self._finalize_run(run_span, RunStatus.PAUSED, results)
                    return self._build_summary(
                        results=results,
                        status=RunStatus.PAUSED,
                        paused=True,
                        pause_reason=f"Reached {self._config.require_human_approval_after} tasks",
                    )

                # Save state periodically
                if self._tasks_this_run % self._config.save_state_every == 0:
                    self.state.save()

                # Invoke callback
                if on_complete:
                    try:
                        callback_result = on_complete(task, result)
                        if asyncio.iscoroutine(callback_result):
                            await callback_result
                    except Exception as e:
                        logger.warning(f"Callback error: {e}")

                # Create git checkpoint if enabled and task succeeded
                if result.status == ExecutionStatus.SUCCESS and self._should_checkpoint():
                    # Extract verification info from result
                    verification_passed = None
                    verification_level = None
                    if result.verification is not None:
                        verification_passed = result.verification.overall
                        # Get highest level that passed
                        if result.verification.levels_run:
                            verification_level = result.verification.levels_run[-1].value

                    checkpoint_result, _ = self._maybe_checkpoint(
                        task,
                        result,
                        verification_passed=verification_passed,
                        verification_level=verification_level,
                    )
                    # Notify callbacks of checkpoint creation
                    if (
                        checkpoint_result
                        and checkpoint_result.success
                        and self._callbacks is not None
                    ):
                        self._callbacks.on_checkpoint_created(
                            checkpoint_id=checkpoint_result.commit_sha or "unknown",
                            task_id=task.id,
                        )

                # Handle failure (SKIPPED is not a failure)
                is_real_failure = result.status in (ExecutionStatus.FAILED, ExecutionStatus.ERROR)
                if is_real_failure and self._config.stop_on_failure:
                    logger.info(f"Stopping due to failure: {task.id}")
                    self._finalize_run(run_span, RunStatus.FAILED, results)
                    return self._build_summary(
                        results=results,
                        status=RunStatus.FAILED,
                    )

            # Save final state
            self.state.save()
            self._finalize_run(run_span, RunStatus.COMPLETED, results)

            return self._build_summary(
                results=results,
                status=RunStatus.COMPLETED,
            )

        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"Executor run failed: {e}")
            run_span.set_status("error", str(e))
            self.tracer.end_span(run_span)
            if self._callbacks is not None:
                self._callbacks.on_run_end()
            raise

    # =========================================================================
    # Todo-Based Execution (Phase 5.12.6)
    # =========================================================================

    async def run_by_todos(
        self,
        on_complete: TaskCompleteCallback | None = None,
    ) -> RunSummary:
        """Execute ROADMAP by normalized todos instead of raw tasks.

        This method iterates over grouped todos, executing all source tasks
        in a single agent call per todo. This reduces LLM calls for related
        tasks and provides better context for grouped work.

        Args:
            on_complete: Callback invoked after each todo completes.
                Note: The callback receives the first source task and combined result.

        Returns:
            Summary of the run including results and statistics.
        """
        # Phase 5.13: Ensure todo manager is initialized (handles LLM normalization)
        await self.ensure_todo_manager()

        self._run_started = datetime.now(UTC)
        self._tasks_this_run = 0
        self._processed_task_ids = set()
        results: list[ExecutionResult] = []

        # Phase 5.12.7: Track files created across todos for cross-todo context
        files_created_this_run: list[str] = []

        # Phase 5.11.4: Reset file write tracking
        self._file_write_tracker = {}
        self._expected_files_per_task = {}

        # Phase 5.8: Initialize run memory
        if self._config.enable_run_memory:
            self._run_memory = RunMemory(
                run_id=self.state.run_id,
                started_at=self._run_started.isoformat(),
            )

        # Initialize observability
        if self._callbacks is not None:
            self._callbacks.set_run_context(run_id=self.state.run_id)

        # Start tracing span
        run_span = self.tracer.start_span(
            "executor.run_by_todos",
            attributes={
                "roadmap": str(self._roadmap_path),
                "max_tasks": self._config.max_tasks,
                "dry_run": self._config.dry_run,
                "execution_mode": "by_todos",
            },
        )

        try:
            # Build project context if needed
            if self._context is None:
                self._context = ProjectContext(
                    self._roadmap_path.parent,
                    max_tokens=self._config.context_max_tokens,
                )
                await self._context.build_structure()

            # Build dependency graph
            await self.build_dependency_graph()

            # Iterate over pending todos
            while True:
                todo = self.todo_manager.next_pending()
                if todo is None:
                    logger.info("No more pending todos")
                    break

                # Check task limits (count source tasks, not todos)
                if self._reached_task_limit():
                    logger.info("Reached max_tasks limit")
                    break

                # Mark todo as in-progress
                self.todo_manager.mark_in_progress(todo.id)

                # Get source tasks for this todo
                source_tasks = self.todo_manager.get_source_tasks(todo, self.roadmap)
                synthetic_task = None

                if not source_tasks:
                    # Phase 5.13: For LLM-normalized ROADMAPs (JSON-only mode),
                    # create synthetic tasks from todos when no checkbox tasks exist
                    if self.todo_manager.uses_json_only:
                        synthetic_task = self.todo_manager.create_synthetic_task(todo)
                        logger.info(f"Todo {todo.id} using synthetic task (LLM-normalized)")
                    else:
                        logger.warning(f"Todo {todo.id} has no source tasks, skipping")
                        self.todo_manager.mark_failed(todo.id, error="No source tasks found")
                        continue

                # Mark all source task IDs as processed
                for task in source_tasks:
                    self._processed_task_ids.add(task.id)
                if synthetic_task:
                    self._processed_task_ids.add(synthetic_task.id)

                # Notify callbacks
                if self._callbacks is not None:
                    task_count = len(source_tasks) if source_tasks else 1
                    self._callbacks.on_task_start(
                        f"todo-{todo.id}",
                        f"{todo.title} ({task_count} tasks)",
                    )

                # Execute grouped todo with retry support
                # Pass files created so far for cross-todo context
                result = await self._execute_todo_with_retry(
                    todo,
                    source_tasks,
                    synthetic_task=synthetic_task,
                    files_created_so_far=files_created_this_run,
                )
                results.append(result)
                self._tasks_this_run += len(source_tasks) if source_tasks else 1

                # Notify callbacks (SKIPPED counts as success for dry-run)
                if self._callbacks is not None:
                    is_success = result.success or result.status == ExecutionStatus.SKIPPED
                    self._callbacks.on_task_end(
                        f"todo-{todo.id}",
                        success=is_success,
                        files_modified=len(result.all_files),
                        error=result.error,
                    )

                # Handle result
                if result.success:
                    # Phase 5.12.7: Track files created for subsequent todos
                    files_created_this_run.extend(result.files_created or [])

                    self.todo_manager.mark_completed(
                        todo.id,
                        files_created=result.files_created,
                        sync_roadmap=True,
                    )

                    # Create checkpoint if enabled
                    if self._should_checkpoint():
                        # Use first source task or synthetic task for checkpoint
                        if source_tasks:
                            primary_task = source_tasks[0]
                            self._maybe_checkpoint(primary_task, result)
                        elif synthetic_task:
                            self._maybe_checkpoint(synthetic_task, result)
                elif result.status == ExecutionStatus.SKIPPED:
                    # Dry run or skipped - don't mark as failed, just continue
                    pass
                else:
                    self.todo_manager.mark_failed(todo.id, error=result.error)

                    if self._config.stop_on_failure:
                        logger.info(f"Stopping due to failure: todo-{todo.id}")
                        self._finalize_run(run_span, RunStatus.FAILED, results)
                        return self._build_summary(results=results, status=RunStatus.FAILED)

                # Invoke callback (pass first source task)
                if on_complete and source_tasks:
                    try:
                        callback_result = on_complete(source_tasks[0], result)
                        if asyncio.iscoroutine(callback_result):
                            await callback_result
                    except Exception as e:
                        logger.warning(f"Callback error: {e}")

                # Save state periodically
                if self._tasks_this_run % self._config.save_state_every == 0:
                    self.state.save()

            # Finalize
            self.state.save()
            self._finalize_run(run_span, RunStatus.COMPLETED, results)
            return self._build_summary(results=results, status=RunStatus.COMPLETED)

        except Exception as e:
            logger.exception(f"Executor run_by_todos failed: {e}")
            run_span.set_status("error", str(e))
            self.tracer.end_span(run_span)
            if self._callbacks is not None:
                self._callbacks.on_run_end()
            raise

    def _build_todo_display_title(
        self,
        todo: TodoItem,
        source_tasks: list[ParsedTask],
    ) -> str:
        """Build a comprehensive display title for a todo.

        Creates a descriptive title that explains what tasks are being done,
        including file paths and action verbs.

        Args:
            todo: The todo item.
            source_tasks: The source ParsedTask objects.

        Returns:
            Comprehensive title string (max 80 chars).
        """
        if len(source_tasks) == 1:
            # Single task: use the full task title
            task = source_tasks[0]
            title = task.title
            # Clean up backticks for display
            title = title.replace("`", "")
            if len(title) > 80:
                title = title[:77] + "..."
            return title

        # Multiple tasks: summarize what's being done
        # Try to extract common file pattern
        all_files: list[str] = []
        for task in source_tasks:
            all_files.extend(task.file_hints or [])

        if all_files:
            # Get unique files
            unique_files = list(dict.fromkeys(all_files))
            if len(unique_files) == 1:
                return f"{todo.title}: {unique_files[0].replace('`', '')}"
            elif len(unique_files) <= 3:
                files_str = ", ".join(f.split("/")[-1].replace("`", "") for f in unique_files)
                return f"{todo.title} ({files_str})"

        # Fallback: section title with task count
        task_count = len(source_tasks)
        return f"{todo.title} ({task_count} tasks)"

    async def _execute_todo(
        self,
        todo: TodoItem,
        source_tasks: list[ParsedTask],
        *,
        synthetic_task: Task | None = None,
        retry_context: str | None = None,
        files_created_so_far: list[str] | None = None,
    ) -> ExecutionResult:
        """Execute a grouped todo with all its source tasks.

        Args:
            todo: The todo item to execute.
            source_tasks: The source ParsedTask objects for this todo.
            synthetic_task: Optional synthetic Task for LLM-normalized ROADMAPs
                without checkbox tasks. Used when source_tasks is empty.
            retry_context: Error context from previous failed attempt.
            files_created_so_far: Files created by previous todos in this run.

        Returns:
            ExecutionResult combining all task outcomes.
        """
        started_at = datetime.now(UTC)

        # Build comprehensive display title
        display_title = self._build_todo_display_title(todo, source_tasks)

        # Determine task count for tracing
        task_count = len(source_tasks) if source_tasks else (1 if synthetic_task else 0)

        # Start tracing span
        span = self.tracer.start_span(
            "executor.execute_todo",
            attributes={
                "todo_id": todo.id,
                "todo_title": display_title,
                "source_task_count": task_count,
                "is_retry": retry_context is not None,
                "is_synthetic": synthetic_task is not None,
            },
        )

        try:
            # Build grouped prompt (include retry context and cross-todo context if present)
            prompt = await self._build_todo_prompt(
                todo,
                source_tasks,
                synthetic_task=synthetic_task,
                retry_context=retry_context,
                files_created_so_far=files_created_so_far,
            )

            if self._config.dry_run:
                logger.info(f"[DRY RUN] Would execute todo: {display_title}")
                self.tracer.end_span(span)
                return ExecutionResult(
                    task_id=f"todo-{todo.id}",
                    title=display_title,
                    status=ExecutionStatus.SKIPPED,
                    started_at=started_at,
                    completed_at=datetime.now(UTC),
                )

            # Check if agent is available
            if self._agent is None:
                logger.warning(f"No agent configured for todo: {display_title}")
                self.tracer.end_span(span)
                return ExecutionResult(
                    task_id=f"todo-{todo.id}",
                    title=display_title,
                    status=ExecutionStatus.SKIPPED,
                    error="No agent configured",
                    started_at=started_at,
                    completed_at=datetime.now(UTC),
                )

            # Execute via agent with timeout
            # Track tokens before/after for this todo
            tokens_before = 0
            if self._callbacks is not None:
                tokens_before = self._callbacks.get_metrics().total_tokens

            with self.tracer.span("executor.agent_run_todo", {"todo.id": todo.id}):
                agent_output = await asyncio.wait_for(
                    self._agent.arun(prompt),
                    timeout=self._config.agent_timeout,
                )

            # Calculate tokens used for this todo
            tokens_after = 0
            if self._callbacks is not None:
                tokens_after = self._callbacks.get_metrics().total_tokens
            tokens_used = tokens_after - tokens_before

            # Parse result
            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            # Parse files modified from agent output
            files_modified = self._parse_files_from_output(agent_output)

            self.tracer.end_span(span)
            return ExecutionResult(
                task_id=f"todo-{todo.id}",
                title=display_title,
                status=ExecutionStatus.SUCCESS,
                files_modified=files_modified,
                files_created=[],  # Will be updated by verification
                agent_output=agent_output,
                token_usage={"total": tokens_used} if tokens_used > 0 else {},
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.exception(f"Failed to execute todo {todo.id}: {e}")
            span.set_status("error", str(e))
            self.tracer.end_span(span)
            return ExecutionResult(
                task_id=f"todo-{todo.id}",
                title=display_title,
                status=ExecutionStatus.ERROR,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )

    async def _execute_todo_with_retry(
        self,
        todo: TodoItem,
        source_tasks: list[ParsedTask],
        *,
        synthetic_task: Task | None = None,
        files_created_so_far: list[str] | None = None,
    ) -> ExecutionResult:
        """Execute a todo with retry support and file verification.

        Phase 5.12.6: Implements retry loop for todo execution.
        Phase 5.12.7: Adds file verification to ensure claimed files exist.
        Phase 5.13: Supports synthetic tasks for LLM-normalized ROADMAPs.

        On failure or missing files, retries with error context so the agent
        can learn from mistakes (e.g., invalid tool calls, permission errors,
        files not created).

        Args:
            todo: The todo item to execute.
            source_tasks: The source ParsedTask objects for this todo.
            files_created_so_far: Files created by previous todos in this run.

        Returns:
            Execution result (final attempt or success).
        """
        max_attempts = self._config.retry_failed
        attempt = 1
        last_error: str | None = None
        last_missing_files: list[str] | None = None

        while attempt <= max_attempts:
            # If this is a retry, log it
            if attempt > 1 and last_error:
                logger.info(
                    "todo_retry_attempt",
                    todo_id=todo.id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    previous_error=last_error,
                )
                # Notify callbacks of retry
                if self._callbacks is not None:
                    self._callbacks.on_task_retry(
                        task_id=f"todo-{todo.id}",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        previous_error=last_error,
                    )

            # Build retry context with missing files info if applicable
            retry_context = None
            if attempt > 1 and last_error:
                retry_context = last_error
                if last_missing_files:
                    retry_context += "\n\nMISSING FILES that must be created:\n" + "\n".join(
                        f"- {f}" for f in last_missing_files[:10]
                    )

            # Execute the todo (with error context on retry and cross-todo context)
            result = await self._execute_todo(
                todo,
                source_tasks,
                synthetic_task=synthetic_task,
                retry_context=retry_context,
                files_created_so_far=files_created_so_far,
            )

            # Handle SKIPPED status (dry-run mode) - return immediately, no retry
            if result.status == ExecutionStatus.SKIPPED:
                return result

            # Check for success and verify files
            if result.success:
                # Phase 5.12.7: Verify files were actually created
                files_ok, missing_files, created_files = self._verify_todo_files_created(
                    todo, source_tasks, result
                )

                if not files_ok and attempt < max_attempts:
                    # Files missing - prepare for retry
                    missing_display = ", ".join(missing_files[:5])
                    if len(missing_files) > 5:
                        missing_display += f", ... ({len(missing_files) - 5} more)"

                    last_error = f"Files not created: {missing_display}"
                    last_missing_files = missing_files

                    logger.warning(
                        "todo_files_missing_will_retry",
                        todo_id=todo.id,
                        missing_files=missing_files,
                        attempt=attempt,
                        next_attempt=attempt + 1,
                    )
                    attempt += 1
                    continue

                # Phase 5.12.7: If files still missing on final attempt, mark as failed
                if not files_ok:
                    missing_display = ", ".join(missing_files[:5])
                    if len(missing_files) > 5:
                        missing_display += f", ... ({len(missing_files) - 5} more)"

                    logger.error(
                        "todo_files_not_created_final",
                        todo_id=todo.id,
                        missing_files=missing_files,
                        attempt=attempt,
                    )

                    display_title = self._build_todo_display_title(todo, source_tasks)
                    return ExecutionResult(
                        task_id=f"todo-{todo.id}",
                        title=display_title,
                        status=ExecutionStatus.ERROR,
                        error=f"Files not created after {attempt} attempts: {missing_display}",
                        files_created=created_files,  # Partial success
                        started_at=result.started_at,
                        completed_at=datetime.now(UTC),
                    )

                # All files verified
                if attempt > 1:
                    logger.info(
                        "todo_retry_succeeded",
                        todo_id=todo.id,
                        attempt=attempt,
                    )

                # Update result with verified created files
                result.files_created = created_files
                return result

            # Failure - prepare for retry
            last_error = result.error or "Unknown error"
            last_missing_files = None  # Clear on regular error
            logger.warning(
                f"Todo {todo.id} failed (attempt {attempt}/{max_attempts}): {last_error}"
            )

            attempt += 1

        # All retries exhausted
        logger.error(f"Todo {todo.id} failed after {max_attempts} attempts: {last_error}")
        # Build display title for the final failure result
        display_title = self._build_todo_display_title(todo, source_tasks)
        return ExecutionResult(
            task_id=f"todo-{todo.id}",
            title=display_title,
            status=ExecutionStatus.ERROR,
            error=f"Failed after {max_attempts} attempts: {last_error}",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

    async def _build_todo_prompt(
        self,
        todo: TodoItem,
        source_tasks: list[ParsedTask],
        *,
        synthetic_task: Task | None = None,
        retry_context: str | None = None,
        files_created_so_far: list[str] | None = None,
    ) -> str:
        """Build a prompt for executing a grouped todo.

        Creates a comprehensive prompt that includes all source tasks
        in the todo, allowing the agent to execute them as a logical unit.

        Args:
            todo: The todo item being executed.
            source_tasks: The source ParsedTask objects.
            synthetic_task: Optional synthetic Task for LLM-normalized ROADMAPs.
            retry_context: Error context from previous failed attempt.
            files_created_so_far: Files created by previous todos in this run.

        Returns:
            Prompt string for the agent.
        """
        sections = []

        # Determine task count
        task_count = len(source_tasks) if source_tasks else (1 if synthetic_task else 0)

        # Todo header
        sections.append(f"# Todo: {todo.title}")
        sections.append("")
        sections.append(f"**Todo ID**: {todo.id}")
        sections.append(f"**Tasks in this unit**: {task_count}")
        sections.append("")

        if todo.description:
            sections.append("## Overview")
            sections.append(todo.description)
            sections.append("")

        # Individual tasks (from source_tasks or synthetic_task)
        sections.append("## Tasks to Complete")
        sections.append("")
        sections.append("Complete all of the following tasks as a single unit of work:")
        sections.append("")

        if source_tasks:
            # Standard checkbox-based tasks
            for i, task in enumerate(source_tasks, 1):
                sections.append(f"### Task {i}: {task.title}")
                sections.append(f"**ID**: {task.id}")
                if task.description:
                    sections.append(task.description)
                if task.subtasks:
                    sections.append("")
                    sections.append("**Subtasks:**")
                    for subtask in task.subtasks:
                        status = "[x]" if subtask.completed else "[ ]"
                        sections.append(f"- {status} {subtask.title}")
                sections.append("")
        elif synthetic_task:
            # LLM-normalized synthetic task
            sections.append(f"### Task 1: {synthetic_task.title}")
            sections.append(f"**ID**: {synthetic_task.id}")
            if synthetic_task.description:
                sections.append(synthetic_task.description)
            sections.append("")

        # Aggregate file hints from all tasks
        all_file_hints: list[str] = []
        for task in source_tasks:
            all_file_hints.extend(task.file_hints or [])
        if synthetic_task and synthetic_task.file_hints:
            all_file_hints.extend(synthetic_task.file_hints)
        if todo.file_hints:
            all_file_hints.extend(todo.file_hints)

        if all_file_hints:
            unique_hints = list(dict.fromkeys(all_file_hints))  # Preserve order, dedupe
            sections.append("## Files to Work With")
            for hint in unique_hints:
                sections.append(f"- `{hint}`")
            sections.append("")

        # Aggregate code context
        all_code_context: list[str] = []
        for task in source_tasks:
            if task.code_context:
                all_code_context.extend(task.code_context)

        if all_code_context:
            sections.append("## Code Examples")
            for code in all_code_context:
                sections.append(code)
            sections.append("")

        # Phase 5.8: Include memory context
        if self._run_memory is not None:
            run_memory_budget = self._config.memory_token_budget // 2
            run_context = self._run_memory.get_context(
                current_task_id=source_tasks[0].id if source_tasks else "",
                max_tokens=run_memory_budget,
            )
            if run_context:
                sections.append("## Recent Task Context (This Run)")
                sections.append(run_context)
                sections.append("")

        if self._project_memory is not None:
            project_memory_budget = self._config.memory_token_budget // 2
            project_context = self._project_memory.get_context(
                task_title=todo.title,
                max_tokens=project_memory_budget,
            )
            if project_context:
                sections.append("## Project History")
                sections.append(project_context)
                sections.append("")

        # Project context
        if self._context:
            header_tokens = sum(len(s) // 4 for s in sections)
            context_budget = max(1000, self._config.context_max_tokens - header_tokens)

            # Use first task for context relevance
            if source_tasks:
                context_result = await self._context.get_task_context_with_budget(
                    task=source_tasks[0],
                    max_tokens=context_budget,
                )
                if context_result.content:
                    sections.append("## Project Context")
                    sections.append(context_result.content)
                    sections.append("")

        # Phase 5.12.7: Cross-todo context - files created by previous todos
        if files_created_so_far:
            sections.append("## Files Created by Previous Todos (This Run)")
            sections.append("")
            sections.append(
                "The following files were created by earlier todos in this run. "
                "These files EXIST and can be imported/referenced:"
            )
            sections.append("")
            for f in files_created_so_far[:20]:  # Limit to prevent prompt overflow
                sections.append(f"- `{f}`")
            if len(files_created_so_far) > 20:
                sections.append(f"- ... and {len(files_created_so_far) - 20} more")
            sections.append("")
            sections.append(
                "**IMPORTANT**: If your task depends on any of these files, "
                "you can import/reference them directly. If a dependency is NOT in this list, "
                "it does NOT exist yet - you must create it first or skip that part."
            )
            sections.append("")

        # Retry context (if this is a retry after failure)
        if retry_context:
            sections.append("## Previous Attempt Failed")
            sections.append("")
            sections.append(
                "**IMPORTANT**: A previous attempt to complete this todo failed. "
                "Learn from the error below and try a different approach:"
            )
            sections.append("")
            sections.append(f"```\n{retry_context}\n```")
            sections.append("")

            # Classify error and provide specific hints
            error_lower = retry_context.lower()
            hints = self._get_retry_hints(error_lower)
            for hint in hints:
                sections.append(f"- {hint}")
            sections.append("")

        # Instructions
        sections.append("## Instructions")
        sections.append("")
        sections.append(
            "Complete ALL tasks listed above as a single unit of work. "
            "These tasks are grouped because they are related and should be "
            "implemented together for consistency."
        )
        sections.append("")

        # Phase 5.12.8: Workspace boundary constraints
        workspace_root = self._roadmap_path.parent
        sections.append("## CRITICAL: Workspace Boundary")
        sections.append("")
        sections.append(f"**Workspace root**: `{workspace_root}`")
        sections.append("")
        sections.append(
            "**ALL file operations MUST stay within this workspace directory.** "
            "You CANNOT access files outside this directory."
        )
        sections.append("")
        sections.append("**When using file tools:**")
        sections.append(
            "- `ls`: Always specify a path relative to workspace root (e.g., `.` or `src/`)"
        )
        sections.append("- `read`: Use workspace-relative paths (e.g., `src/user.py`)")
        sections.append("- `write`: Use workspace-relative paths (e.g., `src/user.py`)")
        sections.append(
            "- `grep`: ALWAYS specify a path parameter (e.g., `path='.'` or `path='src/'`)"
        )
        sections.append("- `glob`: Use workspace-relative patterns (e.g., `src/**/*.py`)")
        sections.append("")
        sections.append("**NEVER:**")
        sections.append("- Access system directories (`/usr`, `/etc`, `/var`, `/bin`, `/sbin`)")
        sections.append("- Use absolute paths outside the workspace")
        sections.append("- Call `grep` without specifying a `path` parameter")
        sections.append("")

        # Phase 5.12.7: Add dependency verification instructions
        sections.append("**Before importing or referencing any file:**")
        sections.append("1. Check if it exists using the `ls` or `read` tool")
        sections.append("2. If importing from a module, verify the module file exists first")
        sections.append(
            "3. If a required file doesn't exist, either create it first or "
            "leave a TODO comment explaining the missing dependency"
        )
        sections.append("")

        return "\n".join(sections)

    def _get_retry_hints(self, error_lower: str) -> list[str]:
        """Get context-specific retry hints based on error type.

        Classifies the error and returns targeted guidance to help
        the agent avoid the same mistake on retry.

        Args:
            error_lower: Lowercase error message for pattern matching.

        Returns:
            List of hint strings for the agent.
        """
        hints: list[str] = []

        # Permission / access errors
        if any(kw in error_lower for kw in ["permission denied", "access denied", "not permitted"]):
            hints.append("**Permission error**: You tried to access files outside the workspace")
            hints.append("CRITICAL: Always specify `path='.'` when using `grep`")
            hints.append("Only use relative paths within the project directory")
            hints.append("Never access system directories (/usr, /etc, /var, /bin, /sbin)")
            hints.append("Example correct grep: `grep(pattern='User', path='.')`")

        # Path traversal errors
        elif any(kw in error_lower for kw in ["path traversal", "outside workspace"]):
            hints.append("**Path traversal blocked**: Stay within the workspace")
            hints.append("Use relative paths from the project root (e.g., `src/file.py`)")
            hints.append("Do not use `../` to navigate outside the workspace")

        # File not found errors
        elif any(kw in error_lower for kw in ["no such file", "file not found", "does not exist"]):
            hints.append("**File not found**: Create files before referencing them")
            hints.append("Check the file path - use project-relative paths")
            hints.append("List directory contents first to verify file locations")

        # Syntax / parse errors
        elif any(kw in error_lower for kw in ["syntax error", "parse error", "invalid syntax"]):
            hints.append("**Syntax error**: Review the code for typos or missing brackets")
            hints.append("Check for proper indentation in Python files")
            hints.append("Verify string quotes and escape characters")

        # Tool call errors
        elif any(kw in error_lower for kw in ["tool call", "invalid tool", "unknown tool"]):
            hints.append("**Tool error**: Use only available tools")
            hints.append("Check tool parameter types and required arguments")
            hints.append("If a tool is unavailable, find an alternative approach")

        # Timeout errors
        elif any(kw in error_lower for kw in ["timeout", "timed out"]):
            hints.append("**Timeout**: Simplify the approach to reduce execution time")
            hints.append("Break complex operations into smaller steps")
            hints.append("Avoid waiting for external resources")

        # Import / module errors
        elif any(
            kw in error_lower for kw in ["import error", "module not found", "no module named"]
        ):
            hints.append("**Import error**: Install required packages first")
            hints.append("Check that module names and paths are correct")
            hints.append("Use standard library modules when possible")

        # Git / version control errors
        elif any(kw in error_lower for kw in ["git", "commit", "repository"]):
            hints.append("**Git error**: Ensure you're in a git repository")
            hints.append("Check git configuration and permissions")
            hints.append("Stage files before committing")

        # JSON / data format errors
        elif any(kw in error_lower for kw in ["json", "decode", "serialize"]):
            hints.append("**Data format error**: Validate JSON structure before parsing")
            hints.append("Escape special characters properly")
            hints.append("Use proper data types (strings, numbers, booleans)")

        # Generic fallback
        else:
            hints.append("Review the error message carefully")
            hints.append("Try a simpler or alternative implementation approach")
            hints.append("If a tool failed, try different parameters or a different tool")

        # Always include these general hints
        hints.append("Do NOT repeat the same action that caused the error")

        return hints

    def _finalize_run(
        self,
        run_span: Span,
        status: RunStatus,
        results: list[ExecutionResult],
    ) -> None:
        """Finalize the run by closing tracing span and notifying callbacks.

        Args:
            run_span: The tracing span for the run.
            status: Final run status.
            results: All execution results from the run.
        """
        # Phase 5.8: Update project memory from run memory
        if self._project_memory is not None and self._run_memory is not None:
            try:
                self._project_memory.update_from_run(self._run_memory)
                logger.debug(
                    "project_memory_updated",
                    outcomes_count=len(self._run_memory.outcomes),
                )
            except Exception as e:
                logger.warning(f"Failed to update project memory: {e}")

        # Phase 5.11.4: Generate and log file write summary if verify_writes enabled
        if self._config.verify_writes:
            file_summary = self._generate_file_write_summary()
            if file_summary.total_expected > 0:
                logger.info(
                    "file_write_summary",
                    total_expected=file_summary.total_expected,
                    total_created=file_summary.total_created,
                    total_verified=file_summary.total_verified,
                    missing_count=len(file_summary.missing_files),
                )
                # Print summary to console
                print(file_summary.summary_text())

        # Set span attributes
        run_span.set_attribute("run.status", status.value)
        run_span.set_attribute("run.tasks_completed", len([r for r in results if r.success]))
        run_span.set_attribute("run.tasks_failed", len([r for r in results if not r.success]))

        # Set error status if failed
        if status == RunStatus.FAILED:
            run_span.set_status("error", "Run failed")

        # End the span
        self.tracer.end_span(run_span)

        # Notify callbacks
        if self._callbacks is not None:
            self._callbacks.on_run_end()

    # =========================================================================
    # Parallel Execution (Phase 5.1)
    # =========================================================================

    async def _execute_parallel_group(
        self,
        group: ParallelGroup,
        *,
        on_complete: TaskCompleteCallback | None = None,
    ) -> list[ExecutionResult]:
        """Execute a group of tasks in parallel.

        Args:
            group: The parallel group to execute.
            on_complete: Optional callback after each task completes.

        Returns:
            List of execution results for all tasks in the group.
        """
        max_concurrent = self._config.parallel_tasks
        if max_concurrent == 0:
            # Auto: use number of tasks in group (capped at 4)
            max_concurrent = min(len(group.tasks), 4)

        logger.info(
            f"Executing parallel group level {group.level}: "
            f"{len(group.tasks)} tasks, max {max_concurrent} concurrent"
        )

        # Use a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(task_id: str) -> ExecutionResult:
            """Execute a single task with semaphore control."""
            async with semaphore:
                # Find the task
                task = None
                for t in self.roadmap.pending_tasks():
                    if t.id == task_id:
                        task = t
                        break

                if task is None:
                    logger.warning(f"Task {task_id} not found in pending tasks")
                    return ExecutionResult(
                        task_id=task_id,
                        status=ExecutionStatus.SKIPPED,
                        error="Task not found",
                        started_at=datetime.now(UTC),
                        completed_at=datetime.now(UTC),
                    )

                # Mark as processed
                self._processed_task_ids.add(task.id)

                # Notify callbacks of task start
                if self._callbacks is not None:
                    self._callbacks.on_task_start(task.id, task.title)

                # Execute the task with retry support (Phase 5.6)
                result = await self._execute_task_with_retry(task)

                # Notify callbacks of task completion
                if self._callbacks is not None:
                    self._callbacks.on_task_end(
                        task.id,
                        success=result.success,
                        files_modified=len(result.all_files),
                        error=result.error,
                    )

                # Invoke callback
                if on_complete:
                    try:
                        callback_result = on_complete(task, result)
                        if asyncio.iscoroutine(callback_result):
                            await callback_result
                    except Exception as e:
                        logger.warning(f"Callback error: {e}")

                return result

        # Execute all tasks in the group concurrently
        tasks = [execute_with_semaphore(task_id) for task_id in group.tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        final_results: list[ExecutionResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_id = group.tasks[i]
                logger.error(f"Task {task_id} raised exception: {result}")
                final_results.append(
                    ExecutionResult(
                        task_id=task_id,
                        status=ExecutionStatus.ERROR,
                        error=str(result),
                        started_at=datetime.now(UTC),
                        completed_at=datetime.now(UTC),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _run_parallel_loop(
        self,
        *,
        on_complete: TaskCompleteCallback | None = None,
    ) -> list[ExecutionResult]:
        """Run the execution loop with parallel task execution.

        This executes tasks in dependency order, running independent
        tasks in parallel while respecting file overlap constraints.

        Args:
            on_complete: Optional callback after each task completes.

        Returns:
            List of all execution results.
        """
        results: list[ExecutionResult] = []

        # Build task dependency graph
        self.build_task_dependency_graph(force=True)

        # Get parallel execution groups
        groups = self.get_parallel_groups()

        if not groups:
            logger.info("No parallel groups to execute")
            return results

        logger.info(
            f"Parallel execution plan: {len(groups)} levels, "
            f"{sum(g.size for g in groups)} total tasks"
        )

        # Execute each level in order
        for group in groups:
            # Check limits before each group
            if self._reached_task_limit():
                logger.info("Reached max_tasks limit during parallel execution")
                break

            # Filter out tasks that shouldn't be processed
            active_tasks = [
                task_id for task_id in group.tasks if task_id not in self._processed_task_ids
            ]

            if not active_tasks:
                continue

            # Limit tasks to remaining budget (Phase 5.1: respect max_tasks)
            if self._config.max_tasks > 0:
                remaining_budget = self._config.max_tasks - self._tasks_this_run
                if remaining_budget <= 0:
                    logger.info("Reached max_tasks limit during parallel execution")
                    break
                if len(active_tasks) > remaining_budget:
                    logger.info(
                        f"Limiting parallel group to {remaining_budget} tasks "
                        f"(max_tasks={self._config.max_tasks})"
                    )
                    active_tasks = active_tasks[:remaining_budget]

            # Create a filtered group
            filtered_group = ParallelGroup(
                tasks=active_tasks,
                level=group.level,
            )

            # Execute the group
            group_results = await self._execute_parallel_group(
                filtered_group,
                on_complete=on_complete,
            )

            # Process results
            for result in group_results:
                results.append(result)
                self._tasks_this_run += 1

                # Analyze dependency impact for successful tasks
                if result.success and result.all_files:
                    impact = self.analyze_change_impact(result.all_files)
                    if impact.has_warnings:
                        for warning in impact.warnings:
                            logger.warning(
                                f"Dependency warning: {warning.dependent_file} "
                                f"may be affected by changes to {warning.changed_file} "
                                f"(impact: {warning.impact_level.value})"
                            )
                            if self._callbacks is not None:
                                self._callbacks.on_dependency_warning(
                                    file_path=str(warning.changed_file),
                                    impact=warning.impact_level.value,
                                    affected_files=[str(warning.dependent_file)],
                                )
                        result.dependency_warnings = impact.warnings

                # Create checkpoint if enabled and task succeeded
                if result.status == ExecutionStatus.SUCCESS and self._should_checkpoint():
                    # Find the task for checkpoint
                    task = None
                    for t in self.roadmap.all_tasks():
                        if t.id == result.task_id:
                            task = t
                            break

                    if task:
                        verification_passed = None
                        verification_level = None
                        if result.verification is not None:
                            verification_passed = result.verification.overall
                            if result.verification.levels_run:
                                verification_level = result.verification.levels_run[-1].value

                        checkpoint_result, _ = self._maybe_checkpoint(
                            task,
                            result,
                            verification_passed=verification_passed,
                            verification_level=verification_level,
                        )
                        if (
                            checkpoint_result
                            and checkpoint_result.success
                            and self._callbacks is not None
                        ):
                            self._callbacks.on_checkpoint_created(
                                checkpoint_id=checkpoint_result.commit_sha or "unknown",
                                task_id=task.id,
                            )

            # Check for failures after group completes
            failed_results = [
                r
                for r in group_results
                if r.status in (ExecutionStatus.FAILED, ExecutionStatus.ERROR)
            ]
            if failed_results and self._config.stop_on_failure:
                logger.info(f"Stopping due to {len(failed_results)} failures in parallel group")
                break

            # Save state after each group
            self.state.save()

            # Rebuild task graph for next level (tasks may have changed status)
            if groups.index(group) < len(groups) - 1:
                self._task_dependency_graph = None  # Force rebuild

        return results

    # =========================================================================
    # Task Execution
    # =========================================================================

    async def _execute_task(
        self,
        task: ParsedTask,
        *,
        retry_context: str | None = None,
    ) -> ExecutionResult:
        """Execute a single task.

        Args:
            task: The task to execute.
            retry_context: Additional context from a failed retry attempt (Phase 5.6).

        Returns:
            Execution result.
        """
        started_at = datetime.now(UTC)
        logger.info(
            "task_execution_started",
            task_id=task.id,
            title=task.title,
        )

        # Handle dry run (don't mark as started - this is just a preview)
        if self._config.dry_run:
            logger.info(
                "task_dry_run",
                task_id=task.id,
            )
            if self._callbacks is not None:
                self._callbacks.on_task_skip(task.id, "dry_run")
            return ExecutionResult(
                task_id=task.id,
                status=ExecutionStatus.SKIPPED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )

        # Mark as started (only for real execution)
        self.state.mark_started(task.id, agent_run_id=self.state.run_id)

        # Start tracing span for task execution
        task_span = self.tracer.start_span(
            "executor.execute_task",
            attributes={
                "task.id": task.id,
                "task.title": task.title,
                "task.phase": task.phase_id,
            },
        )

        try:
            # Build prompt with tracing (include retry context if present)
            with self.tracer.span("executor.build_prompt", {"task.id": task.id}):
                prompt = await self._build_prompt(task, retry_context=retry_context)
                log_task_context(
                    task_id=task.id,
                    context_tokens=len(prompt) // 4,  # Rough estimate
                    files_included=len(task.file_hints or []),
                    file_hints=task.file_hints,
                )

            # Execute via agent
            if self._agent is None:
                # No agent provided - return placeholder
                logger.warning(
                    "task_skipped_no_agent",
                    task_id=task.id,
                )
                result = ExecutionResult(
                    task_id=task.id,
                    status=ExecutionStatus.SKIPPED,
                    error="No agent configured",
                    started_at=started_at,
                    completed_at=datetime.now(UTC),
                )
                self.state.mark_failed(task.id, error="No agent configured")
                task_span.set_status("error", "No agent configured")
                self.tracer.end_span(task_span)
                if self._callbacks is not None:
                    self._callbacks.on_task_skip(task.id, "no_agent")
                return result

            # Run agent with timeout and tracing
            with self.tracer.span("executor.agent_run", {"task.id": task.id}):
                agent_output = await asyncio.wait_for(
                    self._agent.arun(prompt),
                    timeout=self._config.agent_timeout,
                )

            # Parse agent output for files modified
            files_modified = self._parse_files_from_output(agent_output)
            task_span.set_attribute("files.modified", len(files_modified))

            # Verify if enabled (Phase 5.9.2: skip for agent mode)
            verification: VerificationResult | None = None
            should_verify = (
                not self._config.skip_verification
                and self._config.verify_mode != VerifyMode.AGENT
                and self._config.verify_mode != VerifyMode.SKIP
            )
            if should_verify:
                # Determine which levels to verify
                verification_levels = self._get_verification_levels()

                with self.tracer.span("executor.verify", {"task.id": task.id}):
                    verification = await self.verifier.verify(
                        task.to_task(),
                        levels=verification_levels,
                    )

                # Log verification result
                if verification:
                    # Get highest level that was run
                    highest_level = "none"
                    if verification.levels_run:
                        highest_level = verification.levels_run[-1].value
                    log_verification_result(
                        task_id=task.id,
                        passed=verification.overall,
                        verification_level=highest_level,
                        duration_ms=verification.total_duration_ms,
                    )

                if not verification.overall:
                    # Verification failed - get first failure message
                    failures = verification.get_failures() + verification.get_errors()
                    error = (
                        failures[0].error or failures[0].message
                        if failures
                        else "Verification failed"
                    )
                    self.state.mark_failed(
                        task.id,
                        error=error,
                        category=FailureCategory.SYNTAX_ERROR,
                    )
                    self.failure_analyzer.record_failure(
                        task=task.to_task(),
                        category=FailureCategory.SYNTAX_ERROR,
                        error_message=error,
                        verification_result=verification,
                    )

                    task_span.set_status("error", f"Verification failed: {error}")
                    self.tracer.end_span(task_span)

                    return ExecutionResult(
                        task_id=task.id,
                        status=ExecutionStatus.FAILED,
                        files_modified=files_modified,
                        agent_output=agent_output,
                        verification=verification,
                        error=error,
                        started_at=started_at,
                        completed_at=datetime.now(UTC),
                    )

            # Success
            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            self.state.mark_completed(
                task.id,
                files_modified=files_modified,
            )

            # Phase 5.8: Extract outcome and record in run memory
            if self._run_memory is not None:
                try:
                    await self._extract_and_record_outcome(
                        task=task,
                        agent_output=agent_output,
                        duration_seconds=duration_ms / 1000,
                        status="completed",
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract outcome: {e}")

            # Sync completed task to ROADMAP.md immediately via TodoManager
            if self._config.sync_roadmap:
                try:
                    # Use todo manager for smart syncing
                    updated = self.todo_manager.mark_task_completed(task.id, sync_roadmap=True)
                    # Also update state's sync (for backwards compatibility)
                    if updated == 0:
                        # Fallback to state sync if todo manager didn't match
                        updated = self.state.sync_to_roadmap()
                    if updated > 0:
                        logger.info("roadmap_synced", checkboxes_updated=updated)
                except Exception as sync_err:
                    logger.warning("roadmap_sync_failed", error=str(sync_err))

            task_span.set_attribute("task.duration_ms", duration_ms)
            task_span.set_attribute("task.success", True)
            self.tracer.end_span(task_span)

            logger.info(
                "task_execution_completed",
                task_id=task.id,
                duration_ms=duration_ms,
                files_modified=len(files_modified),
            )

            return ExecutionResult(
                task_id=task.id,
                status=ExecutionStatus.SUCCESS,
                files_modified=files_modified,
                agent_output=agent_output,
                verification=verification,
                duration_ms=duration_ms,
                started_at=started_at,
                completed_at=completed_at,
            )

        except TimeoutError:
            error = f"Agent timeout after {self._config.agent_timeout}s"
            logger.error(
                "task_execution_timeout",
                task_id=task.id,
                timeout_seconds=self._config.agent_timeout,
            )
            self.state.mark_failed(task.id, error=error)
            task_span.set_status("error", error)
            self.tracer.end_span(task_span)
            return ExecutionResult(
                task_id=task.id,
                status=ExecutionStatus.ERROR,
                error=error,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )

        except Exception as e:
            error = str(e)
            logger.exception(
                "task_execution_error",
                task_id=task.id,
                error=error,
            )
            self.state.mark_failed(task.id, error=error)
            task_span.set_status("error", error)
            self.tracer.end_span(task_span)
            return ExecutionResult(
                task_id=task.id,
                status=ExecutionStatus.ERROR,
                error=error,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )

    # =========================================================================
    # Outcome Extraction (Phase 5.8)
    # =========================================================================

    async def _extract_and_record_outcome(
        self,
        task: ParsedTask,
        agent_output: str,
        duration_seconds: float,
        status: str,
    ) -> TaskOutcome:
        """Extract outcome from agent output and record in run memory.

        Phase 5.8: Uses outcome_extractor to parse agent response and
        creates a TaskOutcome for the run memory.

        Args:
            task: The task that was executed.
            agent_output: Raw output from the agent.
            duration_seconds: How long the task took.
            status: Task completion status ("completed" or "failed").

        Returns:
            The extracted TaskOutcome.
        """
        # Get LLM for extraction if configured
        llm = None
        if self._config.extract_outcomes_with_llm and self._llm is not None:
            llm = self._llm

        # Extract outcome from agent response
        extraction = await extract_outcome(
            agent_response=agent_output,
            workspace_root=self._roadmap_path.parent,
            task_id=task.id,
            task_title=task.title,
            llm=llm,
        )

        # Create TaskOutcome
        outcome = TaskOutcome(
            task_id=task.id,
            title=task.title,
            status=status,
            files=extraction.files,
            key_decisions=extraction.key_decisions,
            summary=extraction.summary,
            duration_seconds=duration_seconds,
            tokens_used=0,  # Token tracking done elsewhere
        )

        # Record in run memory
        if self._run_memory is not None:
            self._run_memory.add_outcome(outcome)

        logger.debug(
            "outcome_extracted",
            task_id=task.id,
            files_count=len(extraction.files),
            decisions_count=len(extraction.key_decisions),
            summary_length=len(extraction.summary),
        )

        return outcome

    # =========================================================================
    # Retry Logic (Phase 5.6)
    # =========================================================================

    async def _execute_task_with_retry(
        self,
        task: ParsedTask,
        *,
        retry_context: str | None = None,
    ) -> ExecutionResult:
        """Execute a task with retry support using adaptive planning.

        Phase 5.6: Implements retry loop with automatic fix application.

        This method:
        1. Executes the task using _execute_task()
        2. On failure, analyzes the error and applies auto-fixes
        3. Retries with enriched context up to retry_failed times
        4. Tracks retry metrics via callbacks

        Args:
            task: The task to execute.
            retry_context: Additional context from previous failure (for retries).

        Returns:
            Execution result (final attempt or success).
        """
        from ai_infra.executor.adaptive import AdaptiveMode

        max_attempts = self._config.retry_failed
        attempt = 1

        # Track the last result for potential retry
        last_result: ExecutionResult | None = None
        last_error: str | None = None
        # Phase 5.11.3: Track missing files for specialized retry context
        last_missing_files: list[str] | None = None

        while attempt <= max_attempts:
            # Build task with retry context if this is a retry
            if attempt > 1 and (last_error or last_missing_files):
                # Phase 5.11.3: Use specialized context for missing files
                if last_missing_files:
                    retry_context = self._build_missing_files_context(
                        task, last_missing_files, attempt
                    )
                else:
                    retry_context = self._build_retry_context(task, last_error or "", attempt)
                logger.info(
                    "task_retry_attempt",
                    task_id=task.id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    previous_error=last_error,
                    missing_files=last_missing_files,
                )
                # Notify callbacks of retry
                if self._callbacks is not None:
                    self._callbacks.on_task_retry(
                        task_id=task.id,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        previous_error=last_error,
                    )

            # Execute the task
            result = await self._execute_task(task, retry_context=retry_context)

            # Success - verify files were actually created (Phase 5.11.1)
            if result.success:
                # Phase 5.11.4: Use enhanced verification with checksum if enabled
                if self._config.verify_writes:
                    files_ok, missing_files_list = self._verify_and_record_files(task, result)
                else:
                    # Standard verification (Phase 5.11.1)
                    files_ok, missing_files_list = self._verify_files_created(task, result)

                if not files_ok and attempt < max_attempts:
                    # Phase 5.11.3: Files missing - retry with explicit file instructions
                    # Get absolute paths for the log message
                    project_root = self._roadmap_path.parent
                    absolute_paths = [str(project_root / f) for f in missing_files_list]

                    # Build display list for error message
                    missing_display = ", ".join(missing_files_list[:5])
                    if len(missing_files_list) > 5:
                        missing_display += f", ... ({len(missing_files_list) - 5} more)"

                    last_error = f"Files not created: {missing_display}"
                    last_result = result
                    # Phase 5.11.3: Store missing files for specialized retry context
                    last_missing_files = missing_files_list

                    logger.warning(
                        "task_files_missing_will_retry",
                        task_id=task.id,
                        missing_files=missing_files_list,
                        absolute_paths=absolute_paths,
                        attempt=attempt,
                        next_attempt=attempt + 1,
                    )
                    attempt += 1
                    continue

                # All files verified or no retries left
                # Reset missing files tracker on success
                last_missing_files = None
                if attempt > 1:
                    logger.info(
                        "task_retry_succeeded",
                        task_id=task.id,
                        attempt=attempt,
                    )
                    if self._callbacks is not None:
                        self._callbacks.on_retry_success(task.id, attempt)
                return result

            # Failure - check if we should retry
            last_result = result
            last_error = result.error
            # Clear missing files on regular error (not a file creation issue)
            last_missing_files = None

            # Check adaptive mode - only retry in AUTO_FIX mode
            if self.plan_analyzer.mode == AdaptiveMode.NO_ADAPT:
                logger.debug(
                    "no_retry_no_adapt_mode",
                    task_id=task.id,
                )
                return result

            if self.plan_analyzer.mode == AdaptiveMode.SUGGEST:
                # In SUGGEST mode, generate suggestions but don't auto-retry
                suggestions = await self._handle_failure_with_adaptive_planning(task, result)
                result.suggestions = suggestions
                return result

            # AUTO_FIX mode - retry with enhanced context
            # Phase 5.7: Agent figures out fixes via retry context, not hardcoded patterns
            if attempt < max_attempts:
                logger.info(
                    "preparing_retry_with_context",
                    task_id=task.id,
                    next_attempt=attempt + 1,
                    error_preview=last_error[:200] if last_error else None,
                )

            attempt += 1

        # All retries exhausted
        logger.warning(
            "task_retries_exhausted",
            task_id=task.id,
            attempts=max_attempts,
            final_error=last_error,
        )
        if self._callbacks is not None:
            self._callbacks.on_retries_exhausted(
                task_id=task.id,
                attempts=max_attempts,
                final_error=last_error,
            )

        # Return the last result (failure)
        return last_result or result

    async def _apply_pending_fixes(
        self,
        task: ParsedTask,
        result: ExecutionResult,
    ) -> int:
        """Apply auto-fixes suggested by adaptive planning.

        .. deprecated:: Phase 5.7
            Hardcoded auto-fixes are no longer applied. The agent now figures
            out fixes using its existing tools (write_file, edit_file, terminal)
            via the enhanced retry context. This method is kept for backwards
            compatibility but always returns 0.

        Phase 5.6 (legacy): Applied safe fixes before retrying.

        Args:
            task: The failed task.
            result: The failed execution result.

        Returns:
            Number of fixes successfully applied (always 0 in Phase 5.7+).
        """
        # Phase 5.7: Hardcoded fixes disabled.
        # The agent now uses its existing tools to fix issues.
        # This method is kept for backwards compatibility but does nothing.
        logger.debug(
            "hardcoded_fixes_disabled",
            task_id=task.id,
            reason="Phase 5.7: Agent handles fixes via retry context",
        )
        return 0

    def _build_retry_context(
        self,
        task: ParsedTask,
        previous_error: str,
        attempt: int,
    ) -> str:
        """Build enriched context for a retry attempt.

        Phase 5.7: Language-agnostic error recovery context.

        Instead of hardcoded fix types, this provides the agent with:
        - The exact error from the previous attempt
        - Permission to fix ANY file in the project
        - Clear instructions to find and fix the root cause

        The agent uses its existing tools (read_file, write_file, edit_file,
        ls, grep, terminal) to investigate and fix issues.

        Args:
            task: The task being retried.
            previous_error: Error from the previous attempt.
            attempt: Current attempt number.

        Returns:
            Context string to include in the retry prompt.
        """
        context = [
            "",
            f"## Retry Attempt {attempt}",
            "",
            "### Error from Previous Attempt",
            "",
            "```",
            previous_error,
            "```",
            "",
            "### Your Task",
            "",
            "1. **Analyze** the error message above carefully",
            "2. **Investigate** using `ls` and `read_file` to find the problematic code",
            "3. **Identify** the root cause (wrong path, missing file, bad import, syntax error, etc.)",
            "4. **Fix** the issue using `write_file` or `edit_file`",
            "5. **Complete** the original task after fixing the issue",
            "",
            "### Important Rules",
            "",
            "- You can modify **ANY file** in the project, not just task-specific files",
            "- Fix the **root cause**, don't create workarounds or empty placeholder files",
            "- If a previous task created incorrect code, **fix it**",
            "- If the ROADMAP.md task description is wrong, you may **edit it**",
            "- Use `grep` to find all occurrences if needed",
            "",
            "### Common Issues to Look For",
            "",
            "- Wrong import path (module exists but path is incorrect)",
            "- Missing module or package (needs to be created with real implementation)",
            "- Incorrect function/class name (typo or mismatch)",
            "- Missing dependency in config files (package.json, pyproject.toml, etc.)",
            "- Syntax errors in generated code",
            "",
        ]
        return "\n".join(context)

    def _build_missing_files_context(
        self,
        task: ParsedTask,
        missing_files: list[str],
        attempt: int,
    ) -> str:
        """Build retry context specifically for missing file creation.

        Phase 5.11.3: When files expected from a task were not created,
        provide explicit instructions with absolute paths.

        Args:
            task: The task that should have created files.
            missing_files: List of relative file paths that were not created.
            attempt: Current retry attempt number.

        Returns:
            Context string with explicit file creation instructions.
        """
        # Get project root for absolute paths
        project_root = self._roadmap_path.parent

        # Build absolute path list
        absolute_paths: list[str] = []
        for rel_path in missing_files:
            abs_path = project_root / rel_path
            absolute_paths.append(str(abs_path))

        context = [
            "",
            f"## Retry Attempt {attempt} - Missing Files",
            "",
            "### Critical Issue",
            "",
            "The previous attempt reported success but **files were not created on disk**.",
            "This is a file write reliability issue that must be fixed.",
            "",
            "### Files That MUST Be Created",
            "",
            "The following files do not exist and must be created:",
            "",
        ]

        # Add each missing file with absolute path
        for i, (rel_path, abs_path) in enumerate(zip(missing_files, absolute_paths), 1):
            context.append(f"{i}. `{rel_path}`")
            context.append(f"   - Absolute path: `{abs_path}`")

        context.extend(
            [
                "",
                "### Instructions",
                "",
                "1. **Create each file** listed above with real, working content",
                "2. **Do NOT create empty files** - each must have proper implementation",
                "3. **Use write_file** to ensure files are written to disk",
                "4. **Verify** by listing the directory after creation",
                "",
                "### Original Task",
                "",
                f"**{task.title}**",
                "",
                task.description,
                "",
                "Complete the original task by creating the missing files.",
                "",
            ]
        )

        return "\n".join(context)

    # =========================================================================
    # Prompt Building
    # =========================================================================

    async def _build_prompt(
        self,
        task: ParsedTask,
        *,
        retry_context: str | None = None,
    ) -> str:
        """Build a prompt for the agent.

        Uses Phase 5.1 context building with token budget management.

        Args:
            task: The task to build a prompt for.
            retry_context: Additional context from a failed retry (Phase 5.6).

        Returns:
            Prompt string for the agent.
        """
        sections = []

        # Task section
        sections.append(f"# Task: {task.title}")
        sections.append("")
        sections.append(f"**ID**: {task.id}")
        sections.append(f"**Phase**: {task.phase_id}")
        sections.append(f"**Section**: {task.section_id}")
        sections.append("")

        if task.description:
            sections.append("## Description")
            sections.append(task.description)
            sections.append("")

        # Phase 5.6: Include retry context if present
        if retry_context:
            sections.append(retry_context)

        # Phase 5.8: Include run memory context (recent task outcomes)
        if self._run_memory is not None:
            # Allocate portion of memory budget for run memory
            run_memory_budget = self._config.memory_token_budget // 2
            run_context = self._run_memory.get_context(
                current_task_id=task.id,
                max_tokens=run_memory_budget,
            )
            if run_context:
                sections.append("## Recent Task Context (This Run)")
                sections.append(run_context)
                sections.append("")

        # Phase 5.8: Include project memory context (cross-run persistence)
        if self._project_memory is not None:
            # Allocate remaining memory budget for project memory
            project_memory_budget = self._config.memory_token_budget // 2
            project_context = self._project_memory.get_context(
                task_title=task.title,
                max_tokens=project_memory_budget,
            )
            if project_context:
                sections.append("## Project History")
                sections.append(project_context)
                sections.append("")

        # File hints
        if task.file_hints:
            sections.append("## Files to Work With")
            for hint in task.file_hints:
                sections.append(f"- `{hint}`")
            sections.append("")

        # Code context
        if task.code_context:
            sections.append("## Code Examples")
            for code in task.code_context:
                sections.append(code)
            sections.append("")

        # Subtasks
        if task.subtasks:
            sections.append("## Subtasks")
            for subtask in task.subtasks:
                status = "[x]" if subtask.completed else "[ ]"
                sections.append(f"- {status} {subtask.title}")
            sections.append("")

        # Project context (Phase 5.1: with budget management)
        if self._context:
            # Calculate remaining budget for context
            header_tokens = sum(len(s) // 4 for s in sections)  # Rough estimate
            context_budget = max(1000, self._config.context_max_tokens - header_tokens)

            # Get relevant context using budget-aware method
            context_result = await self._context.get_task_context_with_budget(
                task.to_task(),
                max_tokens=context_budget,
            )
            if context_result.content:
                sections.append("## Relevant Project Context")
                sections.append(context_result.content)
                sections.append("")

                # Log context building metrics
                logger.debug(
                    "context_built",
                    task_id=task.id,
                    tokens_used=context_result.tokens_used,
                    sections_included=context_result.sections_included,
                    sections_truncated=context_result.sections_truncated,
                    cache_hits=context_result.cache_hits,
                    build_time_ms=context_result.build_time_ms,
                )

        # Instructions
        sections.append("## Instructions")
        sections.append(
            "Complete this task by making the necessary code changes. "
            "Follow the project's existing patterns and conventions. "
            "Ensure all code is well-documented and tested."
        )

        # Phase 5.9.2: Add verification requirement for agent mode
        if self._config.verify_mode == VerifyMode.AGENT:
            sections.append("")
            sections.append("## VERIFICATION REQUIREMENT")
            sections.append("")
            sections.append("After completing this task, you MUST verify your work by:")
            sections.append("1. Writing a test file (if not already done)")
            sections.append("2. Running the appropriate test command for this project")
            sections.append("3. Ensuring all tests pass before marking complete")
            sections.append("")
            sections.append(
                "Use the terminal to run: `pytest`, `npm test`, `cargo test`, "
                "`go test`, or `make test` based on the project type."
            )
            sections.append("")
            sections.append(
                "Do NOT mark this task as complete until tests pass. "
                "If tests fail, fix the issues and re-run until they pass."
            )

        return "\n".join(sections)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _is_task_pending(self, task: ParsedTask) -> bool:
        """Check if a task is truly pending (not processed or completed in state)."""
        if task.id in self._processed_task_ids:
            return False
        # Check if already completed in state (from a previous run)
        task_status = self.state.get_status(task.id)
        if task_status == TaskStatus.COMPLETED:
            return False
        return True

    def _has_pending_tasks(self) -> bool:
        """Check if there are pending tasks that haven't been processed yet."""
        for task in self.roadmap.pending_tasks():
            if self._is_task_pending(task):
                return True
        return False

    def _get_next_pending_task(self) -> ParsedTask | None:
        """Get the next pending task that hasn't been processed."""
        for task in self.roadmap.pending_tasks():
            if self._is_task_pending(task):
                return task
        return None

    def _reached_task_limit(self) -> bool:
        """Check if max_tasks limit has been reached."""
        if self._config.max_tasks == 0:
            return False
        return self._tasks_this_run >= self._config.max_tasks

    def _should_pause_for_human(self) -> bool:
        """Check if execution should pause for human approval."""
        if self._config.require_human_approval_after == 0:
            return False
        return self._tasks_this_run >= self._config.require_human_approval_after

    def _is_destructive_task(self, task: ParsedTask) -> bool:
        """Check if a task description indicates destructive operations.

        This performs a heuristic check on the task description and file hints
        to determine if the task may involve file deletions.

        Args:
            task: The task to check.

        Returns:
            True if the task appears to involve destructive operations.
        """
        # Keywords that suggest file deletion
        destructive_keywords = [
            "delete",
            "remove",
            "drop",
            "unlink",
            "erase",
            "destroy",
            "clean up",
            "cleanup",
            "prune",
            "purge",
        ]

        # Check task title and description
        text_to_check = f"{task.title} {task.description}".lower()

        for keyword in destructive_keywords:
            if keyword in text_to_check:
                return True

        return False

    def _has_destructive_changes(self, result: ExecutionResult) -> bool:
        """Check if an execution result contains destructive changes.

        Args:
            result: The execution result to check.

        Returns:
            True if files were deleted.
        """
        return len(result.files_deleted) > 0

    async def _handle_failure_with_adaptive_planning(
        self,
        task: ParsedTask,
        result: ExecutionResult,
    ) -> list[PlanSuggestion]:
        """Analyze a failure and optionally apply fixes in adaptive planning mode.

        Phase 5.5: Adaptive Planning support.

        This method:
        1. Analyzes the failure to generate suggestions
        2. In AUTO_FIX mode, applies safe suggestions automatically
        3. In SUGGEST mode, returns suggestions for display
        4. In NO_ADAPT mode, returns an empty list

        Args:
            task: The task that failed.
            result: The failed execution result.

        Returns:
            List of suggestions (empty in NO_ADAPT mode or if all were applied).
        """
        from ai_infra.executor.adaptive import AdaptiveMode

        analyzer = self.plan_analyzer
        suggestions = analyzer.analyze_failure(task, result)

        if not suggestions:
            logger.debug("no_adaptive_suggestions", task_id=task.id)
            return []

        logger.info(
            "adaptive_suggestions_generated",
            task_id=task.id,
            suggestion_count=len(suggestions),
            mode=analyzer.mode.value,
        )

        if analyzer.mode == AdaptiveMode.NO_ADAPT:
            return []

        if analyzer.mode == AdaptiveMode.AUTO_FIX:
            # Apply safe suggestions automatically
            safe_suggestions = analyzer.can_auto_fix(suggestions)
            applied = []
            remaining = []

            for suggestion in suggestions:
                if suggestion in safe_suggestions:
                    apply_result = analyzer.apply_suggestion(suggestion, force=True)
                    if apply_result.success:
                        logger.info(
                            "auto_fix_applied",
                            task_id=task.id,
                            suggestion_type=suggestion.suggestion_type.value,
                            description=suggestion.description,
                        )
                        applied.append(suggestion)
                    else:
                        logger.warning(
                            "auto_fix_failed",
                            task_id=task.id,
                            suggestion_type=suggestion.suggestion_type.value,
                            error=apply_result.message,
                        )
                        remaining.append(suggestion)
                else:
                    # Not safe for auto-fix, keep for user review
                    remaining.append(suggestion)

            return remaining

        # SUGGEST mode - return all suggestions for display
        return suggestions

    def get_pending_suggestions(self) -> list[PlanSuggestion]:
        """Get pending suggestions from the most recent failure analysis.

        Returns:
            List of suggestions that need user approval.
        """
        if hasattr(self, "_pending_suggestions"):
            return self._pending_suggestions
        return []

    def apply_suggestion(
        self,
        suggestion: PlanSuggestion,
    ) -> SuggestionResult:
        """Apply a suggestion that was approved by the user.

        Args:
            suggestion: The suggestion to apply.

        Returns:
            Result of applying the suggestion.
        """

        return self.plan_analyzer.apply_suggestion(suggestion, force=True)

    def _should_checkpoint(self) -> bool:
        """Check if a checkpoint should be created.

        Checkpoints are created every N tasks based on checkpoint_every config.
        """
        if self._config.checkpoint_every == 0:
            return False
        return self._tasks_this_run % self._config.checkpoint_every == 0

    def _maybe_checkpoint(
        self,
        task: ParsedTask,
        result: ExecutionResult,
        *,
        verification_passed: bool | None = None,
        verification_level: str | None = None,
        tags: list[str] | None = None,
    ) -> tuple[CheckpointResult | None, CheckpointMetadata | None]:
        """Create a git checkpoint for the task if checkpointing is enabled.

        Uses RecoveryManager for enhanced metadata when available.

        Args:
            task: The completed task.
            result: The execution result.
            verification_passed: Whether verification passed.
            verification_level: The verification level used.
            tags: Additional tags to apply to the checkpoint.

        Returns:
            Tuple of (CheckpointResult, CheckpointMetadata) if checkpoint was created.
        """
        # Try RecoveryManager first for enhanced metadata
        recovery_mgr = self.recovery_manager
        if recovery_mgr is not None:
            try:
                # Combine config tags with any additional tags
                all_tags = list(self._config.checkpoint_tags)
                if tags:
                    all_tags.extend(tags)

                checkpoint_result, metadata = recovery_mgr.create_checkpoint(
                    task_id=task.id,
                    task_title=task.title,
                    files_modified=result.files_modified,
                    files_created=result.files_created,
                    files_deleted=result.files_deleted,
                    verification_passed=verification_passed,
                    verification_level=verification_level,
                    tags=all_tags if all_tags else None,
                )

                if checkpoint_result.success and checkpoint_result.commit_sha:
                    tag_info = f" with tags: {all_tags}" if all_tags else ""
                    logger.info(
                        f"Created checkpoint {checkpoint_result.commit_sha[:7]} "
                        f"for task {task.id}{tag_info}"
                    )
                elif not checkpoint_result.success:
                    logger.warning(
                        f"Checkpoint failed for task {task.id}: {checkpoint_result.error}"
                    )

                return checkpoint_result, metadata

            except Exception as e:
                logger.warning(f"Enhanced checkpoint error for task {task.id}: {e}")
                # Fall back to basic checkpointer

        # Fall back to basic checkpointer
        checkpointer = self.checkpointer
        if checkpointer is None:
            return None, None

        try:
            checkpoint_result = checkpointer.checkpoint(
                task_id=task.id,
                task_title=task.title,
                files_modified=result.files_modified,
                files_created=result.files_created,
            )

            if checkpoint_result.success and checkpoint_result.commit_sha:
                logger.info(
                    f"Created checkpoint {checkpoint_result.commit_sha[:7]} for task {task.id}"
                )
            elif not checkpoint_result.success:
                logger.warning(f"Checkpoint failed for task {task.id}: {checkpoint_result.error}")

            return checkpoint_result, None

        except Exception as e:
            logger.warning(f"Checkpoint error for task {task.id}: {e}")
            return None, None

    def _get_verification_levels(self) -> list[CheckLevel]:
        """Get the list of verification levels to run based on config."""
        # CheckLevel members in order
        all_levels = [
            CheckLevel.FILES,
            CheckLevel.SYNTAX,
            CheckLevel.IMPORTS,
            CheckLevel.RUNTIME,  # Actually import modules to catch circular imports
            CheckLevel.TESTS,
            CheckLevel.TYPES,
        ]
        max_level = self._config.verification_level
        # Return all levels up to and including the configured level
        levels = []
        for level in all_levels:
            levels.append(level)
            if level == max_level:
                break
        return levels

    def _parse_files_from_output(self, output: str) -> list[str]:
        """Parse file paths from agent output.

        This is a simple heuristic - in practice, the agent would
        return structured output with file paths.

        Args:
            output: Agent output string.

        Returns:
            List of file paths mentioned in output.
        """
        # Simple heuristic: look for file paths
        import re

        pattern = r"[`'\"]?([a-zA-Z0-9_/.-]+\.(py|js|ts|md|yaml|yml|json|toml))[`'\"]?"
        matches = re.findall(pattern, output)
        return list(set(m[0] for m in matches))

    def _extract_expected_files(self, task: ParsedTask) -> list[str]:
        """Extract expected file paths from task description.

        Phase 5.11.1: Parses task title and description to identify files
        that should be created by this task.

        Patterns recognized:
        - Create `path/to/file.ext` - backtick-wrapped paths
        - Create path/to/file.py - bare paths after "Create"
        - **Files**: `path/to/file.ext` - explicit file markers

        Args:
            task: The task to extract file paths from.

        Returns:
            List of relative file paths expected to be created.
        """
        import re

        files: list[str] = []
        text = f"{task.title}\n{task.description}"

        # Pattern 1: Create `path/to/file.ext`
        # Matches: Create `src/utils.js`, Create `tests/test_main.py`
        pattern1 = r"[Cc]reate\s+`([^`]+\.[a-zA-Z0-9]+)`"
        matches1 = re.findall(pattern1, text)
        files.extend(matches1)

        # Pattern 2: Create path/to/file.ext (without backticks)
        # Matches: Create src/utils.py, Create tests/test.js
        # Uses [a-zA-Z0-9_/.\\-]+ instead of \S+ to exclude backticks and special chars
        pattern2 = r"[Cc]reate\s+([a-zA-Z0-9_/.\\-]+\.(?:py|js|ts|jsx|tsx|css|html|json|yaml|yml|toml|md|txt|sql|sh))\b"
        matches2 = re.findall(pattern2, text)
        files.extend(matches2)

        # Pattern 3: **Files**: `path/to/file.ext` or **Files**: `file1`, `file2`
        # Matches: **Files**: `src/main.py`
        pattern3 = r"\*\*[Ff]iles?\*\*:\s*`([^`]+)`"
        matches3 = re.findall(pattern3, text)
        files.extend(matches3)

        # Pattern 4: - [ ] `path/to/file.ext` (file in checkbox format)
        pattern4 = r"-\s*\[[x ]\]\s*`([^`]+\.[a-zA-Z0-9]+)`"
        matches4 = re.findall(pattern4, text)
        files.extend(matches4)

        # Pattern 5: Any backtick-wrapped path with file extension
        # Matches: `src/config.py`, `tests/test.tsx`, etc. anywhere in text
        # This catches files mentioned in lists like "Create `file1.py` and `file2.py`"
        pattern5 = r"`([a-zA-Z0-9_/.\\-]+\.(?:py|js|ts|jsx|tsx|css|html|json|yaml|yml|toml|md|txt|sql|sh))`"
        matches5 = re.findall(pattern5, text)
        files.extend(matches5)

        # Clean up: remove duplicates, filter invalid paths
        seen: set[str] = set()
        result: list[str] = []
        for f in files:
            # Normalize path
            f = f.strip()
            # Skip if already seen
            if f in seen:
                continue
            # Skip if looks like a non-path (e.g., "name.toUpperCase()")
            if "(" in f or ")" in f:
                continue
            # Skip if it's a relative import or module reference
            if f.startswith(".") and "/" not in f:
                continue
            seen.add(f)
            result.append(f)

        return result

    def _verify_files_created(
        self,
        task: ParsedTask,
        result: ExecutionResult,
    ) -> tuple[bool, list[str]]:
        """Verify that files mentioned in task were actually created.

        Phase 5.11.1: Post-task verification to ensure files exist on disk.

        This method:
        1. Extracts expected file paths from task description
        2. Checks each path exists relative to project root
        3. Returns verification status and list of missing files

        Args:
            task: The completed task.
            result: The execution result from the task.

        Returns:
            Tuple of (all_created, missing_files).
            - all_created: True if all expected files exist
            - missing_files: List of file paths that are missing
        """
        expected_files = self._extract_expected_files(task)

        if not expected_files:
            # No files expected, verification passes
            return True, []

        missing: list[str] = []
        for file_path in expected_files:
            # Try relative to project root
            full_path = self._roadmap_path.parent / file_path
            if not full_path.exists():
                # Also try without leading src/ in case it was added
                alt_path = self._roadmap_path.parent / file_path.lstrip("/")
                if not alt_path.exists():
                    missing.append(file_path)

        if missing:
            logger.warning(
                "files_not_created",
                task_id=task.id,
                expected_count=len(expected_files),
                missing_count=len(missing),
                missing_files=missing,
            )
            return False, missing

        logger.debug(
            "files_verified",
            task_id=task.id,
            files_count=len(expected_files),
        )
        return True, []

    def _verify_todo_files_created(
        self,
        todo: TodoItem,
        source_tasks: list[ParsedTask],
        result: ExecutionResult,
    ) -> tuple[bool, list[str], list[str]]:
        """Verify that files mentioned in todo tasks were actually created.

        Phase 5.12.7: Post-todo verification to ensure files exist on disk.

        This method aggregates expected files from all source tasks and verifies
        each one exists. This prevents the agent from claiming success without
        actually creating the files.

        Args:
            todo: The completed todo.
            source_tasks: The source ParsedTask objects for this todo.
            result: The execution result from the todo.

        Returns:
            Tuple of (all_created, missing_files, created_files).
            - all_created: True if all expected files exist
            - missing_files: List of file paths that are missing
            - created_files: List of file paths that were created successfully
        """
        # Aggregate expected files from all source tasks
        all_expected: list[str] = []
        for task in source_tasks:
            expected = self._extract_expected_files(task)
            all_expected.extend(expected)

        # Also check file_hints from todo and tasks
        if todo.file_hints:
            all_expected.extend(todo.file_hints)
        for task in source_tasks:
            if task.file_hints:
                all_expected.extend(task.file_hints)

        # Deduplicate
        all_expected = list(dict.fromkeys(all_expected))

        if not all_expected:
            # No files expected, verification passes
            return True, [], []

        missing: list[str] = []
        created: list[str] = []
        project_root = self._roadmap_path.parent

        for file_path in all_expected:
            # Try relative to project root
            full_path = project_root / file_path
            if full_path.exists():
                created.append(file_path)
            else:
                # Also try without leading src/ in case it was added
                alt_path = project_root / file_path.lstrip("/")
                if alt_path.exists():
                    created.append(file_path)
                else:
                    missing.append(file_path)

        if missing:
            logger.warning(
                "todo_files_not_created",
                todo_id=todo.id,
                expected_count=len(all_expected),
                created_count=len(created),
                missing_count=len(missing),
                missing_files=missing[:10],  # Limit log output
            )
            return False, missing, created

        logger.debug(
            "todo_files_verified",
            todo_id=todo.id,
            files_count=len(created),
        )
        return True, [], created

    def _compute_file_checksum(self, file_path: Path) -> tuple[str, int]:
        """Compute MD5 checksum and size of a file.

        Phase 5.11.4: Used for file write verification.

        Args:
            file_path: Path to the file.

        Returns:
            Tuple of (md5_checksum, size_bytes).
        """
        import hashlib

        hasher = hashlib.md5()
        size = 0
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
                size += len(chunk)
        return hasher.hexdigest(), size

    def _record_file_write(
        self,
        task: ParsedTask,
        file_path: str,
    ) -> FileWriteRecord | None:
        """Record a file write with checksum verification.

        Phase 5.11.4: Records file creation for verification summary.

        Args:
            task: The task that created the file.
            file_path: Relative path to the file.

        Returns:
            FileWriteRecord if file exists, None otherwise.
        """
        full_path = self._roadmap_path.parent / file_path
        if not full_path.exists():
            return None

        try:
            checksum, size = self._compute_file_checksum(full_path)
            record = FileWriteRecord(
                path=file_path,
                absolute_path=str(full_path),
                task_id=task.id,
                size_bytes=size,
                checksum=checksum,
                verified=True,
            )
            self._file_write_tracker[file_path] = record
            logger.debug(
                "file_write_recorded",
                task_id=task.id,
                path=file_path,
                size_bytes=size,
                checksum=checksum[:8],  # Log first 8 chars of checksum
            )
            return record
        except OSError as e:
            logger.warning(
                "file_write_record_failed",
                task_id=task.id,
                path=file_path,
                error=str(e),
            )
            return None

    def _verify_and_record_files(
        self,
        task: ParsedTask,
        result: ExecutionResult,
    ) -> tuple[bool, list[str]]:
        """Verify files and record with checksum (Phase 5.11.4 enhanced verification).

        This method combines Phase 5.11.1 verification with Phase 5.11.4
        checksum recording when verify_writes is enabled.

        Args:
            task: The completed task.
            result: The execution result.

        Returns:
            Tuple of (all_created, missing_files).
        """
        expected_files = self._extract_expected_files(task)

        # Track expected files per task for summary
        if expected_files:
            self._expected_files_per_task[task.id] = expected_files

        if not expected_files:
            return True, []

        missing: list[str] = []
        for file_path in expected_files:
            full_path = self._roadmap_path.parent / file_path
            if not full_path.exists():
                alt_path = self._roadmap_path.parent / file_path.lstrip("/")
                if not alt_path.exists():
                    missing.append(file_path)
                    continue
                full_path = alt_path

            # If verify_writes enabled, record with checksum
            if self._config.verify_writes:
                self._record_file_write(task, file_path)

        if missing:
            logger.warning(
                "files_not_created",
                task_id=task.id,
                expected_count=len(expected_files),
                missing_count=len(missing),
                missing_files=missing,
            )
            return False, missing

        return True, []

    def _generate_file_write_summary(self) -> FileWriteSummary:
        """Generate summary of all file writes during the run.

        Phase 5.11.4: Creates summary report of files created vs expected.

        Returns:
            FileWriteSummary with all file write statistics.
        """
        summary = FileWriteSummary()

        # Calculate totals from tracked data
        all_expected: set[str] = set()
        for files in self._expected_files_per_task.values():
            all_expected.update(files)

        summary.total_expected = len(all_expected)
        summary.verified_files = list(self._file_write_tracker.values())
        summary.total_verified = len(summary.verified_files)

        # Find missing files
        verified_paths = set(self._file_write_tracker.keys())
        summary.missing_files = [f for f in all_expected if f not in verified_paths]

        # Count created (exists on disk, may or may not be verified)
        created_count = 0
        for file_path in all_expected:
            full_path = self._roadmap_path.parent / file_path
            if full_path.exists():
                created_count += 1
        summary.total_created = created_count

        return summary

    def _build_summary(
        self,
        results: list[ExecutionResult],
        status: RunStatus,
        paused: bool = False,
        pause_reason: str = "",
    ) -> RunSummary:
        """Build a run summary.

        Args:
            results: Execution results from this run.
            status: Overall run status.
            paused: Whether the run was paused.
            pause_reason: Why the run was paused.

        Returns:
            Run summary.
        """
        completed_at = datetime.now(UTC)
        duration_ms = 0.0
        if self._run_started:
            duration_ms = (completed_at - self._run_started).total_seconds() * 1000

        tasks_completed = sum(1 for r in results if r.success)
        tasks_failed = sum(
            1 for r in results if r.status in (ExecutionStatus.FAILED, ExecutionStatus.ERROR)
        )
        tasks_skipped = sum(1 for r in results if r.status == ExecutionStatus.SKIPPED)
        total_tokens = sum(sum(r.token_usage.values()) for r in results)

        state_summary = self.state.get_summary()

        return RunSummary(
            status=status,
            tasks_completed=tasks_completed,
            tasks_failed=tasks_failed,
            tasks_skipped=tasks_skipped,
            tasks_remaining=state_summary.pending,
            total_tasks=self.roadmap.total_tasks,
            results=results,
            paused=paused,
            pending_review=self.state.get_in_progress_tasks() if paused else [],
            pause_reason=pause_reason,
            duration_ms=duration_ms,
            total_tokens=total_tokens,
            started_at=self._run_started,
            completed_at=completed_at,
            run_id=self.state.run_id,
        )

    # =========================================================================
    # Resume and Control
    # =========================================================================

    def resume(
        self,
        approved: bool = True,
        rollback: bool = False,
    ) -> RollbackResult | None:
        """Resume execution after a pause.

        Args:
            approved: Whether the paused changes were approved.
                     If False, in-progress tasks are reset in state.
            rollback: Whether to rollback git changes (requires checkpointer).
                     Only effective when approved=False.

        Returns:
            RollbackResult if rollback was attempted, None otherwise.
        """
        rollback_result: RollbackResult | None = None

        if not approved:
            # Get in-progress tasks before resetting state
            in_progress_tasks = self.state.get_in_progress_tasks()

            # Rollback git changes if requested and checkpointer available
            if rollback and in_progress_tasks:
                checkpointer = self.checkpointer
                if checkpointer is not None:
                    # Rollback to the first task's commit (rolls back all subsequent)
                    first_task_id = in_progress_tasks[0]
                    rollback_result = checkpointer.rollback(first_task_id, hard=True)
                    if rollback_result.success:
                        logger.info(
                            f"Rolled back {rollback_result.commits_reverted} commit(s) "
                            f"to {rollback_result.target_sha}"
                        )
                    else:
                        logger.warning(f"Rollback failed: {rollback_result.error}")
                else:
                    logger.warning("Rollback requested but no checkpointer available")
                    rollback_result = RollbackResult(
                        success=False,
                        error="Checkpointing is disabled",
                        message="Cannot rollback: checkpointing is disabled",
                    )

            # Reset in-progress tasks in state
            self.state.recover()
            logger.info("Changes not approved, reset in-progress tasks")
        else:
            logger.info("Changes approved, ready to continue")

        self.state.save()
        return rollback_result

    def get_changes_for_review(self) -> ReviewInfo:
        """Get information about changes pending human review.

        Returns detailed information about what changed during task
        execution, including files modified, commits created, and
        whether destructive operations occurred.

        This method uses results from the last run that triggered a pause.

        Returns:
            ReviewInfo with detailed change information.
        """
        # Collect info from last run results
        task_ids: list[str] = []
        files_modified: list[str] = []
        files_created: list[str] = []
        files_deleted: list[str] = []
        has_destructive = False

        for result in self._last_run_results:
            task_ids.append(result.task_id)
            files_modified.extend(result.files_modified)
            files_created.extend(result.files_created)
            files_deleted.extend(result.files_deleted)

        # Check for destructive changes
        if files_deleted:
            has_destructive = True

        # Get commits for these tasks if checkpointer available
        commits: list[CommitInfo] = []
        checkpointer = self.checkpointer
        if checkpointer is not None:
            for task_id in task_ids:
                commit = checkpointer.get_commit_for_task(task_id)
                if commit:
                    commits.append(commit)

        # Deduplicate
        files_modified = list(set(files_modified))
        files_created = list(set(files_created))
        files_deleted = list(set(files_deleted))

        # Determine pause reason
        pause_reason = "Pending human review"
        if has_destructive:
            pause_reason = "Destructive operations detected"

        return ReviewInfo(
            task_ids=task_ids,
            files_modified=files_modified,
            files_created=files_created,
            files_deleted=files_deleted,
            commits=commits,
            has_destructive=has_destructive,
            pause_reason=pause_reason,
        )

    def reset(self) -> None:
        """Reset executor state completely.

        Re-parses the ROADMAP and reinitializes state.
        """
        self._roadmap = None
        self._state = ExecutorState.from_roadmap(self._roadmap_path)
        self._tasks_this_run = 0
        logger.info("Executor state reset")

    def sync_roadmap(self) -> int:
        """Sync completed tasks back to ROADMAP.md.

        Returns:
            Number of checkboxes updated.
        """
        return self.state.sync_to_roadmap()

    # =========================================================================
    # Recovery Operations (Phase 4.2)
    # =========================================================================

    def preview_rollback(
        self,
        *,
        task_id: str | None = None,
        tag: str | None = None,
        commit_sha: str | None = None,
    ) -> RollbackPreview | None:
        """Preview what a rollback would change.

        Exactly one of task_id, tag, or commit_sha must be provided.

        Args:
            task_id: Rollback to before this task.
            tag: Rollback to this tagged checkpoint.
            commit_sha: Rollback to this specific commit.

        Returns:
            RollbackPreview or None if recovery manager unavailable.

        Example:
            >>> preview = executor.preview_rollback(task_id="1.1.1")
            >>> if preview.is_safe:
            ...     executor.rollback(task_id="1.1.1")
        """
        recovery_mgr = self.recovery_manager
        if recovery_mgr is None:
            return None
        return recovery_mgr.preview_rollback(task_id=task_id, tag=tag, commit_sha=commit_sha)

    def rollback(
        self,
        *,
        task_id: str | None = None,
        tag: str | None = None,
        files: list[str] | None = None,
        hard: bool = True,
        preview: bool = False,
    ) -> RollbackResult | RollbackPreview | SelectiveRollbackResult | None:
        """Rollback to a previous state.

        Supports multiple rollback modes:
        - By task_id: Rollback to before a specific task
        - By tag: Rollback to a named checkpoint
        - By files: Selectively rollback specific files

        Args:
            task_id: Rollback to before this task.
            tag: Rollback to this tagged checkpoint.
            files: Specific files to rollback (requires task_id or tag).
            hard: Discard uncommitted changes (default True).
            preview: Return preview instead of performing rollback.

        Returns:
            Result of the rollback operation.

        Example:
            >>> # Full rollback to before task
            >>> executor.rollback(task_id="1.1.1")
            >>> # Selective rollback of specific files
            >>> executor.rollback(files=["src/main.py"], tag="stable")
        """
        recovery_mgr = self.recovery_manager
        if recovery_mgr is None:
            # Fall back to basic checkpointer
            checkpointer = self.checkpointer
            if checkpointer is None:
                return None
            if task_id:
                return checkpointer.rollback(task_id, hard=hard)
            return None

        # Preview mode
        if preview:
            return recovery_mgr.preview_rollback(task_id=task_id, tag=tag)

        # Selective file rollback
        if files:
            return recovery_mgr.rollback_files(files, to_task_id=task_id, to_tag=tag)

        # Tag rollback
        if tag:
            return recovery_mgr.rollback_to_tag(tag, hard=hard)

        # Task rollback via checkpointer
        if task_id:
            return recovery_mgr.checkpointer.rollback(task_id, hard=hard)

        return None

    def recover(
        self,
        task_id: str,
        *,
        strategy: RecoveryStrategy | None = None,
        failed_files: list[str] | None = None,
        error_context: str | None = None,
    ) -> RecoveryResult | None:
        """Apply a recovery strategy for a failed task.

        Args:
            task_id: The failed task ID.
            strategy: Recovery strategy (default from config).
            failed_files: Files that caused failure (for ROLLBACK_FAILED).
            error_context: Error info (for RETRY_WITH_CONTEXT).

        Returns:
            RecoveryResult or None if recovery unavailable.

        Example:
            >>> # Rollback failed files only
            >>> executor.recover(
            ...     "1.1.1",
            ...     strategy=RecoveryStrategy.ROLLBACK_FAILED,
            ...     failed_files=["src/broken.py"],
            ... )
        """
        recovery_mgr = self.recovery_manager
        if recovery_mgr is None:
            return None

        strategy = strategy or self._config.recovery_strategy

        return recovery_mgr.recover(
            task_id=task_id,
            strategy=strategy,
            failed_files=failed_files,
            error_context=error_context,
        )

    def add_checkpoint_tag(self, tag: str, commit_sha: str | None = None) -> bool:
        """Add a tag to a checkpoint.

        Args:
            tag: Tag name (e.g., "stable", "phase-1-complete").
            commit_sha: Commit to tag (default: current HEAD).

        Returns:
            True if tag was added.

        Example:
            >>> # Tag current state as stable
            >>> executor.add_checkpoint_tag("stable")
        """
        recovery_mgr = self.recovery_manager
        if recovery_mgr is None:
            return False
        return recovery_mgr.add_tag(tag, commit_sha)

    def list_checkpoint_tags(self) -> dict[str, str]:
        """List all checkpoint tags.

        Returns:
            Dictionary mapping tag names to commit SHAs.
        """
        recovery_mgr = self.recovery_manager
        if recovery_mgr is None:
            return {}
        return recovery_mgr.list_tags()

    def get_checkpoint_metadata(self, commit_sha: str) -> CheckpointMetadata | None:
        """Get metadata for a checkpoint.

        Args:
            commit_sha: The commit SHA.

        Returns:
            CheckpointMetadata if found.
        """
        recovery_mgr = self.recovery_manager
        if recovery_mgr is None:
            return None
        return recovery_mgr.get_checkpoint_metadata(commit_sha)
