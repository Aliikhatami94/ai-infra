"""Executor type definitions.

This module contains all the core types used by the executor:
- Enums (VerifyMode, ExecutionStatus, RunStatus)
- Dataclasses (ExecutionResult, RunSummary, ReviewInfo, ExecutorConfig)
- Protocols (AgentProtocol)

Extracted from loop.py as part of Phase 0.5 decomposition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from ai_infra.executor.checkpoint import CommitInfo
from ai_infra.executor.dependencies import DependencyWarning
from ai_infra.executor.recovery import RecoveryStrategy
from ai_infra.executor.verifier import CheckLevel
from ai_infra.mcp import McpServerConfig

if TYPE_CHECKING:
    from ai_infra.executor.roadmap import ParsedTask
    from ai_infra.executor.verifier import VerificationResult


# =============================================================================
# Verification Mode
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


# =============================================================================
# Execution Status
# =============================================================================


class ExecutionStatus(str, Enum):
    """Status of a task execution."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


# =============================================================================
# Execution Result
# =============================================================================


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
# Run Status
# =============================================================================


class RunStatus(str, Enum):
    """Status of an executor run."""

    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"
    STOPPED = "stopped"
    NO_TASKS = "no_tasks"


# =============================================================================
# Run Summary
# =============================================================================


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
        shell_memory_limit_mb: Max memory in MB for shell processes (Phase 2.2).
        shell_cpu_limit_seconds: Max CPU time in seconds for shell processes (Phase 2.2).
        shell_file_limit_mb: Max file size in MB for shell processes (Phase 2.2).
        enable_shell_snapshots: Auto-capture shell snapshots on task boundaries (Phase 16.4).
        docker_isolation: Execute shell commands in Docker containers (Phase 2.3).
        docker_image: Docker image for isolated execution (Phase 2.3).
        docker_allow_network: Allow network access in Docker containers (Phase 2.3).
        mcp_servers: MCP server configurations for external tool integration (Phase 15.1).
        mcp_discover_timeout: Timeout for MCP server discovery in seconds (Phase 15.1).
        mcp_tool_timeout: Timeout for MCP tool calls in seconds (Phase 15.1).
        mcp_auto_discover: Auto-discover MCP tools on executor init (Phase 15.1).
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
    # Phase 2.2: Shell resource limits
    shell_memory_limit_mb: int = 512  # Max memory in MB for shell processes
    shell_cpu_limit_seconds: int = 60  # Max CPU time in seconds for shell processes
    shell_file_limit_mb: int = 100  # Max file size in MB for shell processes
    # Phase 16.4: Shell snapshots on task boundaries
    enable_shell_snapshots: bool = False
    # Phase 2.3: Docker isolation
    docker_isolation: bool = False  # Execute shell commands in Docker containers
    docker_image: str = "python:3.11-slim"  # Docker image for isolated execution
    docker_allow_network: bool = False  # Allow network access in Docker containers
    # Phase 15.1: MCP integration
    mcp_servers: list[McpServerConfig] = field(default_factory=list)
    mcp_discover_timeout: float = 30.0  # Timeout for server discovery
    mcp_tool_timeout: float = 60.0  # Timeout for tool calls
    mcp_auto_discover: bool = True  # Auto-discover on executor init

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
            # Phase 2.2: Shell resource limits
            "shell_memory_limit_mb": self.shell_memory_limit_mb,
            "shell_cpu_limit_seconds": self.shell_cpu_limit_seconds,
            "shell_file_limit_mb": self.shell_file_limit_mb,
            # Phase 16.4: Shell snapshots
            "enable_shell_snapshots": self.enable_shell_snapshots,
            # Phase 2.3: Docker isolation
            "docker_isolation": self.docker_isolation,
            "docker_image": self.docker_image,
            "docker_allow_network": self.docker_allow_network,
            # Phase 15.1: MCP integration
            "mcp_servers": [s.model_dump() for s in self.mcp_servers],
            "mcp_discover_timeout": self.mcp_discover_timeout,
            "mcp_tool_timeout": self.mcp_tool_timeout,
            "mcp_auto_discover": self.mcp_auto_discover,
        }


# =============================================================================
# Type Aliases
# =============================================================================

TaskCompleteCallback = Callable[["ParsedTask", ExecutionResult], Any]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "VerifyMode",
    "ExecutionStatus",
    "RunStatus",
    # Dataclasses
    "ExecutionResult",
    "RunSummary",
    "ReviewInfo",
    "ExecutorConfig",
    # Protocols
    "AgentProtocol",
    # Type aliases
    "TaskCompleteCallback",
]
