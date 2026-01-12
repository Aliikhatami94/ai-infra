"""Observability support for the Executor.

This module provides structured logging, metrics collection, and tracing
for the Executor loop. It integrates with ai-infra's existing observability
infrastructure:

- Structured logging via `ai_infra.logging`
- Metrics via `ai_infra.callbacks.MetricsCallbacks`
- Tracing via `ai_infra.tracing`

Usage:
    from ai_infra.executor.observability import (
        ExecutorCallbacks,
        ExecutorMetrics,
        get_executor_tracer,
    )

    # Create executor with observability
    callbacks = ExecutorCallbacks()
    executor = Executor(
        roadmap="ROADMAP.md",
        callbacks=callbacks,
    )

    # Run tasks
    summary = await executor.run()

    # Get metrics
    metrics = callbacks.get_metrics()
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Task success rate: {metrics.success_rate}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ai_infra.callbacks import (
    Callbacks,
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    ToolEndEvent,
    ToolErrorEvent,
    ToolStartEvent,
)
from ai_infra.executor.metrics import TokenTracker
from ai_infra.logging import get_logger
from ai_infra.tracing import Span, Tracer, get_tracer, trace

if TYPE_CHECKING:
    from ai_infra.executor.roadmap import ParsedTask

logger = get_logger("executor.observability")


# =============================================================================
# Executor Metrics
# =============================================================================


@dataclass
class TaskMetrics:
    """Metrics for a single task execution.

    Attributes:
        task_id: The task ID.
        started_at: When the task started.
        completed_at: When the task completed.
        duration_ms: Total duration in milliseconds.
        llm_calls: Number of LLM calls.
        llm_tokens: Total tokens used by LLM.
        tool_calls: Number of tool calls.
        tool_duration_ms: Total time spent in tools.
        files_modified: Number of files modified.
        success: Whether the task succeeded.
        error: Error message if failed.
    """

    task_id: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float = 0.0
    llm_calls: int = 0
    llm_tokens: int = 0
    tool_calls: int = 0
    tool_duration_ms: float = 0.0
    files_modified: int = 0
    success: bool = False
    error: str | None = None
    # Phase 5.6: Retry tracking
    retry_attempts: int = 0
    retry_success: bool = False
    fixes_applied: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "tool_calls": self.tool_calls,
            "tool_duration_ms": self.tool_duration_ms,
            "files_modified": self.files_modified,
            "success": self.success,
            "error": self.error,
            "retry_attempts": self.retry_attempts,
            "retry_success": self.retry_success,
            "fixes_applied": self.fixes_applied,
        }


@dataclass
class ExecutorMetrics:
    """Aggregated metrics for an executor run.

    Attributes:
        run_id: The executor run ID.
        started_at: When the run started.
        completed_at: When the run completed.
        tasks_started: Number of tasks started.
        tasks_completed: Number of tasks completed successfully.
        tasks_failed: Number of tasks that failed.
        tasks_skipped: Number of tasks skipped.
        total_llm_calls: Total LLM calls across all tasks.
        total_tokens: Total tokens used.
        total_tool_calls: Total tool calls across all tasks.
        total_duration_ms: Total execution duration.
        avg_task_duration_ms: Average task duration.
        avg_llm_calls_per_task: Average LLM calls per task.
        avg_tokens_per_task: Average tokens per task.
        task_metrics: Individual task metrics.
        checkpoints_created: Number of checkpoints created.
        rollbacks_performed: Number of rollbacks performed.
        task_retries: Number of task retry attempts (Phase 5.6).
        fixes_applied: Number of auto-fixes applied (Phase 5.6).
        retry_successes: Number of tasks that succeeded on retry (Phase 5.6).
    """

    run_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    tasks_started: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_skipped: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    total_duration_ms: float = 0.0
    task_metrics: list[TaskMetrics] = field(default_factory=list)
    checkpoints_created: int = 0
    rollbacks_performed: int = 0
    # Phase 5.6: Retry metrics
    task_retries: int = 0
    fixes_applied: int = 0
    retry_successes: int = 0

    @property
    def avg_task_duration_ms(self) -> float:
        """Average task duration in milliseconds."""
        completed = self.tasks_completed + self.tasks_failed
        if completed == 0:
            return 0.0
        return self.total_duration_ms / completed

    @property
    def avg_llm_calls_per_task(self) -> float:
        """Average LLM calls per completed task."""
        completed = self.tasks_completed + self.tasks_failed
        if completed == 0:
            return 0.0
        return self.total_llm_calls / completed

    @property
    def avg_tokens_per_task(self) -> float:
        """Average tokens per completed task."""
        completed = self.tasks_completed + self.tasks_failed
        if completed == 0:
            return 0.0
        return self.total_tokens / completed

    @property
    def success_rate(self) -> float:
        """Task success rate as a percentage."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 0.0
        return (self.tasks_completed / total) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tasks": {
                "started": self.tasks_started,
                "completed": self.tasks_completed,
                "failed": self.tasks_failed,
                "skipped": self.tasks_skipped,
                "success_rate": self.success_rate,
            },
            "llm": {
                "total_calls": self.total_llm_calls,
                "total_tokens": self.total_tokens,
                "avg_calls_per_task": self.avg_llm_calls_per_task,
                "avg_tokens_per_task": self.avg_tokens_per_task,
            },
            "tools": {
                "total_calls": self.total_tool_calls,
            },
            "timing": {
                "total_duration_ms": self.total_duration_ms,
                "avg_task_duration_ms": self.avg_task_duration_ms,
            },
            "recovery": {
                "checkpoints_created": self.checkpoints_created,
                "rollbacks_performed": self.rollbacks_performed,
            },
            "retry": {
                "task_retries": self.task_retries,
                "fixes_applied": self.fixes_applied,
                "retry_successes": self.retry_successes,
            },
            "task_metrics": [m.to_dict() for m in self.task_metrics],
        }

    def summary(self) -> str:
        """Get a human-readable summary of metrics."""
        lines = [
            "Executor Metrics Summary",
            "=" * 40,
            f"Run ID: {self.run_id}",
            "",
            "Tasks:",
            f"  Started:   {self.tasks_started}",
            f"  Completed: {self.tasks_completed}",
            f"  Failed:    {self.tasks_failed}",
            f"  Skipped:   {self.tasks_skipped}",
            f"  Success:   {self.success_rate:.1f}%",
            "",
            "LLM Usage:",
            f"  Total Calls:  {self.total_llm_calls}",
            f"  Total Tokens: {self.total_tokens:,}",
            f"  Avg/Task:     {self.avg_tokens_per_task:.0f} tokens",
            "",
            "Timing:",
            f"  Total:    {self.total_duration_ms:,.0f}ms",
            f"  Avg/Task: {self.avg_task_duration_ms:,.0f}ms",
            "",
            "Recovery:",
            f"  Checkpoints: {self.checkpoints_created}",
            f"  Rollbacks:   {self.rollbacks_performed}",
            "",
            "Retry (Phase 5.6):",
            f"  Retries:     {self.task_retries}",
            f"  Fixes:       {self.fixes_applied}",
            f"  Successes:   {self.retry_successes}",
        ]
        return "\n".join(lines)


# =============================================================================
# Executor Callbacks
# =============================================================================


class ExecutorCallbacks(Callbacks):
    """Callback handler for executor observability.

    Collects metrics and emits structured logs for all executor operations.
    This callback should be passed to the Agent to track LLM and tool usage.

    Example:
        callbacks = ExecutorCallbacks()

        # Set run context
        callbacks.set_run_context(run_id="run-123")

        # Track task execution
        callbacks.on_task_start("task-1", "Implement feature X")
        # ... agent executes ...
        callbacks.on_task_end("task-1", success=True, files_modified=3)

        # Get aggregated metrics
        metrics = callbacks.get_metrics()
    """

    def __init__(self) -> None:
        """Initialize executor callbacks."""
        self._metrics = ExecutorMetrics()
        self._current_task: TaskMetrics | None = None
        self._tracer: Tracer | None = None
        self._current_span: Span | None = None
        self._task_start_time: float = 0.0

    # =========================================================================
    # Run Lifecycle
    # =========================================================================

    def set_run_context(self, run_id: str) -> None:
        """Set the run context for metrics.

        Args:
            run_id: The executor run ID.
        """
        self._metrics.run_id = run_id
        self._metrics.started_at = datetime.now(UTC)
        self._tracer = get_tracer()
        logger.info(
            "executor_run_started",
            run_id=run_id,
        )

    def on_run_end(self) -> None:
        """Called when the executor run completes."""
        self._metrics.completed_at = datetime.now(UTC)
        if self._metrics.started_at:
            duration = (
                self._metrics.completed_at - self._metrics.started_at
            ).total_seconds() * 1000
            self._metrics.total_duration_ms = duration

        logger.info(
            "executor_run_completed",
            run_id=self._metrics.run_id,
            tasks_completed=self._metrics.tasks_completed,
            tasks_failed=self._metrics.tasks_failed,
            total_tokens=self._metrics.total_tokens,
            duration_ms=self._metrics.total_duration_ms,
            success_rate=self._metrics.success_rate,
        )

    # =========================================================================
    # Task Lifecycle
    # =========================================================================

    def on_task_start(self, task_id: str, title: str) -> None:
        """Called when a task starts execution.

        Args:
            task_id: The task ID.
            title: The task title.
        """
        self._metrics.tasks_started += 1
        self._task_start_time = time.time()

        self._current_task = TaskMetrics(
            task_id=task_id,
            started_at=datetime.now(UTC),
        )

        # Start tracing span
        if self._tracer:
            self._current_span = self._tracer.start_span(
                f"executor.task.{task_id}",
                attributes={
                    "task.id": task_id,
                    "task.title": title,
                },
            )

        logger.info(
            "task_started",
            task_id=task_id,
            title=title,
        )

    def on_task_end(
        self,
        task_id: str,
        success: bool,
        files_modified: int = 0,
        error: str | None = None,
    ) -> None:
        """Called when a task completes.

        Args:
            task_id: The task ID.
            success: Whether the task succeeded.
            files_modified: Number of files modified.
            error: Error message if failed.
        """
        duration_ms = (time.time() - self._task_start_time) * 1000

        if self._current_task:
            self._current_task.completed_at = datetime.now(UTC)
            self._current_task.duration_ms = duration_ms
            self._current_task.success = success
            self._current_task.files_modified = files_modified
            self._current_task.error = error
            self._metrics.task_metrics.append(self._current_task)

        # Update aggregated metrics
        if success:
            self._metrics.tasks_completed += 1
        else:
            self._metrics.tasks_failed += 1

        # End tracing span
        if self._current_span:
            self._current_span.set_attribute("task.success", success)
            self._current_span.set_attribute("task.files_modified", files_modified)
            if error:
                self._current_span.set_status("error", error)
            if self._tracer:
                self._tracer.end_span(self._current_span)
            self._current_span = None

        log_fn = logger.info if success else logger.warning
        log_fn(
            "task_completed" if success else "task_failed",
            task_id=task_id,
            success=success,
            duration_ms=duration_ms,
            files_modified=files_modified,
            error=error,
        )

        self._current_task = None

    def on_task_skip(self, task_id: str, reason: str) -> None:
        """Called when a task is skipped.

        Args:
            task_id: The task ID.
            reason: Reason for skipping.
        """
        self._metrics.tasks_skipped += 1
        logger.info(
            "task_skipped",
            task_id=task_id,
            reason=reason,
        )

    # =========================================================================
    # LLM Events
    # =========================================================================

    def on_llm_start(self, event: LLMStartEvent) -> None:
        """Called when LLM call starts."""
        logger.debug(
            "llm_call_started",
            provider=event.provider,
            model=event.model,
            task_id=self._current_task.task_id if self._current_task else None,
        )

    def on_llm_end(self, event: LLMEndEvent) -> None:
        """Called when LLM call completes."""
        self._metrics.total_llm_calls += 1
        if event.total_tokens:
            self._metrics.total_tokens += event.total_tokens

        if self._current_task:
            self._current_task.llm_calls += 1
            if event.total_tokens:
                self._current_task.llm_tokens += event.total_tokens

        # Feed token data to per-node TokenTracker for node-level cost breakdown
        # This connects ExecutorCallbacks to the track_node_metrics decorator
        TokenTracker.add_usage(
            tokens_in=event.input_tokens or 0,
            tokens_out=event.output_tokens or 0,
            llm_calls=1,
        )

        logger.debug(
            "llm_call_completed",
            provider=event.provider,
            model=event.model,
            tokens=event.total_tokens,
            latency_ms=event.latency_ms,
            task_id=self._current_task.task_id if self._current_task else None,
        )

    def on_llm_error(self, event: LLMErrorEvent) -> None:
        """Called when LLM call fails."""
        logger.error(
            "llm_call_failed",
            provider=event.provider,
            model=event.model,
            error=str(event.error),
            task_id=self._current_task.task_id if self._current_task else None,
        )

    # =========================================================================
    # Tool Events
    # =========================================================================

    def on_tool_start(self, event: ToolStartEvent) -> None:
        """Called when tool execution starts."""
        logger.debug(
            "tool_call_started",
            tool=event.tool_name,
            task_id=self._current_task.task_id if self._current_task else None,
        )

    def on_tool_end(self, event: ToolEndEvent) -> None:
        """Called when tool execution completes."""
        self._metrics.total_tool_calls += 1

        if self._current_task:
            self._current_task.tool_calls += 1
            self._current_task.tool_duration_ms += event.latency_ms

        logger.debug(
            "tool_call_completed",
            tool=event.tool_name,
            latency_ms=event.latency_ms,
            task_id=self._current_task.task_id if self._current_task else None,
        )

    def on_tool_error(self, event: ToolErrorEvent) -> None:
        """Called when tool execution fails."""
        logger.warning(
            "tool_call_failed",
            tool=event.tool_name,
            error=str(event.error),
            task_id=self._current_task.task_id if self._current_task else None,
        )

    # =========================================================================
    # Retry Events (Phase 5.6)
    # =========================================================================

    def on_task_retry(
        self,
        task_id: str,
        attempt: int,
        max_attempts: int,
        previous_error: str | None = None,
    ) -> None:
        """Called when a task is being retried.

        Args:
            task_id: The task ID.
            attempt: Current attempt number (2 = first retry).
            max_attempts: Maximum number of attempts allowed.
            previous_error: Error from the previous attempt.
        """
        self._metrics.task_retries += 1
        if self._current_task:
            self._current_task.retry_attempts += 1

        logger.info(
            "task_retry",
            task_id=task_id,
            attempt=attempt,
            max_attempts=max_attempts,
            previous_error=previous_error,
        )

    def on_retry_success(self, task_id: str, attempt: int) -> None:
        """Called when a task succeeds on retry.

        Args:
            task_id: The task ID.
            attempt: The attempt number that succeeded.
        """
        self._metrics.retry_successes += 1
        if self._current_task:
            self._current_task.retry_success = True

        logger.info(
            "task_retry_succeeded",
            task_id=task_id,
            attempt=attempt,
        )

    def on_fix_applied(
        self,
        task_id: str,
        suggestion_type: str,
        description: str,
    ) -> None:
        """Called when an auto-fix is applied before retry.

        Args:
            task_id: The task ID.
            suggestion_type: Type of fix (e.g., "create_init", "create_directory").
            description: Human-readable description of the fix.
        """
        self._metrics.fixes_applied += 1
        if self._current_task:
            self._current_task.fixes_applied += 1

        logger.info(
            "fix_applied",
            task_id=task_id,
            suggestion_type=suggestion_type,
            description=description,
        )

    def on_retries_exhausted(
        self,
        task_id: str,
        attempts: int,
        final_error: str | None = None,
    ) -> None:
        """Called when all retry attempts are exhausted.

        Args:
            task_id: The task ID.
            attempts: Total number of attempts made.
            final_error: The final error message.
        """
        logger.warning(
            "task_retries_exhausted",
            task_id=task_id,
            attempts=attempts,
            final_error=final_error,
        )

    # =========================================================================
    # Recovery Events
    # =========================================================================

    def on_checkpoint_created(self, checkpoint_id: str, task_id: str | None) -> None:
        """Called when a checkpoint is created.

        Args:
            checkpoint_id: The checkpoint/commit ID.
            task_id: The associated task ID, if any.
        """
        self._metrics.checkpoints_created += 1
        logger.info(
            "checkpoint_created",
            checkpoint_id=checkpoint_id,
            task_id=task_id,
        )

    def on_rollback(self, checkpoint_id: str, reason: str) -> None:
        """Called when a rollback is performed.

        Args:
            checkpoint_id: The checkpoint rolled back to.
            reason: Reason for the rollback.
        """
        self._metrics.rollbacks_performed += 1
        logger.warning(
            "rollback_performed",
            checkpoint_id=checkpoint_id,
            reason=reason,
        )

    # =========================================================================
    # Dependency Events
    # =========================================================================

    def on_dependency_warning(
        self,
        file_path: str,
        impact: str,
        affected_files: list[str],
    ) -> None:
        """Called when a dependency warning is raised.

        Args:
            file_path: The file with the warning.
            impact: Impact level (high, medium, low).
            affected_files: List of files that may be affected.
        """
        logger.warning(
            "dependency_warning",
            affected_file=file_path,
            impact=impact,
            affected_count=len(affected_files),
        )

    # =========================================================================
    # Metrics Access
    # =========================================================================

    def get_metrics(self) -> ExecutorMetrics:
        """Get the collected metrics.

        Returns:
            ExecutorMetrics with all collected data.
        """
        return self._metrics

    def get_current_task_metrics(self) -> TaskMetrics | None:
        """Get metrics for the currently executing task.

        Returns:
            TaskMetrics for current task, or None if no task is running.
        """
        return self._current_task

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = ExecutorMetrics()
        self._current_task = None
        self._current_span = None


# =============================================================================
# Tracing Utilities
# =============================================================================


def get_executor_tracer() -> Tracer:
    """Get a tracer configured for executor operations.

    Returns:
        Configured Tracer instance.
    """
    return get_tracer()


@trace(name="executor.build_context")
async def traced_build_context(
    project_root: str,
    task: ParsedTask,
    max_tokens: int,
) -> dict[str, Any]:
    """Traced wrapper for building task context.

    This is a utility for adding tracing to context building.

    Args:
        project_root: Path to project root.
        task: The task being executed.
        max_tokens: Maximum tokens for context.

    Returns:
        Context dictionary.
    """
    # This is a traced wrapper - actual implementation is in the executor
    # The return value indicates this is just a tracing utility
    return {"traced": True, "task_id": task.id, "max_tokens": max_tokens}


@trace(name="executor.verify_task")
async def traced_verify_task(task_id: str, levels: list[str]) -> dict[str, Any]:
    """Traced wrapper for task verification.

    Args:
        task_id: The task ID.
        levels: Verification levels to run.

    Returns:
        Verification result dictionary.
    """
    return {"traced": True, "task_id": task_id, "levels": levels}


# =============================================================================
# Logging Utilities
# =============================================================================


def log_task_context(
    task_id: str,
    context_tokens: int,
    files_included: int,
    file_hints: list[str] | None = None,
) -> None:
    """Log task context information.

    Args:
        task_id: The task ID.
        context_tokens: Number of tokens in context.
        files_included: Number of files included.
        file_hints: File hints from the task.
    """
    logger.info(
        "task_context_built",
        task_id=task_id,
        context_tokens=context_tokens,
        files_included=files_included,
        file_hints=file_hints or [],
    )


def log_verification_result(
    task_id: str,
    passed: bool,
    verification_level: str,
    duration_ms: float,
    details: str | None = None,
) -> None:
    """Log verification result.

    Args:
        task_id: The task ID.
        passed: Whether verification passed.
        verification_level: Verification level.
        duration_ms: Verification duration.
        details: Additional details.
    """
    log_fn = logger.info if passed else logger.warning
    log_fn(
        "verification_result",
        task_id=task_id,
        passed=passed,
        verification_level=verification_level,
        duration_ms=duration_ms,
        details=details,
    )


def log_recovery_action(
    action: str,
    checkpoint_id: str | None = None,
    files_affected: int = 0,
    success: bool = True,
    error: str | None = None,
) -> None:
    """Log recovery action.

    Args:
        action: The recovery action (rollback, restore, skip, etc.).
        checkpoint_id: The checkpoint involved.
        files_affected: Number of files affected.
        success: Whether the action succeeded.
        error: Error message if failed.
    """
    log_fn = logger.info if success else logger.error
    log_fn(
        "recovery_action",
        action=action,
        checkpoint_id=checkpoint_id,
        files_affected=files_affected,
        success=success,
        error=error,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Metrics
    "TaskMetrics",
    "ExecutorMetrics",
    # Callbacks
    "ExecutorCallbacks",
    # Tracing
    "get_executor_tracer",
    "traced_build_context",
    "traced_verify_task",
    # Logging
    "log_task_context",
    "log_verification_result",
    "log_recovery_action",
]
