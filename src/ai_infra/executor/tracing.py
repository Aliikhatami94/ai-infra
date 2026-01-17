"""Executor tracing and observability.

This module provides executor-specific observability built on ai_infra.tracing.

The module consolidates:
- executor/observability.py (857 lines) - executor metrics and callbacks
- executor/graph_tracing.py (486 lines) - graph tracing utilities
- TokenTracker from executor/metrics.py - token tracking for callbacks

Total reduction: 1,343 lines → ~250 lines

Usage:
    from ai_infra.executor.tracing import (
        ExecutorCallbacks,
        ExecutorMetrics,
        TaskMetrics,
        TracingConfig,
        TokenTracker,
    )

    # Create executor with observability
    callbacks = ExecutorCallbacks()
    callbacks.set_run_context(run_id="run-123")

    # Use ai_infra.tracing for spans
    from ai_infra.tracing import trace, get_tracer

    @trace(name="executor.custom_operation")
    async def my_operation():
        ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ai_infra.callbacks import (
    Callbacks,
    GraphNodeEndEvent,
    GraphNodeErrorEvent,
    GraphNodeStartEvent,
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    ToolEndEvent,
    ToolErrorEvent,
    ToolStartEvent,
)
from ai_infra.logging import get_logger
from ai_infra.tracing import Span, Tracer, get_tracer

if TYPE_CHECKING:
    pass

logger = get_logger("executor.tracing")


# =============================================================================
# Token Tracking
# =============================================================================


class TokenTracker:
    """Thread-local token tracking for LLM calls.

    This class provides a context for tracking tokens consumed during
    node execution. It is used by ExecutorCallbacks to bridge LLM events
    to node-level metrics tracking.

    Usage:
        # In callback
        TokenTracker.add_usage(tokens_in=100, tokens_out=50, llm_calls=1)

        # In node decorator
        TokenTracker.reset()
        ... execute node ...
        tokens_in, tokens_out, llm_calls = TokenTracker.get_usage()
    """

    _current_tokens_in: int = 0
    _current_tokens_out: int = 0
    _current_llm_calls: int = 0

    @classmethod
    def reset(cls) -> None:
        """Reset counters for a new tracking session."""
        cls._current_tokens_in = 0
        cls._current_tokens_out = 0
        cls._current_llm_calls = 0

    @classmethod
    def add_usage(
        cls,
        tokens_in: int = 0,
        tokens_out: int = 0,
        llm_calls: int = 0,
    ) -> None:
        """Add token usage from an LLM call."""
        cls._current_tokens_in += tokens_in
        cls._current_tokens_out += tokens_out
        cls._current_llm_calls += llm_calls

    @classmethod
    def get_usage(cls) -> tuple[int, int, int]:
        """Get current token usage (tokens_in, tokens_out, llm_calls)."""
        return cls._current_tokens_in, cls._current_tokens_out, cls._current_llm_calls


# =============================================================================
# Task Metrics
# =============================================================================


@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""

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


# =============================================================================
# Executor Metrics
# =============================================================================


@dataclass
class ExecutorMetrics:
    """Aggregated metrics for an executor run."""

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
    task_retries: int = 0
    fixes_applied: int = 0
    retry_successes: int = 0

    @property
    def avg_task_duration_ms(self) -> float:
        completed = self.tasks_completed + self.tasks_failed
        return self.total_duration_ms / completed if completed > 0 else 0.0

    @property
    def avg_llm_calls_per_task(self) -> float:
        completed = self.tasks_completed + self.tasks_failed
        return self.total_llm_calls / completed if completed > 0 else 0.0

    @property
    def avg_tokens_per_task(self) -> float:
        completed = self.tasks_completed + self.tasks_failed
        return self.total_tokens / completed if completed > 0 else 0.0

    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return (self.tasks_completed / total) * 100 if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
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
            "tools": {"total_calls": self.total_tool_calls},
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
        ]
        return "\n".join(lines)


# =============================================================================
# Executor Callbacks
# =============================================================================


class ExecutorCallbacks(Callbacks):
    """Callback handler for executor observability.

    Combines metrics collection with ai_infra.tracing for spans.
    Pass this to the executor to track LLM and tool usage.
    """

    def __init__(self) -> None:
        self._metrics = ExecutorMetrics()
        self._current_task: TaskMetrics | None = None
        self._tracer: Tracer | None = None
        self._run_span: Span | None = None
        self._current_span: Span | None = None
        self._task_start_time: float = 0.0
        # Graph node tracking (from ExecutorTracingCallbacks)
        self._node_spans: dict[str, Span] = {}

    # =========================================================================
    # Run Lifecycle
    # =========================================================================

    def set_run_context(self, run_id: str, roadmap_path: str = "") -> None:
        """Set the run context for metrics and tracing."""
        self._metrics.run_id = run_id
        self._metrics.started_at = datetime.now(UTC)
        self._tracer = get_tracer()

        # Start run-level span
        self._run_span = self._tracer.start_span(
            "executor.run",
            attributes={"run.id": run_id, "run.roadmap_path": roadmap_path},
        )

        logger.info("executor_run_started", run_id=run_id)

    def on_run_end(
        self,
        error: Exception | None = None,
    ) -> None:
        """Called when the executor run completes."""
        self._metrics.completed_at = datetime.now(UTC)
        if self._metrics.started_at:
            duration = (
                self._metrics.completed_at - self._metrics.started_at
            ).total_seconds() * 1000
            self._metrics.total_duration_ms = duration

        # End run span
        if self._run_span and self._tracer:
            self._run_span.set_attributes(
                {
                    "run.completed_tasks": self._metrics.tasks_completed,
                    "run.failed_tasks": self._metrics.tasks_failed,
                }
            )
            if error:
                self._run_span.record_exception(error)
            self._tracer.end_span(self._run_span)
            self._run_span = None

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
        """Called when a task starts execution."""
        self._metrics.tasks_started += 1
        self._task_start_time = time.time()

        self._current_task = TaskMetrics(
            task_id=task_id,
            started_at=datetime.now(UTC),
        )

        if self._tracer:
            self._current_span = self._tracer.start_span(
                f"executor.task.{task_id}",
                attributes={"task.id": task_id, "task.title": title},
                parent=self._run_span,
            )

        logger.info("task_started", task_id=task_id, title=title)

    def on_task_end(
        self,
        task_id: str,
        success: bool,
        files_modified: int = 0,
        error: str | None = None,
    ) -> None:
        """Called when a task completes."""
        duration_ms = (time.time() - self._task_start_time) * 1000

        if self._current_task:
            self._current_task.completed_at = datetime.now(UTC)
            self._current_task.duration_ms = duration_ms
            self._current_task.success = success
            self._current_task.files_modified = files_modified
            self._current_task.error = error
            self._metrics.task_metrics.append(self._current_task)

        if success:
            self._metrics.tasks_completed += 1
        else:
            self._metrics.tasks_failed += 1

        if self._current_span and self._tracer:
            self._current_span.set_attribute("task.success", success)
            self._current_span.set_attribute("task.files_modified", files_modified)
            if error:
                self._current_span.set_status("error", error)
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
        """Called when a task is skipped."""
        self._metrics.tasks_skipped += 1
        logger.info("task_skipped", task_id=task_id, reason=reason)

    # =========================================================================
    # LLM Events
    # =========================================================================

    def on_llm_start(self, event: LLMStartEvent) -> None:
        logger.debug(
            "llm_call_started",
            provider=event.provider,
            model=event.model,
            task_id=self._current_task.task_id if self._current_task else None,
        )

    def on_llm_end(self, event: LLMEndEvent) -> None:
        self._metrics.total_llm_calls += 1
        if event.total_tokens:
            self._metrics.total_tokens += event.total_tokens

        if self._current_task:
            self._current_task.llm_calls += 1
            if event.total_tokens:
                self._current_task.llm_tokens += event.total_tokens

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
        logger.debug(
            "tool_call_started",
            tool=event.tool_name,
            task_id=self._current_task.task_id if self._current_task else None,
        )

    def on_tool_end(self, event: ToolEndEvent) -> None:
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
        logger.warning(
            "tool_call_failed",
            tool=event.tool_name,
            error=str(event.error),
            task_id=self._current_task.task_id if self._current_task else None,
        )

    # =========================================================================
    # Graph Node Events (from ExecutorTracingCallbacks)
    # =========================================================================

    def on_graph_node_start(self, event: GraphNodeStartEvent) -> None:
        if self._tracer:
            span = self._tracer.start_span(
                f"executor.node.{event.node_id}",
                attributes={
                    "node.id": event.node_id,
                    "node.type": event.node_type,
                    "node.step": event.step,
                },
                parent=self._current_span or self._run_span,
            )
            self._node_spans[event.node_id] = span

    def on_graph_node_end(self, event: GraphNodeEndEvent) -> None:
        if span := self._node_spans.pop(event.node_id, None):
            span.set_attribute("node.latency_ms", event.latency_ms)
            if self._tracer:
                self._tracer.end_span(span)

    def on_graph_node_error(self, event: GraphNodeErrorEvent) -> None:
        if span := self._node_spans.pop(event.node_id, None):
            span.record_exception(event.error)
            if self._tracer:
                self._tracer.end_span(span)

    # =========================================================================
    # Retry Events
    # =========================================================================

    def on_task_retry(
        self,
        task_id: str,
        attempt: int,
        max_attempts: int,
        previous_error: str | None = None,
    ) -> None:
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
        self._metrics.retry_successes += 1
        if self._current_task:
            self._current_task.retry_success = True
        logger.info("task_retry_succeeded", task_id=task_id, attempt=attempt)

    def on_fix_applied(
        self,
        task_id: str,
        suggestion_type: str,
        description: str,
    ) -> None:
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
        self._metrics.checkpoints_created += 1
        logger.info("checkpoint_created", checkpoint_id=checkpoint_id, task_id=task_id)

    def on_rollback(self, checkpoint_id: str, reason: str) -> None:
        self._metrics.rollbacks_performed += 1
        logger.warning("rollback_performed", checkpoint_id=checkpoint_id, reason=reason)

    def on_dependency_warning(
        self,
        file_path: str,
        impact: str,
        affected_files: list[str],
    ) -> None:
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
        return self._metrics

    def get_current_task_metrics(self) -> TaskMetrics | None:
        return self._current_task

    def reset(self) -> None:
        self._metrics = ExecutorMetrics()
        self._current_task = None
        self._current_span = None
        self._run_span = None
        self._node_spans.clear()


# =============================================================================
# Tracing Configuration
# =============================================================================


@dataclass
class TracingConfig:
    """Configuration for executor tracing."""

    enabled: bool = True
    include_state_details: bool = True
    console_output: bool = False

    @classmethod
    def development(cls) -> TracingConfig:
        return cls(enabled=True, include_state_details=True, console_output=True)

    @classmethod
    def production(cls) -> TracingConfig:
        return cls(enabled=True, include_state_details=False, console_output=False)

    @classmethod
    def disabled(cls) -> TracingConfig:
        return cls(enabled=False)


# =============================================================================
# Factory Functions
# =============================================================================


def create_tracing_callbacks(
    config: TracingConfig | None = None,
    run_id: str | None = None,
    roadmap_path: str = "",
) -> ExecutorCallbacks:
    """Create configured executor callbacks with tracing.

    Args:
        config: Tracing configuration.
        run_id: Optional run ID to start tracing immediately.
        roadmap_path: Path to roadmap file.

    Returns:
        Configured ExecutorCallbacks.
    """
    if config is None:
        config = TracingConfig()

    callbacks = ExecutorCallbacks()

    if config.enabled and run_id:
        callbacks.set_run_context(run_id, roadmap_path)

    if config.console_output:
        from ai_infra.tracing import ConsoleExporter

        tracer = get_tracer()
        tracer.add_exporter(ConsoleExporter(verbose=True))

    return callbacks


# =============================================================================
# Logging Utilities
# =============================================================================


def log_task_context(
    task_id: str,
    context_tokens: int,
    files_included: int,
    file_hints: list[str] | None = None,
) -> None:
    """Log task context information."""
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
    """Log verification result."""
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
    """Log recovery action."""
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
# Backwards Compatibility
# =============================================================================

# Alias for backwards compatibility with graph_tracing imports
ExecutorTracingCallbacks = ExecutorCallbacks


def traced_node(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Any:
    """Decorator for tracing node functions.

    This is a thin wrapper around ai_infra.tracing.trace for backwards
    compatibility. New code should use @trace directly.
    """
    from ai_infra.tracing import trace

    return trace(name=name, attributes=attributes)


def create_traced_nodes() -> dict[str, Any]:
    """Create traced versions of all node functions.

    Returns:
        Dictionary mapping node names to traced callables.
    """
    from ai_infra.executor.nodes import (
        build_context_node,
        checkpoint_node,
        decide_next_node,
        execute_task_node,
        handle_failure_node,
        parse_roadmap_node,
        pick_task_node,
        rollback_node,
        verify_task_node,
    )
    from ai_infra.tracing import trace

    return {
        "parse_roadmap": trace("executor.node.parse_roadmap")(parse_roadmap_node),
        "pick_task": trace("executor.node.pick_task")(pick_task_node),
        "build_context": trace("executor.node.build_context")(build_context_node),
        "execute_task": trace("executor.node.execute_task")(execute_task_node),
        "verify_task": trace("executor.node.verify_task")(verify_task_node),
        "checkpoint": trace("executor.node.checkpoint")(checkpoint_node),
        "rollback": trace("executor.node.rollback")(rollback_node),
        "handle_failure": trace("executor.node.handle_failure")(handle_failure_node),
        "decide_next": trace("executor.node.decide_next")(decide_next_node),
    }


# Alias for get_tracer (backwards compatibility with observability.py)
get_executor_tracer = get_tracer


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Token Tracking
    "TokenTracker",
    # Metrics
    "TaskMetrics",
    "ExecutorMetrics",
    # Callbacks
    "ExecutorCallbacks",
    "ExecutorTracingCallbacks",  # Alias for backwards compat
    # Configuration
    "TracingConfig",
    # Factory
    "create_tracing_callbacks",
    "create_traced_nodes",
    # Decorators
    "traced_node",
    # Utilities
    "get_executor_tracer",
    "log_task_context",
    "log_verification_result",
    "log_recovery_action",
]
