"""Tracing support for the Executor Graph.

Phase 1.6.1: Adds tracing to graph execution via ai_infra.tracing.

This module provides:
- TracedNode decorator for wrapping node functions
- ExecutorTracingCallbacks for graph-level tracing
- Integration with ai_infra.tracing infrastructure

Usage:
    from ai_infra.executor.graph_tracing import (
        TracedExecutorGraph,
        create_traced_executor,
    )

    # Create executor with tracing enabled
    executor = create_traced_executor(
        agent,
        "ROADMAP.md",
        tracing_enabled=True,
    )

    # Execution is automatically traced via ai_infra.tracing
    result = await executor.arun()
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from ai_infra.callbacks import (
    Callbacks,
    GraphNodeEndEvent,
    GraphNodeErrorEvent,
    GraphNodeStartEvent,
)
from ai_infra.logging import get_logger
from ai_infra.tracing import Span, Tracer, get_tracer

if TYPE_CHECKING:
    from ai_infra.executor.state import ExecutorGraphState

logger = get_logger("executor.graph_tracing")

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Traced Node Decorator
# =============================================================================


def traced_node(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator to add tracing to a node function.

    Phase 1.6.1: Wraps node functions with ai_infra.tracing spans.

    The decorator:
    1. Creates a span when the node starts
    2. Records relevant state attributes
    3. Handles errors and records exceptions
    4. Ends the span when node completes

    Args:
        name: Custom span name (defaults to function name).
        attributes: Additional span attributes.

    Returns:
        Decorated function with tracing.

    Example:
        @traced_node(name="parse_roadmap")
        async def parse_roadmap_node(state):
            ...
    """

    def decorator(fn: F) -> F:
        span_name = name or f"executor.node.{fn.__name__}"

        @functools.wraps(fn)
        async def async_wrapper(
            state: ExecutorGraphState,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            tracer = get_tracer()

            # Build attributes from state
            span_attributes = {
                "node.name": fn.__name__,
                "node.type": "executor_graph_node",
            }

            # Add task info if available
            current_task = state.get("current_task")
            if current_task:
                span_attributes["task.id"] = str(current_task.id)
                span_attributes["task.title"] = (
                    current_task.title if hasattr(current_task, "title") else str(current_task)
                )

            # Add state metrics
            span_attributes["state.completed_count"] = state.get("completed_count", 0)
            span_attributes["state.retry_count"] = state.get("retry_count", 0)

            # Merge custom attributes
            if attributes:
                span_attributes.update(attributes)

            async with tracer.aspan(span_name, attributes=span_attributes) as span:
                try:
                    result = await fn(state, *args, **kwargs)

                    # Record output metrics
                    if isinstance(result, dict):
                        if result.get("error"):
                            span.set_attribute("node.has_error", True)
                            span.set_attribute(
                                "node.error_type",
                                result["error"].get("error_type", "unknown"),
                            )
                        if result.get("verified") is not None:
                            span.set_attribute("node.verified", result["verified"])
                        if result.get("files_modified"):
                            span.set_attribute(
                                "node.files_modified",
                                len(result["files_modified"]),
                            )

                    return result

                except Exception as e:
                    span.record_exception(e)
                    raise

        return async_wrapper  # type: ignore

    return decorator


# =============================================================================
# Pre-wrapped Traced Node Functions
# =============================================================================


def create_traced_nodes() -> dict[str, Callable]:
    """Create traced versions of all node functions.

    Phase 1.6.1: Wraps each node with tracing.

    Returns:
        Dictionary mapping node names to traced callables.

    Note:
        The actual node functions are imported and wrapped at runtime
        to avoid circular imports.
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

    return {
        "parse_roadmap": traced_node("executor.node.parse_roadmap")(parse_roadmap_node),
        "pick_task": traced_node("executor.node.pick_task")(pick_task_node),
        "build_context": traced_node("executor.node.build_context")(build_context_node),
        "execute_task": traced_node("executor.node.execute_task")(execute_task_node),
        "verify_task": traced_node("executor.node.verify_task")(verify_task_node),
        "checkpoint": traced_node("executor.node.checkpoint")(checkpoint_node),
        "rollback": traced_node("executor.node.rollback")(rollback_node),
        "handle_failure": traced_node("executor.node.handle_failure")(handle_failure_node),
        "decide_next": traced_node("executor.node.decide_next")(decide_next_node),
    }


# =============================================================================
# Executor Tracing Callbacks
# =============================================================================


class ExecutorTracingCallbacks(Callbacks):
    """Callbacks that emit tracing spans for executor operations.

    Phase 1.6.1: Provides tracing integration via callback mechanism.

    This callback handler creates spans for:
    - Graph node executions
    - State transitions
    - LLM and tool calls (if integrated with agent)

    Example:
        callbacks = ExecutorTracingCallbacks()

        # Start run tracing
        callbacks.start_run("my-run-id")

        # ... executor runs with callbacks ...

        callbacks.end_run()
    """

    def __init__(
        self,
        tracer: Tracer | None = None,
        include_state_details: bool = True,
    ):
        """Initialize tracing callbacks.

        Args:
            tracer: Optional custom tracer (defaults to global).
            include_state_details: Whether to include state in spans.
        """
        self._tracer = tracer or get_tracer()
        self._include_state_details = include_state_details
        self._run_span: Span | None = None
        self._node_spans: dict[str, Span] = {}
        self._node_start_times: dict[str, float] = {}

    def start_run(self, run_id: str, roadmap_path: str = "") -> Span:
        """Start tracing for an executor run.

        Args:
            run_id: Unique identifier for the run.
            roadmap_path: Path to the roadmap file.

        Returns:
            The run span.
        """
        self._run_span = self._tracer.start_span(
            "executor.run",
            attributes={
                "run.id": run_id,
                "run.roadmap_path": roadmap_path,
            },
        )
        logger.debug(f"Started tracing for run {run_id}")
        return self._run_span

    def end_run(
        self,
        completed: int = 0,
        failed: int = 0,
        error: Exception | None = None,
    ) -> None:
        """End tracing for an executor run.

        Args:
            completed: Number of completed tasks.
            failed: Number of failed tasks.
            error: Optional run-level error.
        """
        if self._run_span:
            self._run_span.set_attributes(
                {
                    "run.completed_tasks": completed,
                    "run.failed_tasks": failed,
                }
            )

            if error:
                self._run_span.record_exception(error)

            self._tracer.end_span(self._run_span)
            self._run_span = None
            logger.debug("Ended tracing for run")

    def on_graph_node_start(self, event: GraphNodeStartEvent) -> None:
        """Handle graph node start event.

        Args:
            event: The node start event.
        """
        span = self._tracer.start_span(
            f"executor.node.{event.node_id}",
            attributes={
                "node.id": event.node_id,
                "node.type": event.node_type,
                "node.step": event.step,
            },
            parent=self._run_span,
        )
        self._node_spans[event.node_id] = span
        self._node_start_times[event.node_id] = time.time()

        # Add input state details if enabled
        if self._include_state_details and event.inputs:
            self._add_state_attributes(span, event.inputs)

    def on_graph_node_end(self, event: GraphNodeEndEvent) -> None:
        """Handle graph node end event.

        Args:
            event: The node end event.
        """
        span = self._node_spans.pop(event.node_id, None)
        if span:
            span.set_attribute("node.latency_ms", event.latency_ms)

            # Add output state details if enabled
            if self._include_state_details and event.outputs:
                self._add_state_attributes(span, event.outputs, prefix="output")

            self._tracer.end_span(span)
            self._node_start_times.pop(event.node_id, None)

    def on_graph_node_error(self, event: GraphNodeErrorEvent) -> None:
        """Handle graph node error event.

        Args:
            event: The node error event.
        """
        span = self._node_spans.pop(event.node_id, None)
        if span:
            span.record_exception(event.error)
            span.set_attribute("node.latency_ms", event.latency_ms)
            self._tracer.end_span(span)
            self._node_start_times.pop(event.node_id, None)

    def _add_state_attributes(
        self,
        span: Span,
        state: dict[str, Any],
        prefix: str = "state",
    ) -> None:
        """Add state attributes to span.

        Args:
            span: The span to add attributes to.
            state: The state dictionary.
            prefix: Attribute name prefix.
        """
        # Only add simple scalar values to avoid large spans
        safe_keys = [
            "completed_count",
            "retry_count",
            "verified",
            "should_continue",
            "roadmap_path",
        ]

        for key in safe_keys:
            if key in state:
                value = state[key]
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"{prefix}.{key}", value)

        # Add task ID if present
        current_task = state.get("current_task")
        if current_task and hasattr(current_task, "id"):
            span.set_attribute(f"{prefix}.current_task_id", str(current_task.id))

        # Add error info if present
        error = state.get("error")
        if error and isinstance(error, dict):
            span.set_attribute(f"{prefix}.has_error", True)
            span.set_attribute(f"{prefix}.error_type", error.get("error_type", "unknown"))


# =============================================================================
# Tracing Configuration
# =============================================================================


class TracingConfig:
    """Configuration for executor graph tracing.

    Attributes:
        enabled: Whether tracing is enabled.
        include_state_details: Include state info in spans.
        trace_llm_calls: Also trace LLM calls.
        trace_tool_calls: Also trace tool calls.
        console_output: Output spans to console (debug).
    """

    def __init__(
        self,
        enabled: bool = True,
        include_state_details: bool = True,
        trace_llm_calls: bool = True,
        trace_tool_calls: bool = True,
        console_output: bool = False,
    ):
        self.enabled = enabled
        self.include_state_details = include_state_details
        self.trace_llm_calls = trace_llm_calls
        self.trace_tool_calls = trace_tool_calls
        self.console_output = console_output

    @classmethod
    def development(cls) -> TracingConfig:
        """Development configuration with console output."""
        return cls(
            enabled=True,
            include_state_details=True,
            console_output=True,
        )

    @classmethod
    def production(cls) -> TracingConfig:
        """Production configuration for observability backend."""
        return cls(
            enabled=True,
            include_state_details=False,
            console_output=False,
        )

    @classmethod
    def disabled(cls) -> TracingConfig:
        """Disabled tracing."""
        return cls(enabled=False)


# =============================================================================
# Factory Functions
# =============================================================================


def create_tracing_callbacks(
    config: TracingConfig | None = None,
    run_id: str | None = None,
    roadmap_path: str = "",
) -> ExecutorTracingCallbacks:
    """Create configured tracing callbacks.

    Args:
        config: Tracing configuration.
        run_id: Optional run ID to start tracing immediately.
        roadmap_path: Path to roadmap file.

    Returns:
        Configured ExecutorTracingCallbacks.
    """
    if config is None:
        config = TracingConfig()

    if not config.enabled:
        # Return a no-op callbacks instance
        return ExecutorTracingCallbacks(include_state_details=False)

    # Configure tracer if needed
    if config.console_output:
        from ai_infra.tracing import ConsoleExporter

        tracer = get_tracer()
        tracer.add_exporter(ConsoleExporter(verbose=True))

    callbacks = ExecutorTracingCallbacks(
        include_state_details=config.include_state_details,
    )

    # Start run if ID provided
    if run_id:
        callbacks.start_run(run_id, roadmap_path)

    return callbacks


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Decorator
    "traced_node",
    # Factory
    "create_traced_nodes",
    # Callbacks
    "ExecutorTracingCallbacks",
    "create_tracing_callbacks",
    # Configuration
    "TracingConfig",
]
