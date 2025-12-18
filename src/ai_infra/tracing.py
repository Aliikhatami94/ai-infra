"""Tracing support for ai-infra.

Provides optional integration with:
- LangSmith (LangChain's tracing platform)
- OpenTelemetry (vendor-neutral observability)

Tracing is disabled by default and auto-configures from environment:
- LANGSMITH_API_KEY: Enable LangSmith tracing
- OTEL_EXPORTER_OTLP_ENDPOINT: Enable OpenTelemetry tracing

Usage:
    # Auto-detect from environment
    from ai_infra.tracing import get_tracer
    tracer = get_tracer()

    # Explicit configuration
    from ai_infra.tracing import configure_tracing
    configure_tracing(
        langsmith=True,  # or LANGSMITH_API_KEY env var
        opentelemetry=True,  # or OTEL_EXPORTER_OTLP_ENDPOINT env var
    )

    # Use as decorator
    from ai_infra.tracing import trace

    @trace(name="my_operation")
    async def my_function():
        ...

    # Use as context manager
    async with tracer.span("my_operation") as span:
        span.set_attribute("key", "value")
        ...
"""

from __future__ import annotations

import functools
import os
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from .callbacks import (
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

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Span Interface
# =============================================================================


@dataclass
class SpanContext:
    """Context for a tracing span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None

    def to_dict(self) -> dict[str, str]:
        d = {"trace_id": self.trace_id, "span_id": self.span_id}
        if self.parent_span_id:
            d["parent_span_id"] = self.parent_span_id
        return d


class Span:
    """A tracing span representing a unit of work.

    Spans can be nested and form a tree structure representing
    the execution flow.
    """

    def __init__(
        self,
        name: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.parent = parent
        self._attributes: dict[str, Any] = attributes or {}
        self._events: list[dict[str, Any]] = []
        self._start_time = time.time()
        self._end_time: float | None = None
        self._status = "ok"
        self._error: Exception | None = None

        # Generate IDs
        import uuid

        self._span_id: str = uuid.uuid4().hex[:16]
        self._trace_id: str = parent._trace_id if parent else uuid.uuid4().hex[:32]

    @property
    def context(self) -> SpanContext:
        """Get span context for propagation."""
        return SpanContext(
            trace_id=self._trace_id,
            span_id=self._span_id,
            parent_span_id=self.parent._span_id if self.parent else None,
        )

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self._end_time or time.time()
        return (end - self._start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> Span:
        """Set a span attribute."""
        self._attributes[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> Span:
        """Set multiple span attributes."""
        self._attributes.update(attributes)
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        """Add an event to the span."""
        self._events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )
        return self

    def set_status(self, status: str, description: str | None = None) -> Span:
        """Set span status (ok, error)."""
        self._status = status
        if description:
            self._attributes["status_description"] = description
        return self

    def record_exception(self, error: Exception) -> Span:
        """Record an exception on the span."""
        self._error = error
        self._status = "error"
        self._attributes["error"] = True
        self._attributes["error.type"] = type(error).__name__
        self._attributes["error.message"] = str(error)
        return self

    def end(self) -> None:
        """End the span."""
        self._end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "name": self.name,
            "trace_id": self._trace_id,
            "span_id": self._span_id,
            "parent_span_id": self.parent._span_id if self.parent else None,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "duration_ms": self.duration_ms,
            "status": self._status,
            "attributes": self._attributes,
            "events": self._events,
        }


# =============================================================================
# Tracer Interface
# =============================================================================


class Tracer:
    """Tracer for creating and managing spans.

    This is the main entry point for tracing. Get a tracer via
    get_tracer() and use it to create spans.
    """

    def __init__(self, name: str = "ai-infra"):
        self.name = name
        self._exporters: list[SpanExporter] = []
        self._current_span: Span | None = None

    def add_exporter(self, exporter: SpanExporter) -> None:
        """Add a span exporter."""
        self._exporters.append(exporter)

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent: Span | None = None,
    ) -> Span:
        """Start a new span."""
        span = Span(
            name=name,
            parent=parent or self._current_span,
            attributes=attributes,
        )
        self._current_span = span
        return span

    def end_span(self, span: Span) -> None:
        """End a span and export it."""
        span.end()

        # Restore parent as current
        if span == self._current_span:
            self._current_span = span.parent

        # Export to all exporters
        for exporter in self._exporters:
            try:
                exporter.export(span)
            except Exception:
                pass  # Don't let export errors affect tracing

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Context manager for creating a span."""
        span = self.start_span(name, attributes)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            self.end_span(span)

    @asynccontextmanager
    async def aspan(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        """Async context manager for creating a span."""
        span = self.start_span(name, attributes)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            self.end_span(span)


# =============================================================================
# Span Exporters
# =============================================================================


class SpanExporter:
    """Base class for span exporters."""

    def export(self, span: Span) -> None:
        """Export a span."""
        raise NotImplementedError


class ConsoleExporter(SpanExporter):
    """Export spans to console (for debugging)."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def export(self, span: Span) -> None:
        status = "[OK]" if span._status == "ok" else "[ERROR]"
        print(f"{status} {span.name}: {span.duration_ms:.1f}ms")
        if self.verbose:
            for key, value in span._attributes.items():
                print(f"   {key}: {value}")


class LangSmithExporter(SpanExporter):
    """Export spans to LangSmith.

    Requires LANGSMITH_API_KEY environment variable.
    """

    def __init__(self, api_key: str | None = None, project: str | None = None):
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        self.project = project or os.getenv("LANGSMITH_PROJECT", "ai-infra")
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-load LangSmith client."""
        if self._client is None:
            try:
                from langsmith import Client

                self._client = Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "LangSmith integration requires langsmith package. "
                    "Install with: pip install langsmith"
                )
        return self._client

    def export(self, span: Span) -> None:
        """Export span to LangSmith."""
        if not self.api_key:
            return

        try:
            client = self._get_client()

            # Map span to LangSmith run
            run_type = "chain"
            if "llm" in span.name.lower():
                run_type = "llm"
            elif "tool" in span.name.lower():
                run_type = "tool"

            client.create_run(
                name=span.name,
                run_type=run_type,
                project_name=self.project,
                inputs=span._attributes.get("inputs", {}),
                outputs=span._attributes.get("outputs", {}),
                start_time=span._start_time,
                end_time=span._end_time,
                error=str(span._error) if span._error else None,
                extra={"attributes": span._attributes},
                tags=span._attributes.get("tags", []),
            )
        except Exception:
            pass  # Silently fail to not affect main flow


class OpenTelemetryExporter(SpanExporter):
    """Export spans to OpenTelemetry.

    Requires OTEL_EXPORTER_OTLP_ENDPOINT environment variable.
    """

    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self._tracer: Any = None

    def _get_tracer(self) -> Any:
        """Lazy-load OpenTelemetry tracer."""
        if self._tracer is None:
            try:
                from opentelemetry import trace
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                resource = Resource.create({"service.name": "ai-infra"})
                provider = TracerProvider(resource=resource)

                if self.endpoint:
                    exporter = OTLPSpanExporter(endpoint=self.endpoint)
                    processor = BatchSpanProcessor(exporter)
                    provider.add_span_processor(processor)

                trace.set_tracer_provider(provider)
                self._tracer = trace.get_tracer("ai-infra")

            except ImportError:
                raise ImportError(
                    "OpenTelemetry integration requires opentelemetry packages. "
                    "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc"
                )
        return self._tracer

    def export(self, span: Span) -> None:
        """Export span to OpenTelemetry."""
        if not self.endpoint:
            return

        try:
            tracer = self._get_tracer()
            from opentelemetry import trace as otel_trace

            with tracer.start_as_current_span(span.name) as otel_span:
                for key, value in span._attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        otel_span.set_attribute(key, value)

                if span._error:
                    otel_span.record_exception(span._error)
                    otel_span.set_status(
                        otel_trace.StatusCode.ERROR,
                        str(span._error),
                    )
        except Exception:
            pass  # Silently fail


# =============================================================================
# Tracing Callbacks
# =============================================================================


class TracingCallbacks(Callbacks):
    """Callbacks that create spans for ai-infra operations."""

    def __init__(self, tracer: Tracer | None = None):
        self._tracer = tracer or get_tracer()
        self._llm_spans: dict[int, Span] = {}
        self._tool_spans: dict[str, Span] = {}
        self._graph_spans: dict[str, Span] = {}

    def on_llm_start(self, event: LLMStartEvent) -> None:
        span = self._tracer.start_span(
            f"llm.{event.provider}.{event.model}",
            attributes={
                "provider": event.provider,
                "model": event.model,
                "stream": event.stream,
            },
        )
        self._llm_spans[id(event)] = span

    def on_llm_end(self, event: LLMEndEvent) -> None:
        # Find span by provider/model (simplified matching)
        for key, span in list(self._llm_spans.items()):
            if (
                span._attributes.get("provider") == event.provider
                and span._attributes.get("model") == event.model
            ):
                span.set_attributes(
                    {
                        "input_tokens": event.input_tokens,
                        "output_tokens": event.output_tokens,
                        "total_tokens": event.total_tokens,
                        "latency_ms": event.latency_ms,
                    }
                )
                self._tracer.end_span(span)
                del self._llm_spans[key]
                break

    def on_llm_error(self, event: LLMErrorEvent) -> None:
        for key, span in list(self._llm_spans.items()):
            if (
                span._attributes.get("provider") == event.provider
                and span._attributes.get("model") == event.model
            ):
                span.record_exception(event.error)
                self._tracer.end_span(span)
                del self._llm_spans[key]
                break

    def on_tool_start(self, event: ToolStartEvent) -> None:
        span = self._tracer.start_span(
            f"tool.{event.tool_name}",
            attributes={
                "tool_name": event.tool_name,
                "server_name": event.server_name,
            },
        )
        self._tool_spans[event.tool_name] = span

    def on_tool_end(self, event: ToolEndEvent) -> None:
        if span := self._tool_spans.pop(event.tool_name, None):
            span.set_attribute("latency_ms", event.latency_ms)
            self._tracer.end_span(span)

    def on_tool_error(self, event: ToolErrorEvent) -> None:
        if span := self._tool_spans.pop(event.tool_name, None):
            span.record_exception(event.error)
            self._tracer.end_span(span)

    def on_graph_node_start(self, event: GraphNodeStartEvent) -> None:
        span = self._tracer.start_span(
            f"graph.{event.node_id}",
            attributes={
                "node_id": event.node_id,
                "node_type": event.node_type,
                "step": event.step,
            },
        )
        self._graph_spans[event.node_id] = span

    def on_graph_node_end(self, event: GraphNodeEndEvent) -> None:
        if span := self._graph_spans.pop(event.node_id, None):
            span.set_attribute("latency_ms", event.latency_ms)
            self._tracer.end_span(span)

    def on_graph_node_error(self, event: GraphNodeErrorEvent) -> None:
        if span := self._graph_spans.pop(event.node_id, None):
            span.record_exception(event.error)
            self._tracer.end_span(span)


# =============================================================================
# Global Tracer
# =============================================================================

_global_tracer: Tracer | None = None


def get_tracer(name: str = "ai-infra") -> Tracer:
    """Get the global tracer instance.

    Creates and configures a tracer on first call, then returns
    the same instance on subsequent calls.
    """
    global _global_tracer

    if _global_tracer is None:
        _global_tracer = Tracer(name)

        # Auto-configure from environment
        if os.getenv("LANGSMITH_API_KEY"):
            try:
                _global_tracer.add_exporter(LangSmithExporter())
            except ImportError:
                pass

        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            try:
                _global_tracer.add_exporter(OpenTelemetryExporter())
            except ImportError:
                pass

        # Debug mode
        if os.getenv("AI_INFRA_TRACE_DEBUG", "").lower() in ("1", "true"):
            _global_tracer.add_exporter(ConsoleExporter(verbose=True))

    return _global_tracer


def configure_tracing(
    *,
    langsmith: bool = False,
    opentelemetry: bool = False,
    console: bool = False,
    verbose: bool = False,
) -> Tracer:
    """Configure tracing exporters.

    Args:
        langsmith: Enable LangSmith export (requires LANGSMITH_API_KEY)
        opentelemetry: Enable OpenTelemetry export (requires OTEL_EXPORTER_OTLP_ENDPOINT)
        console: Enable console export (for debugging)
        verbose: Verbose console output

    Returns:
        Configured tracer instance
    """
    global _global_tracer

    tracer = Tracer("ai-infra")

    if langsmith:
        tracer.add_exporter(LangSmithExporter())

    if opentelemetry:
        tracer.add_exporter(OpenTelemetryExporter())

    if console:
        tracer.add_exporter(ConsoleExporter(verbose=verbose))

    _global_tracer = tracer
    return tracer


# =============================================================================
# Decorators
# =============================================================================


def trace(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator to trace a function.

    Example:
        @trace(name="process_data")
        def my_function(x: int) -> int:
            return x * 2

        @trace(attributes={"category": "processing"})
        async def async_function():
            ...
    """

    def decorator(fn: F) -> F:
        span_name = name or fn.__name__

        if asyncio_iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                async with tracer.aspan(span_name, attributes):
                    result = await fn(*args, **kwargs)
                    return result

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                with tracer.span(span_name, attributes):
                    result = fn(*args, **kwargs)
                    return result

            return sync_wrapper  # type: ignore

    return decorator


def asyncio_iscoroutinefunction(fn: Any) -> bool:
    """Check if function is async."""
    import asyncio

    return asyncio.iscoroutinefunction(fn)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core
    "Tracer",
    "Span",
    "SpanContext",
    # Configuration
    "get_tracer",
    "configure_tracing",
    # Exporters
    "SpanExporter",
    "ConsoleExporter",
    "LangSmithExporter",
    "OpenTelemetryExporter",
    # Callbacks
    "TracingCallbacks",
    # Decorators
    "trace",
]
