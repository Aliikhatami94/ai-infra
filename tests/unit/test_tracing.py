"""Tests for tracing module."""

import time
from unittest.mock import MagicMock

import pytest

from ai_infra.tracing import (
    ConsoleExporter,
    Span,
    SpanContext,
    SpanExporter,
    Tracer,
    TracingCallbacks,
    configure_tracing,
    get_tracer,
    trace,
)

# =============================================================================
# SpanContext Tests
# =============================================================================


class TestSpanContext:
    """Tests for SpanContext."""

    def test_create_context(self):
        """Test creating span context."""
        ctx = SpanContext(
            trace_id="abc123",
            span_id="def456",
            parent_span_id="ghi789",
        )
        assert ctx.trace_id == "abc123"
        assert ctx.span_id == "def456"
        assert ctx.parent_span_id == "ghi789"

    def test_to_dict_with_parent(self):
        """Test to_dict includes parent_span_id when present."""
        ctx = SpanContext(
            trace_id="abc",
            span_id="def",
            parent_span_id="parent",
        )
        d = ctx.to_dict()
        assert d == {
            "trace_id": "abc",
            "span_id": "def",
            "parent_span_id": "parent",
        }

    def test_to_dict_without_parent(self):
        """Test to_dict excludes parent_span_id when None."""
        ctx = SpanContext(trace_id="abc", span_id="def")
        d = ctx.to_dict()
        assert d == {"trace_id": "abc", "span_id": "def"}
        assert "parent_span_id" not in d


# =============================================================================
# Span Tests
# =============================================================================


class TestSpan:
    """Tests for Span class."""

    def test_create_span(self):
        """Test creating a span."""
        span = Span(name="test-span")
        assert span.name == "test-span"
        assert span.parent is None
        assert span._status == "ok"

    def test_span_with_parent(self):
        """Test span with parent inherits trace_id."""
        parent = Span(name="parent")
        child = Span(name="child", parent=parent)

        assert child.parent is parent
        assert child._trace_id == parent._trace_id

    def test_span_context(self):
        """Test span context generation."""
        span = Span(name="test")
        ctx = span.context

        assert isinstance(ctx, SpanContext)
        assert ctx.trace_id == span._trace_id
        assert ctx.span_id == span._span_id
        assert ctx.parent_span_id is None

    def test_span_context_with_parent(self):
        """Test child span context has parent_span_id."""
        parent = Span(name="parent")
        child = Span(name="child", parent=parent)

        ctx = child.context
        assert ctx.parent_span_id == parent._span_id

    def test_set_attribute(self):
        """Test setting single attribute."""
        span = Span(name="test")
        result = span.set_attribute("key", "value")

        assert span._attributes["key"] == "value"
        assert result is span  # Returns self for chaining

    def test_set_attributes(self):
        """Test setting multiple attributes."""
        span = Span(name="test")
        span.set_attributes({"a": 1, "b": 2})

        assert span._attributes == {"a": 1, "b": 2}

    def test_add_event(self):
        """Test adding event to span."""
        span = Span(name="test")
        span.add_event("my_event", {"data": "value"})

        assert len(span._events) == 1
        event = span._events[0]
        assert event["name"] == "my_event"
        assert event["attributes"] == {"data": "value"}
        assert "timestamp" in event

    def test_add_event_without_attributes(self):
        """Test adding event without attributes."""
        span = Span(name="test")
        span.add_event("simple_event")

        assert span._events[0]["attributes"] == {}

    def test_set_status(self):
        """Test setting span status."""
        span = Span(name="test")
        span.set_status("error", "Something went wrong")

        assert span._status == "error"
        assert span._attributes["status_description"] == "Something went wrong"

    def test_record_exception(self):
        """Test recording exception on span."""
        span = Span(name="test")
        error = ValueError("Test error")
        span.record_exception(error)

        assert span._error is error
        assert span._status == "error"
        assert span._attributes["error"] is True
        assert span._attributes["error.type"] == "ValueError"
        assert span._attributes["error.message"] == "Test error"

    def test_end_span(self):
        """Test ending a span sets end_time."""
        span = Span(name="test")
        assert span._end_time is None

        span.end()
        assert span._end_time is not None

    def test_duration_ms_before_end(self):
        """Test duration_ms before span ends uses current time."""
        span = Span(name="test")
        time.sleep(0.01)  # 10ms
        duration = span.duration_ms

        assert duration >= 10  # At least 10ms

    def test_duration_ms_after_end(self):
        """Test duration_ms after span ends is fixed."""
        span = Span(name="test")
        time.sleep(0.01)
        span.end()

        duration1 = span.duration_ms
        time.sleep(0.01)
        duration2 = span.duration_ms

        assert duration1 == duration2  # Should be same after end

    def test_to_dict(self):
        """Test span serialization to dict."""
        span = Span(name="test", attributes={"key": "value"})
        span.add_event("event1")
        span.end()

        d = span.to_dict()

        assert d["name"] == "test"
        assert d["trace_id"] == span._trace_id
        assert d["span_id"] == span._span_id
        assert d["status"] == "ok"
        assert d["attributes"] == {"key": "value"}
        assert len(d["events"]) == 1
        assert "start_time" in d
        assert "end_time" in d
        assert "duration_ms" in d


# =============================================================================
# Tracer Tests
# =============================================================================


class TestTracer:
    """Tests for Tracer class."""

    def test_create_tracer(self):
        """Test creating a tracer."""
        tracer = Tracer(name="test-tracer")
        assert tracer.name == "test-tracer"
        assert tracer._exporters == []

    def test_add_exporter(self):
        """Test adding an exporter."""
        tracer = Tracer()
        exporter = ConsoleExporter()
        tracer.add_exporter(exporter)

        assert exporter in tracer._exporters

    def test_span_context_manager(self):
        """Test span as context manager."""
        tracer = Tracer()

        with tracer.span("test-span") as span:
            assert isinstance(span, Span)
            assert span.name == "test-span"

        assert span._end_time is not None

    def test_span_with_attributes(self):
        """Test span with initial attributes."""
        tracer = Tracer()

        with tracer.span("test", attributes={"key": "value"}) as span:
            assert span._attributes["key"] == "value"

    def test_span_exports_on_end(self):
        """Test span is exported when context exits."""
        tracer = Tracer()
        mock_exporter = MagicMock(spec=SpanExporter)
        tracer.add_exporter(mock_exporter)

        with tracer.span("test"):
            pass

        mock_exporter.export.assert_called_once()

    def test_span_with_error(self):
        """Test span records exception on error."""
        tracer = Tracer()

        with pytest.raises(ValueError):
            with tracer.span("test") as span:
                raise ValueError("Test error")

        assert span._status == "error"
        assert span._attributes["error.type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_aspan_context_manager(self):
        """Test async span context manager."""
        tracer = Tracer()

        async with tracer.aspan("async-span") as span:
            assert span.name == "async-span"

        assert span._end_time is not None

    @pytest.mark.asyncio
    async def test_aspan_with_error(self):
        """Test async span records exception on error."""
        tracer = Tracer()

        with pytest.raises(RuntimeError):
            async with tracer.aspan("test") as span:
                raise RuntimeError("Async error")

        assert span._status == "error"

    def test_start_span_and_end_span(self):
        """Test manual span start/end."""
        tracer = Tracer()

        span = tracer.start_span("manual-span")
        assert isinstance(span, Span)

        tracer.end_span(span)
        assert span._end_time is not None


# =============================================================================
# ConsoleExporter Tests
# =============================================================================


class TestConsoleExporter:
    """Tests for ConsoleExporter."""

    def test_export_span(self, capsys):
        """Test exporting span to console."""
        exporter = ConsoleExporter()
        span = Span(name="test-span")
        span.set_attribute("key", "value")
        span.end()

        exporter.export(span)

        captured = capsys.readouterr()
        assert "test-span" in captured.out

    def test_verbose_mode(self, capsys):
        """Test verbose console output."""
        exporter = ConsoleExporter(verbose=True)
        span = Span(name="verbose-span")
        span.set_attribute("detail", "info")
        span.end()

        exporter.export(span)

        captured = capsys.readouterr()
        assert "verbose-span" in captured.out


# =============================================================================
# configure_tracing Tests
# =============================================================================


class TestConfigureTracing:
    """Tests for configure_tracing function."""

    def test_configure_console(self):
        """Test configuring console tracing."""
        tracer = configure_tracing(console=True)

        assert isinstance(tracer, Tracer)
        assert any(isinstance(e, ConsoleExporter) for e in tracer._exporters)

    def test_configure_verbose_console(self):
        """Test configuring verbose console tracing."""
        tracer = configure_tracing(console=True, verbose=True)

        console_exporter = next(e for e in tracer._exporters if isinstance(e, ConsoleExporter))
        assert console_exporter.verbose is True


# =============================================================================
# get_tracer Tests
# =============================================================================


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_get_tracer_returns_tracer(self, monkeypatch):
        """Test get_tracer returns a Tracer instance."""
        # Reset global tracer
        import ai_infra.tracing as tracing_module

        monkeypatch.setattr(tracing_module, "_global_tracer", None)

        tracer = get_tracer()
        assert isinstance(tracer, Tracer)

    def test_get_tracer_returns_same_instance(self, monkeypatch):
        """Test get_tracer returns same instance on repeated calls."""
        import ai_infra.tracing as tracing_module

        monkeypatch.setattr(tracing_module, "_global_tracer", None)

        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2

    def test_auto_configure_debug_mode(self, monkeypatch):
        """Test auto-configure debug mode from env."""
        import ai_infra.tracing as tracing_module

        monkeypatch.setattr(tracing_module, "_global_tracer", None)
        monkeypatch.setenv("AI_INFRA_TRACE_DEBUG", "true")

        tracer = get_tracer()
        assert any(isinstance(e, ConsoleExporter) for e in tracer._exporters)


# =============================================================================
# trace Decorator Tests
# =============================================================================


class TestTraceDecorator:
    """Tests for @trace decorator."""

    def test_trace_sync_function(self, monkeypatch):
        """Test tracing a sync function."""
        import ai_infra.tracing as tracing_module

        monkeypatch.setattr(tracing_module, "_global_tracer", None)

        @trace(name="my_function")
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_trace_uses_function_name(self, monkeypatch):
        """Test trace uses function name when no name provided."""
        import ai_infra.tracing as tracing_module

        monkeypatch.setattr(tracing_module, "_global_tracer", None)

        @trace()
        def another_func():
            return "done"

        result = another_func()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_trace_async_function(self, monkeypatch):
        """Test tracing an async function."""
        import ai_infra.tracing as tracing_module

        monkeypatch.setattr(tracing_module, "_global_tracer", None)

        @trace(name="async_op")
        async def async_func(x: int) -> int:
            return x * 3

        result = await async_func(4)
        assert result == 12

    def test_trace_with_attributes(self, monkeypatch):
        """Test trace with custom attributes."""
        import ai_infra.tracing as tracing_module

        monkeypatch.setattr(tracing_module, "_global_tracer", None)

        @trace(attributes={"category": "test"})
        def categorized_func():
            return "result"

        result = categorized_func()
        assert result == "result"


# =============================================================================
# TracingCallbacks Tests
# =============================================================================


class TestTracingCallbacks:
    """Tests for TracingCallbacks."""

    def test_create_callbacks(self):
        """Test creating TracingCallbacks."""
        tracer = Tracer()
        callbacks = TracingCallbacks(tracer)
        assert callbacks._tracer is tracer

    def test_llm_start_creates_span(self):
        """Test on_llm_start creates a span."""
        from ai_infra.callbacks import LLMStartEvent

        tracer = Tracer()
        callbacks = TracingCallbacks(tracer)

        event = LLMStartEvent(
            provider="openai",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        callbacks.on_llm_start(event)

        # Should have a span in the internal dict
        assert len(callbacks._llm_spans) == 1

    def test_llm_end_closes_span(self):
        """Test on_llm_end closes the span."""
        from ai_infra.callbacks import LLMEndEvent, LLMStartEvent

        tracer = Tracer()
        callbacks = TracingCallbacks(tracer)

        start_event = LLMStartEvent(
            provider="openai",
            model="gpt-4",
            messages=[],
        )
        callbacks.on_llm_start(start_event)

        end_event = LLMEndEvent(
            provider="openai",
            model="gpt-4",
            response="Hello!",
            latency_ms=100.0,
            input_tokens=10,
            output_tokens=5,
        )
        callbacks.on_llm_end(end_event)

        # Span should be removed from tracking
        assert len(callbacks._llm_spans) == 0

    def test_tool_start_creates_span(self):
        """Test on_tool_start creates a span."""
        from ai_infra.callbacks import ToolStartEvent

        tracer = Tracer()
        callbacks = TracingCallbacks(tracer)

        event = ToolStartEvent(
            tool_name="search",
            arguments={"query": "test"},
        )
        callbacks.on_tool_start(event)

        assert "search" in callbacks._tool_spans

    def test_tool_error_records_exception(self):
        """Test on_tool_error records exception on span."""
        from ai_infra.callbacks import ToolErrorEvent, ToolStartEvent

        tracer = Tracer()
        callbacks = TracingCallbacks(tracer)

        start_event = ToolStartEvent(
            tool_name="search",
            arguments={},
        )
        callbacks.on_tool_start(start_event)

        error_event = ToolErrorEvent(
            tool_name="search",
            error=ValueError("Tool failed"),
            arguments={},
        )
        callbacks.on_tool_error(error_event)

        assert "search" not in callbacks._tool_spans
