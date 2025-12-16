"""Tests for cross-cutting concerns (errors, logging, callbacks, tracing)."""

import logging
import time

import pytest

from ai_infra.callbacks import (
    CallbackManager,
    Callbacks,
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    MetricsCallbacks,
    ToolEndEvent,
    ToolStartEvent,
)
from ai_infra.errors import (
    AIInfraError,
    ConfigurationError,
    MCPError,
    MCPTimeoutError,
    OpenAPIError,
    ProviderError,
    RateLimitError,
    ValidationError,
)
from ai_infra.logging import (
    HumanFormatter,
    JSONFormatter,
    LLMLogger,
    RequestLogger,
    StructuredLogger,
    configure_logging,
    get_logger,
)
from ai_infra.tracing import ConsoleExporter, Span, Tracer, get_tracer, trace

# =============================================================================
# Error Tests
# =============================================================================


class TestErrors:
    """Tests for error hierarchy."""

    def test_ai_infra_error_base(self):
        """Test base error class."""
        error = AIInfraError("Test error")
        assert error.message == "Test error"
        assert error.details == {}
        assert error.hint is None
        assert error.docs_url is None
        assert str(error) == "Test error"

    def test_ai_infra_error_with_hint(self):
        """Test error with hint and docs."""
        error = AIInfraError(
            "Test error",
            hint="Try this fix",
            docs_url="https://example.com/docs",
        )
        assert error.hint == "Try this fix"
        assert error.docs_url == "https://example.com/docs"
        assert "Hint: Try this fix" in str(error)
        assert "Docs: https://example.com/docs" in str(error)

    def test_provider_error(self):
        """Test provider error with status code."""
        error = ProviderError(
            "Too many requests",
            provider="openai",
            model="gpt-4o",
            status_code=429,
        )
        assert error.provider == "openai"
        assert error.model == "gpt-4o"
        assert error.status_code == 429

    def test_rate_limit_error(self):
        """Test rate limit error with retry_after."""
        error = RateLimitError(
            "Rate limit exceeded",
            provider="openai",
            model="gpt-4o",
            retry_after=30.0,
        )
        assert error.retry_after == 30.0

    def test_mcp_timeout_error(self):
        """Test MCP timeout error."""
        error = MCPTimeoutError(
            "Connection timed out",
            operation="connect",
            timeout=10.0,
        )
        assert error.timeout == 10.0

    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError("Validation failed")
        assert "Validation failed" in str(error)

    def test_configuration_error(self):
        """Test configuration error."""
        error = ConfigurationError(
            "Missing config",
            config_key="api_key",
        )
        assert "Missing config" in str(error)

    def test_error_inheritance(self):
        """Test that all errors inherit from AIInfraError."""
        errors = [
            ProviderError("test", provider="openai"),
            RateLimitError("test", provider="openai"),
            MCPError("test"),
            MCPTimeoutError("test"),
            OpenAPIError("test"),
            ValidationError("test"),
            ConfigurationError("test"),
        ]

        for error in errors:
            assert isinstance(error, AIInfraError)


# =============================================================================
# Logging Tests
# =============================================================================


class TestLogging:
    """Tests for structured logging."""

    def test_configure_logging(self):
        """Test logging configuration."""
        # Should not raise
        configure_logging(level="DEBUG", format="human")

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test")
        assert logger is not None

    def test_structured_logger(self):
        """Test structured logger."""
        logger = StructuredLogger("test")

        # Should not raise
        logger.info("Test message", key="value")
        logger.warning("Warning message", count=42)
        logger.error("Error message", error="Something went wrong")

    def test_request_logger(self):
        """Test request logging."""
        logger = RequestLogger("requests")

        # Start a request log
        log = logger.log_request("GET", "https://api.example.com/data")
        assert log.method == "GET"
        assert log.url == "https://api.example.com/data"

        # Complete the request
        log.complete(status_code=200)
        assert log.status_code == 200
        assert log.latency_ms > 0

    def test_request_logger_sanitizes_sensitive_headers(self):
        """Test that sensitive headers are sanitized."""
        logger = RequestLogger("requests")

        headers = {
            "Authorization": "Bearer sk-secret123",
            "Content-Type": "application/json",
            "X-Api-Key": "my-api-key",
        }

        sanitized = logger._sanitize_headers(headers)
        assert sanitized["Authorization"] == "[REDACTED]"
        assert sanitized["X-Api-Key"] == "[REDACTED]"
        assert sanitized["Content-Type"] == "application/json"

    def test_llm_logger(self):
        """Test LLM call logging."""
        logger = LLMLogger("llm")

        log = logger.log_call_start("openai", "gpt-4o")
        assert log.provider == "openai"
        assert log.model == "gpt-4o"

        log.complete(input_tokens=100, output_tokens=50)
        assert log.input_tokens == 100
        assert log.output_tokens == 50
        assert log.total_tokens == 150
        assert log.latency_ms > 0

    def test_json_formatter(self):
        """Test JSON log formatting."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        assert '"message": "Test message"' in output
        assert '"level": "INFO"' in output

    def test_human_formatter(self):
        """Test human-readable log formatting."""
        formatter = HumanFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        assert "INFO" in output
        assert "Test message" in output


# =============================================================================
# Callbacks Tests
# =============================================================================


class TestCallbacks:
    """Tests for callback system."""

    def test_callbacks_base_class(self):
        """Test that Callbacks base class methods are no-ops."""
        callbacks = Callbacks()

        # None of these should raise
        callbacks.on_llm_start(
            LLMStartEvent(
                provider="openai",
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )
        )
        callbacks.on_llm_end(
            LLMEndEvent(
                provider="openai",
                model="gpt-4o",
                response="Hi!",
            )
        )

    def test_callback_manager(self):
        """Test callback manager with multiple callbacks."""
        cb1 = MetricsCallbacks()
        cb2 = MetricsCallbacks()

        manager = CallbackManager([cb1, cb2])

        # Fire events
        start_event = LLMStartEvent(
            provider="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        end_event = LLMEndEvent(
            provider="openai",
            model="gpt-4o",
            response="Hello!",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_ms=100,
        )

        manager.on_llm_start(start_event)
        manager.on_llm_end(end_event)

        # Both callbacks should have received the event
        assert cb1.llm_calls == 1
        assert cb2.llm_calls == 1
        assert cb1.total_tokens == 15
        assert cb2.total_tokens == 15

    def test_metrics_callbacks(self):
        """Test metrics collection callbacks."""
        metrics = MetricsCallbacks()

        # Simulate 2 LLM calls
        for _ in range(2):
            metrics.on_llm_end(
                LLMEndEvent(
                    provider="openai",
                    model="gpt-4o",
                    response="Hello",
                    total_tokens=100,
                    latency_ms=50.0,
                )
            )

        # Simulate 1 tool call
        metrics.on_tool_end(
            ToolEndEvent(
                tool_name="get_weather",
                result={"temp": 72},
                latency_ms=20.0,
            )
        )

        assert metrics.llm_calls == 2
        assert metrics.total_tokens == 200
        assert metrics.total_latency_ms == 100.0
        assert metrics.tool_calls == 1
        assert metrics.tool_latency_ms == 20.0

    def test_llm_events(self):
        """Test LLM event dataclasses."""
        start = LLMStartEvent(
            provider="anthropic",
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        assert start.provider == "anthropic"
        assert start.model == "claude-3"
        assert start.stream is True
        assert start.timestamp > 0

        error = LLMErrorEvent(
            provider="openai",
            model="gpt-4o",
            error=ValueError("test"),
        )
        assert error.error_type == "ValueError"

    def test_tool_events(self):
        """Test tool event dataclasses."""
        start = ToolStartEvent(
            tool_name="get_weather",
            arguments={"city": "NYC"},
            server_name="weather-server",
        )
        assert start.tool_name == "get_weather"
        assert start.arguments == {"city": "NYC"}
        assert start.server_name == "weather-server"


# =============================================================================
# Tracing Tests
# =============================================================================


class TestTracing:
    """Tests for tracing system."""

    def test_span_creation(self):
        """Test basic span creation."""
        span = Span("test-operation")

        assert span.name == "test-operation"
        assert span._status == "ok"
        assert span._start_time > 0
        assert span._end_time is None

    def test_span_attributes(self):
        """Test span attribute setting."""
        span = Span("test")

        span.set_attribute("key", "value")
        span.set_attributes({"a": 1, "b": 2})

        assert span._attributes["key"] == "value"
        assert span._attributes["a"] == 1
        assert span._attributes["b"] == 2

    def test_span_events(self):
        """Test span events."""
        span = Span("test")

        span.add_event("checkpoint", {"step": 1})
        span.add_event("done")

        assert len(span._events) == 2
        assert span._events[0]["name"] == "checkpoint"
        assert span._events[0]["attributes"]["step"] == 1

    def test_span_error_recording(self):
        """Test span error recording."""
        span = Span("test")
        error = ValueError("test error")

        span.record_exception(error)

        assert span._status == "error"
        assert span._error is error
        assert span._attributes["error"] is True
        assert span._attributes["error.type"] == "ValueError"
        assert span._attributes["error.message"] == "test error"

    def test_span_context(self):
        """Test span context generation."""
        parent = Span("parent")
        child = Span("child", parent=parent)

        context = child.context

        assert context.trace_id == parent._trace_id
        assert context.parent_span_id == parent._span_id
        assert context.span_id == child._span_id

    def test_span_duration(self):
        """Test span duration calculation."""
        span = Span("test")
        time.sleep(0.01)  # 10ms
        span.end()

        assert span.duration_ms >= 10
        assert span._end_time is not None

    def test_tracer_span_context_manager(self):
        """Test tracer span context manager."""
        tracer = Tracer("test")

        with tracer.span("operation") as span:
            span.set_attribute("key", "value")

        assert span._end_time is not None

    def test_tracer_span_error_handling(self):
        """Test that span records exceptions."""
        tracer = Tracer("test")

        with pytest.raises(ValueError):
            with tracer.span("failing") as span:
                raise ValueError("test error")

        assert span._status == "error"
        assert span._attributes["error.type"] == "ValueError"

    def test_console_exporter(self):
        """Test console exporter."""
        exporter = ConsoleExporter(verbose=False)
        span = Span("test-span")
        span.end()

        # Should not raise
        exporter.export(span)

    def test_get_tracer(self):
        """Test global tracer singleton."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()

        assert tracer1 is tracer2

    def test_trace_decorator(self):
        """Test trace decorator for sync functions."""

        @trace(name="decorated-function")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10


@pytest.mark.asyncio
class TestAsyncTracing:
    """Async tests for tracing."""

    async def test_tracer_async_span(self):
        """Test async span context manager."""
        tracer = Tracer("test")

        async with tracer.aspan("async-operation") as span:
            span.set_attribute("async", True)

        assert span._end_time is not None
        assert span._attributes["async"] is True

    async def test_trace_decorator_async(self):
        """Test trace decorator for async functions."""

        @trace(name="async-decorated")
        async def async_function(x: int) -> int:
            return x * 2

        result = await async_function(5)
        assert result == 10
