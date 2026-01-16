"""Tests for ai-infra logging module.

Phase 13.2 of EXECUTOR_4.md - Coverage improvement for logging module.
"""

from __future__ import annotations

import json
import logging

import pytest

from ai_infra.logging import (
    HumanFormatter,
    JSONFormatter,
    LLMCallLog,
    LLMLogger,
    RequestLog,
    RequestLogger,
    StructuredLogger,
    configure_logging,
    get_logger,
    log_async_function,
    log_function,
)

# =============================================================================
# JSONFormatter Tests
# =============================================================================


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_format_basic_message(self) -> None:
        """Test formatting a basic log message."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_format_with_extra(self) -> None:
        """Test formatting includes extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.request_id = "req-123"
        record.user_id = "user-456"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["request_id"] == "req-123"
        assert data["user_id"] == "user-456"

    def test_format_error_includes_location(self) -> None:
        """Test error logs include location info."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "location" in data
        assert data["location"]["line"] == 42

    def test_format_with_exception(self) -> None:
        """Test formatting includes exception info."""
        formatter = JSONFormatter()

        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert "ValueError" in data["exception"]


# =============================================================================
# HumanFormatter Tests
# =============================================================================


class TestHumanFormatter:
    """Tests for HumanFormatter."""

    def test_format_info_message(self) -> None:
        """Test formatting INFO message."""
        formatter = HumanFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "Info message" in output

    def test_format_error_message(self) -> None:
        """Test formatting ERROR message."""
        formatter = HumanFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "ERROR" in output
        assert "Error message" in output


# =============================================================================
# RequestLog Tests
# =============================================================================


class TestRequestLog:
    """Tests for RequestLog dataclass."""

    def test_create_request_log(self) -> None:
        """Test creating request log."""
        log = RequestLog(
            method="GET",
            url="/api/v1/users",
        )

        assert log.method == "GET"
        assert log.url == "/api/v1/users"
        assert log.status_code is None

    def test_complete_request_log(self) -> None:
        """Test completing a request log."""
        log = RequestLog(method="POST", url="/api/v1/users")
        log.complete(status_code=201, response_size=256)

        assert log.status_code == 201
        assert log.response_size == 256
        assert log.latency_ms is not None

    def test_request_log_to_dict(self) -> None:
        """Test converting request log to dictionary."""
        log = RequestLog(method="GET", url="/api/test")
        log.complete(status_code=200)

        d = log.to_dict()

        assert d["method"] == "GET"
        assert d["url"] == "/api/test"
        assert d["status_code"] == 200
        assert "latency_ms" in d


# =============================================================================
# LLMCallLog Tests
# =============================================================================


class TestLLMCallLog:
    """Tests for LLMCallLog dataclass."""

    def test_create_llm_call_log(self) -> None:
        """Test creating LLM call log."""
        log = LLMCallLog(
            provider="openai",
            model="gpt-4o",
        )

        assert log.provider == "openai"
        assert log.model == "gpt-4o"
        assert log.input_tokens is None

    def test_complete_llm_call_log(self) -> None:
        """Test completing an LLM call log."""
        log = LLMCallLog(provider="anthropic", model="claude-3")
        log.complete(input_tokens=100, output_tokens=50)

        assert log.input_tokens == 100
        assert log.output_tokens == 50
        assert log.total_tokens == 150
        assert log.latency_ms is not None

    def test_llm_call_log_to_dict(self) -> None:
        """Test converting LLM call log to dictionary."""
        log = LLMCallLog(provider="openai", model="gpt-4o", stream=True)
        log.complete(input_tokens=50, output_tokens=25)

        d = log.to_dict()

        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o"
        assert d["stream"] is True
        assert d["total_tokens"] == 75

    def test_llm_call_log_with_error(self) -> None:
        """Test LLM call log with error."""
        log = LLMCallLog(provider="openai", model="gpt-4o")
        log.complete(error="Rate limit exceeded")

        assert log.error == "Rate limit exceeded"
        assert "error" in log.to_dict()


# =============================================================================
# configure_logging Tests
# =============================================================================


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_debug_level(self) -> None:
        """Test configuring debug log level."""
        configure_logging(level="DEBUG")

        logger = logging.getLogger("ai_infra")
        assert logger.level == logging.DEBUG

    def test_configure_with_json_format(self) -> None:
        """Test configuring with JSON format."""
        configure_logging(format="json")

    def test_configure_with_human_format(self) -> None:
        """Test configuring with human format."""
        configure_logging(format="human")


# =============================================================================
# get_logger Tests
# =============================================================================


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger(self) -> None:
        """Test getting a logger."""
        logger = get_logger("test.component")

        assert logger is not None
        assert isinstance(logger, StructuredLogger)

    def test_get_logger_returns_structured_logger(self) -> None:
        """Test get_logger returns StructuredLogger."""
        logger = get_logger("ai_infra.llm")

        assert isinstance(logger, StructuredLogger)


# =============================================================================
# StructuredLogger Tests
# =============================================================================


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    def test_create_structured_logger(self) -> None:
        """Test creating structured logger."""
        logger = StructuredLogger("test")

        assert logger is not None

    def test_info_method(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test info logging."""
        logger = StructuredLogger("test")

        with caplog.at_level(logging.INFO):
            logger.info("Test message", key1="value1")

        assert "Test message" in caplog.text

    def test_error_method(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test error logging."""
        logger = StructuredLogger("test")

        with caplog.at_level(logging.ERROR):
            logger.error("Error occurred", error_code=500)

        assert "Error occurred" in caplog.text

    def test_warning_method(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning logging."""
        logger = StructuredLogger("test")

        with caplog.at_level(logging.WARNING):
            logger.warning("Warning message")

        assert "Warning message" in caplog.text

    def test_child_logger(self) -> None:
        """Test creating child logger."""
        parent = StructuredLogger("parent")
        child = parent.child("child")

        assert child is not None
        assert isinstance(child, StructuredLogger)


# =============================================================================
# log_function Tests
# =============================================================================


class TestLogFunction:
    """Tests for log_function decorator."""

    def test_log_function_decorator(self) -> None:
        """Test log_function with synchronous function."""

        @log_function()
        def my_function() -> str:
            return "result"

        result = my_function()

        assert result == "result"

    def test_log_function_preserves_return(self) -> None:
        """Test log_function preserves function return value."""

        @log_function()
        def add_numbers(a: int, b: int) -> int:
            return a + b

        result = add_numbers(2, 3)

        assert result == 5

    def test_log_function_with_exception(self) -> None:
        """Test log_function logs on exception."""

        @log_function()
        def failing_function() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError):
            failing_function()


# =============================================================================
# log_async_function Tests
# =============================================================================


class TestLogAsyncFunction:
    """Tests for log_async_function decorator."""

    @pytest.mark.asyncio
    async def test_log_async_function_decorator(self) -> None:
        """Test log_async_function with async function."""

        @log_async_function()
        async def my_async_function() -> str:
            return "async result"

        result = await my_async_function()

        assert result == "async result"

    @pytest.mark.asyncio
    async def test_log_async_function_preserves_return(self) -> None:
        """Test log_async_function preserves function return value."""

        @log_async_function()
        async def async_add(a: int, b: int) -> int:
            return a + b

        result = await async_add(5, 7)

        assert result == 12


# =============================================================================
# RequestLogger Tests
# =============================================================================


class TestRequestLogger:
    """Tests for RequestLogger."""

    def test_create_request_logger(self) -> None:
        """Test creating request logger."""
        logger = RequestLogger("test.requests")

        assert logger is not None

    def test_log_request(self) -> None:
        """Test logging request."""
        logger = RequestLogger("test")
        log = logger.log_request(method="GET", url="/api/test")

        assert log.method == "GET"
        assert log.url == "/api/test"

    def test_sanitize_url_with_api_key(self) -> None:
        """Test URL sanitization removes API key."""
        logger = RequestLogger("test")
        log = logger.log_request(method="GET", url="/api/test?api_key=secret123&name=test")

        assert "secret123" not in log.url
        assert "REDACTED" in log.url  # URL-encoded as %5BREDACTED%5D
        assert "name=test" in log.url


# =============================================================================
# LLMLogger Tests
# =============================================================================


class TestLLMLogger:
    """Tests for LLMLogger."""

    def test_create_llm_logger(self) -> None:
        """Test creating LLM logger."""
        logger = LLMLogger("test.llm")

        assert logger is not None

    def test_log_call_start(self) -> None:
        """Test logging LLM call start."""
        logger = LLMLogger("test")
        log = logger.log_call_start(provider="openai", model="gpt-4o")

        assert log.provider == "openai"
        assert log.model == "gpt-4o"
