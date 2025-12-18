"""Structured logging for ai-infra.

Provides consistent, structured logging across all ai-infra components with:
- JSON-formatted logs for production
- Human-readable logs for development
- Request/response logging with sanitization
- Performance metrics logging

Usage:
    from ai_infra.logging import get_logger, configure_logging

    # Configure logging level
    configure_logging(level="DEBUG")

    # Get a logger
    log = get_logger("my_component")
    log.info("Processing request", extra={"request_id": "123"})

    # Use structured logger
    from ai_infra.logging import StructuredLogger
    logger = StructuredLogger("llm")
    logger.info("LLM call", provider="openai", model="gpt-4o", tokens=150)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, TypeVar

# Context for request tracking
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)

# Type var for decorators
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# JSON Formatter
# =============================================================================


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Outputs logs as JSON objects with consistent fields:
    - timestamp: ISO 8601 timestamp
    - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - logger: Logger name
    - message: Log message
    - ... extra fields from record
    """

    SKIP_FIELDS = {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Build base log entry
        log_entry: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info for errors
        if record.levelno >= logging.ERROR:
            log_entry["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add request/trace IDs from context
        if req_id := _request_id.get():
            log_entry["request_id"] = req_id
        if trace := _trace_id.get():
            log_entry["trace_id"] = trace

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in self.SKIP_FIELDS and not key.startswith("_"):
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable formatter for development.

    Outputs logs in a format like:
    2024-01-15 10:30:45 INFO  [ai_infra.llm] Processing request provider=openai model=gpt-4o
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Level with color
        level = record.levelname.ljust(5)
        if self.use_colors and record.levelname in self.COLORS:
            level = f"{self.COLORS[record.levelname]}{level}{self.RESET}"

        # Logger name (shortened)
        name = record.name
        if name.startswith("ai_infra."):
            name = name[9:]  # Remove prefix

        # Message
        message = record.getMessage()

        # Extra fields
        extras = []
        for key, value in record.__dict__.items():
            if key not in JSONFormatter.SKIP_FIELDS and not key.startswith("_"):
                if isinstance(value, str) and " " in value:
                    extras.append(f'{key}="{value}"')
                else:
                    extras.append(f"{key}={value}")

        extra_str = " ".join(extras)
        if extra_str:
            message = f"{message} {extra_str}"

        # Build line
        line = f"{timestamp} {level} [{name}] {message}"

        # Add exception
        if record.exc_info:
            line += f"\n{self.formatException(record.exc_info)}"

        return line


# =============================================================================
# Structured Logger
# =============================================================================


class StructuredLogger:
    """Logger with structured field support.

    Wraps a standard logger but provides convenient methods for
    adding structured fields to log messages.

    Example:
        log = StructuredLogger("llm")
        log.info("Request completed", provider="openai", tokens=150, latency_ms=234)
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(f"ai_infra.{name}")

    def _log(self, level: int, message: str, **fields: Any) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, message, extra=fields)

    def debug(self, message: str, **fields: Any) -> None:
        self._log(logging.DEBUG, message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._log(logging.INFO, message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._log(logging.WARNING, message, **fields)

    def error(self, message: str, exc_info: bool = False, **fields: Any) -> None:
        if exc_info:
            self._logger.error(message, exc_info=True, extra=fields)
        else:
            self._log(logging.ERROR, message, **fields)

    def critical(self, message: str, **fields: Any) -> None:
        self._log(logging.CRITICAL, message, **fields)

    def exception(self, message: str, **fields: Any) -> None:
        self._logger.exception(message, extra=fields)

    def is_enabled_for(self, level: int) -> bool:
        return self._logger.isEnabledFor(level)

    def child(self, suffix: str) -> StructuredLogger:
        """Create a child logger with a suffix."""
        return StructuredLogger(f"{self._logger.name}.{suffix}")


# =============================================================================
# Request Logger
# =============================================================================


@dataclass
class RequestLog:
    """Structured log entry for HTTP/API requests."""

    method: str
    url: str
    start_time: float = field(default_factory=time.time)
    status_code: int | None = None
    latency_ms: float | None = None
    request_size: int | None = None
    response_size: int | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def complete(
        self,
        status_code: int,
        response_size: int | None = None,
        error: str | None = None,
    ) -> RequestLog:
        """Mark request as complete and calculate latency."""
        self.status_code = status_code
        self.latency_ms = (time.time() - self.start_time) * 1000
        self.response_size = response_size
        self.error = error
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        d: dict[str, Any] = {
            "method": self.method,
            "url": self.url,
        }
        if self.status_code is not None:
            d["status_code"] = self.status_code
        if self.latency_ms is not None:
            d["latency_ms"] = round(self.latency_ms, 2)
        if self.request_size is not None:
            d["request_size"] = self.request_size
        if self.response_size is not None:
            d["response_size"] = self.response_size
        if self.error:
            d["error"] = self.error
        d.update(self.extra)
        return d


class RequestLogger:
    """Logger for HTTP/API requests with automatic sanitization."""

    SENSITIVE_HEADERS = {
        "authorization",
        "api-key",
        "x-api-key",
        "apikey",
        "cookie",
        "set-cookie",
        "x-auth-token",
    }
    SENSITIVE_PARAMS = {
        "api_key",
        "apikey",
        "key",
        "token",
        "secret",
        "password",
        "pwd",
        "auth",
    }

    def __init__(self, name: str = "http"):
        self._logger = StructuredLogger(name)

    def log_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: Any | None = None,
    ) -> RequestLog:
        """Log outgoing request and return a RequestLog for completion."""
        request_log = RequestLog(method=method, url=self._sanitize_url(url))

        if self._logger.is_enabled_for(logging.DEBUG):
            self._logger.debug(
                "Request started",
                method=method,
                url=request_log.url,
                headers=self._sanitize_headers(headers) if headers else None,
            )

        return request_log

    def log_response(self, request_log: RequestLog) -> None:
        """Log completed request."""
        level = logging.INFO
        if request_log.status_code and request_log.status_code >= 400:
            level = logging.WARNING
        if request_log.status_code and request_log.status_code >= 500:
            level = logging.ERROR

        self._logger._log(
            level,
            "Request completed",
            **request_log.to_dict(),
        )

    def _sanitize_url(self, url: str) -> str:
        """Remove sensitive query params from URL."""
        from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

        parsed = urlparse(url)
        if not parsed.query:
            return url

        params = parse_qs(parsed.query)
        for key in list(params.keys()):
            if key.lower() in self.SENSITIVE_PARAMS:
                params[key] = ["[REDACTED]"]

        sanitized = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                urlencode(params, doseq=True),
                parsed.fragment,
            )
        )
        return sanitized

    def _sanitize_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Redact sensitive headers."""
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in self.SENSITIVE_HEADERS:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized


# =============================================================================
# LLM Logger
# =============================================================================


@dataclass
class LLMCallLog:
    """Structured log entry for LLM API calls."""

    provider: str
    model: str
    start_time: float = field(default_factory=time.time)
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    cached: bool = False
    stream: bool = False
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def complete(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error: str | None = None,
    ) -> LLMCallLog:
        """Mark call as complete."""
        self.latency_ms = (time.time() - self.start_time) * 1000
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if input_tokens and output_tokens:
            self.total_tokens = input_tokens + output_tokens
        self.error = error
        return self

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
        }
        if self.input_tokens is not None:
            d["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            d["output_tokens"] = self.output_tokens
        if self.total_tokens is not None:
            d["total_tokens"] = self.total_tokens
        if self.latency_ms is not None:
            d["latency_ms"] = round(self.latency_ms, 2)
        if self.cached:
            d["cached"] = True
        if self.stream:
            d["stream"] = True
        if self.error:
            d["error"] = self.error
        d.update(self.extra)
        return d


class LLMLogger:
    """Logger for LLM API calls with token tracking."""

    def __init__(self, name: str = "llm"):
        self._logger = StructuredLogger(name)

    def log_call_start(
        self,
        provider: str,
        model: str,
        stream: bool = False,
        **extra: Any,
    ) -> LLMCallLog:
        """Log start of LLM call and return log entry."""
        call_log = LLMCallLog(
            provider=provider,
            model=model,
            stream=stream,
            extra=extra,
        )

        if self._logger.is_enabled_for(logging.DEBUG):
            self._logger.debug(
                "LLM call started",
                provider=provider,
                model=model,
                stream=stream,
            )

        return call_log

    def log_call_complete(self, call_log: LLMCallLog) -> None:
        """Log completed LLM call."""
        level = logging.INFO if not call_log.error else logging.ERROR
        self._logger._log(level, "LLM call completed", **call_log.to_dict())


# =============================================================================
# Configuration
# =============================================================================


def configure_logging(
    level: str | int = "INFO",
    format: str = "human",  # "human" or "json"
    log_requests: bool = False,
    log_responses: bool = False,
) -> None:
    """Configure ai-infra logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ("human" for dev, "json" for production)
        log_requests: Log outgoing HTTP requests
        log_responses: Log HTTP responses

    Example:
        # Development
        configure_logging(level="DEBUG", format="human")

        # Production
        configure_logging(level="INFO", format="json")
    """
    # Get root ai_infra logger
    logger = logging.getLogger("ai_infra")

    # Parse level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Add handler with formatter
    handler = logging.StreamHandler(sys.stdout)
    if format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(HumanFormatter())

    logger.addHandler(handler)

    # Configure sub-loggers
    if not log_requests:
        logging.getLogger("ai_infra.http").setLevel(logging.WARNING)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for a component.

    Args:
        name: Component name (will be prefixed with "ai_infra.")

    Returns:
        StructuredLogger instance

    Example:
        log = get_logger("mcp.client")
        log.info("Connected to server", server="my-server")
    """
    return StructuredLogger(name)


# =============================================================================
# Decorators
# =============================================================================


def log_function(
    logger: StructuredLogger | None = None,
    level: int = logging.DEBUG,
) -> Callable[[F], F]:
    """Decorator to log function entry/exit.

    Example:
        @log_function()
        def process_data(x: int) -> int:
            return x * 2
    """

    def decorator(fn: F) -> F:
        _logger = logger or get_logger(fn.__module__.split(".")[-1])

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger._log(level, f"Entering {fn.__name__}")
            start = time.time()
            try:
                result = fn(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                _logger._log(level, f"Exiting {fn.__name__}", latency_ms=round(elapsed, 2))
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                _logger.error(
                    f"Error in {fn.__name__}: {e}",
                    exc_info=True,
                    latency_ms=round(elapsed, 2),
                )
                raise

        return wrapper  # type: ignore

    return decorator


def log_async_function(
    logger: StructuredLogger | None = None,
    level: int = logging.DEBUG,
) -> Callable[[F], F]:
    """Decorator to log async function entry/exit."""

    def decorator(fn: F) -> F:
        _logger = logger or get_logger(fn.__module__.split(".")[-1])

        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger._log(level, f"Entering {fn.__name__}")
            start = time.time()
            try:
                result = await fn(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                _logger._log(level, f"Exiting {fn.__name__}", latency_ms=round(elapsed, 2))
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                _logger.error(
                    f"Error in {fn.__name__}: {e}",
                    exc_info=True,
                    latency_ms=round(elapsed, 2),
                )
                raise

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Auto-configure from environment
# =============================================================================


def _auto_configure() -> None:
    """Auto-configure logging from environment variables."""
    level = os.getenv("AI_INFRA_LOG_LEVEL", "INFO")
    format = os.getenv("AI_INFRA_LOG_FORMAT", "human")

    # Only configure if not already configured
    logger = logging.getLogger("ai_infra")
    if not logger.handlers:
        configure_logging(level=level, format=format)


# Auto-configure on import
_auto_configure()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "configure_logging",
    "get_logger",
    # Loggers
    "StructuredLogger",
    "RequestLogger",
    "LLMLogger",
    # Log entries
    "RequestLog",
    "LLMCallLog",
    # Formatters
    "JSONFormatter",
    "HumanFormatter",
    # Decorators
    "log_function",
    "log_async_function",
]
