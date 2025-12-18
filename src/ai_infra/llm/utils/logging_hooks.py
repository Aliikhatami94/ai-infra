"""Logging hooks for LLM request/response observability.

Provides configurable callbacks for logging LLM interactions:
- on_request: Called before invoking the model
- on_response: Called after successful model response
- on_error: Called when an error occurs

Example:
    ```python
    from ai_infra.llm import LLM
    import logging

    logger = logging.getLogger(__name__)

    llm = LLM()
    llm.set_logging_hooks(
        on_request=lambda req: logger.info("Request: %s", req),
        on_response=lambda resp: logger.info("Response: %s", resp),
        on_error=lambda err: logger.error("Error: %s", err),
    )
    ```
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ErrorContext", "LoggingHooks", "RequestContext", "ResponseContext"]

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    """Context passed to on_request hook."""

    user_msg: str
    system: str | None
    provider: str
    model_name: str
    model_kwargs: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_msg": self.user_msg[:200] + "..." if len(self.user_msg) > 200 else self.user_msg,
            "system": (
                self.system[:100] + "..." if self.system and len(self.system) > 100 else self.system
            ),
            "provider": self.provider,
            "model_name": self.model_name,
            "model_kwargs": {k: v for k, v in self.model_kwargs.items() if k != "api_key"},
            "timestamp": self.timestamp,
        }


@dataclass
class ResponseContext:
    """Context passed to on_response hook."""

    request: RequestContext
    response: Any
    duration_ms: float
    token_usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        content = getattr(self.response, "content", str(self.response))
        if isinstance(content, str) and len(content) > 200:
            content = content[:200] + "..."
        return {
            "provider": self.request.provider,
            "model_name": self.request.model_name,
            "content_preview": content,
            "duration_ms": round(self.duration_ms, 2),
            "token_usage": self.token_usage,
        }


@dataclass
class ErrorContext:
    """Context passed to on_error hook."""

    request: RequestContext
    error: BaseException
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.request.provider,
            "model_name": self.request.model_name,
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "duration_ms": round(self.duration_ms, 2),
        }


class LoggingHooks:
    """Configuration for request/response logging hooks.

    Hooks are optional callbacks that receive context about LLM calls:
    - on_request(ctx: RequestContext): Called before model invocation
    - on_response(ctx: ResponseContext): Called after successful response
    - on_error(ctx: ErrorContext): Called when an error occurs

    Both sync and async versions are supported. If only sync is provided,
    it will be run in a thread for async calls.
    """

    def __init__(
        self,
        *,
        on_request: Callable[[RequestContext], None] | None = None,
        on_response: Callable[[ResponseContext], None] | None = None,
        on_error: Callable[[ErrorContext], None] | None = None,
        on_request_async: Callable[[RequestContext], Any] | None = None,
        on_response_async: Callable[[ResponseContext], Any] | None = None,
        on_error_async: Callable[[ErrorContext], Any] | None = None,
    ):
        self.on_request = on_request
        self.on_response = on_response
        self.on_error = on_error
        self.on_request_async = on_request_async
        self.on_response_async = on_response_async
        self.on_error_async = on_error_async

    def set(
        self,
        *,
        on_request: Callable[[RequestContext], None] | None = None,
        on_response: Callable[[ResponseContext], None] | None = None,
        on_error: Callable[[ErrorContext], None] | None = None,
        on_request_async: Callable[[RequestContext], Any] | None = None,
        on_response_async: Callable[[ResponseContext], Any] | None = None,
        on_error_async: Callable[[ErrorContext], Any] | None = None,
    ) -> LoggingHooks:
        """Update hooks. Only non-None values are updated."""
        if on_request is not None:
            self.on_request = on_request
        if on_response is not None:
            self.on_response = on_response
        if on_error is not None:
            self.on_error = on_error
        if on_request_async is not None:
            self.on_request_async = on_request_async
        if on_response_async is not None:
            self.on_response_async = on_response_async
        if on_error_async is not None:
            self.on_error_async = on_error_async
        return self

    def clear(self) -> LoggingHooks:
        """Clear all hooks."""
        self.on_request = None
        self.on_response = None
        self.on_error = None
        self.on_request_async = None
        self.on_response_async = None
        self.on_error_async = None
        return self

    @property
    def enabled(self) -> bool:
        """Check if any hooks are configured."""
        return any(
            [
                self.on_request,
                self.on_response,
                self.on_error,
                self.on_request_async,
                self.on_response_async,
                self.on_error_async,
            ]
        )

    # Sync call helpers
    def call_request_sync(self, ctx: RequestContext) -> None:
        """Call on_request hook synchronously."""
        if self.on_request:
            try:
                self.on_request(ctx)
            except Exception as e:
                logger.warning("on_request hook failed: %s", e)

    def call_response_sync(self, ctx: ResponseContext) -> None:
        """Call on_response hook synchronously."""
        if self.on_response:
            try:
                self.on_response(ctx)
            except Exception as e:
                logger.warning("on_response hook failed: %s", e)

    def call_error_sync(self, ctx: ErrorContext) -> None:
        """Call on_error hook synchronously."""
        if self.on_error:
            try:
                self.on_error(ctx)
            except Exception as e:
                logger.warning("on_error hook failed: %s", e)

    # Async call helpers
    async def call_request_async(self, ctx: RequestContext) -> None:
        """Call on_request hook asynchronously."""
        if self.on_request_async:
            try:
                await self.on_request_async(ctx)
            except Exception as e:
                logger.warning("on_request_async hook failed: %s", e)
        elif self.on_request:
            try:
                await asyncio.to_thread(self.on_request, ctx)
            except Exception as e:
                logger.warning("on_request hook failed: %s", e)

    async def call_response_async(self, ctx: ResponseContext) -> None:
        """Call on_response hook asynchronously."""
        if self.on_response_async:
            try:
                await self.on_response_async(ctx)
            except Exception as e:
                logger.warning("on_response_async hook failed: %s", e)
        elif self.on_response:
            try:
                await asyncio.to_thread(self.on_response, ctx)
            except Exception as e:
                logger.warning("on_response hook failed: %s", e)

    async def call_error_async(self, ctx: ErrorContext) -> None:
        """Call on_error hook asynchronously."""
        if self.on_error_async:
            try:
                await self.on_error_async(ctx)
            except Exception as e:
                logger.warning("on_error_async hook failed: %s", e)
        elif self.on_error:
            try:
                await asyncio.to_thread(self.on_error, ctx)
            except Exception as e:
                logger.warning("on_error hook failed: %s", e)
