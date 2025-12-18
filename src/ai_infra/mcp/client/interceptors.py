"""Interceptors for MCP tool call lifecycle.

This module provides a middleware pattern for intercepting tool calls,
enabling cross-cutting concerns like caching, retry, rate limiting, and
request/response modification.

Interceptors wrap tool execution in an "onion" pattern - the first
interceptor in the list is the outermost layer.

Example:
    ```python
    from ai_infra.mcp import MCPClient
    from ai_infra.mcp.client.interceptors import (
        CachingInterceptor,
        RetryInterceptor,
        LoggingInterceptor,
    )

    mcp = MCPClient(
        [config],
        interceptors=[
            LoggingInterceptor(),      # First: logs all calls
            RetryInterceptor(max_attempts=3),  # Retries on failure
            CachingInterceptor(ttl=60),  # Innermost: caches results
        ],
    )
    ```
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from mcp.types import CallToolResult, TextContent


@dataclass
class MCPToolCallRequest:
    """Tool execution request passed to interceptors.

    Interceptors can read and modify these fields before the tool
    executes. Use `override()` to create modified copies.

    Attributes:
        name: Tool name (modifiable).
        args: Tool arguments (modifiable).
        server_name: Server name (read-only context).
        headers: HTTP headers for streamable_http transport (modifiable).
        metadata: Custom metadata for inter-interceptor communication.
    """

    name: str
    args: dict[str, Any]
    server_name: str
    headers: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def override(self, **kwargs: Any) -> MCPToolCallRequest:
        """Create a copy with overridden fields.

        Args:
            **kwargs: Fields to override.

        Returns:
            New request with specified fields changed.

        Example:
            ```python
            new_req = request.override(args={"param": "new_value"})
            ```
        """
        return replace(self, **kwargs)

    def cache_key(self) -> str:
        """Generate a cache key for this request.

        Returns:
            Hash string suitable for caching lookups.
        """
        key_data = {
            "name": self.name,
            "args": self.args,
            "server": self.server_name,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]


# Type alias for the handler function
MCPToolCallHandler = Callable[[MCPToolCallRequest], Awaitable[CallToolResult]]


@runtime_checkable
class ToolCallInterceptor(Protocol):
    """Protocol for tool call interceptors.

    Interceptors wrap tool execution in an "onion" pattern:
    - First in list = outermost layer (sees request first, response last)
    - Can modify request before passing to next handler
    - Can modify response before returning
    - Can short-circuit by returning without calling handler
    - Can retry by calling handler multiple times

    Example:
        ```python
        class LoggingInterceptor:
            async def __call__(self, request, handler):
                print(f"Calling {request.name}")
                start = time.time()
                result = await handler(request)
                print(f"Completed in {time.time() - start:.2f}s")
                return result
        ```
    """

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: MCPToolCallHandler,
    ) -> CallToolResult:
        """Intercept a tool call.

        Args:
            request: The tool call request (can be modified).
            handler: The next handler in the chain (may be the actual
                tool call or another interceptor).

        Returns:
            The tool call result (can be modified or replaced).
        """
        ...


def build_interceptor_chain(
    base_handler: MCPToolCallHandler,
    interceptors: list[ToolCallInterceptor] | None,
) -> MCPToolCallHandler:
    """Build an onion-pattern handler chain from interceptors.

    Args:
        base_handler: The innermost handler (actual tool call).
        interceptors: List of interceptors to wrap around the handler.
            First interceptor = outermost layer.

    Returns:
        A handler that passes through all interceptors.

    Example:
        ```python
        # Order: logging → retry → cache → base_handler
        chain = build_interceptor_chain(
            base_handler,
            [LoggingInterceptor(), RetryInterceptor(), CachingInterceptor()],
        )
        result = await chain(request)
        ```
    """
    handler = base_handler
    if interceptors:
        # Reverse so first interceptor is outermost
        for interceptor in reversed(interceptors):
            current_handler = handler
            current_interceptor = interceptor

            async def wrapped(
                req: MCPToolCallRequest,
                *,
                _int: ToolCallInterceptor = current_interceptor,
                _h: MCPToolCallHandler = current_handler,
            ) -> CallToolResult:
                return await _int(req, _h)

            handler = wrapped
    return handler


# ---------------------------------------------------------------------------
# Built-in Interceptors
# ---------------------------------------------------------------------------


@dataclass
class CachingInterceptor:
    """Cache tool call results for a configurable TTL.

    Caches based on tool name, arguments, and server name.
    Useful for expensive operations with deterministic results.

    Attributes:
        ttl_seconds: Time-to-live for cache entries (default: 300).

    Example:
        ```python
        cache = CachingInterceptor(ttl_seconds=60)
        mcp = MCPClient([config], interceptors=[cache])
        ```
    """

    ttl_seconds: int = 300
    _cache: dict[str, tuple[CallToolResult, float]] = field(default_factory=dict, repr=False)

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: MCPToolCallHandler,
    ) -> CallToolResult:
        """Check cache or execute and cache result."""
        key = request.cache_key()
        now = time.time()

        # Check cache
        if key in self._cache:
            result, cached_at = self._cache[key]
            if now - cached_at < self.ttl_seconds:
                return result
            # Expired - remove
            del self._cache[key]

        # Execute and cache
        result = await handler(request)
        self._cache[key] = (result, now)
        return result

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def invalidate(self, request: MCPToolCallRequest) -> bool:
        """Invalidate a specific cache entry.

        Args:
            request: The request to invalidate.

        Returns:
            True if entry was found and removed, False otherwise.
        """
        key = request.cache_key()
        if key in self._cache:
            del self._cache[key]
            return True
        return False


@dataclass
class RetryInterceptor:
    """Retry failed tool calls with exponential backoff.

    Retries on any exception, with configurable delay and max attempts.

    Attributes:
        max_attempts: Maximum number of attempts (default: 3).
        delay_seconds: Initial delay between retries (default: 1.0).
        exponential: Whether to use exponential backoff (default: True).
        retry_on: Optional tuple of exception types to retry on.
            If None, retries on all exceptions.

    Example:
        ```python
        retry = RetryInterceptor(max_attempts=3, delay_seconds=0.5)
        mcp = MCPClient([config], interceptors=[retry])
        ```
    """

    max_attempts: int = 3
    delay_seconds: float = 1.0
    exponential: bool = True
    retry_on: tuple[type[Exception], ...] | None = None

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: MCPToolCallHandler,
    ) -> CallToolResult:
        """Execute with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.max_attempts):
            try:
                return await handler(request)
            except Exception as e:
                # Check if we should retry this exception
                if self.retry_on is not None and not isinstance(e, self.retry_on):
                    raise

                last_error = e
                if attempt < self.max_attempts - 1:
                    delay = self.delay_seconds
                    if self.exponential:
                        delay *= 2**attempt
                    await asyncio.sleep(delay)

        # All attempts failed
        assert last_error is not None
        raise last_error


@dataclass
class RateLimitInterceptor:
    """Rate limit tool calls to a maximum calls per second.

    Uses a simple token bucket algorithm with configurable rate.

    Attributes:
        calls_per_second: Maximum calls allowed per second (default: 10.0).

    Example:
        ```python
        rate_limit = RateLimitInterceptor(calls_per_second=5.0)
        mcp = MCPClient([config], interceptors=[rate_limit])
        ```
    """

    calls_per_second: float = 10.0
    _last_call_time: float = field(default=0.0, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: MCPToolCallHandler,
    ) -> CallToolResult:
        """Enforce rate limit before executing."""
        async with self._lock:
            now = time.time()
            min_interval = 1.0 / self.calls_per_second
            elapsed = now - self._last_call_time

            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            self._last_call_time = time.time()

        return await handler(request)


@dataclass
class LoggingInterceptor:
    """Log tool calls for debugging and observability.

    Logs tool name, arguments, execution time, and result status.

    Attributes:
        log_fn: Logging function to use (default: print).
        include_args: Whether to include arguments in logs (default: False).
        include_result: Whether to include result preview (default: False).

    Example:
        ```python
        import logging
        logger = logging.getLogger(__name__)

        log = LoggingInterceptor(log_fn=logger.info, include_args=True)
        mcp = MCPClient([config], interceptors=[log])
        ```
    """

    log_fn: Callable[[str], None] = field(default=print, repr=False)
    include_args: bool = False
    include_result: bool = False

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: MCPToolCallHandler,
    ) -> CallToolResult:
        """Log tool call execution."""
        args_str = ""
        if self.include_args:
            args_str = f" args={request.args}"

        self.log_fn(f"[MCP] Calling {request.server_name}/{request.name}{args_str}")
        start = time.time()

        try:
            result = await handler(request)
            elapsed = time.time() - start

            result_str = ""
            if self.include_result:
                # Preview first 100 chars of result
                content = getattr(result, "content", [])
                if content and hasattr(content[0], "text"):
                    preview = content[0].text[:100]
                    result_str = f" result={preview!r}..."

            self.log_fn(f"[MCP] {request.name} completed in {elapsed:.3f}s{result_str}")
            return result

        except Exception as e:
            elapsed = time.time() - start
            self.log_fn(f"[MCP] {request.name} failed after {elapsed:.3f}s: {e}")
            raise


@dataclass
class HeaderInjectionInterceptor:
    """Inject headers into tool call requests.

    Useful for adding authentication, tracing, or custom headers
    to streamable_http transport calls.

    Attributes:
        headers: Headers to inject into every request.
        header_fn: Optional async function to dynamically generate headers.

    Example:
        ```python
        # Static headers
        auth = HeaderInjectionInterceptor(headers={"X-API-Key": "secret"})

        # Dynamic headers
        async def get_headers(request):
            return {"X-Request-ID": str(uuid.uuid4())}

        trace = HeaderInjectionInterceptor(header_fn=get_headers)
        ```
    """

    headers: dict[str, str] = field(default_factory=dict)
    header_fn: Callable[[MCPToolCallRequest], Awaitable[dict[str, str]]] | None = None

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: MCPToolCallHandler,
    ) -> CallToolResult:
        """Inject headers into request."""
        new_headers = dict(request.headers or {})

        # Add static headers
        new_headers.update(self.headers)

        # Add dynamic headers
        if self.header_fn:
            dynamic = await self.header_fn(request)
            new_headers.update(dynamic)

        return await handler(request.override(headers=new_headers))


def create_mock_result(content: str) -> CallToolResult:
    """Create a mock CallToolResult for testing.

    Args:
        content: Text content for the result.

    Returns:
        A CallToolResult with the given text content.
    """
    return CallToolResult(
        content=[TextContent(type="text", text=content)],
        isError=False,
    )
