"""Unit tests for MCP interceptors."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from ai_infra.mcp.client.interceptors import (
    CachingInterceptor,
    HeaderInjectionInterceptor,
    LoggingInterceptor,
    MCPToolCallRequest,
    RateLimitInterceptor,
    RetryInterceptor,
    ToolCallInterceptor,
    build_interceptor_chain,
    create_mock_result,
)

# ---------------------------------------------------------------------------
# MCPToolCallRequest tests
# ---------------------------------------------------------------------------


class TestMCPToolCallRequest:
    """Tests for MCPToolCallRequest dataclass."""

    def test_create_request(self):
        """Test creating a request."""
        req = MCPToolCallRequest(
            name="test_tool",
            args={"key": "value"},
            server_name="test-server",
        )
        assert req.name == "test_tool"
        assert req.args == {"key": "value"}
        assert req.server_name == "test-server"
        assert req.headers is None
        assert req.metadata == {}

    def test_override_creates_copy(self):
        """Test override creates a new request with modified fields."""
        req = MCPToolCallRequest(
            name="test_tool",
            args={"key": "value"},
            server_name="test-server",
        )
        new_req = req.override(name="new_tool", args={"new": "args"})

        # Original unchanged
        assert req.name == "test_tool"
        assert req.args == {"key": "value"}

        # New request has changes
        assert new_req.name == "new_tool"
        assert new_req.args == {"new": "args"}
        assert new_req.server_name == "test-server"  # Unchanged

    def test_cache_key_deterministic(self):
        """Test cache_key is deterministic."""
        req = MCPToolCallRequest(
            name="test_tool",
            args={"key": "value"},
            server_name="test-server",
        )
        key1 = req.cache_key()
        key2 = req.cache_key()
        assert key1 == key2

    def test_cache_key_different_for_different_requests(self):
        """Test cache_key differs for different requests."""
        req1 = MCPToolCallRequest(
            name="tool_a",
            args={"key": "value"},
            server_name="server",
        )
        req2 = MCPToolCallRequest(
            name="tool_b",
            args={"key": "value"},
            server_name="server",
        )
        assert req1.cache_key() != req2.cache_key()


# ---------------------------------------------------------------------------
# build_interceptor_chain tests
# ---------------------------------------------------------------------------


class TestBuildInterceptorChain:
    """Tests for build_interceptor_chain function."""

    @pytest.mark.asyncio
    async def test_no_interceptors_calls_base_handler(self):
        """Test with no interceptors - base handler called directly."""
        mock_result = create_mock_result("base result")
        base_handler = AsyncMock(return_value=mock_result)

        chain = build_interceptor_chain(base_handler, None)
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")
        result = await chain(req)

        base_handler.assert_called_once_with(req)
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_empty_list_calls_base_handler(self):
        """Test with empty interceptor list."""
        mock_result = create_mock_result("base result")
        base_handler = AsyncMock(return_value=mock_result)

        chain = build_interceptor_chain(base_handler, [])
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")
        result = await chain(req)

        base_handler.assert_called_once_with(req)
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_single_interceptor(self):
        """Test single interceptor wraps handler."""
        mock_result = create_mock_result("result")
        base_handler = AsyncMock(return_value=mock_result)
        interceptor_calls = []

        async def interceptor(request, handler):
            interceptor_calls.append(("before", request.name))
            result = await handler(request)
            interceptor_calls.append(("after", request.name))
            return result

        chain = build_interceptor_chain(base_handler, [interceptor])
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")
        await chain(req)

        assert interceptor_calls == [("before", "tool"), ("after", "tool")]
        base_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_interceptor_order(self):
        """Test interceptors are called in correct order (first = outermost)."""
        mock_result = create_mock_result("result")
        base_handler = AsyncMock(return_value=mock_result)
        calls = []

        async def interceptor_a(request, handler):
            calls.append("A-before")
            result = await handler(request)
            calls.append("A-after")
            return result

        async def interceptor_b(request, handler):
            calls.append("B-before")
            result = await handler(request)
            calls.append("B-after")
            return result

        chain = build_interceptor_chain(base_handler, [interceptor_a, interceptor_b])
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")
        await chain(req)

        # Onion pattern: A wraps B wraps handler
        assert calls == ["A-before", "B-before", "B-after", "A-after"]

    @pytest.mark.asyncio
    async def test_interceptor_can_modify_request(self):
        """Test interceptor can modify request before passing to handler."""
        mock_result = create_mock_result("result")
        base_handler = AsyncMock(return_value=mock_result)

        async def modify_interceptor(request, handler):
            new_req = request.override(name="modified_tool")
            return await handler(new_req)

        chain = build_interceptor_chain(base_handler, [modify_interceptor])
        req = MCPToolCallRequest(name="original", args={}, server_name="server")
        await chain(req)

        # Handler was called with modified request
        called_req = base_handler.call_args[0][0]
        assert called_req.name == "modified_tool"

    @pytest.mark.asyncio
    async def test_interceptor_can_short_circuit(self):
        """Test interceptor can return without calling handler."""
        base_handler = AsyncMock()
        short_circuit_result = create_mock_result("short-circuit")

        async def short_circuit_interceptor(request, handler):
            # Don't call handler - return directly
            return short_circuit_result

        chain = build_interceptor_chain(base_handler, [short_circuit_interceptor])
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")
        result = await chain(req)

        assert result == short_circuit_result
        base_handler.assert_not_called()


# ---------------------------------------------------------------------------
# CachingInterceptor tests
# ---------------------------------------------------------------------------


class TestCachingInterceptor:
    """Tests for CachingInterceptor."""

    @pytest.mark.asyncio
    async def test_caches_result(self):
        """Test result is cached on first call."""
        mock_result = create_mock_result("cached")
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            return mock_result

        cache = CachingInterceptor(ttl_seconds=60)
        req = MCPToolCallRequest(name="tool", args={"x": 1}, server_name="server")

        # First call
        await cache(req, handler)
        assert call_count == 1

        # Second call - should be cached
        await cache(req, handler)
        assert call_count == 1  # Still 1, not called again

    @pytest.mark.asyncio
    async def test_different_requests_not_cached(self):
        """Test different requests are not cached together."""
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            return create_mock_result(f"result-{call_count}")

        cache = CachingInterceptor(ttl_seconds=60)

        req1 = MCPToolCallRequest(name="tool", args={"x": 1}, server_name="server")
        req2 = MCPToolCallRequest(name="tool", args={"x": 2}, server_name="server")

        await cache(req1, handler)
        await cache(req2, handler)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test cache can be cleared."""
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            return create_mock_result("result")

        cache = CachingInterceptor(ttl_seconds=60)
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")

        await cache(req, handler)
        assert call_count == 1

        cache.clear()

        await cache(req, handler)
        assert call_count == 2  # Called again after clear

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """Test specific cache entry can be invalidated."""
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            return create_mock_result("result")

        cache = CachingInterceptor(ttl_seconds=60)
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")

        await cache(req, handler)
        assert call_count == 1

        removed = cache.invalidate(req)
        assert removed is True

        await cache(req, handler)
        assert call_count == 2


# ---------------------------------------------------------------------------
# RetryInterceptor tests
# ---------------------------------------------------------------------------


class TestRetryInterceptor:
    """Tests for RetryInterceptor."""

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        """Test no retry when handler succeeds."""
        mock_result = create_mock_result("success")
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            return mock_result

        retry = RetryInterceptor(max_attempts=3, delay_seconds=0.01)
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")

        result = await retry(req, handler)
        assert result == mock_result
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retries on failure."""
        mock_result = create_mock_result("success")
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return mock_result

        retry = RetryInterceptor(max_attempts=3, delay_seconds=0.01)
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")

        result = await retry(req, handler)
        assert result == mock_result
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        """Test raises last error after max attempts."""

        async def handler(request):
            raise RuntimeError("Always fails")

        retry = RetryInterceptor(max_attempts=3, delay_seconds=0.01)
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")

        with pytest.raises(RuntimeError, match="Always fails"):
            await retry(req, handler)

    @pytest.mark.asyncio
    async def test_retry_on_specific_exceptions(self):
        """Test retry only on specific exception types."""
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        retry = RetryInterceptor(
            max_attempts=3,
            delay_seconds=0.01,
            retry_on=(RuntimeError,),  # Only retry RuntimeError
        )
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")

        with pytest.raises(ValueError):
            await retry(req, handler)

        # Should not retry - only 1 call
        assert call_count == 1


# ---------------------------------------------------------------------------
# RateLimitInterceptor tests
# ---------------------------------------------------------------------------


class TestRateLimitInterceptor:
    """Tests for RateLimitInterceptor."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        """Test rate limit delays calls."""
        import time

        mock_result = create_mock_result("result")

        async def handler(request):
            return mock_result

        # 10 calls per second = 0.1s between calls
        rate_limit = RateLimitInterceptor(calls_per_second=10.0)
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")

        start = time.time()
        await rate_limit(req, handler)
        await rate_limit(req, handler)
        elapsed = time.time() - start

        # Should take at least 0.1s (1/10 second)
        assert elapsed >= 0.09


# ---------------------------------------------------------------------------
# LoggingInterceptor tests
# ---------------------------------------------------------------------------


class TestLoggingInterceptor:
    """Tests for LoggingInterceptor."""

    @pytest.mark.asyncio
    async def test_logs_call(self):
        """Test logs tool call."""
        mock_result = create_mock_result("result")
        logs = []

        async def handler(request):
            return mock_result

        log = LoggingInterceptor(log_fn=logs.append)
        req = MCPToolCallRequest(name="my_tool", args={}, server_name="my-server")

        await log(req, handler)

        assert len(logs) == 2
        assert "[MCP] Calling my-server/my_tool" in logs[0]
        assert "completed" in logs[1]

    @pytest.mark.asyncio
    async def test_logs_failure(self):
        """Test logs failure."""
        logs = []

        async def handler(request):
            raise RuntimeError("Failed!")

        log = LoggingInterceptor(log_fn=logs.append)
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")

        with pytest.raises(RuntimeError):
            await log(req, handler)

        assert len(logs) == 2
        assert "Calling" in logs[0]
        assert "failed" in logs[1]


# ---------------------------------------------------------------------------
# HeaderInjectionInterceptor tests
# ---------------------------------------------------------------------------


class TestHeaderInjectionInterceptor:
    """Tests for HeaderInjectionInterceptor."""

    @pytest.mark.asyncio
    async def test_injects_static_headers(self):
        """Test injects static headers."""
        mock_result = create_mock_result("result")
        captured_request = None

        async def handler(request):
            nonlocal captured_request
            captured_request = request
            return mock_result

        inject = HeaderInjectionInterceptor(headers={"X-API-Key": "secret", "X-Custom": "value"})
        req = MCPToolCallRequest(name="tool", args={}, server_name="server")

        await inject(req, handler)

        assert captured_request is not None
        assert captured_request.headers == {
            "X-API-Key": "secret",
            "X-Custom": "value",
        }

    @pytest.mark.asyncio
    async def test_injects_dynamic_headers(self):
        """Test injects dynamic headers from function."""
        mock_result = create_mock_result("result")
        captured_request = None

        async def handler(request):
            nonlocal captured_request
            captured_request = request
            return mock_result

        async def header_fn(request):
            return {"X-Tool": request.name}

        inject = HeaderInjectionInterceptor(header_fn=header_fn)
        req = MCPToolCallRequest(name="my_tool", args={}, server_name="server")

        await inject(req, handler)

        assert captured_request is not None
        assert captured_request.headers == {"X-Tool": "my_tool"}

    @pytest.mark.asyncio
    async def test_merges_existing_headers(self):
        """Test merges with existing headers."""
        mock_result = create_mock_result("result")
        captured_request = None

        async def handler(request):
            nonlocal captured_request
            captured_request = request
            return mock_result

        inject = HeaderInjectionInterceptor(headers={"X-New": "new"})
        req = MCPToolCallRequest(
            name="tool",
            args={},
            server_name="server",
            headers={"X-Existing": "existing"},
        )

        await inject(req, handler)

        assert captured_request is not None
        assert captured_request.headers == {
            "X-Existing": "existing",
            "X-New": "new",
        }


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


class TestToolCallInterceptorProtocol:
    """Tests for ToolCallInterceptor protocol conformance."""

    def test_caching_interceptor_is_protocol(self):
        """Test CachingInterceptor conforms to protocol."""
        assert isinstance(CachingInterceptor(), ToolCallInterceptor)

    def test_retry_interceptor_is_protocol(self):
        """Test RetryInterceptor conforms to protocol."""
        assert isinstance(RetryInterceptor(), ToolCallInterceptor)

    def test_rate_limit_interceptor_is_protocol(self):
        """Test RateLimitInterceptor conforms to protocol."""
        assert isinstance(RateLimitInterceptor(), ToolCallInterceptor)

    def test_logging_interceptor_is_protocol(self):
        """Test LoggingInterceptor conforms to protocol."""
        assert isinstance(LoggingInterceptor(), ToolCallInterceptor)

    def test_header_injection_interceptor_is_protocol(self):
        """Test HeaderInjectionInterceptor conforms to protocol."""
        assert isinstance(HeaderInjectionInterceptor(), ToolCallInterceptor)


# ---------------------------------------------------------------------------
# create_mock_result tests
# ---------------------------------------------------------------------------


class TestCreateMockResult:
    """Tests for create_mock_result helper."""

    def test_creates_result_with_text(self):
        """Test creates result with text content."""
        result = create_mock_result("hello world")
        assert isinstance(result, CallToolResult)
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "hello world"
        assert result.isError is False
