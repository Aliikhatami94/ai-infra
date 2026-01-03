"""Tests for MCP tool execution.

Tests cover:
- Tool invocation via call_tool
- Tool argument passing
- Tool result handling
- Tool timeout handling
- Interceptor chain execution
- Callback integration during tool execution

Phase 1.1.4 of production readiness test plan.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai_infra.mcp.client import (
    MCPClient,
    MCPServerError,
    MCPToolCallRequest,
    MCPToolError,
    build_interceptor_chain,
)

# =============================================================================
# Call Tool Basic Tests
# =============================================================================


class TestCallToolBasic:
    """Tests for basic tool calling."""

    @pytest.mark.asyncio
    async def test_call_tool_unknown_server_raises(self):
        """call_tool with unknown server raises MCPServerError."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._discovered = True
        client._by_name = {"filesystem": MagicMock()}

        with pytest.raises(MCPServerError, match="Unknown server"):
            await client.call_tool("nonexistent", "read_file", {})

    @pytest.mark.asyncio
    async def test_call_tool_suggests_server_name(self):
        """call_tool suggests similar server names on typo."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._discovered = True
        client._by_name = {"filesystem": MagicMock()}

        with pytest.raises(MCPServerError, match="filesystem"):
            await client.call_tool("filesyste", "read_file", {})  # Typo


# =============================================================================
# Tool Timeout Tests
# =============================================================================


class TestToolTimeout:
    """Tests for tool timeout handling."""

    def test_tool_timeout_stored(self):
        """tool_timeout option is stored."""
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            tool_timeout=30.0,
        )
        assert client._tool_timeout == 30.0

    def test_default_tool_timeout(self):
        """Default tool timeout is 60 seconds."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        assert client._tool_timeout == 60.0


# =============================================================================
# Interceptor Tests
# =============================================================================


class TestInterceptors:
    """Tests for tool call interceptors."""

    def test_interceptor_stored(self):
        """Interceptors are stored on client."""

        class TestInterceptor:
            async def __call__(self, request, handler):
                return await handler(request)

        interceptor = TestInterceptor()
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            interceptors=[interceptor],
        )
        assert client._interceptors == [interceptor]

    def test_build_interceptor_chain_no_interceptors(self):
        """build_interceptor_chain with no interceptors returns handler."""

        async def base_handler(request):
            return "result"

        chain = build_interceptor_chain(base_handler, None)
        assert chain is base_handler

    @pytest.mark.asyncio
    async def test_interceptor_modifies_request(self):
        """Interceptor can modify the request."""

        class ModifyInterceptor:
            async def __call__(self, request: MCPToolCallRequest, handler):
                # Modify args
                request.args["injected"] = True
                return await handler(request)

        captured_request = None

        async def base_handler(request):
            nonlocal captured_request
            captured_request = request
            return MagicMock(content=[])

        chain = build_interceptor_chain(base_handler, [ModifyInterceptor()])
        request = MCPToolCallRequest(name="test", args={}, server_name="server")
        await chain(request)

        assert captured_request.args.get("injected") is True

    @pytest.mark.asyncio
    async def test_interceptor_modifies_response(self):
        """Interceptor can modify the response."""

        class ResponseInterceptor:
            async def __call__(self, request, handler):
                result = await handler(request)
                result.modified = True
                return result

        async def base_handler(request):
            return MagicMock(content=[], modified=False)

        chain = build_interceptor_chain(base_handler, [ResponseInterceptor()])
        request = MCPToolCallRequest(name="test", args={}, server_name="server")
        result = await chain(request)

        assert result.modified is True

    @pytest.mark.asyncio
    async def test_multiple_interceptors_chain(self):
        """Multiple interceptors execute in order."""
        execution_order = []

        class FirstInterceptor:
            async def __call__(self, request, handler):
                execution_order.append("first_before")
                result = await handler(request)
                execution_order.append("first_after")
                return result

        class SecondInterceptor:
            async def __call__(self, request, handler):
                execution_order.append("second_before")
                result = await handler(request)
                execution_order.append("second_after")
                return result

        async def base_handler(request):
            execution_order.append("base")
            return MagicMock(content=[])

        chain = build_interceptor_chain(base_handler, [FirstInterceptor(), SecondInterceptor()])
        request = MCPToolCallRequest(name="test", args={}, server_name="server")
        await chain(request)

        assert execution_order == [
            "first_before",
            "second_before",
            "base",
            "second_after",
            "first_after",
        ]


# =============================================================================
# MCPToolCallRequest Tests
# =============================================================================


class TestMCPToolCallRequest:
    """Tests for MCPToolCallRequest model."""

    def test_request_basic(self):
        """Basic request creation."""
        request = MCPToolCallRequest(
            name="read_file",
            args={"path": "/tmp/test.txt"},
            server_name="filesystem",
        )

        assert request.name == "read_file"
        assert request.args == {"path": "/tmp/test.txt"}
        assert request.server_name == "filesystem"

    def test_request_empty_args(self):
        """Request with empty args."""
        request = MCPToolCallRequest(
            name="list_tools",
            args={},
            server_name="server",
        )

        assert request.args == {}


# =============================================================================
# Tool Error Handling Tests
# =============================================================================


class TestToolErrorHandling:
    """Tests for tool error handling."""

    def test_mcp_tool_error_attributes(self):
        """MCPToolError stores tool and server names."""
        error = MCPToolError(
            "Tool failed",
            tool_name="read_file",
            server_name="filesystem",
        )

        assert error.tool_name == "read_file"
        assert error.server_name == "filesystem"
        assert "Tool failed" in str(error)

    def test_mcp_tool_error_with_details(self):
        """MCPToolError can include details."""
        error = MCPToolError(
            "Tool failed",
            tool_name="execute",
            server_name="shell",
            details={"exit_code": 1},
        )

        assert error.details["exit_code"] == 1


# =============================================================================
# Callback Integration Tests
# =============================================================================


class TestCallbackIntegration:
    """Tests for callback integration during tool execution."""

    def test_client_accepts_callbacks(self):
        """Client accepts callbacks parameter."""
        from ai_infra.callbacks import Callbacks

        callbacks = Callbacks()
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            callbacks=callbacks,
        )
        assert client._callbacks is not None

    def test_callbacks_normalized_to_manager(self):
        """Callbacks are normalized to CallbackManager."""
        from ai_infra.callbacks import CallbackManager, Callbacks

        callbacks = Callbacks()
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            callbacks=callbacks,
        )
        assert isinstance(client._callbacks, CallbackManager)


# =============================================================================
# Get Client Tests
# =============================================================================


class TestGetClient:
    """Tests for get_client method."""

    def test_get_client_unknown_server_raises(self):
        """get_client with unknown server raises ValueError."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._by_name = {"known": MagicMock()}

        with pytest.raises(ValueError, match="Unknown server"):
            client.get_client("unknown")

    def test_get_client_suggests_similar(self):
        """get_client suggests similar names."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._by_name = {"filesystem": MagicMock()}

        with pytest.raises(ValueError, match="filesystem"):
            client.get_client("filesyste")  # Typo
