"""Tests for ai_infra.mcp.client module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.mcp.client import (
    MCPClient,
    MCPConnectionError,
    MCPError,
    McpServerConfig,
    MCPServerError,
    MCPTimeoutError,
    MCPToolError,
)

# ============================================================================
# Exception Tests
# ============================================================================


class TestExceptions:
    """Tests for MCP exception classes."""

    def test_mcp_error_base(self):
        """Test base MCPError."""
        error = MCPError("test error", details={"key": "value"})
        assert str(error) == "test error"
        assert error.message == "test error"
        assert error.details == {"key": "value"}

    def test_mcp_server_error(self):
        """Test MCPServerError with server_name."""
        error = MCPServerError("server failed", server_name="myserver")
        assert error.server_name == "myserver"
        assert "server failed" in str(error)

    def test_mcp_tool_error(self):
        """Test MCPToolError with tool and server names."""
        error = MCPToolError("tool failed", tool_name="read_file", server_name="fs")
        assert error.tool_name == "read_file"
        assert error.server_name == "fs"

    def test_mcp_timeout_error(self):
        """Test MCPTimeoutError with operation and timeout."""
        error = MCPTimeoutError("timed out", operation="discover", timeout=30.0)
        assert error.operation == "discover"
        assert error.timeout == 30.0

    def test_mcp_connection_error(self):
        """Test MCPConnectionError is subclass of MCPServerError."""
        error = MCPConnectionError("connection failed", server_name="myserver")
        assert isinstance(error, MCPServerError)
        assert error.server_name == "myserver"


# ============================================================================
# MCPClient Initialization Tests
# ============================================================================


class TestMCPClientInit:
    """Tests for MCPClient initialization."""

    def test_init_with_dict_config(self):
        """Test initialization with dict configs."""
        client = MCPClient(
            [
                {"transport": "stdio", "command": "npx", "args": ["-y", "server"]},
            ]
        )
        assert len(client._configs) == 1
        assert client._configs[0].transport == "stdio"

    def test_init_with_pydantic_config(self):
        """Test initialization with Pydantic configs."""
        config = McpServerConfig(transport="stdio", command="npx", args=[])
        client = MCPClient([config])
        assert len(client._configs) == 1
        assert client._configs[0] is config

    def test_init_with_connection_options(self):
        """Test initialization with connection management options."""
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            auto_reconnect=True,
            reconnect_delay=2.0,
            max_reconnect_attempts=10,
            tool_timeout=30.0,
            discover_timeout=15.0,
            pool_size=20,
        )
        assert client._auto_reconnect is True
        assert client._reconnect_delay == 2.0
        assert client._max_reconnect_attempts == 10
        assert client._tool_timeout == 30.0
        assert client._discover_timeout == 15.0
        assert client._pool_size == 20

    def test_init_with_invalid_config_type(self):
        """Test initialization with invalid config type raises."""
        with pytest.raises(TypeError, match="must be a list"):
            MCPClient({"command": "npx"})  # type: ignore

    def test_init_with_http_config(self):
        """Test initialization with HTTP configs."""
        client = MCPClient(
            [
                {"transport": "streamable_http", "url": "http://localhost:3000/mcp"},
                {"transport": "sse", "url": "http://localhost:3001/mcp"},
            ]
        )
        assert len(client._configs) == 2
        assert client._configs[0].transport == "streamable_http"
        assert client._configs[1].transport == "sse"


# ============================================================================
# MCPClient Config Validation Tests
# ============================================================================


class TestMCPClientConfigValidation:
    """Tests for config validation."""

    def test_stdio_requires_command(self):
        """Test that stdio transport requires command."""
        with pytest.raises(ValueError, match="requires 'command'"):
            MCPClient([{"transport": "stdio"}])

    def test_http_requires_url(self):
        """Test that HTTP transports require url."""
        with pytest.raises(ValueError, match="requires 'url'"):
            MCPClient([{"transport": "streamable_http"}])

        with pytest.raises(ValueError, match="requires 'url'"):
            MCPClient([{"transport": "sse"}])


# ============================================================================
# MCPClient Helper Method Tests
# ============================================================================


class TestMCPClientHelpers:
    """Tests for helper methods."""

    def test_cfg_identity_stdio(self):
        """Test config identity for stdio transport."""
        client = MCPClient(
            [{"transport": "stdio", "command": "npx", "args": ["-y", "server"]}]
        )
        identity = client._cfg_identity(client._configs[0])
        assert "stdio" in identity
        assert "npx" in identity
        assert "-y server" in identity

    def test_cfg_identity_http(self):
        """Test config identity for HTTP transport."""
        client = MCPClient(
            [{"transport": "streamable_http", "url": "http://localhost:3000"}]
        )
        identity = client._cfg_identity(client._configs[0])
        assert "streamable_http" in identity
        assert "localhost:3000" in identity

    def test_uniq_name_no_collision(self):
        """Test unique name generation without collisions."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        used = set()
        name = client._uniq_name("server", used)
        assert name == "server"

    def test_uniq_name_with_collision(self):
        """Test unique name generation with collisions."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        used = {"server", "server#2"}
        name = client._uniq_name("server", used)
        assert name == "server#3"

    def test_last_errors_empty(self):
        """Test last_errors returns empty list initially."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        assert client.last_errors() == []


# ============================================================================
# MCPClient Async Context Manager Tests
# ============================================================================


class TestMCPClientContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_calls_discover(self):
        """Test that entering context manager calls discover."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])

        with patch.object(client, "discover", new_callable=AsyncMock) as mock_discover:
            mock_discover.return_value = {}

            async with client as c:
                assert c is client
                mock_discover.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_calls_close(self):
        """Test that exiting context manager calls close."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])

        with patch.object(client, "discover", new_callable=AsyncMock):
            with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
                async with client:
                    pass
                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_resets_state(self):
        """Test that close resets client state."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._discovered = True
        client._by_name = {"test": MagicMock()}
        client._health_status = {"test": "healthy"}

        await client.close()

        assert client._discovered is False
        assert client._by_name == {}
        assert client._health_status == {}


# ============================================================================
# MCPClient Server Names Tests
# ============================================================================


class TestMCPClientServerNames:
    """Tests for server_names method."""

    def test_server_names_empty_before_discover(self):
        """Test server_names returns empty before discovery."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        assert client.server_names() == []

    def test_server_names_after_discover(self):
        """Test server_names returns names after discovery."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._by_name = {"server1": MagicMock(), "server2": MagicMock()}
        assert client.server_names() == ["server1", "server2"]


# ============================================================================
# MCPClient list_tools Tests
# ============================================================================


class TestMCPClientListTools:
    """Tests for list_tools method."""

    @pytest.mark.asyncio
    async def test_list_tools_unknown_server_raises(self):
        """Test list_tools with unknown server raises MCPServerError."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._discovered = True
        client._by_name = {"filesystem": MagicMock()}

        with pytest.raises(MCPServerError, match="Unknown server"):
            await client.list_tools(server="nonexistent")

    @pytest.mark.asyncio
    async def test_list_tools_suggests_similar_names(self):
        """Test list_tools suggests similar server names."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._discovered = True
        client._by_name = {"filesystem": MagicMock()}

        with pytest.raises(MCPServerError, match="filesystem"):
            await client.list_tools(server="filesyste")


# ============================================================================
# MCPClient call_tool Tests
# ============================================================================


class TestMCPClientCallTool:
    """Tests for call_tool method."""

    @pytest.mark.asyncio
    async def test_call_tool_unknown_server_raises(self):
        """Test call_tool with unknown server raises MCPServerError."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._discovered = True
        client._by_name = {"filesystem": MagicMock()}

        with pytest.raises(MCPServerError, match="Unknown server"):
            await client.call_tool("nonexistent", "read_file", {})


# ============================================================================
# MCPClient Timeout Tests
# ============================================================================


class TestMCPClientTimeouts:
    """Tests for timeout handling."""

    def test_timeout_config_stored(self):
        """Test that timeout configs are stored."""
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            tool_timeout=30.0,
            discover_timeout=10.0,
        )
        assert client._tool_timeout == 30.0
        assert client._discover_timeout == 10.0


# ============================================================================
# MCPClient Health Check Tests
# ============================================================================


class TestMCPClientHealthCheck:
    """Tests for health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self):
        """Test health_check returns status dict."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])

        # Mock the session opening to succeed
        mock_session = MagicMock()
        mock_session.mcp_server_info = {"name": "test-server"}

        # Create an async context manager mock
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_ctx():
            yield mock_session

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.return_value = mock_ctx()
            result = await client.health_check()

        assert isinstance(result, dict)
        assert "test-server" in result
        assert result["test-server"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_server(self):
        """Test health_check marks failed server as unhealthy."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.side_effect = Exception("Connection failed")
            result = await client.health_check()

        assert isinstance(result, dict)
        assert any(status == "unhealthy" for status in result.values())
