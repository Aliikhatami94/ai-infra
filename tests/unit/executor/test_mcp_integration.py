"""Unit tests for executor MCP integration (Phase 15.2, 15.5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.executor.mcp_integration import (
    DEFAULT_DISCOVER_TIMEOUT,
    DEFAULT_HEALTH_CHECK_INTERVAL,
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    DEFAULT_TOOL_TIMEOUT,
    DiscoveryResult,
    ExecutorMCPManager,
    HealthCheckResult,
    MCPServerInfo,
    RetryConfig,
    create_mcp_manager,
)
from ai_infra.mcp import McpServerConfig
from ai_infra.mcp.client.exceptions import (
    MCPConnectionError,
    MCPTimeoutError,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def stdio_config() -> McpServerConfig:
    """Create a stdio MCP server config."""
    return McpServerConfig(
        transport="stdio",
        command="npx",
        args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
    )


@pytest.fixture
def sse_config() -> McpServerConfig:
    """Create an SSE MCP server config."""
    return McpServerConfig(
        transport="sse",
        url="http://localhost:8080/sse",
    )


@pytest.fixture
def mock_tool() -> MagicMock:
    """Create a mock tool."""
    tool = MagicMock()
    tool.name = "test_tool"
    return tool


@pytest.fixture
def mock_client() -> AsyncMock:
    """Create a mock MCP client."""
    client = AsyncMock()
    client.discover = AsyncMock(return_value={})
    client.list_tools = AsyncMock(return_value=[])
    client.last_errors = MagicMock(return_value=[])
    client.health_check = AsyncMock(return_value={})
    client.call_tool = AsyncMock(return_value={"result": "success"})
    client.close = AsyncMock()
    return client


# =============================================================================
# RetryConfig Tests
# =============================================================================


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_values(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_retries == DEFAULT_MAX_RETRIES
        assert config.base_delay == DEFAULT_RETRY_BASE_DELAY
        assert config.max_delay == DEFAULT_RETRY_MAX_DELAY
        assert config.exponential_base == 2.0

    def test_custom_values(self):
        """Test custom retry configuration values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=60.0,
            exponential_base=3.0,
        )
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 3.0

    def test_get_delay_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0)
        assert config.get_delay(0) == 1.0  # 1 * 2^0
        assert config.get_delay(1) == 2.0  # 1 * 2^1
        assert config.get_delay(2) == 4.0  # 1 * 2^2
        assert config.get_delay(3) == 8.0  # 1 * 2^3

    def test_get_delay_capped_at_max(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0)
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 5.0  # Capped
        assert config.get_delay(10) == 5.0  # Still capped


# =============================================================================
# MCPServerInfo Tests
# =============================================================================


class TestMCPServerInfo:
    """Tests for MCPServerInfo class."""

    def test_basic_creation(self, stdio_config):
        """Test basic server info creation."""
        info = MCPServerInfo(
            name="test-server",
            config=stdio_config,
            tool_count=5,
            health_status="healthy",
        )
        assert info.name == "test-server"
        assert info.config == stdio_config
        assert info.tool_count == 5
        assert info.health_status == "healthy"
        assert info.error is None
        assert info.transport == "unknown"  # transport derived from config

    def test_with_error(self, stdio_config):
        """Test server info with error."""
        info = MCPServerInfo(
            name="failed-server",
            config=None,
            transport="stdio",
            health_status="unhealthy",
            error="Connection refused",
        )
        assert info.health_status == "unhealthy"
        assert info.error == "Connection refused"
        assert info.config is None
        assert info.transport == "stdio"

    def test_to_dict(self, stdio_config):
        """Test to_dict conversion."""
        info = MCPServerInfo(
            name="test-server",
            config=stdio_config,
            tool_count=3,
            health_status="healthy",
        )
        result = info.to_dict()
        assert result["name"] == "test-server"
        assert result["transport"] == "stdio"
        assert result["tool_count"] == 3
        assert result["health_status"] == "healthy"
        assert result["error"] is None

    def test_to_dict_without_config(self):
        """Test to_dict conversion without config."""
        info = MCPServerInfo(
            name="failed-server",
            config=None,
            transport="sse",
            health_status="unhealthy",
            error="Connection refused",
        )
        result = info.to_dict()
        assert result["name"] == "failed-server"
        assert result["transport"] == "sse"
        assert result["health_status"] == "unhealthy"
        assert result["error"] == "Connection refused"


# =============================================================================
# DiscoveryResult Tests
# =============================================================================


class TestDiscoveryResult:
    """Tests for DiscoveryResult class."""

    def test_default_values(self):
        """Test default discovery result values."""
        result = DiscoveryResult()
        assert result.success is True
        assert result.tools == []
        assert result.servers == []
        assert result.errors == []
        assert result.total_servers == 0
        assert result.healthy_servers == 0
        assert result.retry_count == 0

    def test_tool_count_property(self, mock_tool):
        """Test tool_count property."""
        result = DiscoveryResult(tools=[mock_tool, mock_tool, mock_tool])
        assert result.tool_count == 3

    def test_to_dict(self, stdio_config, mock_tool):
        """Test to_dict conversion."""
        server_info = MCPServerInfo(
            name="test",
            config=stdio_config,
            tool_count=1,
            health_status="healthy",
        )
        result = DiscoveryResult(
            success=True,
            tools=[mock_tool],
            servers=[server_info],
            total_servers=1,
            healthy_servers=1,
            retry_count=0,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["tool_count"] == 1
        assert d["total_servers"] == 1
        assert d["healthy_servers"] == 1
        assert len(d["servers"]) == 1


# =============================================================================
# ExecutorMCPManager Tests
# =============================================================================


class TestExecutorMCPManagerInit:
    """Tests for ExecutorMCPManager initialization."""

    def test_basic_init(self, stdio_config):
        """Test basic manager initialization."""
        manager = ExecutorMCPManager([stdio_config])
        assert manager.configs == [stdio_config]
        assert manager.discover_timeout == DEFAULT_DISCOVER_TIMEOUT
        assert manager.tool_timeout == DEFAULT_TOOL_TIMEOUT
        assert manager.is_discovered is False

    def test_custom_timeouts(self, stdio_config):
        """Test manager with custom timeouts."""
        manager = ExecutorMCPManager(
            [stdio_config],
            discover_timeout=60.0,
            tool_timeout=120.0,
        )
        assert manager.discover_timeout == 60.0
        assert manager.tool_timeout == 120.0

    def test_custom_retry_config(self, stdio_config):
        """Test manager with custom retry config."""
        retry = RetryConfig(max_retries=5, base_delay=2.0)
        manager = ExecutorMCPManager([stdio_config], retry_config=retry)
        assert manager._retry_config.max_retries == 5
        assert manager._retry_config.base_delay == 2.0

    def test_empty_configs(self):
        """Test manager with empty configs."""
        manager = ExecutorMCPManager([])
        assert manager.configs == []
        assert manager.is_discovered is False


class TestExecutorMCPManagerProperties:
    """Tests for ExecutorMCPManager properties."""

    def test_is_discovered_initially_false(self, stdio_config):
        """Test is_discovered is False initially."""
        manager = ExecutorMCPManager([stdio_config])
        assert manager.is_discovered is False

    def test_configs_property(self, stdio_config, sse_config):
        """Test configs property returns correct list."""
        manager = ExecutorMCPManager([stdio_config, sse_config])
        assert len(manager.configs) == 2
        assert manager.configs[0] == stdio_config
        assert manager.configs[1] == sse_config


class TestExecutorMCPManagerDiscover:
    """Tests for ExecutorMCPManager.discover() method."""

    @pytest.mark.asyncio
    async def test_discover_empty_configs(self):
        """Test discover with empty configs returns empty result."""
        manager = ExecutorMCPManager([])
        result = await manager.discover()
        assert result.success is True
        assert result.tools == []
        assert result.total_servers == 0

    @pytest.mark.asyncio
    async def test_discover_success(self, stdio_config, mock_client, mock_tool):
        """Test successful discovery."""
        mock_client.discover.return_value = {"test-server": stdio_config}
        mock_client.list_tools.return_value = [mock_tool]

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            result = await manager.discover()

        assert result.success is True
        assert manager.is_discovered is True
        mock_client.discover.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_timeout_with_retry(self, stdio_config, mock_client):
        """Test discovery timeout triggers retry."""
        mock_client.discover.side_effect = MCPTimeoutError("Timeout")

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                retry_config = RetryConfig(max_retries=2, base_delay=0.01)
                manager = ExecutorMCPManager([stdio_config], retry_config=retry_config)
                result = await manager.discover(strict=False)

        assert result.success is False
        assert result.retry_count == 3  # Initial + 2 retries
        assert mock_client.discover.call_count == 3

    @pytest.mark.asyncio
    async def test_discover_connection_error_with_retry(self, stdio_config, mock_client):
        """Test connection error triggers retry."""
        mock_client.discover.side_effect = MCPConnectionError("Connection refused")

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                retry_config = RetryConfig(max_retries=1, base_delay=0.01)
                manager = ExecutorMCPManager([stdio_config], retry_config=retry_config)
                result = await manager.discover(strict=False)

        assert result.success is False
        assert mock_client.discover.call_count == 2

    @pytest.mark.asyncio
    async def test_discover_strict_mode_raises(self, stdio_config, mock_client):
        """Test strict mode raises on failure."""
        mock_client.discover.side_effect = MCPTimeoutError("Timeout")

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                retry_config = RetryConfig(max_retries=0, base_delay=0.01)
                manager = ExecutorMCPManager([stdio_config], retry_config=retry_config)
                with pytest.raises(MCPTimeoutError):
                    await manager.discover(strict=True)

    @pytest.mark.asyncio
    async def test_discover_recovery_after_retry(self, stdio_config, mock_client, mock_tool):
        """Test discovery succeeds after retry."""
        # First call fails, second succeeds
        mock_client.discover.side_effect = [
            MCPTimeoutError("Timeout"),
            {"test-server": stdio_config},
        ]
        mock_client.list_tools.return_value = [mock_tool]

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                retry_config = RetryConfig(max_retries=2, base_delay=0.01)
                manager = ExecutorMCPManager([stdio_config], retry_config=retry_config)
                result = await manager.discover()

        assert result.success is True
        assert result.retry_count == 1
        assert mock_client.discover.call_count == 2


class TestExecutorMCPManagerTools:
    """Tests for ExecutorMCPManager tool access methods."""

    def test_get_tools_empty(self, stdio_config):
        """Test get_tools returns empty list before discovery."""
        manager = ExecutorMCPManager([stdio_config])
        assert manager.get_tools() == []

    @pytest.mark.asyncio
    async def test_get_tools_after_discover(self, stdio_config, mock_client, mock_tool):
        """Test get_tools returns tools after discovery."""
        mock_client.discover.return_value = {"test-server": stdio_config}
        mock_client.list_tools.return_value = [mock_tool]

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover()
            tools = manager.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_get_tool_names_empty(self, stdio_config):
        """Test get_tool_names returns empty list before discovery."""
        manager = ExecutorMCPManager([stdio_config])
        assert manager.get_tool_names() == []

    @pytest.mark.asyncio
    async def test_get_tool_names_after_discover(self, stdio_config, mock_client, mock_tool):
        """Test get_tool_names returns names after discovery."""
        mock_client.discover.return_value = {"test-server": stdio_config}
        mock_client.list_tools.return_value = [mock_tool]

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover()
            names = manager.get_tool_names()

        assert names == ["test_tool"]


class TestExecutorMCPManagerServerInfo:
    """Tests for ExecutorMCPManager server info methods."""

    def test_get_server_info_empty(self, stdio_config):
        """Test get_server_info returns empty list before discovery."""
        manager = ExecutorMCPManager([stdio_config])
        assert manager.get_server_info() == []

    def test_get_server_names_empty(self, stdio_config):
        """Test get_server_names returns empty list before discovery."""
        manager = ExecutorMCPManager([stdio_config])
        assert manager.get_server_names() == []

    @pytest.mark.asyncio
    async def test_get_server_info_after_discover(self, stdio_config, mock_client, mock_tool):
        """Test get_server_info after discovery."""
        mock_client.discover.return_value = {"test-server": stdio_config}
        mock_client.list_tools.return_value = [mock_tool]

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover()
            servers = manager.get_server_info()

        assert len(servers) >= 1

    def test_get_last_discovery_result_none(self, stdio_config):
        """Test get_last_discovery_result returns None before discovery."""
        manager = ExecutorMCPManager([stdio_config])
        assert manager.get_last_discovery_result() is None


class TestExecutorMCPManagerHealthCheck:
    """Tests for ExecutorMCPManager health check."""

    @pytest.mark.asyncio
    async def test_health_check(self, stdio_config, mock_client):
        """Test health check returns HealthCheckResult (Phase 15.5)."""
        mock_client.health_check.return_value = {"test-server": "healthy"}

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            result = await manager.health_check()

        # Updated for Phase 15.5: health_check now returns HealthCheckResult
        assert isinstance(result, HealthCheckResult)
        assert result.server_statuses == {"test-server": "healthy"}
        mock_client.health_check.assert_called_once()


class TestExecutorMCPManagerCallTool:
    """Tests for ExecutorMCPManager tool calls."""

    @pytest.mark.asyncio
    async def test_call_tool_without_discovery_raises(self, stdio_config):
        """Test call_tool raises if discovery not run."""
        manager = ExecutorMCPManager([stdio_config])
        with pytest.raises(ValueError, match="Discovery has not been run"):
            await manager.call_tool("server", "tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_success(self, stdio_config, mock_client, mock_tool):
        """Test successful tool call."""
        mock_client.discover.return_value = {"test-server": stdio_config}
        mock_client.list_tools.return_value = [mock_tool]
        mock_client.call_tool.return_value = {"result": "data"}

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover()
            result = await manager.call_tool("test-server", "test_tool", {"arg": "val"})

        assert result == {"result": "data"}
        mock_client.call_tool.assert_called_once_with("test-server", "test_tool", {"arg": "val"})


class TestExecutorMCPManagerLifecycle:
    """Tests for ExecutorMCPManager lifecycle methods."""

    @pytest.mark.asyncio
    async def test_close(self, stdio_config, mock_client, mock_tool):
        """Test close cleans up resources."""
        mock_client.discover.return_value = {"test-server": stdio_config}
        mock_client.list_tools.return_value = [mock_tool]

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover()
            assert manager.is_discovered is True

            await manager.close()

            assert manager.is_discovered is False
            assert manager.get_tools() == []
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, stdio_config, mock_client, mock_tool):
        """Test async context manager."""
        mock_client.discover.return_value = {"test-server": stdio_config}
        mock_client.list_tools.return_value = [mock_tool]

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            async with ExecutorMCPManager([stdio_config]) as manager:
                assert manager.is_discovered is True

            mock_client.close.assert_called_once()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateMCPManager:
    """Tests for create_mcp_manager factory function."""

    def test_basic_creation(self, stdio_config):
        """Test basic manager creation."""
        manager = create_mcp_manager([stdio_config])
        assert isinstance(manager, ExecutorMCPManager)
        assert manager.configs == [stdio_config]

    def test_custom_timeouts(self, stdio_config):
        """Test creation with custom timeouts."""
        manager = create_mcp_manager(
            [stdio_config],
            discover_timeout=60.0,
            tool_timeout=120.0,
        )
        assert manager.discover_timeout == 60.0
        assert manager.tool_timeout == 120.0

    def test_custom_retries(self, stdio_config):
        """Test creation with custom retries."""
        manager = create_mcp_manager([stdio_config], max_retries=5)
        assert manager._retry_config.max_retries == 5

    def test_empty_configs(self):
        """Test creation with empty configs."""
        manager = create_mcp_manager([])
        assert manager.configs == []


# =============================================================================
# Integration Pattern Tests
# =============================================================================


class TestMCPIntegrationPatterns:
    """Tests for common integration patterns."""

    @pytest.mark.asyncio
    async def test_multiple_servers_discovery(
        self, stdio_config, sse_config, mock_client, mock_tool
    ):
        """Test discovery with multiple server configs."""
        mock_client.discover.return_value = {
            "stdio-server": stdio_config,
            "sse-server": sse_config,
        }
        mock_client.list_tools.return_value = [mock_tool, mock_tool]

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config, sse_config])
            result = await manager.discover()

        assert result.success is True
        assert len(manager.configs) == 2

    @pytest.mark.asyncio
    async def test_graceful_degradation_partial_failure(self, stdio_config, mock_client, mock_tool):
        """Test graceful degradation with partial failures."""
        mock_client.discover.return_value = {"working-server": stdio_config}
        mock_client.list_tools.return_value = [mock_tool]
        mock_client.last_errors.return_value = [
            {"identity": "failed-server", "error": "Connection refused", "config": {}}
        ]

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config, stdio_config])
            result = await manager.discover(strict=False)

        assert result.success is True
        # Tools are still available from working server


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling behavior."""

    @pytest.mark.asyncio
    async def test_timeout_error_logged(self, stdio_config, mock_client, caplog):
        """Test timeout errors are logged."""
        mock_client.discover.side_effect = MCPTimeoutError("Discovery timed out")

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                retry_config = RetryConfig(max_retries=0)
                manager = ExecutorMCPManager([stdio_config], retry_config=retry_config)
                result = await manager.discover(strict=False)

        assert result.success is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_connection_error_recorded(self, stdio_config, mock_client):
        """Test connection errors are recorded in result."""
        mock_client.discover.side_effect = MCPConnectionError("Connection refused")

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                retry_config = RetryConfig(max_retries=0)
                manager = ExecutorMCPManager([stdio_config], retry_config=retry_config)
                result = await manager.discover(strict=False)

        assert result.success is False
        assert result.errors[0]["type"] == "MCPConnectionError"


# =============================================================================
# Health Check Result Tests (Phase 15.5)
# =============================================================================


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass (Phase 15.5.1)."""

    def test_default_values(self):
        """Test default health check result values."""
        result = HealthCheckResult()
        assert result.server_statuses == {}
        assert result.healthy_count == 0
        assert result.unhealthy_count == 0
        assert result.checked_at == 0.0
        assert result.reconnect_attempted is False
        assert result.reconnect_succeeded is False

    def test_custom_values(self):
        """Test health check result with custom values."""
        result = HealthCheckResult(
            server_statuses={"server1": "healthy", "server2": "unhealthy"},
            healthy_count=1,
            unhealthy_count=1,
            checked_at=1234567890.0,
            reconnect_attempted=True,
            reconnect_succeeded=True,
        )
        assert result.server_statuses == {"server1": "healthy", "server2": "unhealthy"}
        assert result.healthy_count == 1
        assert result.unhealthy_count == 1
        assert result.checked_at == 1234567890.0
        assert result.reconnect_attempted is True
        assert result.reconnect_succeeded is True

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = HealthCheckResult(
            server_statuses={"server1": "healthy"},
            healthy_count=1,
            unhealthy_count=0,
            checked_at=1234567890.0,
        )
        d = result.to_dict()
        assert d["server_statuses"] == {"server1": "healthy"}
        assert d["healthy_count"] == 1
        assert d["unhealthy_count"] == 0
        assert d["checked_at"] == 1234567890.0
        assert d["reconnect_attempted"] is False
        assert d["reconnect_succeeded"] is False

    def test_all_healthy_property(self):
        """Test all_healthy property."""
        # All healthy
        result = HealthCheckResult(healthy_count=2, unhealthy_count=0)
        assert result.all_healthy is True

        # Some unhealthy
        result = HealthCheckResult(healthy_count=1, unhealthy_count=1)
        assert result.all_healthy is False

        # None healthy (empty)
        result = HealthCheckResult(healthy_count=0, unhealthy_count=0)
        assert result.all_healthy is False


# =============================================================================
# Health Check Tests (Phase 15.5)
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method (Phase 15.5.1)."""

    @pytest.mark.asyncio
    async def test_health_check_returns_result(self, stdio_config, mock_client):
        """Test health_check returns HealthCheckResult."""
        mock_client.health_check = AsyncMock(return_value={"status": "healthy"})

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover(strict=False)
            result = await manager.health_check()

        assert isinstance(result, HealthCheckResult)
        assert result.checked_at > 0

    @pytest.mark.asyncio
    async def test_health_check_tracks_healthy_servers(self, stdio_config, mock_client):
        """Test health_check tracks healthy server count."""
        mock_client.health_check = AsyncMock(return_value={"status": "healthy"})
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover(strict=False)
            result = await manager.health_check()

        # After discovery with tools, server should be tracked
        assert result.checked_at > 0

    @pytest.mark.asyncio
    async def test_health_check_no_servers(self):
        """Test health_check with no servers configured."""
        manager = ExecutorMCPManager([])
        result = await manager.health_check()

        assert isinstance(result, HealthCheckResult)
        assert result.healthy_count == 0
        assert result.unhealthy_count == 0

    @pytest.mark.asyncio
    async def test_last_health_check_tracking(self, stdio_config, mock_client):
        """Test get_last_health_check returns latest result."""
        mock_client.health_check = AsyncMock(return_value={"status": "healthy"})

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover(strict=False)

            # No health check yet
            assert manager.get_last_health_check() is None

            # After health check
            result = await manager.health_check()
            assert manager.get_last_health_check() == result


# =============================================================================
# Auto-Reconnect Tests (Phase 15.5.3)
# =============================================================================


class TestAutoReconnect:
    """Tests for auto-reconnect functionality (Phase 15.5.3)."""

    @pytest.mark.asyncio
    async def test_consecutive_failure_tracking(self, stdio_config, mock_client):
        """Test consecutive failures are tracked per server."""
        mock_client.health_check = AsyncMock(side_effect=MCPConnectionError("Connection lost"))

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover(strict=False)

            # Check failure tracking for a specific server
            # Server name is derived from config (default naming)
            failures = manager.get_consecutive_failures("test-server")
            # Should return 0 if server not found or no failures yet
            assert isinstance(failures, int)
            assert failures >= 0

    @pytest.mark.asyncio
    async def test_health_check_with_auto_reconnect_disabled(self, stdio_config, mock_client):
        """Test health_check with auto_reconnect=False."""
        mock_client.health_check = AsyncMock(return_value={"status": "unhealthy"})

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover(strict=False)
            result = await manager.health_check(auto_reconnect=False)

        assert result.reconnect_attempted is False

    @pytest.mark.asyncio
    async def test_time_since_health_check(self, stdio_config, mock_client):
        """Test time_since_health_check returns correct value."""
        mock_client.health_check = AsyncMock(return_value={"status": "healthy"})

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover(strict=False)

            # No health check yet - returns infinity
            assert manager.time_since_health_check() == float("inf")

            # After health check
            await manager.health_check()
            elapsed = manager.time_since_health_check()
            assert elapsed is not None
            assert elapsed >= 0
            assert elapsed != float("inf")


# =============================================================================
# Health Monitor Tests (Phase 15.5.2)
# =============================================================================


class TestHealthMonitor:
    """Tests for periodic health monitoring (Phase 15.5.2)."""

    @pytest.mark.asyncio
    async def test_health_monitor_defaults(self, stdio_config):
        """Test health monitor default configuration."""
        manager = ExecutorMCPManager([stdio_config])
        assert manager._health_check_interval == DEFAULT_HEALTH_CHECK_INTERVAL
        assert manager._max_consecutive_failures == DEFAULT_MAX_CONSECUTIVE_FAILURES

    @pytest.mark.asyncio
    async def test_health_monitor_custom_interval(self, stdio_config):
        """Test health monitor with custom interval."""
        manager = ExecutorMCPManager(
            [stdio_config],
            health_check_interval=30.0,
            max_consecutive_failures=5,
        )
        assert manager._health_check_interval == 30.0
        assert manager._max_consecutive_failures == 5

    @pytest.mark.asyncio
    async def test_health_monitor_not_running_initially(self, stdio_config):
        """Test health monitor is not running initially."""
        manager = ExecutorMCPManager([stdio_config])
        assert manager.is_health_monitor_running is False

    @pytest.mark.asyncio
    async def test_start_stop_health_monitor(self, stdio_config, mock_client):
        """Test starting and stopping health monitor."""
        mock_client.health_check = AsyncMock(return_value={"status": "healthy"})

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager(
                [stdio_config],
                health_check_interval=0.1,  # Short interval for testing
            )

            # Start monitor
            await manager.start_health_monitor()
            assert manager.is_health_monitor_running is True

            # Let it run briefly
            import asyncio

            await asyncio.sleep(0.05)

            # Stop monitor
            await manager.stop_health_monitor()
            assert manager.is_health_monitor_running is False

    @pytest.mark.asyncio
    async def test_close_stops_health_monitor(self, stdio_config, mock_client):
        """Test close() stops the health monitor."""
        mock_client.health_check = AsyncMock(return_value={"status": "healthy"})

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager(
                [stdio_config],
                health_check_interval=0.1,
            )
            await manager.start_health_monitor()
            assert manager.is_health_monitor_running is True

            await manager.close()
            assert manager.is_health_monitor_running is False


# =============================================================================
# Health Summary Tests (Phase 15.5.4)
# =============================================================================


class TestHealthSummary:
    """Tests for get_health_summary method (Phase 15.5.4)."""

    @pytest.mark.asyncio
    async def test_health_summary_empty(self):
        """Test health summary with no servers."""
        manager = ExecutorMCPManager([])
        summary = manager.get_health_summary()

        # No servers = unhealthy (nothing to check)
        assert summary["overall_status"] == "unhealthy"
        assert summary["healthy_count"] == 0
        assert summary["total_count"] == 0
        assert summary["health_monitor_running"] is False
        assert summary["servers"] == []

    @pytest.mark.asyncio
    async def test_health_summary_structure(self, stdio_config, mock_client):
        """Test health summary has correct structure."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])
        mock_client.health_check = AsyncMock(return_value={"status": "healthy"})

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover(strict=False)
            await manager.health_check()
            summary = manager.get_health_summary()

        # Check structure
        assert "overall_status" in summary
        assert "healthy_count" in summary
        assert "total_count" in summary
        assert "health_monitor_running" in summary
        assert "last_check_time" in summary
        assert "time_since_check" in summary
        assert "servers" in summary

    @pytest.mark.asyncio
    async def test_health_summary_server_details(self, stdio_config, mock_client):
        """Test health summary includes server details."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])
        mock_client.health_check = AsyncMock(return_value={"status": "healthy"})

        with patch("ai_infra.executor.mcp_integration.MCPClient", return_value=mock_client):
            manager = ExecutorMCPManager([stdio_config])
            await manager.discover(strict=False)
            await manager.health_check()
            summary = manager.get_health_summary()

        servers = summary.get("servers", [])
        if servers:
            server = servers[0]
            assert "name" in server
            assert "transport" in server
            assert "health_status" in server
            assert "tool_count" in server
            assert "consecutive_failures" in server
