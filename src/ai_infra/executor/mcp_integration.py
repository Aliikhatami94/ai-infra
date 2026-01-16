"""MCP integration for the Executor module (Phase 15.2).

Provides ExecutorMCPManager for integrating external MCP servers with the executor:
- Tool discovery with timeout handling
- Server health monitoring
- Graceful degradation on connection failures
- Retry logic with exponential backoff
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ai_infra.logging import get_logger
from ai_infra.mcp import MCPClient, McpServerConfig
from ai_infra.mcp.client.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

logger = get_logger("executor.mcp")


# =============================================================================
# Constants
# =============================================================================

DEFAULT_DISCOVER_TIMEOUT = 30.0
DEFAULT_TOOL_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0
DEFAULT_RETRY_MAX_DELAY = 30.0

# Phase 15.5: Health monitoring constants
DEFAULT_HEALTH_CHECK_INTERVAL = 60.0  # seconds
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3


# =============================================================================
# Health Status (Phase 15.5)
# =============================================================================


@dataclass
class HealthCheckResult:
    """Result of a health check operation.

    Attributes:
        server_statuses: Dict mapping server names to their health status.
        healthy_count: Number of healthy servers.
        unhealthy_count: Number of unhealthy servers.
        checked_at: Timestamp of the health check.
        reconnect_attempted: Whether reconnection was attempted.
        reconnect_succeeded: Whether reconnection succeeded (if attempted).
    """

    server_statuses: dict[str, str] = field(default_factory=dict)
    healthy_count: int = 0
    unhealthy_count: int = 0
    checked_at: float = 0.0  # time.time() value
    reconnect_attempted: bool = False
    reconnect_succeeded: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_statuses": self.server_statuses,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "checked_at": self.checked_at,
            "reconnect_attempted": self.reconnect_attempted,
            "reconnect_succeeded": self.reconnect_succeeded,
        }

    @property
    def all_healthy(self) -> bool:
        """Check if all servers are healthy."""
        return self.unhealthy_count == 0 and self.healthy_count > 0


# =============================================================================
# Server Info
# =============================================================================


@dataclass
class MCPServerInfo:
    """Information about a discovered MCP server.

    Attributes:
        name: Server name from discovery.
        config: Original server configuration (may be None for failed servers).
        tool_count: Number of tools discovered.
        health_status: Current health status.
        error: Error message if discovery failed.
        transport: Transport type (derived from config or error info).
    """

    name: str
    config: McpServerConfig | None = None
    tool_count: int = 0
    health_status: str = "unknown"
    error: str | None = None
    transport: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "transport": self.config.transport if self.config else self.transport,
            "tool_count": self.tool_count,
            "health_status": self.health_status,
            "error": self.error,
        }


# =============================================================================
# Retry Configuration
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds (caps exponential backoff).
        exponential_base: Base for exponential backoff (default 2).
    """

    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_RETRY_BASE_DELAY
    max_delay: float = DEFAULT_RETRY_MAX_DELAY
    exponential_base: float = 2.0

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed).

        Uses exponential backoff: base_delay * (exponential_base ^ attempt)
        Capped at max_delay.
        """
        delay = self.base_delay * (self.exponential_base**attempt)
        return min(delay, self.max_delay)


# =============================================================================
# Discovery Result
# =============================================================================


@dataclass
class DiscoveryResult:
    """Result of MCP server discovery.

    Attributes:
        success: Whether discovery completed successfully.
        tools: List of discovered tools.
        servers: Information about each server.
        errors: List of errors encountered.
        total_servers: Total number of configured servers.
        healthy_servers: Number of healthy servers.
        retry_count: Number of retries performed.
    """

    success: bool = True
    tools: list[Any] = field(default_factory=list)
    servers: list[MCPServerInfo] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    total_servers: int = 0
    healthy_servers: int = 0
    retry_count: int = 0

    @property
    def tool_count(self) -> int:
        """Total number of discovered tools."""
        return len(self.tools)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "tool_count": self.tool_count,
            "total_servers": self.total_servers,
            "healthy_servers": self.healthy_servers,
            "retry_count": self.retry_count,
            "servers": [s.to_dict() for s in self.servers],
            "errors": self.errors,
        }


# =============================================================================
# ExecutorMCPManager
# =============================================================================


class ExecutorMCPManager:
    """Manages MCP server connections for executor.

    Wraps ai_infra.mcp.MCPClient for executor-specific use cases:
    - Tool discovery with timeout handling
    - Server health monitoring
    - Graceful degradation on connection failures
    - Retry logic with exponential backoff

    Example:
        >>> from ai_infra.mcp import McpServerConfig
        >>> configs = [
        ...     McpServerConfig(
        ...         transport="stdio",
        ...         command="npx",
        ...         args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
        ...     )
        ... ]
        >>> manager = ExecutorMCPManager(configs)
        >>> result = await manager.discover()
        >>> tools = manager.get_tools()

    Attributes:
        configs: List of MCP server configurations.
        discover_timeout: Timeout for discovery in seconds.
        tool_timeout: Timeout for tool calls in seconds.
        retry_config: Configuration for retry behavior.
    """

    def __init__(
        self,
        configs: list[McpServerConfig],
        discover_timeout: float = DEFAULT_DISCOVER_TIMEOUT,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
        retry_config: RetryConfig | None = None,
        health_check_interval: float = DEFAULT_HEALTH_CHECK_INTERVAL,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    ):
        """Initialize MCP manager.

        Args:
            configs: List of MCP server configurations.
            discover_timeout: Timeout for discovery in seconds (default 30.0).
            tool_timeout: Timeout for tool calls in seconds (default 60.0).
            retry_config: Retry configuration (optional, uses defaults if None).
            health_check_interval: Interval between periodic health checks (default 60.0).
            max_consecutive_failures: Max failures before auto-reconnect (default 3).
        """
        self._configs = configs
        self._discover_timeout = discover_timeout
        self._tool_timeout = tool_timeout
        self._retry_config = retry_config or RetryConfig()
        self._client: MCPClient | None = None
        self._tools: list[Any] = []
        self._discovered = False
        self._server_info: list[MCPServerInfo] = []
        self._last_discovery_result: DiscoveryResult | None = None

        # Phase 15.5: Health monitoring state
        self._health_check_interval = health_check_interval
        self._max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures: dict[str, int] = {}  # server_name -> failure count
        self._last_health_check: float = 0.0
        self._last_health_result: HealthCheckResult | None = None
        self._health_monitor_task: asyncio.Task[None] | None = None
        self._health_monitor_running = False

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_discovered(self) -> bool:
        """Whether discovery has completed successfully."""
        return self._discovered

    @property
    def configs(self) -> list[McpServerConfig]:
        """Get configured MCP server configs."""
        return self._configs

    @property
    def discover_timeout(self) -> float:
        """Get discovery timeout in seconds."""
        return self._discover_timeout

    @property
    def tool_timeout(self) -> float:
        """Get tool call timeout in seconds."""
        return self._tool_timeout

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    async def discover(self, *, strict: bool = False) -> DiscoveryResult:
        """Discover tools from all configured MCP servers.

        Performs discovery with retry logic for transient failures.
        On success, tools are available via get_tools().

        Args:
            strict: If True, raise on any server failure.
                   If False (default), continue with partial results.

        Returns:
            DiscoveryResult with discovered tools and server info.

        Raises:
            MCPConnectionError: If strict=True and connection fails.
            MCPTimeoutError: If discovery times out after all retries.
        """
        if not self._configs:
            logger.debug("No MCP servers configured, skipping discovery")
            result = DiscoveryResult(
                success=True,
                tools=[],
                servers=[],
                total_servers=0,
                healthy_servers=0,
            )
            self._last_discovery_result = result
            return result

        logger.info(
            f"Discovering tools from {len(self._configs)} MCP server(s) (timeout={self._discover_timeout}s)"
        )

        last_error: Exception | None = None
        retry_count = 0

        for attempt in range(self._retry_config.max_retries + 1):
            try:
                result = await self._do_discovery(strict=strict)
                result.retry_count = retry_count
                self._last_discovery_result = result
                return result

            except MCPTimeoutError as e:
                last_error = e
                retry_count = attempt + 1

                if attempt < self._retry_config.max_retries:
                    delay = self._retry_config.get_delay(attempt)
                    logger.warning(
                        f"MCP discovery timed out (attempt {attempt + 1}/{self._retry_config.max_retries + 1}), retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"MCP discovery failed after {self._retry_config.max_retries + 1} attempts: {e}"
                    )

            except MCPConnectionError as e:
                last_error = e
                retry_count = attempt + 1

                if attempt < self._retry_config.max_retries:
                    delay = self._retry_config.get_delay(attempt)
                    logger.warning(
                        f"MCP connection error (attempt {attempt + 1}/{self._retry_config.max_retries + 1}), retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"MCP discovery failed after {self._retry_config.max_retries + 1} attempts: {e}"
                    )

        # All retries exhausted
        if strict and last_error:
            raise last_error

        # Return partial result on graceful degradation
        result = DiscoveryResult(
            success=False,
            tools=[],
            servers=[],
            errors=[{"error": str(last_error), "type": type(last_error).__name__}],
            total_servers=len(self._configs),
            healthy_servers=0,
            retry_count=retry_count,
        )
        self._last_discovery_result = result
        return result

    async def _do_discovery(self, *, strict: bool) -> DiscoveryResult:
        """Internal discovery implementation.

        Args:
            strict: If True, raise on failures.

        Returns:
            DiscoveryResult with tools and server info.

        Raises:
            MCPTimeoutError: If discovery times out.
            MCPConnectionError: If connection fails.
        """
        # Create client with timeouts
        self._client = MCPClient(
            self._configs,
            discover_timeout=self._discover_timeout,
            tool_timeout=self._tool_timeout,
        )

        # Discover servers
        server_map = await self._client.discover(strict=strict)

        # Collect server info
        self._server_info = []
        errors = self._client.last_errors()

        for name, config in server_map.items():
            info = MCPServerInfo(
                name=name,
                config=config,
                health_status="healthy",
            )
            self._server_info.append(info)

        # Add error entries for failed servers
        for err in errors:
            err_config = err.get("config", {})
            info = MCPServerInfo(
                name=err.get("identity", "unknown"),
                config=None,
                transport=err_config.get("transport", "unknown"),
                health_status="unhealthy",
                error=err.get("error"),
            )
            self._server_info.append(info)

        # Get all tools
        self._tools = await self._client.list_tools()
        self._discovered = True

        # Update tool counts per server
        for info in self._server_info:
            if info.health_status == "healthy" and info.name in server_map:
                try:
                    server_tools = await self._client.list_tools(server=info.name)
                    info.tool_count = len(server_tools)
                except MCPError:
                    pass  # Keep count as 0

        healthy_count = sum(1 for s in self._server_info if s.health_status == "healthy")

        logger.info(
            f"MCP discovery complete: {len(self._tools)} tools from {healthy_count}/{len(self._configs)} healthy server(s)"
        )

        # Log per-server details
        for info in self._server_info:
            transport = info.config.transport if info.config else info.transport
            if info.health_status == "healthy":
                logger.debug(f"  Server '{info.name}' ({transport}): {info.tool_count} tools")
            else:
                logger.debug(
                    f"  Server '{info.name}' ({transport}): UNHEALTHY - {info.error or 'unknown error'}"
                )

        return DiscoveryResult(
            success=True,
            tools=list(self._tools),
            servers=list(self._server_info),
            errors=errors,
            total_servers=len(self._configs),
            healthy_servers=healthy_count,
        )

    # -------------------------------------------------------------------------
    # Tool Access
    # -------------------------------------------------------------------------

    def get_tools(self) -> list[BaseTool]:
        """Get discovered tools.

        Returns:
            List of tools discovered from MCP servers.
            Empty list if discovery has not been run.
        """
        return list(self._tools)

    def get_tool_names(self) -> list[str]:
        """Get names of discovered tools.

        Returns:
            List of tool names.
        """
        names = []
        for tool in self._tools:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
            names.append(name)
        return names

    # -------------------------------------------------------------------------
    # Server Info
    # -------------------------------------------------------------------------

    def get_server_info(self) -> list[MCPServerInfo]:
        """Get information about discovered servers.

        Returns:
            List of MCPServerInfo for each configured server.
        """
        return list(self._server_info)

    def get_server_names(self) -> list[str]:
        """Get names of healthy servers.

        Returns:
            List of server names that are healthy.
        """
        return [s.name for s in self._server_info if s.health_status == "healthy"]

    def get_last_discovery_result(self) -> DiscoveryResult | None:
        """Get the result of the last discovery attempt.

        Returns:
            DiscoveryResult from last discover() call, or None if never called.
        """
        return self._last_discovery_result

    # -------------------------------------------------------------------------
    # Health Check (Phase 15.5)
    # -------------------------------------------------------------------------

    async def health_check(self, *, auto_reconnect: bool = True) -> HealthCheckResult:
        """Check health of all configured servers (Phase 15.5).

        Performs health check and optionally attempts reconnection for
        servers that have exceeded the failure threshold.

        Args:
            auto_reconnect: If True, attempt to reconnect failed servers.

        Returns:
            HealthCheckResult with detailed server statuses.
        """
        if self._client is None:
            self._client = MCPClient(
                self._configs,
                discover_timeout=self._discover_timeout,
                tool_timeout=self._tool_timeout,
            )

        current_time = time.time()
        statuses: dict[str, str] = {}
        healthy_count = 0
        unhealthy_count = 0
        reconnect_attempted = False
        reconnect_succeeded = False

        try:
            raw_statuses = await self._client.health_check()
            statuses = raw_statuses

            for server_name, status in raw_statuses.items():
                if status == "healthy":
                    healthy_count += 1
                    # Reset failure counter on success
                    self._consecutive_failures[server_name] = 0
                else:
                    unhealthy_count += 1
                    # Increment failure counter
                    self._consecutive_failures[server_name] = (
                        self._consecutive_failures.get(server_name, 0) + 1
                    )

                    # Check if auto-reconnect should be attempted
                    if (
                        auto_reconnect
                        and self._consecutive_failures[server_name]
                        >= self._max_consecutive_failures
                    ):
                        logger.warning(
                            f"Server '{server_name}' has failed {self._consecutive_failures[server_name]} "
                            f"consecutive health checks, attempting reconnect"
                        )
                        reconnect_attempted = True
                        try:
                            await self._reconnect_server(server_name)
                            reconnect_succeeded = True
                            self._consecutive_failures[server_name] = 0
                            logger.info(f"Successfully reconnected to server '{server_name}'")
                        except MCPError as e:
                            logger.error(f"Failed to reconnect to server '{server_name}': {e}")

            # Update server info health statuses
            for info in self._server_info:
                if info.name in statuses:
                    info.health_status = statuses[info.name]

        except MCPError as e:
            logger.error(f"Health check failed: {e}")
            # Mark all servers as unhealthy
            for info in self._server_info:
                statuses[info.name] = "unhealthy"
                unhealthy_count += 1
                info.health_status = "unhealthy"

        result = HealthCheckResult(
            server_statuses=statuses,
            healthy_count=healthy_count,
            unhealthy_count=unhealthy_count,
            checked_at=current_time,
            reconnect_attempted=reconnect_attempted,
            reconnect_succeeded=reconnect_succeeded,
        )

        self._last_health_check = current_time
        self._last_health_result = result

        logger.debug(f"Health check complete: {healthy_count} healthy, {unhealthy_count} unhealthy")

        return result

    async def _reconnect_server(self, server_name: str) -> None:
        """Attempt to reconnect to a specific server (Phase 15.5).

        Args:
            server_name: Name of the server to reconnect.

        Raises:
            MCPError: If reconnection fails.
        """
        logger.info(f"Attempting to reconnect to server '{server_name}'")

        # Find the config for this server
        config = None
        for cfg in self._configs:
            # Try to match by server identity/name
            if hasattr(cfg, "identity") and cfg.identity == server_name:
                config = cfg
                break

        if config is None:
            # Try to find by other means or reconnect via client
            logger.debug(f"No direct config match for '{server_name}', using client reconnect")

        # Let the client handle reconnection
        if self._client is not None:
            # Re-discover to refresh connections
            await self._client.discover(strict=False)

    def get_last_health_check(self) -> HealthCheckResult | None:
        """Get the result of the last health check.

        Returns:
            HealthCheckResult from last health_check() call, or None if never called.
        """
        return self._last_health_result

    def get_consecutive_failures(self, server_name: str) -> int:
        """Get the number of consecutive failures for a server.

        Args:
            server_name: Name of the server.

        Returns:
            Number of consecutive health check failures.
        """
        return self._consecutive_failures.get(server_name, 0)

    def time_since_health_check(self) -> float:
        """Get time in seconds since last health check.

        Returns:
            Seconds since last health check, or infinity if never checked.
        """
        if self._last_health_check == 0.0:
            return float("inf")
        return time.time() - self._last_health_check

    # -------------------------------------------------------------------------
    # Periodic Health Monitoring (Phase 15.5)
    # -------------------------------------------------------------------------

    async def start_health_monitor(self) -> None:
        """Start periodic health monitoring in the background (Phase 15.5).

        Health checks will run at the configured interval until
        stop_health_monitor() is called.
        """
        if self._health_monitor_running:
            logger.debug("Health monitor already running")
            return

        self._health_monitor_running = True
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info(f"Started MCP health monitor (interval={self._health_check_interval}s)")

    async def stop_health_monitor(self) -> None:
        """Stop periodic health monitoring (Phase 15.5)."""
        if not self._health_monitor_running:
            return

        self._health_monitor_running = False
        if self._health_monitor_task is not None:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
            self._health_monitor_task = None
        logger.info("Stopped MCP health monitor")

    async def _health_monitor_loop(self) -> None:
        """Internal health monitoring loop (Phase 15.5)."""
        while self._health_monitor_running:
            try:
                await asyncio.sleep(self._health_check_interval)
                if self._health_monitor_running:
                    await self.health_check(auto_reconnect=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    @property
    def is_health_monitor_running(self) -> bool:
        """Check if the health monitor is currently running."""
        return self._health_monitor_running

    # -------------------------------------------------------------------------
    # Health Status Summary (Phase 15.5)
    # -------------------------------------------------------------------------

    def get_health_summary(self) -> dict[str, Any]:
        """Get a comprehensive health summary for all servers (Phase 15.5).

        Returns:
            Dict with overall health status and per-server details.
        """
        servers = []
        for info in self._server_info:
            server_data = {
                "name": info.name,
                "transport": info.config.transport if info.config else info.transport,
                "health_status": info.health_status,
                "tool_count": info.tool_count,
                "consecutive_failures": self._consecutive_failures.get(info.name, 0),
            }
            if info.config:
                if info.config.transport == "stdio":
                    server_data["command"] = info.config.command
                else:
                    server_data["url"] = info.config.url
            if info.error:
                server_data["error"] = info.error
            servers.append(server_data)

        healthy_count = sum(1 for s in servers if s["health_status"] == "healthy")
        total_count = len(servers)

        return {
            "overall_status": "healthy"
            if healthy_count == total_count and total_count > 0
            else "unhealthy",
            "healthy_count": healthy_count,
            "total_count": total_count,
            "health_monitor_running": self._health_monitor_running,
            "last_check_time": self._last_health_check if self._last_health_check > 0 else None,
            "time_since_check": self.time_since_health_check()
            if self._last_health_check > 0
            else None,
            "servers": servers,
        }

    # -------------------------------------------------------------------------
    # Tool Calls
    # -------------------------------------------------------------------------

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on a specific server.

        Args:
            server_name: Name of the server.
            tool_name: Name of the tool.
            arguments: Arguments to pass to the tool.

        Returns:
            Result from the tool call.

        Raises:
            MCPError: If the tool call fails.
            ValueError: If discovery has not been run.
        """
        if self._client is None:
            raise ValueError("Discovery has not been run. Call discover() first.")

        logger.debug(
            f"Calling MCP tool: server={server_name}, tool={tool_name}, args={list(arguments.keys())}"
        )

        try:
            result = await self._client.call_tool(server_name, tool_name, arguments)
            logger.debug(f"MCP tool call complete: server={server_name}, tool={tool_name}")
            return result
        except MCPError as e:
            logger.error(f"MCP tool call failed: server={server_name}, tool={tool_name}, error={e}")
            raise

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def close(self) -> None:
        """Close all MCP server connections."""
        # Phase 15.5: Stop health monitor first
        await self.stop_health_monitor()

        if self._client:
            logger.debug("Closing MCP client connections")
            await self._client.close()
            self._client = None
        self._discovered = False
        self._tools = []
        self._server_info = []
        self._consecutive_failures = {}
        self._last_health_result = None

    async def __aenter__(self) -> ExecutorMCPManager:
        """Enter async context - discover servers."""
        await self.discover()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context - cleanup."""
        await self.close()


# =============================================================================
# Factory Function
# =============================================================================


def create_mcp_manager(
    configs: list[McpServerConfig],
    discover_timeout: float = DEFAULT_DISCOVER_TIMEOUT,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ExecutorMCPManager:
    """Create an ExecutorMCPManager with the given configuration.

    Factory function for creating MCP managers with sensible defaults.

    Args:
        configs: List of MCP server configurations.
        discover_timeout: Timeout for discovery in seconds.
        tool_timeout: Timeout for tool calls in seconds.
        max_retries: Maximum number of retry attempts.

    Returns:
        Configured ExecutorMCPManager instance.

    Example:
        >>> from ai_infra.mcp import McpServerConfig
        >>> configs = [McpServerConfig(transport="stdio", command="mcp-server")]
        >>> manager = create_mcp_manager(configs)
    """
    retry_config = RetryConfig(max_retries=max_retries)
    return ExecutorMCPManager(
        configs=configs,
        discover_timeout=discover_timeout,
        tool_timeout=tool_timeout,
        retry_config=retry_config,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main class
    "ExecutorMCPManager",
    # Data classes
    "MCPServerInfo",
    "DiscoveryResult",
    "RetryConfig",
    "HealthCheckResult",  # Phase 15.5
    # Factory
    "create_mcp_manager",
    # Constants
    "DEFAULT_DISCOVER_TIMEOUT",
    "DEFAULT_TOOL_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BASE_DELAY",
    "DEFAULT_RETRY_MAX_DELAY",
    "DEFAULT_HEALTH_CHECK_INTERVAL",  # Phase 15.5
    "DEFAULT_MAX_CONSECUTIVE_FAILURES",  # Phase 15.5
]
