"""Tests for MCP tool discovery.

Tests cover:
- Server discovery process
- Tool listing from servers
- Tool schema extraction
- Multi-server tool discovery
- Discovery timeout handling
- Discovery error handling

Phase 1.1.4 of production readiness test plan.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest

from ai_infra.mcp.client import (
    MCPClient,
    MCPServerError,
    MCPTimeoutError,
)

# =============================================================================
# Discovery Tests
# =============================================================================


class TestDiscovery:
    """Tests for server discovery."""

    @pytest.mark.asyncio
    async def test_discover_populates_by_name(self):
        """discover() populates _by_name mapping."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])

        mock_session = MagicMock()
        mock_session.mcp_server_info = {"name": "test-server"}

        @asynccontextmanager
        async def mock_ctx():
            yield mock_session

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.return_value = mock_ctx()
            result = await client.discover()

        assert "test-server" in result
        assert client._discovered is True

    @pytest.mark.asyncio
    async def test_discover_sets_discovered_flag(self):
        """discover() sets _discovered flag."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])

        mock_session = MagicMock()
        mock_session.mcp_server_info = {"name": "server"}

        @asynccontextmanager
        async def mock_ctx():
            yield mock_session

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.return_value = mock_ctx()
            await client.discover()

        assert client._discovered is True

    @pytest.mark.asyncio
    async def test_discover_records_errors(self):
        """discover() records errors for failed servers."""
        client = MCPClient([{"transport": "stdio", "command": "failing-server"}])

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.side_effect = Exception("Connection failed")
            await client.discover()

        assert len(client._errors) > 0
        assert "Connection failed" in str(client._errors[0])

    @pytest.mark.asyncio
    async def test_discover_strict_mode_raises(self):
        """discover(strict=True) raises ExceptionGroup on failure."""
        client = MCPClient([{"transport": "stdio", "command": "failing-server"}])

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.side_effect = Exception("Connection failed")

            with pytest.raises(ExceptionGroup) as exc_info:
                await client.discover(strict=True)

            assert len(exc_info.value.exceptions) == 1


# =============================================================================
# Discovery Timeout Tests
# =============================================================================


class TestDiscoveryTimeout:
    """Tests for discovery timeout handling."""

    def test_discover_timeout_stored(self):
        """discover_timeout option is stored."""
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            discover_timeout=15.0,
        )
        assert client._discover_timeout == 15.0

    @pytest.mark.asyncio
    async def test_discover_timeout_raises(self):
        """discover() raises MCPTimeoutError on timeout."""
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            discover_timeout=0.01,  # Very short timeout
        )

        async def slow_discover():
            import asyncio

            await asyncio.sleep(1)

        @asynccontextmanager
        async def slow_ctx():
            await slow_discover()
            yield MagicMock()

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.return_value = slow_ctx()

            with pytest.raises(MCPTimeoutError) as exc_info:
                await client.discover()

            assert exc_info.value.operation == "discover"


# =============================================================================
# List Tools Tests
# =============================================================================


class TestListTools:
    """Tests for listing tools from servers."""

    @pytest.mark.asyncio
    async def test_list_tools_unknown_server_raises(self):
        """list_tools with unknown server raises MCPServerError."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._discovered = True
        client._by_name = {"filesystem": MagicMock()}

        with pytest.raises(MCPServerError, match="Unknown server"):
            await client.list_tools(server="nonexistent")

    @pytest.mark.asyncio
    async def test_list_tools_suggests_similar_names(self):
        """list_tools suggests similar server names on typo."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._discovered = True
        client._by_name = {"filesystem": MagicMock()}

        with pytest.raises(MCPServerError, match="filesystem"):
            await client.list_tools(server="filesyste")  # Typo


# =============================================================================
# Multi-Server Discovery Tests
# =============================================================================


class TestMultiServerDiscovery:
    """Tests for multi-server discovery."""

    @pytest.mark.asyncio
    async def test_discover_multiple_servers(self):
        """discover() handles multiple servers."""
        client = MCPClient(
            [
                {"transport": "stdio", "command": "server1"},
                {"transport": "stdio", "command": "server2"},
            ]
        )

        call_count = 0

        def make_mock_ctx():
            @asynccontextmanager
            async def mock_ctx():
                nonlocal call_count
                call_count += 1
                mock_session = MagicMock()
                mock_session.mcp_server_info = {"name": f"server{call_count}"}
                yield mock_session

            return mock_ctx()

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.side_effect = lambda cfg: make_mock_ctx()
            result = await client.discover()

        # Both servers discovered
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_discover_handles_partial_failure(self):
        """discover() continues after single server failure."""
        client = MCPClient(
            [
                {"transport": "stdio", "command": "good-server"},
                {"transport": "stdio", "command": "bad-server"},
            ]
        )

        call_count = 0

        def make_mock_ctx():
            nonlocal call_count
            call_count += 1
            current_call = call_count

            @asynccontextmanager
            async def mock_ctx():
                if current_call == 2:
                    raise Exception("Server 2 failed")
                mock_session = MagicMock()
                mock_session.mcp_server_info = {"name": "good-server"}
                yield mock_session

            return mock_ctx()

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.side_effect = lambda cfg: make_mock_ctx()
            result = await client.discover()

        # One server discovered, one error recorded
        assert len(result) == 1
        assert len(client._errors) == 1


# =============================================================================
# Unique Name Collision Tests
# =============================================================================


class TestNameCollisionHandling:
    """Tests for handling duplicate server names."""

    @pytest.mark.asyncio
    async def test_duplicate_names_get_suffix(self):
        """Duplicate server names get #N suffix."""
        client = MCPClient(
            [
                {"transport": "stdio", "command": "server1"},
                {"transport": "stdio", "command": "server2"},
            ]
        )

        def make_mock_ctx():
            @asynccontextmanager
            async def mock_ctx():
                mock_session = MagicMock()
                # Both servers report same name
                mock_session.mcp_server_info = {"name": "server"}
                yield mock_session

            return mock_ctx()

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.side_effect = lambda cfg: make_mock_ctx()
            result = await client.discover()

        # Should have unique names
        names = list(result.keys())
        assert len(names) == 2
        assert len(set(names)) == 2  # All unique


# =============================================================================
# Health Status During Discovery Tests
# =============================================================================


class TestHealthStatusDuringDiscovery:
    """Tests for health status tracking during discovery."""

    @pytest.mark.asyncio
    async def test_successful_discover_marks_healthy(self):
        """Successful discovery marks server as healthy."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])

        mock_session = MagicMock()
        mock_session.mcp_server_info = {"name": "healthy-server"}

        @asynccontextmanager
        async def mock_ctx():
            yield mock_session

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.return_value = mock_ctx()
            await client.discover()

        assert client._health_status.get("healthy-server") == "healthy"
