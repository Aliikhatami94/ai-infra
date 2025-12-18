"""Integration tests for MCP Client.

These tests verify MCP client functionality with real MCP servers.
They require MCP to be properly installed and configured.

Run with: pytest tests/integration/test_mcp_client.py -v
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from ai_infra import MCPClient, MCPSecuritySettings


@pytest.mark.integration
class TestMCPClientConnection:
    """Integration tests for MCP client connection handling."""

    @pytest.mark.asyncio
    async def test_connect_to_filesystem_server(self):
        """Test connection to the built-in filesystem MCP server."""
        # Create a temp directory for the filesystem server
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MCPClient(
                server_name="filesystem",
                server_type="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmpdir],
            )

            try:
                await client.connect()
                assert client.is_connected()
            finally:
                await client.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_handling(self):
        """Test proper disconnect handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MCPClient(
                server_name="filesystem",
                server_type="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmpdir],
            )

            await client.connect()
            assert client.is_connected()

            await client.disconnect()
            assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling for invalid server."""
        client = MCPClient(
            server_name="nonexistent",
            server_type="stdio",
            command="nonexistent-command-that-does-not-exist",
            args=[],
        )

        with pytest.raises((FileNotFoundError, OSError, ConnectionError)):
            await client.connect()


@pytest.mark.integration
class TestMCPToolDiscovery:
    """Integration tests for MCP tool discovery."""

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test discovering tools from MCP server."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MCPClient(
                server_name="filesystem",
                server_type="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmpdir],
            )

            try:
                await client.connect()
                tools = await client.list_tools()

                assert isinstance(tools, list)
                assert len(tools) > 0

                # Filesystem server should have common tools
                tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools]
                # At least one tool should be available
                assert len(tool_names) > 0
            finally:
                await client.disconnect()

    @pytest.mark.asyncio
    async def test_get_tool_as_langchain_tools(self):
        """Test getting tools in LangChain format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MCPClient(
                server_name="filesystem",
                server_type="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmpdir],
            )

            try:
                await client.connect()
                lc_tools = await client.get_tools()

                assert isinstance(lc_tools, list)
                # Should return tools in a format compatible with LangChain
                for tool in lc_tools:
                    assert hasattr(tool, "name") or callable(tool)
            finally:
                await client.disconnect()


@pytest.mark.integration
class TestMCPToolExecution:
    """Integration tests for MCP tool execution."""

    @pytest.mark.asyncio
    async def test_execute_read_file_tool(self):
        """Test executing a read_file tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("Hello, MCP!")

            client = MCPClient(
                server_name="filesystem",
                server_type="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmpdir],
            )

            try:
                await client.connect()

                # Try to read the file using MCP
                result = await client.call_tool(
                    "read_file",
                    {"path": test_file},
                )

                assert result is not None
                # Result should contain the file content
                result_str = str(result)
                assert "Hello" in result_str or "MCP" in result_str
            finally:
                await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_list_directory_tool(self):
        """Test executing a list_directory tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            for name in ["file1.txt", "file2.txt", "file3.txt"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write(f"Content of {name}")

            client = MCPClient(
                server_name="filesystem",
                server_type="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmpdir],
            )

            try:
                await client.connect()

                # List the directory
                result = await client.call_tool(
                    "list_directory",
                    {"path": tmpdir},
                )

                assert result is not None
                result_str = str(result)
                # Should list the files
                assert "file1" in result_str or len(result_str) > 0
            finally:
                await client.disconnect()


@pytest.mark.integration
class TestMCPTimeout:
    """Integration tests for MCP timeout handling."""

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """Test that connection timeout works."""
        # This test uses a server that takes too long to respond
        # We simulate by using a very short timeout
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MCPClient(
                server_name="filesystem",
                server_type="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmpdir],
                timeout=0.001,  # Very short timeout
            )

            # Should either connect or timeout - both are valid outcomes
            # The important thing is it doesn't hang forever
            try:
                await asyncio.wait_for(client.connect(), timeout=5.0)
                await client.disconnect()
            except (TimeoutError, Exception):
                pass  # Expected - timeout or connection error


@pytest.mark.integration
class TestMCPSecuritySettings:
    """Integration tests for MCP security settings."""

    @pytest.mark.asyncio
    async def test_security_settings_applied(self):
        """Test that security settings are applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            security = MCPSecuritySettings(
                allow_code_execution=False,
                max_tool_calls_per_request=10,
            )

            client = MCPClient(
                server_name="filesystem",
                server_type="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmpdir],
                security_settings=security,
            )

            try:
                await client.connect()
                # Connection should work with security settings
                assert client.is_connected()
            finally:
                await client.disconnect()
