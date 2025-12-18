"""Integration tests for MCP Server.

These tests verify MCP server functionality, including tool registration
and execution via client connections.

Run with: pytest tests/integration/test_mcp_server.py -v
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_infra import MCPServer
from ai_infra.mcp.server.tools import mcp_from_functions


@pytest.mark.integration
class TestMCPServerStartup:
    """Integration tests for MCP server startup."""

    @pytest.mark.asyncio
    async def test_server_creation(self):
        """Test creating an MCP server."""
        server = MCPServer(name="test-server")
        assert server is not None
        assert server.name == "test-server"

    @pytest.mark.asyncio
    async def test_server_with_description(self):
        """Test creating a server with description."""
        server = MCPServer(
            name="test-server",
            description="A test MCP server for integration testing",
        )
        assert server.name == "test-server"
        assert "test" in server.description.lower()


@pytest.mark.integration
class TestMCPToolRegistration:
    """Integration tests for MCP tool registration."""

    @pytest.mark.asyncio
    async def test_register_simple_tool(self):
        """Test registering a simple tool."""
        server = MCPServer(name="test-server")

        @server.tool()
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        # Tool should be registered
        tools = server.list_tools()
        tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools]
        assert "add_numbers" in tool_names

    @pytest.mark.asyncio
    async def test_register_async_tool(self):
        """Test registering an async tool."""
        server = MCPServer(name="test-server")

        @server.tool()
        async def fetch_data(url: str) -> str:
            """Fetch data from a URL (mock)."""
            await asyncio.sleep(0.01)  # Simulate async operation
            return f"Data from {url}"

        tools = server.list_tools()
        tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools]
        assert "fetch_data" in tool_names

    @pytest.mark.asyncio
    async def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        server = MCPServer(name="test-server")

        @server.tool()
        def tool_one() -> str:
            """First tool."""
            return "one"

        @server.tool()
        def tool_two() -> str:
            """Second tool."""
            return "two"

        @server.tool()
        def tool_three() -> str:
            """Third tool."""
            return "three"

        tools = server.list_tools()
        assert len(tools) >= 3


@pytest.mark.integration
class TestMCPFromFunctions:
    """Integration tests for mcp_from_functions helper."""

    def test_create_server_from_functions(self):
        """Test creating a server from a list of functions."""

        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"

        def calculate(a: int, b: int, operation: str = "add") -> int:
            """Perform a calculation."""
            if operation == "add":
                return a + b
            elif operation == "subtract":
                return a - b
            else:
                return a * b

        server = mcp_from_functions(
            [greet, calculate],
            name="calculator-server",
        )

        assert server is not None
        tools = server.list_tools()
        tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools]
        assert "greet" in tool_names
        assert "calculate" in tool_names

    def test_function_with_complex_types(self):
        """Test registering functions with complex types."""

        def complex_func(
            items: list[str],
            config: dict[str, Any] | None = None,
            count: int | None = None,
        ) -> dict[str, Any]:
            """A function with complex parameter types."""
            return {
                "items": items,
                "config": config,
                "count": count,
            }

        server = mcp_from_functions([complex_func], name="complex-server")
        tools = server.list_tools()
        assert len(tools) >= 1


@pytest.mark.integration
class TestMCPToolExecution:
    """Integration tests for executing tools via MCP server."""

    @pytest.mark.asyncio
    async def test_execute_registered_tool(self):
        """Test executing a registered tool directly."""
        server = MCPServer(name="test-server")

        @server.tool()
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        # Execute the tool
        result = await server.call_tool("multiply", {"a": 6, "b": 7})
        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        """Test executing an async tool."""
        server = MCPServer(name="test-server")

        @server.tool()
        async def async_multiply(a: int, b: int) -> int:
            """Multiply two numbers asynchronously."""
            await asyncio.sleep(0.01)
            return a * b

        result = await server.call_tool("async_multiply", {"a": 3, "b": 4})
        assert result == 12

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test that tool errors are properly handled."""
        server = MCPServer(name="test-server")

        @server.tool()
        def failing_tool() -> str:
            """A tool that always fails."""
            raise ValueError("Intentional test error")

        with pytest.raises(Exception) as exc_info:
            await server.call_tool("failing_tool", {})

        assert "error" in str(exc_info.value).lower() or "Intentional" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tool_with_optional_params(self):
        """Test executing a tool with optional parameters."""
        server = MCPServer(name="test-server")

        @server.tool()
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        # Without optional param
        result1 = await server.call_tool("greet", {"name": "World"})
        assert "Hello" in result1

        # With optional param
        result2 = await server.call_tool("greet", {"name": "World", "greeting": "Hi"})
        assert "Hi" in result2


@pytest.mark.integration
class TestMCPConcurrency:
    """Integration tests for MCP server concurrency."""

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self):
        """Test that multiple concurrent tool calls work correctly."""
        server = MCPServer(name="test-server")

        @server.tool()
        async def slow_add(a: int, b: int) -> int:
            """Add numbers with a small delay."""
            await asyncio.sleep(0.05)
            return a + b

        # Execute multiple calls concurrently
        tasks = [
            server.call_tool("slow_add", {"a": 1, "b": 1}),
            server.call_tool("slow_add", {"a": 2, "b": 2}),
            server.call_tool("slow_add", {"a": 3, "b": 3}),
        ]

        results = await asyncio.gather(*tasks)

        assert results[0] == 2
        assert results[1] == 4
        assert results[2] == 6

    @pytest.mark.asyncio
    async def test_many_concurrent_requests(self):
        """Test handling many concurrent requests."""
        server = MCPServer(name="test-server")

        @server.tool()
        async def identity(x: int) -> int:
            """Return the input."""
            await asyncio.sleep(0.01)
            return x

        # Execute 20 concurrent calls
        tasks = [server.call_tool("identity", {"x": i}) for i in range(20)]
        results = await asyncio.gather(*tasks)

        assert results == list(range(20))
