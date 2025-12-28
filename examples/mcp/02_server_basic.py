#!/usr/bin/env python
"""Basic MCP Server Example.

This example demonstrates:
- Creating an MCP server from plain Python functions
- Running the server with different transports
- Exposing tools, prompts, and resources
- Security settings and auto-detection

MCP (Model Context Protocol) allows you to expose tools that AI agents
can discover and use. This server can be consumed by:
- Claude Desktop
- AI agents using MCPClient
- Any MCP-compatible client

Transports:
- stdio: For local CLI usage (Claude Desktop, etc.)
- streamable_http: For web deployment (recommended)
- sse: For simpler HTTP streaming
"""

import asyncio
import random
from datetime import datetime

from ai_infra import MCPSecuritySettings, mcp_from_functions

# =============================================================================
# Define Tools as Plain Functions
# =============================================================================
# MCP tools are just Python functions. ai-infra automatically:
# - Extracts the tool name from the function name
# - Extracts the description from the docstring
# - Infers parameter types from type hints


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city (e.g., "New York", "London", "Tokyo")

    Returns:
        A description of the current weather conditions.
    """
    # Simulated weather data
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly cloudy", "Stormy"]
    temp = random.randint(40, 90)
    condition = random.choice(conditions)
    return f"Weather in {city}: {condition}, {temp}°F"


def get_time(timezone: str = "UTC") -> str:
    """Get the current time.

    Args:
        timezone: The timezone name (e.g., "UTC", "America/New_York")

    Returns:
        The current time as a formatted string.
    """
    now = datetime.now()
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression (e.g., "2 + 2", "sqrt(16)")

    Returns:
        The result of the calculation.

    Examples:
        calculate("2 + 2") → "4"
        calculate("sqrt(16)") → "4.0"
        calculate("3.14 * 2**2") → "12.56"
    """
    import math

    # Safe evaluation with limited namespace
    allowed = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search_knowledge(query: str, max_results: int = 3) -> str:
    """Search a knowledge base.

    Args:
        query: The search query
        max_results: Maximum number of results to return (1-10)

    Returns:
        Matching results from the knowledge base.
    """
    # Simulated knowledge base
    knowledge = {
        "python": "Python is a high-level programming language known for readability.",
        "mcp": "MCP (Model Context Protocol) is a standard for AI tool discovery.",
        "ai-infra": "ai-infra is a Python SDK for building AI applications.",
        "agents": "AI agents are systems that can use tools to accomplish tasks.",
        "llm": "Large Language Models are AI systems trained on text data.",
    }

    results = []
    query_lower = query.lower()
    for topic, info in knowledge.items():
        if query_lower in topic.lower() or query_lower in info.lower():
            results.append(f"• {topic}: {info}")
            if len(results) >= max_results:
                break

    if results:
        return "\n".join(results)
    return f"No results found for: {query}"


def echo(message: str) -> str:
    """Echo a message back (useful for testing).

    Args:
        message: The message to echo

    Returns:
        The same message, echoed back.
    """
    return f"Echo: {message}"


# =============================================================================
# Example 1: Create Server from Functions
# =============================================================================


def create_simple_server():
    """Create a simple MCP server with auto-detected security."""
    print("=" * 60)
    print("Creating MCP Server from Functions")
    print("=" * 60)

    # Create server - security is auto-configured from environment
    mcp = mcp_from_functions(
        name="demo-tools",
        functions=[get_weather, get_time, calculate, search_knowledge, echo],
    )

    print("\n✓ Server created with tools:")
    print("  - get_weather")
    print("  - get_time")
    print("  - calculate")
    print("  - search_knowledge")
    print("  - echo")

    return mcp


# =============================================================================
# Example 2: Run with stdio (for Claude Desktop)
# =============================================================================


def run_stdio():
    """Run the server with stdio transport.

    This is how Claude Desktop and other CLI tools connect to MCP servers.
    The server reads from stdin and writes to stdout.

    To use with Claude Desktop, add to claude_desktop_config.json:
    {
        "mcpServers": {
            "demo": {
                "command": "python",
                "args": ["examples/mcp/02_server_basic.py", "--stdio"]
            }
        }
    }
    """
    print("\nStarting MCP server with stdio transport...")
    print("(Press Ctrl+C to stop)")

    mcp = create_simple_server()
    mcp.run(transport="stdio")


# =============================================================================
# Example 3: Run with HTTP (for web deployment)
# =============================================================================


def run_http(host: str = "127.0.0.1", port: int = 8000):
    """Run the server with streamable HTTP transport.

    This is recommended for web deployment. Clients connect via HTTP
    with streaming responses.

    Args:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to bind to (default: 8000)
    """
    print(f"\nStarting MCP server at http://{host}:{port}/mcp")
    print("(Press Ctrl+C to stop)")
    print("\nTest with:")
    print(f'  curl http://{host}:{port}/mcp -H "Content-Type: application/json"')

    mcp = create_simple_server()

    # For HTTP, we need to configure security
    # By default, it auto-detects from environment (Railway, Render, etc.)
    # For local development, localhost is always allowed

    import uvicorn

    # Get the streamable HTTP app
    app = mcp.streamable_http_app()
    uvicorn.run(app, host=host, port=port)


# =============================================================================
# Example 4: Custom Security Settings
# =============================================================================


def create_server_with_security():
    """Create server with explicit security settings."""
    print("\n" + "=" * 60)
    print("Server with Custom Security")
    print("=" * 60)

    # Disable security (development only!)
    dev_security = MCPSecuritySettings(enable_security=False)

    # Custom domains (production)
    prod_security = MCPSecuritySettings(domains=["api.example.com", "tools.example.com"])

    # Use auto-detection (recommended)
    auto_security = MCPSecuritySettings()  # Detects from environment

    print("\nSecurity configurations:")
    print(f"  dev_security.enabled = {dev_security.enabled}")
    print(f"  prod_security.allowed_hosts = {prod_security.allowed_hosts[:3]}...")
    print(f"  auto_security.allowed_hosts = {auto_security.allowed_hosts[:3]}...")

    # Create server with custom security
    mcp = mcp_from_functions(
        name="secure-tools",
        functions=[echo],
        security=auto_security,
    )

    print("\n✓ Server created with auto-detected security")
    return mcp


# =============================================================================
# Example 5: Using ToolDef for Advanced Configuration
# =============================================================================


def create_server_with_tooldef():
    """Create server with explicit tool definitions."""
    from ai_infra.mcp.server.tools import ToolDef

    print("\n" + "=" * 60)
    print("Server with ToolDef Configuration")
    print("=" * 60)

    # ToolDef allows customizing tool name and description
    tools = [
        # Simple function
        get_weather,
        # ToolDef with custom name/description
        ToolDef(
            fn=calculate,
            name="math_eval",  # Override function name
            description="Evaluate mathematical expressions safely",  # Override docstring
        ),
        ToolDef(
            fn=echo,
            name="test_echo",
            description="Test tool that echoes input",
        ),
    ]

    mcp = mcp_from_functions(name="custom-tools", functions=tools)

    print("\n✓ Server created with custom tool definitions:")
    print("  - get_weather (from function)")
    print("  - math_eval (from ToolDef, was 'calculate')")
    print("  - test_echo (from ToolDef, was 'echo')")

    return mcp


# =============================================================================
# Example 6: Async Tools
# =============================================================================


async def async_example():
    """Demonstrate async tools."""
    print("\n" + "=" * 60)
    print("Async Tools")
    print("=" * 60)

    # Async tools work the same way
    async def fetch_data(url: str) -> str:
        """Fetch data from a URL (simulated).

        Args:
            url: The URL to fetch

        Returns:
            The fetched content.
        """
        await asyncio.sleep(0.1)  # Simulate network delay
        return f"Fetched content from: {url}"

    async def process_async(data: str) -> str:
        """Process data asynchronously.

        Args:
            data: Data to process

        Returns:
            Processed result.
        """
        await asyncio.sleep(0.05)
        return f"Processed: {data.upper()}"

    mcp = mcp_from_functions(
        name="async-tools",
        functions=[fetch_data, process_async],
    )

    print("\n✓ Server created with async tools:")
    print("  - fetch_data (async)")
    print("  - process_async (async)")

    return mcp


# =============================================================================
# Main
# =============================================================================


def main():
    """Run examples or start server based on command line args."""
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--stdio":
            run_stdio()
        elif sys.argv[1] == "--http":
            host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
            port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
            run_http(host, port)
        elif sys.argv[1] == "--help":
            print("MCP Server Example")
            print()
            print("Usage:")
            print("  python 02_server_basic.py           # Show examples")
            print("  python 02_server_basic.py --stdio   # Run with stdio")
            print("  python 02_server_basic.py --http    # Run with HTTP")
            print("  python 02_server_basic.py --http 0.0.0.0 8080  # Custom host/port")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Show examples
        print("\n" + "=" * 60)
        print("MCP Server Examples")
        print("=" * 60)
        print("\nThis example shows how to create MCP servers.")
        print("Run with --stdio or --http to start a real server.\n")

        create_simple_server()
        create_server_with_security()
        create_server_with_tooldef()
        asyncio.run(async_example())

        print("\n" + "=" * 60)
        print("To start a server:")
        print("  python 02_server_basic.py --stdio   # For Claude Desktop")
        print("  python 02_server_basic.py --http    # For web clients")
        print("=" * 60)


if __name__ == "__main__":
    main()
