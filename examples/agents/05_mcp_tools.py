#!/usr/bin/env python
"""Agent with MCP (Model Context Protocol) Tools Example.

This example demonstrates:
- Loading tools from MCP servers
- Using MCP tools with Agent
- Caching MCP tool discovery
- Multi-server MCP configurations

MCP allows external tools to be discovered and used dynamically.
Common MCP servers include:
- Filesystem access
- GitHub operations
- Documentation search
- Database queries

Required API Keys:
- OPENAI_API_KEY or ANTHROPIC_API_KEY (for the agent)

Note: This example uses placeholder URLs. Replace with real MCP server URLs.
"""

import asyncio
import os

from ai_infra import Agent, MCPClient, load_mcp_tools_cached

# =============================================================================
# Basic MCP Tool Loading
# =============================================================================


async def basic_mcp_tools():
    """Load tools from an MCP server."""
    print("=" * 60)
    print("Basic MCP Tool Loading")
    print("=" * 60)

    # Example URL - replace with a real MCP server
    # Many documentation sites expose MCP endpoints
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

    print(f"\nAttempting to connect to MCP server: {mcp_url}")
    print("Note: You need a running MCP server for this to work.")
    print()

    try:
        # load_mcp_tools_cached handles caching automatically
        tools = await load_mcp_tools_cached(
            mcp_url,
            transport="streamable_http",  # or "sse", "stdio"
        )

        print(f"✓ Loaded {len(tools)} tools from MCP server")
        for tool in tools:
            name = getattr(tool, "name", str(tool))
            desc = getattr(tool, "description", "")[:80]
            print(f"  - {name}: {desc}")

        # Create agent with MCP tools
        agent = Agent(tools=tools)

        # The agent can now use any of the discovered tools
        result = await agent.arun("What tools do you have available?")
        print(f"\nAgent response: {result}")

    except Exception as e:
        print(f"Could not connect to MCP server: {e}")
        print("This is expected if no MCP server is running.")


# =============================================================================
# MCPClient for Multi-Server Setup
# =============================================================================


async def multi_server_mcp():
    """Connect to multiple MCP servers."""
    print("\n" + "=" * 60)
    print("Multi-Server MCP Configuration")
    print("=" * 60)

    # Configuration for multiple MCP servers
    # Replace these with your actual MCP server configurations
    configs = [
        # Streamable HTTP server (e.g., documentation)
        {
            "transport": "streamable_http",
            "url": os.getenv("MCP_DOCS_URL", "http://localhost:8000/mcp/docs"),
        },
        # SSE server (e.g., GitHub)
        {
            "transport": "sse",
            "url": os.getenv("MCP_GITHUB_URL", "http://localhost:8001/mcp"),
        },
    ]

    print("\nConfigured servers:")
    for cfg in configs:
        print(f"  - {cfg['transport']}: {cfg['url']}")

    try:
        # MCPClient manages multiple server connections
        async with MCPClient(configs, discover_timeout=10.0) as mcp:
            # Discover available servers
            servers = mcp.server_names()
            print(f"\n✓ Discovered {len(servers)} servers: {servers}")

            # List all tools from all servers
            all_tools = await mcp.list_tools()
            print(f"✓ Total tools available: {len(all_tools)}")

            # Create agent with all discovered tools
            agent = Agent(tools=all_tools)

            result = await agent.arun("List the available operations you can perform.")
            print(f"\nAgent response: {result}")

    except Exception as e:
        print(f"\nCould not connect to MCP servers: {e}")
        print("This is expected if no MCP servers are running.")


# =============================================================================
# Stdio MCP Server (NPX/Local Commands)
# =============================================================================


async def stdio_mcp_server():
    """Connect to an MCP server via stdio (subprocess)."""
    print("\n" + "=" * 60)
    print("Stdio MCP Server (NPX)")
    print("=" * 60)

    # Stdio transport runs a local command and communicates via stdin/stdout
    # This is common for npm-based MCP servers
    config = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
    }

    print(f"\nCommand: {config['command']} {' '.join(config['args'])}")
    print("Note: Requires npx (Node.js) to be installed.")
    print()

    try:
        async with MCPClient([config], discover_timeout=30.0) as mcp:
            servers = mcp.server_names()
            print(f"✓ Connected to: {servers}")

            tools = await mcp.list_tools()
            print(f"✓ Available tools: {len(tools)}")
            for tool in tools[:5]:  # Show first 5
                name = getattr(tool, "name", str(tool))
                print(f"  - {name}")

            # Create agent with filesystem tools
            agent = Agent(tools=tools)

            result = await agent.arun("List the files in /tmp directory")
            print(f"\nAgent response: {result[:500]}...")

    except FileNotFoundError:
        print("npx not found. Install Node.js to use stdio MCP servers.")
    except Exception as e:
        print(f"Could not start MCP server: {e}")


# =============================================================================
# Direct Tool Call (Without Agent)
# =============================================================================


async def direct_tool_call():
    """Call MCP tools directly without using an agent."""
    print("\n" + "=" * 60)
    print("Direct MCP Tool Calls")
    print("=" * 60)

    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

    print(f"\nConnecting to: {mcp_url}")

    try:
        configs = [{"transport": "streamable_http", "url": mcp_url}]

        async with MCPClient(configs, tool_timeout=30.0) as mcp:
            servers = mcp.server_names()
            if not servers:
                print("No servers discovered")
                return

            server_name = servers[0]
            print(f"✓ Connected to server: {server_name}")

            # List available tools
            tools = await mcp.list_tools(server=server_name)
            print("✓ Available tools:")
            for tool in tools[:5]:
                name = getattr(tool, "name", str(tool))
                desc = getattr(tool, "description", "")[:60]
                print(f"  - {name}: {desc}")

            if tools:
                # Call a tool directly
                tool_name = getattr(tools[0], "name", "")
                print(f"\n→ Calling tool: {tool_name}")

                result = await mcp.call_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments={},  # Provide required arguments
                )
                print(f"← Result: {result}")

    except Exception as e:
        print(f"Error: {e}")


# =============================================================================
# Cached Tool Loading Pattern
# =============================================================================


async def cached_loading_pattern():
    """Demonstrate cached MCP tool loading."""
    print("\n" + "=" * 60)
    print("Cached MCP Tool Loading")
    print("=" * 60)

    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

    try:
        import time

        # First load - hits the server
        print("\nFirst load (network call)...")
        start = time.time()
        tools1 = await load_mcp_tools_cached(mcp_url)
        elapsed1 = time.time() - start
        print(f"  Loaded {len(tools1)} tools in {elapsed1:.3f}s")

        # Second load - uses cache
        print("\nSecond load (cached)...")
        start = time.time()
        tools2 = await load_mcp_tools_cached(mcp_url)
        elapsed2 = time.time() - start
        print(f"  Loaded {len(tools2)} tools in {elapsed2:.3f}s")

        # Force refresh
        print("\nForce refresh (network call)...")
        start = time.time()
        tools3 = await load_mcp_tools_cached(mcp_url, force_refresh=True)
        elapsed3 = time.time() - start
        print(f"  Loaded {len(tools3)} tools in {elapsed3:.3f}s")

        print(f"\nCache speedup: {elapsed1 / elapsed2:.1f}x faster")

    except Exception as e:
        print(f"Could not connect: {e}")


# =============================================================================
# Agent with MCP and Custom Tools Combined
# =============================================================================


async def combined_tools():
    """Combine MCP tools with custom Python tools."""
    print("\n" + "=" * 60)
    print("Combined MCP + Custom Tools")
    print("=" * 60)

    # Custom Python tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: Mathematical expression like "2 + 2" or "10 * 5".

        Returns:
            Result of the calculation.
        """
        import ast
        import operator

        # Safe math operations
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                return ops[type(node.op)](left, right)
            raise ValueError("Unsupported expression")

        tree = ast.parse(expression, mode="eval")
        result = eval_node(tree.body)
        return str(result)

    def get_current_time() -> str:
        """Get the current date and time."""
        from datetime import datetime

        return datetime.now().isoformat()

    # Custom tools list
    custom_tools = [calculate, get_current_time]
    print(f"Custom tools: {[t.__name__ for t in custom_tools]}")

    # Try to load MCP tools
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
    mcp_tools = []

    try:
        mcp_tools = await load_mcp_tools_cached(mcp_url)
        print(f"MCP tools: {len(mcp_tools)} loaded from {mcp_url}")
    except Exception:
        print("No MCP server available - using custom tools only")

    # Combine all tools
    all_tools = custom_tools + mcp_tools
    print(f"Total tools: {len(all_tools)}")

    # Create agent with combined toolset
    agent = Agent(tools=all_tools)

    # Test with custom tool
    result = await agent.arun("What is 42 * 17?")
    print(f"\nMath result: {result}")

    result = await agent.arun("What time is it?")
    print(f"Time result: {result}")


# =============================================================================
# Health Check and Server Status
# =============================================================================


async def health_check():
    """Check MCP server health."""
    print("\n" + "=" * 60)
    print("MCP Server Health Check")
    print("=" * 60)

    configs = [
        {"transport": "streamable_http", "url": "http://localhost:8000/mcp"},
        {"transport": "streamable_http", "url": "http://localhost:8001/mcp"},
        {"transport": "streamable_http", "url": "http://localhost:9999/mcp"},  # Should fail
    ]

    print("\nChecking server health...")
    mcp = MCPClient(configs, discover_timeout=5.0)

    status = await mcp.health_check()
    print("\nServer Status:")
    for server, health in status.items():
        icon = "✓" if health == "healthy" else "✗"
        print(f"  {icon} {server}: {health}")

    # Get error details for failed servers
    errors = mcp.last_errors()
    if errors:
        print("\nError details:")
        for error in errors:
            print(f"  - {error.get('identity', 'unknown')}: {error.get('error', '')}")


if __name__ == "__main__":
    print("MCP Tools Examples")
    print("=" * 60)
    print("\nNote: These examples require running MCP servers.")
    print("Set MCP_SERVER_URL environment variable to your MCP server.")
    print()

    # Run examples
    asyncio.run(basic_mcp_tools())
    asyncio.run(multi_server_mcp())
    asyncio.run(stdio_mcp_server())
    asyncio.run(direct_tool_call())
    asyncio.run(cached_loading_pattern())
    asyncio.run(combined_tools())
    asyncio.run(health_check())
