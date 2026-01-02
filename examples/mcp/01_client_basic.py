#!/usr/bin/env python
"""Basic MCP Client Example.

This example demonstrates:
- Connecting to an MCP server using MCPClient
- Discovering available tools from the server
- Calling tools and handling responses
- Using async context manager for cleanup
- Error handling and timeouts

MCP (Model Context Protocol) is a standard protocol for AI tool discovery
and execution. This client can connect to any MCP-compliant server.

Supported transports:
- stdio: Local process communication (most common for CLI tools)
- sse: Server-Sent Events over HTTP
- streamable_http: HTTP with streaming (recommended for web services)

Required:
- A running MCP server (or use stdio transport with npx)

Example MCP servers:
- npx -y @anthropic/mcp-server-filesystem /tmp
- npx -y @anthropic/mcp-server-github
- Your own FastMCP server
"""

import asyncio
import os

from ai_infra import MCPClient

# =============================================================================
# Example 1: Connect via stdio (Local Process)
# =============================================================================


async def connect_stdio():
    """Connect to an MCP server via stdio (subprocess).

    This is the most common way to use MCP - launching a local process
    that communicates via stdin/stdout.
    """
    print("=" * 60)
    print("1. Connect to MCP Server via stdio")
    print("=" * 60)

    # Filesystem MCP server - gives access to file operations
    # The command launches the server as a subprocess
    config = [
        {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
        }
    ]

    print("\nConnecting to filesystem MCP server...")
    print("Command: npx -y @anthropic/mcp-server-filesystem /tmp")

    try:
        async with MCPClient(config) as mcp:
            # List available tools
            tools = await mcp.list_tools()
            print(f"\n[OK] Connected! Found {len(tools)} tools:")
            for tool in tools:
                name = getattr(tool, "name", str(tool))
                desc = getattr(tool, "description", "")[:60]
                print(f"  - {name}: {desc}...")

            # Call a tool
            print("\n-> Calling 'list_directory' tool...")
            result = await mcp.call_tool("list_directory", {"path": "/tmp"})
            print(f"Result: {str(result)[:200]}...")

    except FileNotFoundError:
        print("[X] npx not found. Install Node.js to run this example.")
    except Exception as e:
        print(f"[X] Error connecting: {e}")


# =============================================================================
# Example 2: Connect via HTTP (streamable_http)
# =============================================================================


async def connect_http():
    """Connect to an MCP server via HTTP with streaming.

    This is recommended for web-based MCP servers. The server must
    support the streamable_http transport.
    """
    print("\n" + "=" * 60)
    print("2. Connect to MCP Server via HTTP (streamable_http)")
    print("=" * 60)

    # Get URL from environment or use default
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

    config = [
        {
            "transport": "streamable_http",
            "url": mcp_url,
        }
    ]

    print(f"\nConnecting to: {mcp_url}")
    print("Note: You need a running MCP server for this to work.")

    try:
        async with MCPClient(
            config,
            tool_timeout=30.0,  # 30 second timeout for tool calls
            discover_timeout=10.0,  # 10 second timeout for discovery
        ) as mcp:
            tools = await mcp.list_tools()
            print(f"\n[OK] Connected! Found {len(tools)} tools")

            for tool in tools:
                name = getattr(tool, "name", str(tool))
                print(f"  - {name}")

    except Exception as e:
        print(f"[X] Could not connect: {e}")
        print("   Start an MCP server first, or set MCP_SERVER_URL")


# =============================================================================
# Example 3: Connect via SSE (Server-Sent Events)
# =============================================================================


async def connect_sse():
    """Connect to an MCP server via Server-Sent Events.

    SSE is a simpler transport that works well with load balancers
    and proxies that may have trouble with WebSockets.
    """
    print("\n" + "=" * 60)
    print("3. Connect to MCP Server via SSE")
    print("=" * 60)

    mcp_url = os.getenv("MCP_SSE_URL", "http://localhost:8001/mcp")

    config = [
        {
            "transport": "sse",
            "url": mcp_url,
        }
    ]

    print(f"\nConnecting to: {mcp_url}")

    try:
        async with MCPClient(config) as mcp:
            tools = await mcp.list_tools()
            print(f"\n[OK] Connected! Found {len(tools)} tools")

    except Exception as e:
        print(f"[X] Could not connect: {e}")


# =============================================================================
# Example 4: Multi-Server Configuration
# =============================================================================


async def connect_multiple_servers():
    """Connect to multiple MCP servers simultaneously.

    MCPClient can connect to multiple servers at once. Tools from
    different servers are prefixed with the server name to avoid
    conflicts (e.g., "filesystem__read_file", "github__create_issue").
    """
    print("\n" + "=" * 60)
    print("4. Multi-Server Configuration")
    print("=" * 60)

    configs = [
        # Filesystem server (stdio)
        {
            "name": "files",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
        },
        # Another server could be added here:
        # {
        #     "name": "github",
        #     "transport": "sse",
        #     "url": "http://localhost:8001/mcp",
        # },
    ]

    print("\nConfigured servers:")
    for cfg in configs:
        name = cfg.get("name", "unnamed")
        transport = cfg.get("transport", "unknown")
        print(f"  - {name}: {transport}")

    try:
        async with MCPClient(configs) as mcp:
            tools = await mcp.list_tools()
            print(f"\n[OK] Connected to all servers! Total tools: {len(tools)}")

            # Group tools by server prefix
            by_server: dict[str, list[str]] = {}
            for tool in tools:
                name = getattr(tool, "name", str(tool))
                if "__" in name:
                    server, tool_name = name.split("__", 1)
                else:
                    server, tool_name = "default", name
                by_server.setdefault(server, []).append(tool_name)

            for server, tool_names in by_server.items():
                print(f"\n  Server '{server}':")
                for t in tool_names[:5]:  # Show first 5
                    print(f"    - {t}")
                if len(tool_names) > 5:
                    print(f"    ... and {len(tool_names) - 5} more")

    except Exception as e:
        print(f"[X] Error: {e}")


# =============================================================================
# Example 5: Error Handling and Health Checks
# =============================================================================


async def error_handling():
    """Demonstrate error handling patterns."""
    print("\n" + "=" * 60)
    print("5. Error Handling Patterns")
    print("=" * 60)

    from ai_infra.mcp import MCPServerError, MCPTimeoutError

    config = [
        {
            "transport": "streamable_http",
            "url": "http://localhost:9999/mcp",  # Intentionally wrong
        }
    ]

    print("\nAttempting to connect to non-existent server...")

    try:
        mcp = MCPClient(config, discover_timeout=3.0)
        await mcp.discover()
    except MCPTimeoutError as e:
        print(f"⏱  Timeout: {e}")
    except MCPServerError as e:
        print(f" Server error: {e}")
    except Exception as e:
        print(f"[X] Connection failed (expected): {type(e).__name__}: {e}")

    print("\n[OK] Error was caught and handled gracefully")


# =============================================================================
# Example 6: Using Prompts and Resources
# =============================================================================


async def prompts_and_resources():
    """Use MCP prompts and resources.

    MCP servers can also expose:
    - Prompts: Pre-defined prompt templates
    - Resources: Files and data that can be read
    """
    print("\n" + "=" * 60)
    print("6. MCP Prompts and Resources")
    print("=" * 60)

    config = [
        {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
        }
    ]

    print("\nConnecting to filesystem server...")

    try:
        async with MCPClient(config) as mcp:
            # List prompts
            prompts = await mcp.list_prompts()
            print(f"\nPrompts: {len(prompts)} available")
            for p in prompts[:3]:
                print(f"  - {p.name}: {p.description[:50]}...")

            # List resources
            resources = await mcp.list_resources()
            print(f"\nResources: {len(resources)} available")
            for r in resources[:3]:
                print(f"  - {r.uri}: {r.name}")

    except Exception as e:
        print(f"Note: {e}")
        print("(Not all MCP servers support prompts/resources)")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MCP Client Examples")
    print("=" * 60)
    print("\nThis example demonstrates connecting to MCP servers.")
    print("Make sure you have npx installed for stdio examples.\n")

    # Run examples
    await connect_stdio()
    await connect_http()
    # await connect_sse()  # Uncomment if you have an SSE server
    await connect_multiple_servers()
    await error_handling()
    await prompts_and_resources()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
