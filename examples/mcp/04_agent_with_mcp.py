#!/usr/bin/env python
"""Agent with MCP Tools Example.

This example demonstrates:
- Loading tools from MCP servers into an Agent
- Using cached tool loading for performance
- Combining MCP tools with local function tools
- Production patterns for MCP + Agent integration
- Error handling and fallbacks

This combines the power of:
- Agent: Autonomous tool-using AI
- MCP: Dynamic tool discovery from external servers

Required API Keys (at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
"""

import asyncio
import os

from ai_infra import Agent, MCPClient, load_mcp_tools_cached

# =============================================================================
# Local Tools (Always Available)
# =============================================================================


def get_current_time() -> str:
    """Get the current date and time.

    Returns:
        Current timestamp in ISO format.
    """
    from datetime import datetime

    return datetime.now().isoformat()


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like "2 + 2" or "sqrt(16)"

    Returns:
        The calculation result.
    """
    import math

    allowed = {
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Example 1: Basic Agent with MCP Tools
# =============================================================================


async def basic_agent_with_mcp():
    """Create an agent that uses tools from an MCP server."""
    print("=" * 60)
    print("1. Basic Agent with MCP Tools")
    print("=" * 60)

    # MCP server URL (replace with your actual server)
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

    print(f"\nConnecting to MCP server: {mcp_url}")

    try:
        # Load tools from MCP server (cached for performance)
        mcp_tools = await load_mcp_tools_cached(
            mcp_url,
            transport="streamable_http",
        )

        print(f"[OK] Loaded {len(mcp_tools)} tools from MCP server")

        # Create agent with MCP tools
        agent = Agent(tools=mcp_tools)

        # Use the agent
        result = await agent.arun("What tools do you have available? List them briefly.")
        print(f"\nAgent: {result}")

    except Exception as e:
        print(f"[X] Could not connect to MCP server: {e}")
        print("   Make sure an MCP server is running, or set MCP_SERVER_URL")


# =============================================================================
# Example 2: Combining MCP + Local Tools
# =============================================================================


async def combined_tools():
    """Combine MCP tools with local Python functions."""
    print("\n" + "=" * 60)
    print("2. Combined MCP + Local Tools")
    print("=" * 60)

    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

    # Local tools (always available)
    local_tools = [get_current_time, calculate]

    print("\nLocal tools:")
    for tool in local_tools:
        print(f"  - {tool.__name__}")

    try:
        # Load MCP tools
        mcp_tools = await load_mcp_tools_cached(mcp_url)
        print(f"\nMCP tools: {len(mcp_tools)} loaded")

        # Combine all tools
        all_tools = local_tools + list(mcp_tools)

        # Create agent with combined tools
        agent = Agent(tools=all_tools)

        print(f"\n[OK] Agent created with {len(all_tools)} total tools")

        # The agent can use both local and MCP tools
        result = await agent.arun("What is the current time, and also calculate sqrt(144)?")
        print(f"\nAgent: {result}")

    except Exception as e:
        print(f"\n[!]  MCP unavailable: {e}")
        print("   Falling back to local tools only...")

        # Fallback: agent with only local tools
        agent = Agent(tools=local_tools)
        result = await agent.arun("What is the current time?")
        print(f"\nAgent (local only): {result}")


# =============================================================================
# Example 3: Using MCPClient Directly with Agent
# =============================================================================


async def agent_with_mcp_client():
    """Use MCPClient for more control over MCP connections."""
    print("\n" + "=" * 60)
    print("3. Agent with MCPClient (Direct Control)")
    print("=" * 60)

    # Multi-server configuration
    configs = [
        {
            "name": "filesystem",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
        },
        # Add more servers as needed:
        # {
        #     "name": "docs",
        #     "transport": "streamable_http",
        #     "url": "http://localhost:8001/mcp",
        # },
    ]

    print("\nConfigured MCP servers:")
    for cfg in configs:
        print(f"  - {cfg['name']}: {cfg['transport']}")

    try:
        async with MCPClient(configs) as mcp:
            # Get tools from MCP
            mcp_tools = await mcp.list_tools()
            print(f"\n[OK] Discovered {len(mcp_tools)} tools")

            # Create agent
            agent = Agent(
                tools=list(mcp_tools) + [calculate],
                recursion_limit=10,  # Safety limit
            )

            # Run agent
            result = await agent.arun(
                "List the files in /tmp directory and tell me how many there are."
            )
            print(f"\nAgent: {result}")

    except FileNotFoundError:
        print("[X] npx not found. Install Node.js for this example.")
    except Exception as e:
        print(f"[X] Error: {e}")


# =============================================================================
# Example 4: Production Pattern with Caching
# =============================================================================


async def production_pattern():
    """Production-ready pattern with caching and fallbacks."""
    print("\n" + "=" * 60)
    print("4. Production Pattern (Caching + Fallbacks)")
    print("=" * 60)

    from ai_infra import get_cache_stats

    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

    # Check cache stats
    stats = get_cache_stats()
    print(f"\nCache stats: {stats}")

    # Load with caching (fast on subsequent calls)
    try:
        # First load - may be slow
        tools_1 = await load_mcp_tools_cached(mcp_url, transport="streamable_http")
        print(f"[OK] First load: {len(tools_1)} tools")

        # Second load - instant from cache
        tools_2 = await load_mcp_tools_cached(mcp_url, transport="streamable_http")
        print(f"[OK] Cached load: {len(tools_2)} tools (instant)")

        # Force refresh if needed
        # clear_mcp_cache(mcp_url)
        # tools = await load_mcp_tools_cached(mcp_url, transport="streamable_http")

    except Exception as e:
        print(f"[!]  MCP unavailable: {e}")
        tools_1 = []

    print("\n[OK] Production best practices:")
    print("  - Use load_mcp_tools_cached() for performance")
    print("  - Always have fallback local tools")
    print("  - Set recursion_limit on agents")
    print("  - Use tool_timeout for MCP calls")


# =============================================================================
# Example 5: Agent with Callbacks for MCP Events
# =============================================================================


async def agent_with_callbacks():
    """Use callbacks to observe MCP tool calls."""
    print("\n" + "=" * 60)
    print("5. Agent with Callbacks (Observability)")
    print("=" * 60)

    from ai_infra.callbacks import Callbacks, ToolEndEvent, ToolStartEvent

    class MCPObserver(Callbacks):
        """Observer for MCP tool calls."""

        def on_tool_start(self, event: ToolStartEvent) -> None:
            print(f"   Tool starting: {event.tool_name}")
            if event.tool_args:
                print(f"     Args: {event.tool_args}")

        def on_tool_end(self, event: ToolEndEvent) -> None:
            result_preview = str(event.result)[:100]
            print(f"  [OK] Tool finished: {event.tool_name}")
            print(f"     Result: {result_preview}...")
            if event.latency_ms:
                print(f"     Latency: {event.latency_ms:.0f}ms")

    print("\nUsing callbacks to observe tool execution:")

    # Create agent with callbacks
    agent = Agent(
        tools=[get_current_time, calculate],
        callbacks=MCPObserver(),
    )

    # Run with observation
    result = await agent.arun("What time is it, and what is 25 * 4?")
    print(f"\nFinal result: {result}")


# =============================================================================
# Example 6: Error Handling and Timeouts
# =============================================================================


async def error_handling():
    """Handle MCP errors gracefully."""
    print("\n" + "=" * 60)
    print("6. Error Handling and Timeouts")
    print("=" * 60)

    from ai_infra.mcp import MCPServerError, MCPTimeoutError

    # Bad URL for testing
    bad_url = "http://localhost:9999/mcp"

    print(f"\nAttempting to connect to: {bad_url}")

    try:
        # Set short timeout for faster failure
        async with MCPClient(
            [{"transport": "streamable_http", "url": bad_url}],
            discover_timeout=3.0,
            tool_timeout=5.0,
        ) as mcp:
            await mcp.list_tools()

    except MCPTimeoutError as e:
        print(f"⏱  Timeout error: {e}")
        print("   → Increase timeout or check server responsiveness")

    except MCPServerError as e:
        print(f"🔌 Server error: {e}")
        print("   → Check if server is running")

    except Exception as e:
        print(f"[X] Connection error: {type(e).__name__}: {e}")
        print("   → Server is not reachable")

    print("\n[OK] Error handled gracefully - agent can fall back to local tools")

    # Fallback pattern
    agent = Agent(tools=[get_current_time, calculate])
    result = await agent.arun("What is 2 + 2?")
    print(f"\nFallback agent result: {result}")


# =============================================================================
# Example 7: MCP Tools with HITL (Human-in-the-Loop)
# =============================================================================


async def mcp_with_hitl():
    """Use MCP tools with human approval required."""
    print("\n" + "=" * 60)
    print("7. MCP Tools with Human Approval (HITL)")
    print("=" * 60)

    # Simulating MCP tools that need approval
    def dangerous_operation(action: str) -> str:
        """Perform a dangerous operation (simulated MCP tool).

        Args:
            action: The action to perform

        Returns:
            Result of the action.
        """
        return f"Executed dangerous action: {action}"

    print("\nCreating agent with approval required for dangerous tools...")

    # Agent with approval required
    _agent = Agent(
        tools=[get_current_time, dangerous_operation],
        require_approval=["dangerous_operation"],  # Only this tool needs approval
    )

    print("[OK] Agent created with HITL for 'dangerous_operation'")
    print()
    print("In production, the agent would pause and ask for approval")
    print("before executing any call to 'dangerous_operation'.")
    print()
    print("Approval options:")
    print("  - require_approval=True  # All tools need approval")
    print("  - require_approval=['tool1', 'tool2']  # Specific tools")
    print("  - require_approval=lambda name, args: ...  # Dynamic")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Agent with MCP Tools - Examples")
    print("=" * 60)
    print()
    print("These examples show how to combine AI Agents with MCP tools.")
    print("Make sure you have at least one LLM API key set.")
    print()

    # Run examples
    await basic_agent_with_mcp()
    await combined_tools()
    await agent_with_mcp_client()
    await production_pattern()
    await agent_with_callbacks()
    await error_handling()
    await mcp_with_hitl()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  1. Use load_mcp_tools_cached() for performance")
    print("  2. Combine MCP tools with local tools for reliability")
    print("  3. Always have fallbacks when MCP servers are unavailable")
    print("  4. Use callbacks for observability")
    print("  5. Set timeouts and recursion limits for safety")


if __name__ == "__main__":
    asyncio.run(main())
