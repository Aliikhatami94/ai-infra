#!/usr/bin/env python
"""Basic Agent with Tools Example.

This example demonstrates:
- Creating an Agent with simple Python functions as tools
- Tool auto-discovery from type hints and docstrings
- Sync and async agent execution
- Tool execution with error handling

Required API Keys (at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
"""

import asyncio
from datetime import datetime

from ai_infra import Agent

# =============================================================================
# Define Simple Tools (Plain Python Functions)
# =============================================================================
# Tools are just functions! ai-infra auto-generates schemas from:
# - Function name -> tool name
# - Docstring -> tool description
# - Type hints -> parameter types
# - Default values -> optional parameters


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name to get weather for.

    Returns:
        A string describing the current weather.
    """
    # In a real app, this would call a weather API
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 55°F",
        "Tokyo": "Rainy, 65°F",
        "Paris": "Partly cloudy, 68°F",
        "Sydney": "Clear, 78°F",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


def get_time(timezone: str = "UTC") -> str:
    """Get the current time in a timezone.

    Args:
        timezone: The timezone name (default: UTC).

    Returns:
        Current time as a string.
    """
    # Simplified - in real app, use pytz or zoneinfo
    return f"Current time in {timezone}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression like "2 + 2" or "sqrt(16)".

    Returns:
        The result as a string.
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
        return f"Error evaluating '{expression}': {e}"


def search_knowledge(query: str, max_results: int = 3) -> str:
    """Search a knowledge base for information.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        Search results as a formatted string.
    """
    # Simulated knowledge base
    knowledge = {
        "python": "Python is a high-level programming language known for readability.",
        "javascript": "JavaScript is a scripting language primarily used for web development.",
        "rust": "Rust is a systems programming language focused on safety and performance.",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
        "machine learning": "ML is a subset of AI that learns from data without explicit programming.",
    }

    results = []
    query_lower = query.lower()
    for key, value in knowledge.items():
        if query_lower in key or key in query_lower:
            results.append(f"- {key.title()}: {value}")
            if len(results) >= max_results:
                break

    if results:
        return "Found:\n" + "\n".join(results)
    return f"No results found for '{query}'"


# =============================================================================
# Basic Agent Usage
# =============================================================================


def main():
    print("=" * 60)
    print("Basic Agent with Tools")
    print("=" * 60)

    # Create agent with tools - it's this simple!
    agent = Agent(tools=[get_weather, get_time, calculate])

    # The agent will automatically call tools when needed
    print("\n1. Simple tool call:")
    result = agent.run("What's the weather in Tokyo?")
    print("   Q: What's the weather in Tokyo?")
    print(f"   A: {result}")

    # Agent can chain multiple tool calls
    print("\n2. Multi-tool query:")
    result = agent.run("What's the weather in London and what time is it there?")
    print("   Q: What's the weather in London and what time is it there?")
    print(f"   A: {result}")

    # Agent uses tools intelligently
    print("\n3. Calculation:")
    result = agent.run("What is the square root of 144 plus 10?")
    print("   Q: What is the square root of 144 plus 10?")
    print(f"   A: {result}")


def agent_with_system_prompt():
    """Agent with custom system prompt."""
    print("\n" + "=" * 60)
    print("Agent with System Prompt")
    print("=" * 60)

    agent = Agent(
        tools=[get_weather, search_knowledge],
        system="You are a helpful research assistant. Always cite your sources.",
    )

    result = agent.run("Tell me about Python programming and check the weather in New York")
    print(f"\nAgent response:\n{result}")


def agent_with_provider_selection():
    """Specify which provider to use."""
    from ai_infra import LLM

    print("\n" + "=" * 60)
    print("Agent with Provider Selection")
    print("=" * 60)

    configured = LLM.list_configured_providers()
    print(f"Configured providers: {configured}")

    if not configured:
        print("No providers configured. Set an API key.")
        return

    # Use specific provider
    provider = configured[0]
    agent = Agent(
        tools=[calculate],
        provider=provider,
    )

    result = agent.run("Calculate 15% of 250")
    print(f"\nUsing {provider}:")
    print(f"Result: {result}")


def agent_with_error_handling():
    """Demonstrate tool error handling."""
    print("\n" + "=" * 60)
    print("Agent Tool Error Handling")
    print("=" * 60)

    def risky_tool(value: int) -> str:
        """A tool that might fail.

        Args:
            value: A number (negative values cause errors).
        """
        if value < 0:
            raise ValueError("Cannot process negative values!")
        return f"Processed: {value * 2}"

    # Default: return_error - agent sees the error and can recover
    agent = Agent(
        tools=[risky_tool],
        on_tool_error="return_error",  # Agent sees error message
    )

    result = agent.run("Process the value -5")
    print("\nWith return_error mode:")
    print(f"Result: {result}")

    # Note: Other modes are "retry" (retry failed tools) and "abort" (raise exception)


def agent_with_timeout():
    """Agent with tool timeout."""
    import time

    print("\n" + "=" * 60)
    print("Agent with Tool Timeout")
    print("=" * 60)

    def slow_tool(seconds: int = 5) -> str:
        """A slow operation that takes time.

        Args:
            seconds: How long to wait.
        """
        time.sleep(seconds)
        return "Done!"

    agent = Agent(
        tools=[slow_tool],
        tool_timeout=2.0,  # 2 second timeout
        on_tool_error="return_error",
    )

    result = agent.run("Run the slow tool for 1 second")
    print("\nWith 2s timeout (1s operation):")
    print(f"Result: {result}")


async def async_agent_example():
    """Async agent execution."""
    print("\n" + "=" * 60)
    print("Async Agent Execution")
    print("=" * 60)

    agent = Agent(tools=[get_weather, calculate])

    # Async run
    result = await agent.arun("What's the weather in Paris?")
    print(f"\nAsync result: {result}")

    # Multiple concurrent agent calls
    print("\nRunning multiple queries concurrently...")
    results = await asyncio.gather(
        agent.arun("Weather in Sydney?"),
        agent.arun("Calculate 7 * 8"),
        agent.arun("Weather in New York?"),
    )

    for i, r in enumerate(results, 1):
        print(f"  Result {i}: {r[:80]}...")


def agent_with_validation():
    """Agent with tool result validation."""
    print("\n" + "=" * 60)
    print("Agent with Tool Validation")
    print("=" * 60)

    def typed_tool(x: int, y: int) -> int:
        """Add two numbers.

        Args:
            x: First number.
            y: Second number.

        Returns:
            The sum as an integer.
        """
        return x + y

    agent = Agent(
        tools=[typed_tool],
        validate_tool_results=True,  # Validate return types match annotations
    )

    result = agent.run("Add 5 and 7")
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
    agent_with_system_prompt()
    agent_with_provider_selection()
    agent_with_error_handling()
    agent_with_timeout()
    asyncio.run(async_agent_example())
    agent_with_validation()
