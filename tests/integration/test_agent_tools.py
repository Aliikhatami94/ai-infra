"""Integration tests for Agent with Tools.

These tests verify Agent functionality with real LLM providers and tools.
They require an LLM API key and make real API calls.

Run with: pytest tests/integration/test_agent_tools.py -v
"""

from __future__ import annotations

import os

import pytest

from ai_infra import Agent

# Skip marker for tests requiring OpenAI API key (most reliable for agents)
SKIP_NO_OPENAI = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

# Skip marker for tests requiring Anthropic API key
SKIP_NO_ANTHROPIC = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

# Skip if no LLM provider is available
SKIP_NO_LLM = pytest.mark.skipif(
    not (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    ),
    reason="No LLM API key set",
)


def get_available_provider() -> tuple[str, str]:
    """Get an available provider and model for testing."""
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-3-5-haiku-latest"
    elif os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "google_genai", "gemini-2.0-flash-exp"
    else:
        return "openai", "gpt-4o-mini"  # Default


@SKIP_NO_LLM
@pytest.mark.integration
class TestAgentWithTools:
    """Integration tests for Agent with tool calling."""

    def test_agent_with_simple_tool(self):
        """Test agent using a simple calculation tool."""
        provider, model = get_available_provider()

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        agent = Agent(
            tools=[add],
            provider=provider,
            model_name=model,
        )

        result = agent.run("What is 15 + 27? Use the add tool.")

        assert result is not None
        # The result should contain 42
        assert "42" in str(result)

    def test_agent_with_multiple_tools(self):
        """Test agent using multiple tools."""
        provider, model = get_available_provider()

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers together."""
            return a * b

        agent = Agent(
            tools=[add, multiply],
            provider=provider,
            model_name=model,
        )

        result = agent.run("What is 5 times 6? Use the multiply tool.")

        assert result is not None
        assert "30" in str(result)

    @pytest.mark.asyncio
    async def test_agent_async_run(self):
        """Test agent running asynchronously."""
        provider, model = get_available_provider()

        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"

        agent = Agent(
            tools=[greet],
            provider=provider,
            model_name=model,
        )

        result = await agent.arun("Greet Alice using the greet tool.")

        assert result is not None
        assert "Hello" in str(result) or "Alice" in str(result)


@SKIP_NO_LLM
@pytest.mark.integration
class TestAgentMultiStep:
    """Integration tests for multi-step agent reasoning."""

    def test_multi_step_calculation(self):
        """Test agent performing multi-step reasoning."""
        provider, model = get_available_provider()

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers together."""
            return a * b

        agent = Agent(
            tools=[add, multiply],
            provider=provider,
            model_name=model,
            max_iterations=5,
        )

        # This requires the agent to use multiple tools
        result = agent.run(
            "First add 10 and 5, then multiply the result by 2. "
            "Use the add tool first, then the multiply tool. "
            "What is the final answer?"
        )

        assert result is not None
        # 10 + 5 = 15, 15 * 2 = 30
        assert "30" in str(result)


@SKIP_NO_LLM
@pytest.mark.integration
class TestAgentErrorHandling:
    """Integration tests for agent error handling."""

    def test_tool_error_handled(self):
        """Test that tool errors are handled gracefully."""
        provider, model = get_available_provider()

        def failing_tool(x: int) -> int:
            """A tool that fails for negative numbers."""
            if x < 0:
                raise ValueError("Cannot process negative numbers")
            return x * 2

        agent = Agent(
            tools=[failing_tool],
            provider=provider,
            model_name=model,
            max_iterations=3,
        )

        # The agent should handle the error and report it
        result = agent.run("Use the failing_tool with the value -5. Report what happens.")

        assert result is not None
        # Should mention the error or failure
        result_str = str(result).lower()
        assert "error" in result_str or "negative" in result_str or "cannot" in result_str

    def test_no_infinite_loops(self):
        """Test that agent doesn't loop infinitely."""
        provider, model = get_available_provider()

        call_count = 0

        def counter() -> int:
            """Increment and return a counter."""
            nonlocal call_count
            call_count += 1
            return call_count

        agent = Agent(
            tools=[counter],
            provider=provider,
            model_name=model,
            max_iterations=3,  # Limit iterations
        )

        # This prompt might cause looping, but should be limited
        agent.run("Keep calling the counter tool until you get 100.")

        # Should stop before 100 due to iteration limit
        assert call_count <= 10  # Well under 100


@SKIP_NO_LLM
@pytest.mark.integration
class TestAgentRecursionLimit:
    """Integration tests for agent recursion limit enforcement."""

    def test_recursion_limit_enforced(self):
        """Test that recursion limit is enforced."""
        provider, model = get_available_provider()

        def echo(message: str) -> str:
            """Echo back a message."""
            return f"Echo: {message}"

        agent = Agent(
            tools=[echo],
            provider=provider,
            model_name=model,
            max_iterations=2,  # Very low limit
        )

        # Run and verify it completes
        result = agent.run("Call the echo tool 10 times with different messages.")

        assert result is not None
        # Should have stopped due to iteration limit

    def test_default_recursion_limit(self):
        """Test that default recursion limit exists."""
        provider, model = get_available_provider()

        def noop() -> str:
            """Do nothing."""
            return "done"

        agent = Agent(
            tools=[noop],
            provider=provider,
            model_name=model,
            # No explicit max_iterations - should use default
        )

        # Agent should have a reasonable default
        assert agent.max_iterations is not None or hasattr(agent, "_max_iterations")


@SKIP_NO_OPENAI
@pytest.mark.integration
class TestAgentWithOpenAI:
    """OpenAI-specific agent integration tests."""

    def test_openai_function_calling(self):
        """Test OpenAI native function calling."""

        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            # Mock weather data
            return f"The weather in {city} is sunny, 72°F"

        agent = Agent(
            tools=[get_weather],
            provider="openai",
            model_name="gpt-4o-mini",
        )

        result = agent.run("What's the weather like in San Francisco?")

        assert result is not None
        assert "San Francisco" in str(result) or "sunny" in str(result).lower()


@SKIP_NO_ANTHROPIC
@pytest.mark.integration
class TestAgentWithAnthropic:
    """Anthropic-specific agent integration tests."""

    def test_anthropic_tool_use(self):
        """Test Anthropic tool use."""

        def calculate_tip(bill: float, percentage: float = 18.0) -> float:
            """Calculate tip amount for a bill."""
            return round(bill * (percentage / 100), 2)

        agent = Agent(
            tools=[calculate_tip],
            provider="anthropic",
            model_name="claude-3-5-haiku-latest",
        )

        result = agent.run("Calculate a 20% tip on a $50 bill.")

        assert result is not None
        assert "10" in str(result)  # 20% of 50 = 10
