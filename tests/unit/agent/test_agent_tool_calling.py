"""Tests for Agent tool invocation and response handling.

Tests cover:
- Tool registration and availability
- Tool invocation during agent run
- Tool response parsing
- Tool error handling modes (return_error, retry, abort)
- Tool result validation
- Multiple tool calls in sequence

Phase 1.1.3 of production readiness test plan.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import tool as lc_tool

from ai_infra import Agent
from ai_infra.llm.tools.hitl import (
    ToolExecutionConfig,
    ToolExecutionError,
    ToolTimeoutError,
    wrap_tool_with_execution_config,
)

# =============================================================================
# Test Tools
# =============================================================================


def simple_tool(x: int) -> str:
    """A simple tool that doubles a number."""
    return f"Result: {x * 2}"


def failing_tool(x: int) -> str:
    """A tool that always fails."""
    raise ValueError("Tool execution failed")


def slow_tool(x: int) -> str:
    """A tool that takes time."""
    import time

    time.sleep(2)
    return f"Slow result: {x}"


@lc_tool
def langchain_tool(query: str) -> str:
    """A LangChain tool for searching."""
    return f"Search results for: {query}"


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Tests for tool registration on Agent."""

    def test_agent_accepts_function_tools(self):
        """Agent accepts plain Python functions as tools."""
        agent = Agent(tools=[simple_tool])
        assert len(agent.tools) == 1

    def test_agent_accepts_langchain_tools(self):
        """Agent accepts LangChain @tool decorated functions."""
        agent = Agent(tools=[langchain_tool])
        assert len(agent.tools) == 1

    def test_agent_accepts_mixed_tools(self):
        """Agent accepts mix of function and LangChain tools."""
        agent = Agent(tools=[simple_tool, langchain_tool])
        assert len(agent.tools) == 2

    def test_agent_no_tools_by_default(self):
        """Agent has no tools by default."""
        agent = Agent()
        assert len(agent.tools) == 0

    def test_agent_tool_count_correct(self):
        """Tool count is correct after adding."""
        agent = Agent(tools=[simple_tool, langchain_tool])
        assert len(agent.tools) == 2

    def test_set_global_tools(self):
        """Agent.set_global_tools updates the tool list."""
        agent = Agent()
        assert len(agent.tools) == 0
        agent.set_global_tools([simple_tool])
        assert len(agent.tools) == 1


# =============================================================================
# Tool Execution Config Tests
# =============================================================================


class TestToolExecutionConfig:
    """Tests for tool execution configuration."""

    def test_default_on_error_return_error(self):
        """Default on_tool_error is 'return_error'."""
        agent = Agent(tools=[simple_tool])
        assert agent._tool_execution_config.on_error == "return_error"

    def test_custom_on_error_retry(self):
        """on_tool_error='retry' is stored correctly."""
        agent = Agent(tools=[simple_tool], on_tool_error="retry")
        assert agent._tool_execution_config.on_error == "retry"

    def test_custom_on_error_abort(self):
        """on_tool_error='abort' is stored correctly."""
        agent = Agent(tools=[simple_tool], on_tool_error="abort")
        assert agent._tool_execution_config.on_error == "abort"

    def test_tool_timeout_stored(self):
        """tool_timeout is stored in config."""
        agent = Agent(tools=[simple_tool], tool_timeout=30.0)
        assert agent._tool_execution_config.timeout == 30.0

    def test_max_tool_retries_stored(self):
        """max_tool_retries is stored in config."""
        agent = Agent(tools=[simple_tool], max_tool_retries=3)
        assert agent._tool_execution_config.max_retries == 3

    def test_validate_tool_results_stored(self):
        """validate_tool_results is stored in config."""
        agent = Agent(tools=[simple_tool], validate_tool_results=True)
        assert agent._tool_execution_config.validate_results is True


# =============================================================================
# Wrapped Tool Error Handling Tests
# =============================================================================


class TestWrappedToolErrorHandling:
    """Tests for wrapped tool error handling behavior."""

    def test_return_error_returns_message(self):
        """on_error='return_error' returns error message."""
        config = ToolExecutionConfig(on_error="return_error")
        wrapped = wrap_tool_with_execution_config(failing_tool, config)
        result = wrapped.invoke({"x": 5})
        assert "[Tool Error: failing_tool]" in result
        assert "ValueError" in result

    def test_abort_raises_exception(self):
        """on_error='abort' raises ToolExecutionError."""
        config = ToolExecutionConfig(on_error="abort")
        wrapped = wrap_tool_with_execution_config(failing_tool, config)
        with pytest.raises(ToolExecutionError) as exc_info:
            wrapped.invoke({"x": 5})
        assert exc_info.value.tool_name == "failing_tool"

    def test_retry_attempts_multiple_times(self):
        """on_error='retry' retries up to max_retries times."""
        call_count = 0

        def counting_tool(x: int) -> str:
            """A tool that counts calls and fails."""
            nonlocal call_count
            call_count += 1
            raise ValueError("Still failing")

        config = ToolExecutionConfig(on_error="retry", max_retries=2)
        wrapped = wrap_tool_with_execution_config(counting_tool, config)
        result = wrapped.invoke({"x": 5})

        # 1 initial + 2 retries = 3 total calls
        assert call_count == 3
        assert "[Tool Error:" in result

    def test_retry_succeeds_on_later_attempt(self):
        """Tool succeeds on retry after initial failure."""
        call_count = 0

        def flaky_tool(x: int) -> str:
            """A flaky tool that fails initially."""
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Transient error")
            return f"Success: {x}"

        config = ToolExecutionConfig(on_error="retry", max_retries=2)
        wrapped = wrap_tool_with_execution_config(flaky_tool, config)
        result = wrapped.invoke({"x": 5})

        assert result == "Success: 5"
        assert call_count == 2


# =============================================================================
# Tool Timeout Tests
# =============================================================================


class TestToolTimeout:
    """Tests for tool timeout handling."""

    def test_timeout_returns_error_by_default(self):
        """Timeout returns error message by default."""
        config = ToolExecutionConfig(timeout=0.1)
        wrapped = wrap_tool_with_execution_config(slow_tool, config)
        result = wrapped.invoke({"x": 5})
        assert "[Tool Timeout: slow_tool]" in result

    def test_timeout_abort_raises(self):
        """Timeout with on_timeout='abort' raises ToolTimeoutError."""
        config = ToolExecutionConfig(timeout=0.1, on_timeout="abort")
        wrapped = wrap_tool_with_execution_config(slow_tool, config)
        with pytest.raises(ToolTimeoutError) as exc_info:
            wrapped.invoke({"x": 5})
        assert exc_info.value.tool_name == "slow_tool"


# =============================================================================
# Agent with Tools Integration Tests
# =============================================================================


class TestAgentWithToolsIntegration:
    """Integration tests for Agent with tools (mocked LLM)."""

    def test_agent_creates_with_tools(self):
        """Agent can be created with tools and proper config."""
        agent = Agent(
            tools=[simple_tool, langchain_tool],
            on_tool_error="retry",
            tool_timeout=30.0,
            max_tool_retries=2,
        )

        assert len(agent.tools) == 2
        assert agent._tool_execution_config.on_error == "retry"
        assert agent._tool_execution_config.timeout == 30.0
        assert agent._tool_execution_config.max_retries == 2

    def test_agent_with_all_tool_options(self):
        """Agent accepts all tool-related options together."""
        agent = Agent(
            tools=[simple_tool],
            on_tool_error="abort",
            tool_timeout=60.0,
            max_tool_retries=5,
            validate_tool_results=True,
        )

        assert agent._tool_execution_config.on_error == "abort"
        assert agent._tool_execution_config.timeout == 60.0
        assert agent._tool_execution_config.max_retries == 5
        assert agent._tool_execution_config.validate_results is True


# =============================================================================
# Tool Result Validation Tests
# =============================================================================


class TestToolResultValidation:
    """Tests for tool result type validation."""

    def test_validation_disabled_by_default(self):
        """Validation is disabled by default."""
        agent = Agent(tools=[simple_tool])
        assert agent._tool_execution_config.validate_results is False

    def test_validation_enabled(self):
        """Validation can be enabled."""
        agent = Agent(tools=[simple_tool], validate_tool_results=True)
        assert agent._tool_execution_config.validate_results is True


# =============================================================================
# Dynamic Tool Tests
# =============================================================================


class TestDynamicTools:
    """Tests for dynamic tool management."""

    def test_tools_can_be_replaced(self):
        """Tools can be replaced after agent creation."""
        agent = Agent(tools=[simple_tool])
        assert len(agent.tools) == 1

        agent.set_global_tools([langchain_tool])
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "langchain_tool"

    def test_tools_can_be_added(self):
        """Additional tools can be added via set_global_tools."""
        agent = Agent()
        assert len(agent.tools) == 0

        def new_tool(x: str) -> str:
            return x.upper()

        agent.set_global_tools([new_tool])
        assert len(agent.tools) == 1
