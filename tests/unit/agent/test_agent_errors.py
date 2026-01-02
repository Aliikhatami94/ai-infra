"""Tests for Agent error handling.

Tests cover:
- Tool errors and recovery modes
- Recursion limit enforcement
- Timeout handling
- Exception propagation
- Error callback firing

Phase 1.1.3 of production readiness test plan.
"""

from __future__ import annotations

import pytest

from ai_infra import Agent
from ai_infra.callbacks import (
    CallbackManager,
    Callbacks,
    LLMErrorEvent,
    ToolErrorEvent,
)
from ai_infra.llm.tools.hitl import (
    ToolExecutionConfig,
    ToolExecutionError,
    ToolTimeoutError,
    ToolValidationError,
    wrap_tool_with_execution_config,
)

# =============================================================================
# Test Tools
# =============================================================================


def working_tool(x: int) -> str:
    """A working tool."""
    return f"Result: {x}"


def failing_tool(x: int) -> str:
    """A tool that always fails."""
    raise ValueError("Tool failed")


def type_error_tool(x: int) -> str:
    """A tool with type mismatch."""
    return 123  # Should be str


def timeout_tool(x: int) -> str:
    """A tool that times out."""
    import time

    time.sleep(10)
    return "Never returned"


# =============================================================================
# Recursion Limit Tests
# =============================================================================


class TestRecursionLimit:
    """Tests for agent recursion limit configuration."""

    def test_default_recursion_limit(self):
        """Default recursion limit is 50."""
        agent = Agent()
        assert agent._recursion_limit == 50

    def test_custom_recursion_limit(self):
        """Custom recursion limit is stored."""
        agent = Agent(recursion_limit=100)
        assert agent._recursion_limit == 100

    def test_low_recursion_limit(self):
        """Low recursion limit is accepted."""
        agent = Agent(recursion_limit=5)
        assert agent._recursion_limit == 5

    def test_zero_recursion_limit(self):
        """Zero recursion limit is accepted (disables limit)."""
        agent = Agent(recursion_limit=0)
        assert agent._recursion_limit == 0

    def test_large_recursion_limit_warning(self):
        """Large recursion limits should work but are risky."""
        agent = Agent(recursion_limit=1000)
        assert agent._recursion_limit == 1000


# =============================================================================
# Tool Error Mode Tests
# =============================================================================


class TestToolErrorModes:
    """Tests for different tool error handling modes."""

    def test_return_error_mode(self):
        """return_error mode returns error message to agent."""
        config = ToolExecutionConfig(on_error="return_error")
        wrapped = wrap_tool_with_execution_config(failing_tool, config)

        result = wrapped.invoke({"x": 5})

        assert "[Tool Error: failing_tool]" in result
        assert "ValueError" in result
        assert "Tool failed" in result

    def test_abort_mode_raises(self):
        """abort mode raises ToolExecutionError."""
        config = ToolExecutionConfig(on_error="abort")
        wrapped = wrap_tool_with_execution_config(failing_tool, config)

        with pytest.raises(ToolExecutionError) as exc_info:
            wrapped.invoke({"x": 5})

        assert exc_info.value.tool_name == "failing_tool"
        assert isinstance(exc_info.value.original_error, ValueError)

    def test_retry_mode_retries(self):
        """retry mode attempts retries."""
        attempts = []

        def counting_tool(x: int) -> str:
            """A tool that counts and fails."""
            attempts.append(1)
            raise ValueError("Still failing")

        config = ToolExecutionConfig(on_error="retry", max_retries=3)
        wrapped = wrap_tool_with_execution_config(counting_tool, config)

        result = wrapped.invoke({"x": 5})

        # 1 initial + 3 retries = 4 total
        assert len(attempts) == 4
        assert "[Tool Error:" in result


# =============================================================================
# Tool Timeout Tests
# =============================================================================


class TestToolTimeout:
    """Tests for tool timeout handling."""

    def test_timeout_error_message(self):
        """Timeout returns error message by default."""

        def slow_tool(x: int) -> str:
            """A slow tool."""
            import time

            time.sleep(2)
            return "done"

        config = ToolExecutionConfig(timeout=0.1, on_timeout="return_error")
        wrapped = wrap_tool_with_execution_config(slow_tool, config)

        result = wrapped.invoke({"x": 5})

        assert "[Tool Timeout: slow_tool]" in result

    def test_timeout_abort_raises(self):
        """Timeout with abort mode raises ToolTimeoutError."""

        def slow_tool(x: int) -> str:
            """A slow tool."""
            import time

            time.sleep(2)
            return "done"

        config = ToolExecutionConfig(timeout=0.1, on_timeout="abort")
        wrapped = wrap_tool_with_execution_config(slow_tool, config)

        with pytest.raises(ToolTimeoutError) as exc_info:
            wrapped.invoke({"x": 5})

        assert exc_info.value.tool_name == "slow_tool"


# =============================================================================
# Tool Validation Error Tests
# =============================================================================


class TestToolValidationErrors:
    """Tests for tool result validation errors."""

    def test_validation_error_on_wrong_type(self):
        """Wrong return type raises ToolValidationError."""
        config = ToolExecutionConfig(validate_results=True)
        wrapped = wrap_tool_with_execution_config(type_error_tool, config, expected_return_type=str)

        with pytest.raises(ToolValidationError) as exc_info:
            wrapped.invoke({"x": 5})

        assert exc_info.value.tool_name == "type_error_tool"

    def test_validation_disabled_allows_wrong_type(self):
        """Disabled validation allows wrong types."""
        config = ToolExecutionConfig(validate_results=False)
        wrapped = wrap_tool_with_execution_config(type_error_tool, config, expected_return_type=str)

        result = wrapped.invoke({"x": 5})
        assert result == 123  # Wrong type but allowed


# =============================================================================
# Async Error Handling Tests
# =============================================================================


class TestAsyncErrorHandling:
    """Tests for async error handling."""

    @pytest.mark.asyncio
    async def test_async_return_error_mode(self):
        """Async return_error mode returns error message."""
        config = ToolExecutionConfig(on_error="return_error")
        wrapped = wrap_tool_with_execution_config(failing_tool, config)

        result = await wrapped.ainvoke({"x": 5})

        assert "[Tool Error: failing_tool]" in result

    @pytest.mark.asyncio
    async def test_async_abort_mode_raises(self):
        """Async abort mode raises exception."""
        config = ToolExecutionConfig(on_error="abort")
        wrapped = wrap_tool_with_execution_config(failing_tool, config)

        with pytest.raises(ToolExecutionError):
            await wrapped.ainvoke({"x": 5})

    @pytest.mark.asyncio
    async def test_async_timeout_error(self):
        """Async timeout returns error message."""

        async def async_slow(x: int) -> str:
            """An async slow tool."""
            import asyncio

            await asyncio.sleep(2)
            return "done"

        config = ToolExecutionConfig(timeout=0.1)
        wrapped = wrap_tool_with_execution_config(async_slow, config)

        result = await wrapped.ainvoke({"x": 5})

        assert "[Tool Timeout:" in result


# =============================================================================
# Error Callback Tests
# =============================================================================


class TestErrorCallbacks:
    """Tests for error callback firing."""

    def test_agent_accepts_error_callbacks(self):
        """Agent accepts callbacks that handle errors."""
        error_events = []

        class ErrorTracker(Callbacks):
            def on_llm_error(self, event: LLMErrorEvent) -> None:
                error_events.append(event)

            def on_tool_error(self, event: ToolErrorEvent) -> None:
                error_events.append(event)

        agent = Agent(callbacks=ErrorTracker())
        assert agent._callbacks is not None

    def test_callback_manager_stores_error_handlers(self):
        """CallbackManager correctly stores error handlers."""
        error_events = []

        class ErrorTracker(Callbacks):
            def on_tool_error(self, event: ToolErrorEvent) -> None:
                error_events.append(event)

        manager = CallbackManager([ErrorTracker()])
        agent = Agent(callbacks=manager)

        assert len(agent._callbacks._callbacks) == 1


# =============================================================================
# Agent Error Configuration Tests
# =============================================================================


class TestAgentErrorConfiguration:
    """Tests for Agent-level error configuration."""

    def test_agent_on_tool_error_default(self):
        """Default on_tool_error is 'return_error'."""
        agent = Agent()
        assert agent._tool_execution_config.on_error == "return_error"

    def test_agent_on_tool_error_abort(self):
        """on_tool_error='abort' is stored."""
        agent = Agent(on_tool_error="abort")
        assert agent._tool_execution_config.on_error == "abort"

    def test_agent_on_tool_error_retry(self):
        """on_tool_error='retry' is stored."""
        agent = Agent(on_tool_error="retry")
        assert agent._tool_execution_config.on_error == "retry"

    def test_agent_with_all_error_options(self):
        """Agent accepts all error-related options."""
        agent = Agent(
            on_tool_error="retry",
            max_tool_retries=5,
            tool_timeout=30.0,
            validate_tool_results=True,
            recursion_limit=100,
        )

        assert agent._tool_execution_config.on_error == "retry"
        assert agent._tool_execution_config.max_retries == 5
        assert agent._tool_execution_config.timeout == 30.0
        assert agent._tool_execution_config.validate_results is True
        assert agent._recursion_limit == 100
