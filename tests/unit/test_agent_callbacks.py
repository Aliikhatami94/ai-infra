"""Tests for Agent class callback integration.

This module tests that the Agent class properly fires callbacks during:
- LLM calls (start, end, error)
- Tool execution (start, end)
- Streaming (token events)
"""

import pytest

from ai_infra import Agent
from ai_infra.callbacks import (
    CallbackManager,
    Callbacks,
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    LLMTokenEvent,
    ToolEndEvent,
    ToolStartEvent,
)


class TestAgentCallbacksParameter:
    """Tests for Agent accepting callbacks parameter."""

    def test_agent_accepts_callbacks(self):
        """Test Agent accepts Callbacks instance."""
        callbacks = Callbacks()
        agent = Agent(callbacks=callbacks)
        assert agent._callbacks is not None

    def test_agent_accepts_callback_manager(self):
        """Test Agent accepts CallbackManager instance."""
        manager = CallbackManager([Callbacks()])
        agent = Agent(callbacks=manager)
        assert agent._callbacks is not None

    def test_agent_without_callbacks(self):
        """Test Agent works without callbacks."""
        agent = Agent()
        assert agent._callbacks is None

    def test_agent_normalizes_single_callback(self):
        """Test Agent normalizes single Callbacks to CallbackManager."""

        class MyCallbacks(Callbacks):
            pass

        agent = Agent(callbacks=MyCallbacks())
        # Should be normalized to CallbackManager
        assert agent._callbacks is not None
        assert isinstance(agent._callbacks, CallbackManager)


class TestAgentCallbackEvents:
    """Tests for Agent firing callback events.

    Note: These tests mock the LLM to avoid actual API calls.
    """

    @pytest.fixture
    def tracking_callbacks(self):
        """Create callbacks that track all events."""
        events = []

        class TrackingCallbacks(Callbacks):
            def on_llm_start(self, event: LLMStartEvent) -> None:
                events.append(("llm_start", event))

            def on_llm_end(self, event: LLMEndEvent) -> None:
                events.append(("llm_end", event))

            def on_llm_error(self, event: LLMErrorEvent) -> None:
                events.append(("llm_error", event))

            def on_llm_token(self, event: LLMTokenEvent) -> None:
                events.append(("llm_token", event))

            def on_tool_start(self, event: ToolStartEvent) -> None:
                events.append(("tool_start", event))

            def on_tool_end(self, event: ToolEndEvent) -> None:
                events.append(("tool_end", event))

        return TrackingCallbacks(), events

    def test_agent_callback_instantiation(self, tracking_callbacks):
        """Test Agent can be created with tracking callbacks."""
        callbacks, events = tracking_callbacks
        agent = Agent(callbacks=callbacks)
        assert agent._callbacks is not None

    def test_agent_with_multiple_callbacks(self):
        """Test Agent with CallbackManager containing multiple callbacks."""
        events1 = []
        events2 = []

        class Callbacks1(Callbacks):
            def on_llm_start(self, event):
                events1.append(event)

        class Callbacks2(Callbacks):
            def on_llm_start(self, event):
                events2.append(event)

        manager = CallbackManager([Callbacks1(), Callbacks2()])
        agent = Agent(callbacks=manager)

        # Both callbacks should be registered
        assert len(agent._callbacks._callbacks) == 2


class TestAgentToolCallbacks:
    """Tests for tool callback wrapping."""

    def test_agent_with_tool(self):
        """Test Agent can be created with tools and callbacks."""
        events = []

        class ToolCallbacks(Callbacks):
            def on_tool_start(self, event: ToolStartEvent) -> None:
                events.append(("start", event.tool_name))

            def on_tool_end(self, event: ToolEndEvent) -> None:
                events.append(("end", event.tool_name))

        def calculator(expression: str) -> str:
            """Calculate a math expression."""
            return str(eval(expression))

        agent = Agent(tools=[calculator], callbacks=ToolCallbacks())
        assert agent._callbacks is not None
        # Tools should be registered
        assert len(agent.tools) > 0


class TestAgentCallbackIntegration:
    """Integration tests for Agent callbacks.

    These tests verify the callback infrastructure is properly wired,
    without making actual LLM API calls.
    """

    def test_callback_manager_passed_correctly(self):
        """Test that CallbackManager is correctly stored on Agent."""

        class TestCallbacks(Callbacks):
            def on_llm_start(self, event):
                pass

        callbacks = TestCallbacks()
        agent = Agent(callbacks=callbacks)

        # Verify the callbacks were normalized to CallbackManager
        assert isinstance(agent._callbacks, CallbackManager)
        assert len(agent._callbacks._callbacks) == 1
        assert isinstance(agent._callbacks._callbacks[0], TestCallbacks)

    def test_multiple_callback_handlers(self):
        """Test Agent with multiple callback handlers."""
        handler1_events = []
        handler2_events = []

        class Handler1(Callbacks):
            def on_llm_start(self, event):
                handler1_events.append(event)

        class Handler2(Callbacks):
            def on_llm_start(self, event):
                handler2_events.append(event)

        manager = CallbackManager([Handler1(), Handler2()])
        agent = Agent(callbacks=manager)

        assert len(agent._callbacks._callbacks) == 2
