"""Tests for Phase 1.2: Token Streaming.

Tests for streaming LLM tokens through the executor.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_infra.executor.streaming import (
    StreamEventType,
    StreamingConfig,
    create_llm_done_event,
    create_llm_error_event,
    create_llm_thinking_event,
    create_llm_token_event,
    create_llm_tool_end_event,
    create_llm_tool_start_event,
)

# =============================================================================
# StreamEventType LLM Token Tests
# =============================================================================


class TestLLMEventTypes:
    """Tests for LLM token event types."""

    def test_llm_event_types_exist(self) -> None:
        """Test all LLM event types are defined."""
        expected = [
            "LLM_THINKING",
            "LLM_TOKEN",
            "LLM_TOOL_START",
            "LLM_TOOL_END",
            "LLM_DONE",
            "LLM_ERROR",
        ]
        for name in expected:
            assert hasattr(StreamEventType, name)

    def test_llm_event_type_values(self) -> None:
        """Test LLM event types have string values."""
        assert StreamEventType.LLM_THINKING.value == "llm_thinking"
        assert StreamEventType.LLM_TOKEN.value == "llm_token"
        assert StreamEventType.LLM_TOOL_START.value == "llm_tool_start"
        assert StreamEventType.LLM_TOOL_END.value == "llm_tool_end"
        assert StreamEventType.LLM_DONE.value == "llm_done"
        assert StreamEventType.LLM_ERROR.value == "llm_error"


# =============================================================================
# StreamingConfig Token Options Tests
# =============================================================================


class TestStreamingConfigTokens:
    """Tests for StreamingConfig token streaming options."""

    def test_default_token_streaming_off(self) -> None:
        """Test token streaming is off by default."""
        config = StreamingConfig()
        assert config.stream_tokens is False
        assert config.show_llm_thinking is True
        assert config.show_llm_tools is True
        assert config.token_visibility == "standard"

    def test_verbose_enables_tokens(self) -> None:
        """Test verbose mode enables token streaming."""
        config = StreamingConfig.verbose()
        assert config.stream_tokens is True
        assert config.show_llm_thinking is True
        assert config.show_llm_tools is True
        assert config.token_visibility == "detailed"

    def test_minimal_disables_tokens(self) -> None:
        """Test minimal mode disables token streaming."""
        config = StreamingConfig.minimal()
        assert config.stream_tokens is False

    def test_json_output_enables_tokens(self) -> None:
        """Test JSON output enables token streaming."""
        config = StreamingConfig.json_output()
        assert config.stream_tokens is True
        assert config.token_visibility == "detailed"

    def test_tokens_only_config(self) -> None:
        """Test tokens_only() configuration."""
        config = StreamingConfig.tokens_only()
        assert config.stream_tokens is True
        assert config.show_llm_thinking is False
        assert config.show_llm_tools is False
        assert config.show_node_transitions is False
        assert config.show_task_progress is True
        assert config.token_visibility == "minimal"


# =============================================================================
# LLM Event Builder Tests
# =============================================================================


class TestLLMThinkingEvent:
    """Tests for create_llm_thinking_event."""

    def test_basic_thinking_event(self) -> None:
        """Test basic thinking event creation."""
        event = create_llm_thinking_event()
        assert event.event_type == StreamEventType.LLM_THINKING
        assert event.node_name == "execute_task"
        assert event.message == "Agent is thinking..."
        assert event.data == {"model": None}

    def test_thinking_event_with_model(self) -> None:
        """Test thinking event with model name."""
        event = create_llm_thinking_event(model="claude-sonnet-4-20250514")
        assert event.data["model"] == "claude-sonnet-4-20250514"

    def test_thinking_event_custom_node(self) -> None:
        """Test thinking event with custom node name."""
        event = create_llm_thinking_event(node_name="custom_node")
        assert event.node_name == "custom_node"


class TestLLMTokenEvent:
    """Tests for create_llm_token_event."""

    def test_basic_token_event(self) -> None:
        """Test basic token event creation."""
        event = create_llm_token_event("Hello")
        assert event.event_type == StreamEventType.LLM_TOKEN
        assert event.node_name == "execute_task"
        assert event.message == "Hello"
        assert event.data == {"content": "Hello"}

    def test_token_event_with_whitespace(self) -> None:
        """Test token event preserves whitespace."""
        event = create_llm_token_event("  \n\t")
        assert event.message == "  \n\t"
        assert event.data["content"] == "  \n\t"

    def test_token_event_custom_node(self) -> None:
        """Test token event with custom node name."""
        event = create_llm_token_event("test", node_name="my_node")
        assert event.node_name == "my_node"


class TestLLMToolStartEvent:
    """Tests for create_llm_tool_start_event."""

    def test_basic_tool_start(self) -> None:
        """Test basic tool start event."""
        event = create_llm_tool_start_event(tool="search_docs")
        assert event.event_type == StreamEventType.LLM_TOOL_START
        assert event.message == "Calling tool: search_docs"
        assert event.data["tool"] == "search_docs"
        assert "tool_id" not in event.data
        assert "arguments" not in event.data

    def test_tool_start_with_id(self) -> None:
        """Test tool start with tool ID."""
        event = create_llm_tool_start_event(
            tool="search_docs",
            tool_id="call_abc123",
        )
        assert event.data["tool_id"] == "call_abc123"

    def test_tool_start_with_arguments(self) -> None:
        """Test tool start with arguments."""
        args = {"query": "authentication", "limit": 5}
        event = create_llm_tool_start_event(
            tool="search_docs",
            tool_id="call_123",
            arguments=args,
        )
        assert event.data["arguments"] == args


class TestLLMToolEndEvent:
    """Tests for create_llm_tool_end_event."""

    def test_basic_tool_end(self) -> None:
        """Test basic tool end event."""
        event = create_llm_tool_end_event(tool="search_docs")
        assert event.event_type == StreamEventType.LLM_TOOL_END
        assert "search_docs" in event.message
        assert event.data["tool"] == "search_docs"

    def test_tool_end_with_latency(self) -> None:
        """Test tool end with latency."""
        event = create_llm_tool_end_event(
            tool="search_docs",
            latency_ms=234.5,
        )
        assert event.data["latency_ms"] == 234.5
        assert event.duration_ms == 234.5
        assert "234.5ms" in event.message

    def test_tool_end_with_result(self) -> None:
        """Test tool end with result."""
        result = {"results": ["doc1", "doc2"], "count": 2}
        event = create_llm_tool_end_event(
            tool="search_docs",
            tool_id="call_123",
            result=result,
        )
        assert event.data["result"] == result
        assert event.data["tool_id"] == "call_123"

    def test_tool_end_with_preview(self) -> None:
        """Test tool end with preview."""
        event = create_llm_tool_end_event(
            tool="search_docs",
            preview="Found 2 results: doc1, doc2...",
        )
        assert event.data["preview"] == "Found 2 results: doc1, doc2..."


class TestLLMDoneEvent:
    """Tests for create_llm_done_event."""

    def test_basic_done_event(self) -> None:
        """Test basic done event."""
        event = create_llm_done_event()
        assert event.event_type == StreamEventType.LLM_DONE
        assert event.message == "Agent response complete"
        assert event.data == {"tools_called": 0}

    def test_done_event_with_tools_called(self) -> None:
        """Test done event with tools called count."""
        event = create_llm_done_event(tools_called=3)
        assert event.data["tools_called"] == 3


class TestLLMErrorEvent:
    """Tests for create_llm_error_event."""

    def test_basic_error_event(self) -> None:
        """Test basic error event."""
        event = create_llm_error_event(error="Connection timeout")
        assert event.event_type == StreamEventType.LLM_ERROR
        assert event.message == "LLM error: Connection timeout"
        assert event.data["error"] == "Connection timeout"


# =============================================================================
# execute_task_streaming Tests
# =============================================================================


class TestExecuteTaskStreaming:
    """Tests for execute_task_streaming function."""

    @pytest.fixture
    def mock_state(self) -> dict[str, Any]:
        """Create mock executor state."""
        mock_task = MagicMock()
        mock_task.id = "task-1"
        mock_task.title = "Test task"
        return {
            "current_task": mock_task,
            "prompt": "Test prompt for the agent",
            "task_plan": {},
            "retry_count": 0,
        }

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create mock agent."""
        agent = MagicMock()
        agent._tools = []
        agent._system_prompt = None
        return agent

    @pytest.mark.asyncio
    async def test_streaming_no_prompt(self) -> None:
        """Test streaming with no prompt yields error."""
        from ai_infra.executor.nodes.execute import execute_task_streaming

        state = {"current_task": None, "prompt": ""}
        events = []
        async for event in execute_task_streaming(state, agent=MagicMock()):
            events.append(event)

        assert len(events) == 1
        assert events[0].event_type == StreamEventType.LLM_ERROR
        assert "No prompt" in events[0].data["error"]

    @pytest.mark.asyncio
    async def test_streaming_no_agent(self) -> None:
        """Test streaming with no agent yields error."""
        from ai_infra.executor.nodes.execute import execute_task_streaming

        state = {"current_task": None, "prompt": "test"}
        events = []
        async for event in execute_task_streaming(state, agent=None):
            events.append(event)

        assert len(events) == 1
        assert events[0].event_type == StreamEventType.LLM_ERROR
        assert "No agent" in events[0].data["error"]

    @pytest.mark.asyncio
    async def test_streaming_tokens(self, mock_state: dict, mock_agent: MagicMock) -> None:
        """Test streaming yields token events."""
        from ai_infra.executor.nodes.execute import execute_task_streaming
        from ai_infra.llm.streaming import StreamEvent

        # Mock agent.astream to yield test events
        async def mock_astream(prompt, visibility="standard"):
            yield StreamEvent(type="thinking", model="test-model")
            yield StreamEvent(type="token", content="Hello")
            yield StreamEvent(type="token", content=" world")
            yield StreamEvent(type="done", tools_called=0)

        mock_agent.astream = mock_astream

        config = StreamingConfig.verbose()
        events = []
        async for event in execute_task_streaming(
            mock_state,
            agent=mock_agent,
            streaming_config=config,
        ):
            events.append(event)

        # Check we got expected events
        event_types = [e.event_type for e in events]
        assert StreamEventType.LLM_THINKING in event_types
        assert StreamEventType.LLM_TOKEN in event_types
        assert StreamEventType.LLM_DONE in event_types

        # Check token content
        token_events = [e for e in events if e.event_type == StreamEventType.LLM_TOKEN]
        assert len(token_events) == 2
        assert token_events[0].data["content"] == "Hello"
        assert token_events[1].data["content"] == " world"

    @pytest.mark.asyncio
    async def test_streaming_tools(self, mock_state: dict, mock_agent: MagicMock) -> None:
        """Test streaming yields tool events."""
        from ai_infra.executor.nodes.execute import execute_task_streaming
        from ai_infra.llm.streaming import StreamEvent

        async def mock_astream(prompt, visibility="standard"):
            yield StreamEvent(type="thinking", model="test-model")
            yield StreamEvent(
                type="tool_start",
                tool="search_docs",
                tool_id="call_1",
                arguments={"query": "test"},
            )
            yield StreamEvent(
                type="tool_end",
                tool="search_docs",
                tool_id="call_1",
                latency_ms=100.0,
                result="Found 2 docs",
            )
            yield StreamEvent(type="token", content="Based on the docs...")
            yield StreamEvent(type="done", tools_called=1)

        mock_agent.astream = mock_astream

        config = StreamingConfig.verbose()
        events = []
        async for event in execute_task_streaming(
            mock_state,
            agent=mock_agent,
            streaming_config=config,
        ):
            events.append(event)

        event_types = [e.event_type for e in events]
        assert StreamEventType.LLM_TOOL_START in event_types
        assert StreamEventType.LLM_TOOL_END in event_types

        # Check tool start data
        tool_start = next(e for e in events if e.event_type == StreamEventType.LLM_TOOL_START)
        assert tool_start.data["tool"] == "search_docs"
        assert tool_start.data["arguments"] == {"query": "test"}  # detailed visibility

        # Check tool end data
        tool_end = next(e for e in events if e.event_type == StreamEventType.LLM_TOOL_END)
        assert tool_end.data["tool"] == "search_docs"
        assert tool_end.data["latency_ms"] == 100.0

    @pytest.mark.asyncio
    async def test_streaming_respects_config(self, mock_state: dict, mock_agent: MagicMock) -> None:
        """Test streaming respects StreamingConfig options."""
        from ai_infra.executor.nodes.execute import execute_task_streaming
        from ai_infra.llm.streaming import StreamEvent

        async def mock_astream(prompt, visibility="standard"):
            yield StreamEvent(type="thinking", model="test-model")
            yield StreamEvent(type="token", content="Hello")
            yield StreamEvent(type="done", tools_called=0)

        mock_agent.astream = mock_astream

        # Test with tokens_only config - should hide thinking
        config = StreamingConfig.tokens_only()
        events = []
        async for event in execute_task_streaming(
            mock_state,
            agent=mock_agent,
            streaming_config=config,
        ):
            events.append(event)

        event_types = [e.event_type for e in events]
        # tokens_only hides thinking events
        assert StreamEventType.LLM_THINKING not in event_types
        assert StreamEventType.LLM_TOKEN in event_types

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, mock_state: dict, mock_agent: MagicMock) -> None:
        """Test streaming handles errors gracefully."""
        from ai_infra.executor.nodes.execute import execute_task_streaming
        from ai_infra.llm.streaming import StreamEvent

        async def mock_astream(prompt, visibility="standard"):
            yield StreamEvent(type="thinking", model="test-model")
            yield StreamEvent(type="error", error="API rate limit exceeded")

        mock_agent.astream = mock_astream

        config = StreamingConfig.verbose()
        events = []
        async for event in execute_task_streaming(
            mock_state,
            agent=mock_agent,
            streaming_config=config,
        ):
            events.append(event)

        # Should have error event
        error_events = [e for e in events if e.event_type == StreamEventType.LLM_ERROR]
        assert len(error_events) == 1
        assert "rate limit" in error_events[0].data["error"]


# =============================================================================
# ExecutorStreamEvent to_dict Tests
# =============================================================================


class TestExecutorStreamEventToDict:
    """Tests for ExecutorStreamEvent.to_dict() with LLM events."""

    def test_token_event_serialization(self) -> None:
        """Test token event serializes correctly."""
        event = create_llm_token_event("Hello")
        data = event.to_dict()

        assert data["event_type"] == "llm_token"
        assert data["node_name"] == "execute_task"
        assert data["message"] == "Hello"
        assert data["data"]["content"] == "Hello"
        assert "timestamp" in data

    def test_tool_event_serialization(self) -> None:
        """Test tool event serializes correctly."""
        event = create_llm_tool_start_event(
            tool="search_docs",
            tool_id="call_123",
            arguments={"query": "test"},
        )
        data = event.to_dict()

        assert data["event_type"] == "llm_tool_start"
        assert data["data"]["tool"] == "search_docs"
        assert data["data"]["tool_id"] == "call_123"
        assert data["data"]["arguments"] == {"query": "test"}
