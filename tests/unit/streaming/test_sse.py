"""Tests for SSE streaming callbacks."""

import pytest

from ai_infra.callbacks import LLMEndEvent, LLMStartEvent, ToolEndEvent, ToolStartEvent
from ai_infra.streaming import SSECallbacks


@pytest.mark.asyncio
async def test_sse_callbacks_stream():
    """Test SSE event streaming."""
    callbacks = SSECallbacks(visibility="standard")

    # Fire events
    callbacks.on_llm_start(LLMStartEvent(provider="openai", model="gpt-4o", messages=[]))
    callbacks.on_tool_start(ToolStartEvent(tool_name="search", arguments={}))
    callbacks.mark_done()

    # Collect events
    events = []
    async for event in callbacks.stream():
        events.append(event)

    # Check events (tool_start + done at minimum)
    assert len(events) >= 2
    assert events[-1].event == "done"
    assert "tools_called" in events[-1].data


def test_visibility_filtering():
    """Test visibility level filtering."""
    minimal = SSECallbacks(visibility="minimal")
    assert not minimal._should_emit("standard")
    assert not minimal._should_emit("debug")

    standard = SSECallbacks(visibility="standard")
    assert standard._should_emit("standard")
    assert not standard._should_emit("debug")

    debug = SSECallbacks(visibility="debug")
    assert debug._should_emit("debug")
    assert debug._should_emit("standard")


def test_tool_count_tracking():
    """Test that tool calls are counted."""
    callbacks = SSECallbacks(visibility="standard")

    # Call multiple tools
    callbacks.on_tool_start(ToolStartEvent(tool_name="tool1", arguments={}))
    callbacks.on_tool_start(ToolStartEvent(tool_name="tool2", arguments={}))
    callbacks.on_tool_start(ToolStartEvent(tool_name="tool3", arguments={}))

    assert callbacks._stats["tools_called"] == 3


def test_token_tracking():
    """Test that tokens are tracked."""
    callbacks = SSECallbacks(visibility="standard")

    # Fire LLM end events with tokens
    callbacks.on_llm_end(
        LLMEndEvent(provider="openai", model="gpt-4o", response="test", total_tokens=100)
    )
    callbacks.on_llm_end(
        LLMEndEvent(provider="openai", model="gpt-4o", response="test", total_tokens=50)
    )

    assert callbacks._stats["total_tokens"] == 150


@pytest.mark.asyncio
async def test_event_data_structure():
    """Test SSE event data structure."""
    callbacks = SSECallbacks(visibility="detailed")

    callbacks.on_tool_start(ToolStartEvent(tool_name="my_tool", arguments={"arg": "value"}))
    callbacks.mark_done()

    events = []
    async for event in callbacks.stream():
        events.append(event)

    # Find tool_start event
    tool_event = next(e for e in events if e.event == "tool_start")
    assert tool_event.data["name"] == "my_tool"
    assert tool_event.data["args"] == {"arg": "value"}  # Should be visible in detailed mode


@pytest.mark.asyncio
async def test_done_event_structure():
    """Test that done event includes stats."""
    callbacks = SSECallbacks(visibility="standard")

    callbacks.on_tool_start(ToolStartEvent(tool_name="tool1", arguments={}))
    callbacks.on_llm_end(
        LLMEndEvent(provider="openai", model="gpt-4o", response="test", total_tokens=42)
    )
    callbacks.mark_done()

    events = []
    async for event in callbacks.stream():
        events.append(event)

    done_event = events[-1]
    assert done_event.event == "done"
    assert done_event.data["tools_called"] == 1
    assert done_event.data["total_tokens"] == 42
    assert "duration_ms" in done_event.data
