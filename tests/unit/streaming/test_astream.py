"""Tests for Agent.astream normalized streaming events."""

from typing import Any, Iterable, Tuple

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from ai_infra.llm.agent import Agent
from ai_infra.llm.streaming import StreamConfig


class DummyAgent(Agent):
    """Agent with a stubbed token stream for astream()."""

    def __init__(self, token_stream: Iterable[Tuple[Any, Any]]):
        super().__init__()
        self._token_stream = list(token_stream)

    async def astream_agent_tokens(self, *args, **kwargs):
        for token in self._token_stream:
            yield token


@pytest.mark.asyncio
async def test_astream_emits_tool_events_and_done():
    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "index": 0,
                "id": "call-1",
                "name": "search_docs",
                "args": '{"query": "pricing"}',
            }
        ],
    )
    tool_result = ToolMessage(content="result", tool_call_id="call-1", name="search_docs")
    final_chunk = AIMessageChunk(content="all done")

    agent = DummyAgent([(tool_call_chunk, {}), (tool_result, {}), (final_chunk, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="debug"),
        )
    ]

    assert [event.type for event in events] == [
        "thinking",
        "tool_start",
        "tool_end",
        "token",
        "done",
    ]
    tool_start = events[1]
    assert tool_start.tool == "search_docs"
    assert tool_start.arguments == {"query": "pricing"}
    tool_end = events[2]
    assert tool_end.tool_id == "call-1"
    assert tool_end.preview == "result"
    assert events[-1].tools_called == 1


@pytest.mark.asyncio
async def test_astream_standard_hides_tool_arguments():
    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "index": 0,
                "id": "call-2",
                "name": "search_docs",
                "args": '{"query": "pricing"}',
            }
        ],
    )
    tool_result = ToolMessage(content="result", tool_call_id="call-2", name="search_docs")

    agent = DummyAgent([(tool_call_chunk, {}), (tool_result, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="standard"),
        )
    ]

    tool_start_events = [event for event in events if event.type == "tool_start"]
    assert len(tool_start_events) == 1
    assert tool_start_events[0].arguments is None


@pytest.mark.asyncio
async def test_astream_minimal_visibility_only_tokens():
    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"index": 0, "id": "call-3", "name": "search_docs", "args": "{}"}],
    )
    tool_result = ToolMessage(content="done", tool_call_id="call-3", name="search_docs")
    token_chunk = AIMessageChunk(content="final")

    agent = DummyAgent([(tool_call_chunk, {}), (tool_result, {}), (token_chunk, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="minimal"),
        )
    ]

    assert [event.type for event in events] == ["token"]
    assert events[0].content == "final"


@pytest.mark.asyncio
async def test_astream_deduplicates_tool_starts():
    first_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"index": 0, "id": "call-4", "name": "search_docs", "args": "{}"}],
    )
    duplicate_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"index": 0, "id": "call-4", "name": "search_docs", "args": "{}"}],
    )

    agent = DummyAgent([(first_chunk, {}), (duplicate_chunk, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="standard"),
        )
    ]

    tool_start_events = [event for event in events if event.type == "tool_start"]
    assert len(tool_start_events) == 1
