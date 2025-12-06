"""Server-Sent Events (SSE) streaming callbacks."""

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Literal

from ai_infra.callbacks import (
    Callbacks,
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    LLMTokenEvent,
    ToolEndEvent,
    ToolStartEvent,
)

VisibilityLevel = Literal["minimal", "standard", "detailed", "debug"]


@dataclass
class SSEEvent:
    """Server-Sent Event."""

    event: str
    data: dict


class SSECallbacks(Callbacks):
    """
    Bridge ai-infra callbacks to Server-Sent Events stream.

    Converts callback events (LLMStartEvent, ToolStartEvent, etc.) into
    SSE-formatted events with configurable visibility levels.

    Example:
        from ai_infra import Agent
        from ai_infra.streaming import SSECallbacks

        callbacks = SSECallbacks(visibility="standard")
        agent = Agent(tools=[...], callbacks=callbacks)

        # Stream events
        async for event in callbacks.stream():
            if event.event == "content_delta":
                print(event.data["delta"])
            elif event.event == "tool_start":
                print(f"Tool: {event.data['name']}")
    """

    def __init__(
        self,
        visibility: VisibilityLevel = "standard",
    ):
        self._visibility = visibility
        self._queue: asyncio.Queue[SSEEvent] = asyncio.Queue()
        self._done = False
        self._stats = {
            "tools_called": 0,
            "total_tokens": 0,
            "start_time": time.time(),
        }

    async def stream(self) -> AsyncIterator[SSEEvent]:
        """Stream SSE events until done."""
        while not self._done or not self._queue.empty():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

        # Final done event
        yield SSEEvent(
            event="done",
            data={
                "tools_called": self._stats["tools_called"],
                "total_tokens": self._stats["total_tokens"],
                "duration_ms": int((time.time() - self._stats["start_time"]) * 1000),
            },
        )

    def _should_emit(self, level: str) -> bool:
        """Check if event should be emitted based on visibility."""
        levels = {"minimal": 0, "standard": 1, "detailed": 2, "debug": 3}
        return levels.get(level, 0) <= levels.get(self._visibility, 1)

    def _emit(self, event: str, data: dict) -> None:
        """Emit an SSE event."""
        self._queue.put_nowait(SSEEvent(event=event, data=data))

    # Callback handlers

    def on_llm_start(self, event: LLMStartEvent) -> None:
        if self._should_emit("debug"):
            self._emit("llm_start", {"provider": event.provider, "model": event.model})

    def on_llm_token(self, event: LLMTokenEvent) -> None:
        # Stream content deltas (always visible)
        if hasattr(event, "delta"):
            self._emit("content_delta", {"delta": event.delta})

    def on_llm_end(self, event: LLMEndEvent) -> None:
        if hasattr(event, "total_tokens") and event.total_tokens:
            self._stats["total_tokens"] += event.total_tokens
        if self._should_emit("debug"):
            self._emit("llm_end", {"tokens": getattr(event, "total_tokens", 0)})

    def on_tool_start(self, event: ToolStartEvent) -> None:
        self._stats["tools_called"] += 1
        if self._should_emit("standard"):
            self._emit(
                "tool_start",
                {
                    "name": event.tool_name,
                    "args": event.arguments if self._should_emit("detailed") else None,
                },
            )

    def on_tool_end(self, event: ToolEndEvent) -> None:
        if self._should_emit("standard"):
            self._emit(
                "tool_end",
                {
                    "name": event.tool_name,
                    "result": event.result if self._should_emit("detailed") else None,
                },
            )

    def on_llm_error(self, event: LLMErrorEvent) -> None:
        self._emit("error", {"message": str(event.error)})
        self._done = True

    def mark_done(self) -> None:
        """Mark stream as complete."""
        self._done = True
