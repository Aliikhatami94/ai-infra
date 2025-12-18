"""Callbacks and hooks for ai-infra operations.

Provides a callback system for observing and responding to events during:
- LLM calls (start, end, error, tokens)
- Tool execution (start, end, error)
- MCP operations (connect, disconnect, tool calls)
- Graph execution (node start, end, error)

Usage:
    from ai_infra.callbacks import Callbacks, CallbackManager

    class MyCallbacks(Callbacks):
        def on_llm_start(self, provider: str, model: str, messages: list[dict]):
            print(f"Starting LLM call to {provider}/{model}")

        def on_llm_end(self, response: str, usage: dict):
            print(f"LLM call complete, tokens: {usage.get('total_tokens')}")

    # Use with LLM
    callbacks = CallbackManager([MyCallbacks()])
    llm = LLM(callbacks=callbacks)
"""

from __future__ import annotations

import time
from abc import ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

# Type for callback functions
CallbackFn = Callable[..., None]
AsyncCallbackFn = Callable[..., Any]  # Coroutine

T = TypeVar("T")


# =============================================================================
# Event Types
# =============================================================================


@dataclass
class LLMStartEvent:
    """Event fired when LLM call starts."""

    provider: str
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    stream: bool = False
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMEndEvent:
    """Event fired when LLM call completes."""

    provider: str
    model: str
    response: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = 0
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    cached: bool = False
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMErrorEvent:
    """Event fired when LLM call fails."""

    provider: str
    model: str
    error: Exception
    error_type: str = ""
    latency_ms: float = 0
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.error_type:
            self.error_type = type(self.error).__name__


@dataclass
class LLMTokenEvent:
    """Event fired for each streaming token."""

    provider: str
    model: str
    token: str
    index: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolStartEvent:
    """Event fired when tool execution starts."""

    tool_name: str
    arguments: dict[str, Any]
    server_name: str | None = None
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolEndEvent:
    """Event fired when tool execution completes."""

    tool_name: str
    result: Any
    server_name: str | None = None
    latency_ms: float = 0
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolErrorEvent:
    """Event fired when tool execution fails."""

    tool_name: str
    error: Exception
    arguments: dict[str, Any]
    server_name: str | None = None
    error_type: str = ""
    latency_ms: float = 0
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.error_type:
            self.error_type = type(self.error).__name__


@dataclass
class MCPConnectEvent:
    """Event fired when MCP server connects."""

    server_name: str
    transport: str
    tools_count: int = 0
    resources_count: int = 0
    prompts_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class MCPDisconnectEvent:
    """Event fired when MCP server disconnects."""

    server_name: str
    reason: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class MCPProgressEvent:
    """Event fired when MCP tool reports progress.

    This event is fired during long-running MCP tool executions
    when the tool reports incremental progress.

    Example:
        class MyCallbacks(Callbacks):
            async def on_mcp_progress_async(self, event: MCPProgressEvent):
                print(f"[{event.server_name}/{event.tool_name}] {event.progress:.0%}")
    """

    server_name: str
    tool_name: str | None
    progress: float
    total: float | None = None
    message: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class MCPLoggingEvent:
    """Event fired when MCP server sends a log message.

    This event is fired when an MCP server emits logging notifications
    during tool execution or other operations.

    Example:
        class MyCallbacks(Callbacks):
            async def on_mcp_logging_async(self, event: MCPLoggingEvent):
                print(f"[{event.server_name}] {event.level}: {event.data}")
    """

    server_name: str
    tool_name: str | None
    level: str  # debug, info, warning, error
    data: Any
    logger_name: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class GraphNodeStartEvent:
    """Event fired when graph node execution starts."""

    node_id: str
    node_type: str
    inputs: dict[str, Any]
    step: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class GraphNodeEndEvent:
    """Event fired when graph node execution completes."""

    node_id: str
    node_type: str
    outputs: dict[str, Any]
    step: int = 0
    latency_ms: float = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class GraphNodeErrorEvent:
    """Event fired when graph node execution fails."""

    node_id: str
    node_type: str
    error: Exception
    step: int = 0
    latency_ms: float = 0
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Callback Utilities
# =============================================================================


def normalize_callbacks(
    callbacks: Any | None,
) -> CallbackManager | None:
    """Convert callbacks to CallbackManager.

    Accepts either a single Callbacks instance or a CallbackManager,
    and normalizes to CallbackManager for consistent dispatch.

    This is a shared utility used by LLM, Agent, and MCPClient to normalize
    callback parameters.

    Args:
        callbacks: Single callback handler, CallbackManager, or None

    Returns:
        CallbackManager or None

    Raises:
        ValueError: If callbacks is not None, Callbacks, or CallbackManager

    Example:
        >>> cb = normalize_callbacks(MyCallbacks())
        >>> isinstance(cb, CallbackManager)
        True
        >>> normalize_callbacks(None)
        None
    """
    if callbacks is None:
        return None

    if isinstance(callbacks, CallbackManager):
        return callbacks
    if isinstance(callbacks, Callbacks):
        return CallbackManager([callbacks])
    raise ValueError(
        f"Invalid callbacks type: {type(callbacks)}. "
        "Expected Callbacks or CallbackManager instance."
    )


# =============================================================================
# Callbacks Base Class
# =============================================================================


class Callbacks(ABC):
    """Base class for callback handlers.

    Override methods to receive events. All methods have default no-op
    implementations, so you only need to override the ones you care about.

    Example:
        class MetricsCallbacks(Callbacks):
            def __init__(self):
                self.total_tokens = 0
                self.call_count = 0

            def on_llm_end(self, event: LLMEndEvent):
                self.call_count += 1
                if event.total_tokens:
                    self.total_tokens += event.total_tokens
    """

    # LLM events
    def on_llm_start(self, event: LLMStartEvent) -> None:
        """Called when LLM call starts."""
        pass

    def on_llm_end(self, event: LLMEndEvent) -> None:
        """Called when LLM call completes."""
        pass

    def on_llm_error(self, event: LLMErrorEvent) -> None:
        """Called when LLM call fails."""
        pass

    def on_llm_token(self, event: LLMTokenEvent) -> None:
        """Called for each streaming token."""
        pass

    # Tool events
    def on_tool_start(self, event: ToolStartEvent) -> None:
        """Called when tool execution starts."""
        pass

    def on_tool_end(self, event: ToolEndEvent) -> None:
        """Called when tool execution completes."""
        pass

    def on_tool_error(self, event: ToolErrorEvent) -> None:
        """Called when tool execution fails."""
        pass

    # MCP events
    def on_mcp_connect(self, event: MCPConnectEvent) -> None:
        """Called when MCP server connects."""
        pass

    def on_mcp_disconnect(self, event: MCPDisconnectEvent) -> None:
        """Called when MCP server disconnects."""
        pass

    def on_mcp_progress(self, event: MCPProgressEvent) -> None:
        """Called when MCP tool reports progress.

        Override this for sync callbacks. For async callbacks,
        override on_mcp_progress_async instead.
        """
        pass

    async def on_mcp_progress_async(self, event: MCPProgressEvent) -> None:
        """Async version of on_mcp_progress.

        Called during async MCP operations. Default implementation
        calls the sync version.

        Example:
            class MyCallbacks(Callbacks):
                async def on_mcp_progress_async(self, event):
                    await notify_user(f"Progress: {event.progress:.0%}")
        """
        self.on_mcp_progress(event)

    def on_mcp_logging(self, event: MCPLoggingEvent) -> None:
        """Called when MCP server sends log message.

        Override this for sync callbacks. For async callbacks,
        override on_mcp_logging_async instead.
        """
        pass

    async def on_mcp_logging_async(self, event: MCPLoggingEvent) -> None:
        """Async version of on_mcp_logging.

        Called during async MCP operations. Default implementation
        calls the sync version.
        """
        self.on_mcp_logging(event)

    # Graph events
    def on_graph_node_start(self, event: GraphNodeStartEvent) -> None:
        """Called when graph node execution starts."""
        pass

    def on_graph_node_end(self, event: GraphNodeEndEvent) -> None:
        """Called when graph node execution completes."""
        pass

    def on_graph_node_error(self, event: GraphNodeErrorEvent) -> None:
        """Called when graph node execution fails."""
        pass


# =============================================================================
# Callback Manager
# =============================================================================


class CallbackManager:
    """Manages multiple callback handlers.

    Dispatches events to all registered callbacks. Errors in callbacks
    are caught and logged, not propagated.

    Example:
        manager = CallbackManager([
            LoggingCallbacks(),
            MetricsCallbacks(),
        ])

        # Fire event
        manager.on_llm_start(LLMStartEvent(...))

        # Or use context manager for timing
        with manager.llm_call("openai", "gpt-4o", messages) as ctx:
            response = await do_llm_call()
            ctx.set_response(response, tokens=150)
    """

    def __init__(
        self,
        callbacks: Sequence[Callbacks] | None = None,
        critical_callbacks: Sequence[Callbacks] | None = None,
    ):
        """Initialize CallbackManager.

        Args:
            callbacks: List of callback handlers (errors logged but not propagated)
            critical_callbacks: List of critical callback handlers (errors propagate).
                Use for security audit callbacks that MUST succeed.
        """
        self._callbacks: list[Callbacks] = list(callbacks or [])
        self._critical_callbacks: list[Callbacks] = list(critical_callbacks or [])

    def add(self, callback: Callbacks, critical: bool = False) -> None:
        """Add a callback handler.

        Args:
            callback: Callback handler to add
            critical: If True, errors in this callback will propagate (not swallowed)
        """
        if critical:
            self._critical_callbacks.append(callback)
        else:
            self._callbacks.append(callback)

    def remove(self, callback: Callbacks) -> None:
        """Remove a callback handler."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass
        try:
            self._critical_callbacks.remove(callback)
        except ValueError:
            pass

    def _dispatch(self, method: str, event: Any) -> None:
        """Dispatch event to all callbacks.

        Critical callbacks are called first and errors propagate.
        Regular callbacks have errors logged but not propagated.
        """
        import logging

        logger = logging.getLogger("ai_infra.callbacks")

        # Critical callbacks first - errors propagate
        for callback in self._critical_callbacks:
            handler = getattr(callback, method, None)
            if handler:
                try:
                    handler(event)
                except Exception:
                    logger.error(
                        f"Critical callback error in {callback.__class__.__name__}.{method}",
                        exc_info=True,
                    )
                    raise  # Propagate critical callback errors

        # Regular callbacks - errors logged but not propagated
        for callback in self._callbacks:
            try:
                handler = getattr(callback, method, None)
                if handler:
                    handler(event)
            except Exception as e:
                logger.warning(f"Callback error in {callback.__class__.__name__}.{method}: {e}")

    async def _dispatch_async(self, method: str, event: Any) -> None:
        """Dispatch async event to all callbacks.

        Critical callbacks are called first and errors propagate.
        Regular callbacks have errors logged but not propagated.
        """
        import logging

        logger = logging.getLogger("ai_infra.callbacks")

        # Critical callbacks first - errors propagate
        for callback in self._critical_callbacks:
            handler = getattr(callback, method, None)
            if handler:
                try:
                    await handler(event)
                except Exception:
                    logger.error(
                        f"Critical callback error in {callback.__class__.__name__}.{method}",
                        exc_info=True,
                    )
                    raise  # Propagate critical callback errors

        # Regular callbacks - errors logged but not propagated
        for callback in self._callbacks:
            try:
                handler = getattr(callback, method, None)
                if handler:
                    await handler(event)
            except Exception as e:
                logger.warning(f"Callback error in {callback.__class__.__name__}.{method}: {e}")

    # LLM events
    def on_llm_start(self, event: LLMStartEvent) -> None:
        self._dispatch("on_llm_start", event)

    async def on_llm_start_async(self, event: LLMStartEvent) -> None:
        await self._dispatch_async("on_llm_start_async", event)

    def on_llm_end(self, event: LLMEndEvent) -> None:
        self._dispatch("on_llm_end", event)

    async def on_llm_end_async(self, event: LLMEndEvent) -> None:
        await self._dispatch_async("on_llm_end_async", event)

    def on_llm_error(self, event: LLMErrorEvent) -> None:
        self._dispatch("on_llm_error", event)

    async def on_llm_error_async(self, event: LLMErrorEvent) -> None:
        await self._dispatch_async("on_llm_error_async", event)

    def on_llm_token(self, event: LLMTokenEvent) -> None:
        self._dispatch("on_llm_token", event)

    async def on_llm_token_async(self, event: LLMTokenEvent) -> None:
        await self._dispatch_async("on_llm_token_async", event)

    # Tool events
    def on_tool_start(self, event: ToolStartEvent) -> None:
        self._dispatch("on_tool_start", event)

    async def on_tool_start_async(self, event: ToolStartEvent) -> None:
        await self._dispatch_async("on_tool_start_async", event)

    def on_tool_end(self, event: ToolEndEvent) -> None:
        self._dispatch("on_tool_end", event)

    async def on_tool_end_async(self, event: ToolEndEvent) -> None:
        await self._dispatch_async("on_tool_end_async", event)

    def on_tool_error(self, event: ToolErrorEvent) -> None:
        self._dispatch("on_tool_error", event)

    async def on_tool_error_async(self, event: ToolErrorEvent) -> None:
        await self._dispatch_async("on_tool_error_async", event)

    # MCP events
    def on_mcp_connect(self, event: MCPConnectEvent) -> None:
        self._dispatch("on_mcp_connect", event)

    def on_mcp_disconnect(self, event: MCPDisconnectEvent) -> None:
        self._dispatch("on_mcp_disconnect", event)

    def on_mcp_progress(self, event: MCPProgressEvent) -> None:
        self._dispatch("on_mcp_progress", event)

    async def on_mcp_progress_async(self, event: MCPProgressEvent) -> None:
        await self._dispatch_async("on_mcp_progress_async", event)

    def on_mcp_logging(self, event: MCPLoggingEvent) -> None:
        self._dispatch("on_mcp_logging", event)

    async def on_mcp_logging_async(self, event: MCPLoggingEvent) -> None:
        await self._dispatch_async("on_mcp_logging_async", event)

    # Graph events
    def on_graph_node_start(self, event: GraphNodeStartEvent) -> None:
        self._dispatch("on_graph_node_start", event)

    def on_graph_node_end(self, event: GraphNodeEndEvent) -> None:
        self._dispatch("on_graph_node_end", event)

    def on_graph_node_error(self, event: GraphNodeErrorEvent) -> None:
        self._dispatch("on_graph_node_error", event)

    # Context managers for auto-timing
    def llm_call(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        **extra: Any,
    ) -> LLMCallContext:
        """Context manager for LLM calls with auto-timing."""
        return LLMCallContext(self, provider, model, messages, extra)

    def tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        server_name: str | None = None,
        **extra: Any,
    ) -> ToolCallContext:
        """Context manager for tool calls with auto-timing."""
        return ToolCallContext(self, tool_name, arguments, server_name, extra)


# =============================================================================
# Context Managers for Auto-Timing
# =============================================================================


class LLMCallContext:
    """Context manager for tracking LLM call lifecycle."""

    def __init__(
        self,
        manager: CallbackManager,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        extra: dict[str, Any],
    ):
        self._manager = manager
        self._provider = provider
        self._model = model
        self._messages = messages
        self._extra = extra
        self._start_time = 0.0
        self._response: str | None = None
        self._tokens: dict[str, int] | None = None
        self._error: Exception | None = None

    def __enter__(self) -> LLMCallContext:
        self._start_time = time.time()
        self._manager.on_llm_start(
            LLMStartEvent(
                provider=self._provider,
                model=self._model,
                messages=self._messages,
                extra=self._extra,
            )
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        latency_ms = (time.time() - self._start_time) * 1000

        if exc_val is not None:
            self._manager.on_llm_error(
                LLMErrorEvent(
                    provider=self._provider,
                    model=self._model,
                    error=exc_val,
                    latency_ms=latency_ms,
                )
            )
        elif self._response is not None:
            self._manager.on_llm_end(
                LLMEndEvent(
                    provider=self._provider,
                    model=self._model,
                    response=self._response,
                    input_tokens=self._tokens.get("input") if self._tokens else None,
                    output_tokens=self._tokens.get("output") if self._tokens else None,
                    total_tokens=self._tokens.get("total") if self._tokens else None,
                    latency_ms=latency_ms,
                )
            )

    def set_response(
        self,
        response: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Set the response for the end event."""
        self._response = response
        self._tokens = {}
        if input_tokens is not None:
            self._tokens["input"] = input_tokens
        if output_tokens is not None:
            self._tokens["output"] = output_tokens
        if total_tokens is not None:
            self._tokens["total"] = total_tokens


class ToolCallContext:
    """Context manager for tracking tool call lifecycle."""

    def __init__(
        self,
        manager: CallbackManager,
        tool_name: str,
        arguments: dict[str, Any],
        server_name: str | None,
        extra: dict[str, Any],
    ):
        self._manager = manager
        self._tool_name = tool_name
        self._arguments = arguments
        self._server_name = server_name
        self._extra = extra
        self._start_time = 0.0
        self._result: Any = None
        self._has_result = False

    def __enter__(self) -> ToolCallContext:
        self._start_time = time.time()
        self._manager.on_tool_start(
            ToolStartEvent(
                tool_name=self._tool_name,
                arguments=self._arguments,
                server_name=self._server_name,
                extra=self._extra,
            )
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        latency_ms = (time.time() - self._start_time) * 1000

        if exc_val is not None:
            self._manager.on_tool_error(
                ToolErrorEvent(
                    tool_name=self._tool_name,
                    error=exc_val,
                    arguments=self._arguments,
                    server_name=self._server_name,
                    latency_ms=latency_ms,
                )
            )
        elif self._has_result:
            self._manager.on_tool_end(
                ToolEndEvent(
                    tool_name=self._tool_name,
                    result=self._result,
                    server_name=self._server_name,
                    latency_ms=latency_ms,
                )
            )

    def set_result(self, result: Any) -> None:
        """Set the result for the end event."""
        self._result = result
        self._has_result = True


# =============================================================================
# Built-in Callback Implementations
# =============================================================================


class LoggingCallbacks(Callbacks):
    """Callback that logs all events to the ai-infra logger."""

    def __init__(self, level: str = "DEBUG"):
        import logging

        self._logger = logging.getLogger("ai_infra.callbacks")
        self._level = getattr(logging, level.upper(), logging.DEBUG)

    def on_llm_start(self, event: LLMStartEvent) -> None:
        self._logger.log(
            self._level,
            f"LLM call started: {event.provider}/{event.model}",
        )

    def on_llm_end(self, event: LLMEndEvent) -> None:
        self._logger.log(
            self._level,
            f"LLM call completed: {event.provider}/{event.model} "
            f"tokens={event.total_tokens} latency={event.latency_ms:.0f}ms",
        )

    def on_llm_error(self, event: LLMErrorEvent) -> None:
        self._logger.error(
            f"LLM call failed: {event.provider}/{event.model} "
            f"error={event.error_type}: {event.error}",
        )

    def on_tool_start(self, event: ToolStartEvent) -> None:
        self._logger.log(
            self._level,
            f"Tool call started: {event.tool_name}",
        )

    def on_tool_end(self, event: ToolEndEvent) -> None:
        self._logger.log(
            self._level,
            f"Tool call completed: {event.tool_name} latency={event.latency_ms:.0f}ms",
        )

    def on_tool_error(self, event: ToolErrorEvent) -> None:
        self._logger.error(
            f"Tool call failed: {event.tool_name} error={event.error_type}: {event.error}",
        )


class MetricsCallbacks(Callbacks):
    """Callback that collects metrics about operations."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.llm_errors = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.tool_calls = 0
        self.tool_errors = 0
        self.tool_latency_ms = 0.0

    def on_llm_end(self, event: LLMEndEvent) -> None:
        self.llm_calls += 1
        if event.total_tokens:
            self.total_tokens += event.total_tokens
        self.total_latency_ms += event.latency_ms

    def on_llm_error(self, event: LLMErrorEvent) -> None:
        self.llm_errors += 1
        self.total_latency_ms += event.latency_ms

    def on_tool_end(self, event: ToolEndEvent) -> None:
        self.tool_calls += 1
        self.tool_latency_ms += event.latency_ms

    def on_tool_error(self, event: ToolErrorEvent) -> None:
        self.tool_errors += 1
        self.tool_latency_ms += event.latency_ms

    def get_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        return {
            "llm": {
                "calls": self.llm_calls,
                "errors": self.llm_errors,
                "total_tokens": self.total_tokens,
                "avg_latency_ms": self.total_latency_ms / max(self.llm_calls, 1),
            },
            "tools": {
                "calls": self.tool_calls,
                "errors": self.tool_errors,
                "avg_latency_ms": self.tool_latency_ms / max(self.tool_calls, 1),
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.llm_calls = 0
        self.llm_errors = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.tool_calls = 0
        self.tool_errors = 0
        self.tool_latency_ms = 0.0


class PrintCallbacks(Callbacks):
    """Simple callback that prints events to stdout.

    Useful for debugging and development.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def on_llm_start(self, event: LLMStartEvent) -> None:
        print(f"LLM call started: {event.provider}/{event.model}")

    def on_llm_end(self, event: LLMEndEvent) -> None:
        tokens = f" ({event.total_tokens} tokens)" if event.total_tokens else ""
        print(f"LLM call completed in {event.latency_ms:.0f}ms{tokens}")
        if self.verbose and event.response:
            print(f"   Response: {event.response[:100]}...")

    def on_llm_error(self, event: LLMErrorEvent) -> None:
        print(f"[ERROR] LLM call failed: {event.error}")

    def on_tool_start(self, event: ToolStartEvent) -> None:
        print(f"Tool call started: {event.tool_name}")
        if self.verbose:
            print(f"   Args: {event.arguments}")

    def on_tool_end(self, event: ToolEndEvent) -> None:
        print(f"Tool call completed in {event.latency_ms:.0f}ms")

    def on_tool_error(self, event: ToolErrorEvent) -> None:
        print(f"[ERROR] Tool call failed: {event.tool_name}: {event.error}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base
    "Callbacks",
    "CallbackManager",
    # Utility
    "normalize_callbacks",
    # Events
    "LLMStartEvent",
    "LLMEndEvent",
    "LLMErrorEvent",
    "LLMTokenEvent",
    "ToolStartEvent",
    "ToolEndEvent",
    "ToolErrorEvent",
    "MCPConnectEvent",
    "MCPDisconnectEvent",
    "MCPProgressEvent",
    "MCPLoggingEvent",
    "GraphNodeStartEvent",
    "GraphNodeEndEvent",
    "GraphNodeErrorEvent",
    # Context managers
    "LLMCallContext",
    "ToolCallContext",
    # Built-in callbacks
    "LoggingCallbacks",
    "MetricsCallbacks",
    "PrintCallbacks",
]
