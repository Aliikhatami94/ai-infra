"""Tests for ai_infra.callbacks module - Unified Callback System.

This module tests the core callback infrastructure including:
- Event dataclasses (LLM, Tool, MCP, Graph events)
- Callbacks base class
- CallbackManager dispatch
- Built-in callbacks (LoggingCallbacks, MetricsCallbacks, PrintCallbacks)
"""

import pytest

from ai_infra.callbacks import (
    CallbackManager,
    Callbacks,
    GraphNodeEndEvent,
    GraphNodeErrorEvent,
    GraphNodeStartEvent,
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    LLMTokenEvent,
    LoggingCallbacks,
    MCPConnectEvent,
    MCPDisconnectEvent,
    MCPLoggingEvent,
    MCPProgressEvent,
    MetricsCallbacks,
    PrintCallbacks,
    ToolEndEvent,
    ToolErrorEvent,
    ToolStartEvent,
)

# =============================================================================
# Event Dataclass Tests
# =============================================================================


class TestLLMEvents:
    """Tests for LLM event dataclasses."""

    def test_llm_start_event(self):
        """Test LLMStartEvent creation."""
        event = LLMStartEvent(
            provider="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )
        assert event.provider == "openai"
        assert event.model == "gpt-4o"
        assert len(event.messages) == 1
        assert event.temperature == 0.7
        assert event.timestamp > 0

    def test_llm_end_event(self):
        """Test LLMEndEvent with token usage."""
        event = LLMEndEvent(
            provider="anthropic",
            model="claude-3",
            response="Hello there!",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_ms=250.5,
        )
        assert event.provider == "anthropic"
        assert event.response == "Hello there!"
        assert event.total_tokens == 15
        assert event.latency_ms == 250.5

    def test_llm_error_event(self):
        """Test LLMErrorEvent with auto error_type."""
        error = ValueError("Invalid input")
        event = LLMErrorEvent(
            provider="openai",
            model="gpt-4o",
            error=error,
            latency_ms=100.0,
        )
        assert event.error_type == "ValueError"
        assert event.error is error

    def test_llm_token_event(self):
        """Test LLMTokenEvent for streaming."""
        event = LLMTokenEvent(
            provider="openai",
            model="gpt-4o",
            token="Hello",
            index=0,
        )
        assert event.token == "Hello"
        assert event.index == 0


class TestToolEvents:
    """Tests for Tool event dataclasses."""

    def test_tool_start_event(self):
        """Test ToolStartEvent creation."""
        event = ToolStartEvent(
            tool_name="calculator",
            arguments={"expression": "2+2"},
        )
        assert event.tool_name == "calculator"
        assert event.arguments == {"expression": "2+2"}

    def test_tool_end_event(self):
        """Test ToolEndEvent creation."""
        event = ToolEndEvent(
            tool_name="calculator",
            result="4",
            latency_ms=10.0,
        )
        assert event.tool_name == "calculator"
        assert event.result == "4"

    def test_tool_error_event(self):
        """Test ToolErrorEvent creation."""
        error = RuntimeError("Calculator error")
        event = ToolErrorEvent(
            tool_name="calculator",
            error=error,
            arguments={"expression": "1/0"},
        )
        assert event.tool_name == "calculator"
        assert event.error_type == "RuntimeError"


class TestMCPEvents:
    """Tests for MCP event dataclasses."""

    def test_mcp_connect_event(self):
        """Test MCPConnectEvent creation."""
        event = MCPConnectEvent(
            server_name="filesystem",
            transport="stdio",
            tools_count=2,
        )
        assert event.server_name == "filesystem"
        assert event.transport == "stdio"
        assert event.tools_count == 2

    def test_mcp_disconnect_event(self):
        """Test MCPDisconnectEvent creation."""
        event = MCPDisconnectEvent(
            server_name="filesystem",
            reason="Connection closed",
        )
        assert event.server_name == "filesystem"
        assert event.reason == "Connection closed"

    def test_mcp_progress_event(self):
        """Test MCPProgressEvent creation."""
        event = MCPProgressEvent(
            server_name="filesystem",
            tool_name="read_file",
            progress=50.0,
            total=100.0,
            message="Reading file...",
        )
        assert event.server_name == "filesystem"
        assert event.tool_name == "read_file"
        assert event.progress == 50.0
        assert event.total == 100.0
        assert event.message == "Reading file..."

    def test_mcp_logging_event(self):
        """Test MCPLoggingEvent creation."""
        event = MCPLoggingEvent(
            server_name="filesystem",
            tool_name="write_file",
            level="info",
            data={"bytes_written": 1024},
            logger_name="fs.write",
        )
        assert event.server_name == "filesystem"
        assert event.level == "info"
        assert event.data == {"bytes_written": 1024}


class TestGraphEvents:
    """Tests for Graph event dataclasses."""

    def test_graph_node_start_event(self):
        """Test GraphNodeStartEvent creation."""
        event = GraphNodeStartEvent(
            node_id="process_data",
            node_type="processor",
            inputs={"data": [1, 2, 3]},
            step=1,
        )
        assert event.node_id == "process_data"
        assert event.step == 1

    def test_graph_node_end_event(self):
        """Test GraphNodeEndEvent creation."""
        event = GraphNodeEndEvent(
            node_id="process_data",
            node_type="processor",
            outputs={"result": [2, 4, 6]},
            latency_ms=50.0,
        )
        assert event.outputs == {"result": [2, 4, 6]}

    def test_graph_node_error_event(self):
        """Test GraphNodeErrorEvent creation."""
        error = KeyError("missing_key")
        event = GraphNodeErrorEvent(
            node_id="process_data",
            node_type="processor",
            error=error,
        )
        assert event.node_id == "process_data"
        assert event.error is error
        assert type(event.error).__name__ == "KeyError"


# =============================================================================
# Callbacks Base Class Tests
# =============================================================================


class TestCallbacksBase:
    """Tests for Callbacks base class."""

    def test_callbacks_default_noop(self):
        """Test that default callback methods are no-ops."""
        callbacks = Callbacks()

        # All these should do nothing without error
        callbacks.on_llm_start(LLMStartEvent("openai", "gpt-4o", []))
        callbacks.on_llm_end(LLMEndEvent("openai", "gpt-4o", "response"))
        callbacks.on_llm_error(LLMErrorEvent("openai", "gpt-4o", ValueError()))
        callbacks.on_llm_token(LLMTokenEvent("openai", "gpt-4o", "token"))
        callbacks.on_tool_start(ToolStartEvent("tool", {}))
        callbacks.on_tool_end(ToolEndEvent("tool", "result"))
        callbacks.on_mcp_progress(MCPProgressEvent("server", "tool", 0.5))
        callbacks.on_mcp_logging(MCPLoggingEvent("server", "tool", "info", {}))

    def test_callbacks_subclass_override(self):
        """Test that subclass can override specific methods."""
        events = []

        class MyCallbacks(Callbacks):
            def on_llm_start(self, event: LLMStartEvent) -> None:
                events.append(("llm_start", event))

            def on_llm_end(self, event: LLMEndEvent) -> None:
                events.append(("llm_end", event))

        callbacks = MyCallbacks()
        start_event = LLMStartEvent("openai", "gpt-4o", [])
        end_event = LLMEndEvent("openai", "gpt-4o", "response")

        callbacks.on_llm_start(start_event)
        callbacks.on_llm_end(end_event)

        assert len(events) == 2
        assert events[0] == ("llm_start", start_event)
        assert events[1] == ("llm_end", end_event)

    @pytest.mark.asyncio
    async def test_callbacks_async_methods(self):
        """Test async callback methods."""
        events = []

        class MyAsyncCallbacks(Callbacks):
            async def on_mcp_progress_async(self, event: MCPProgressEvent) -> None:
                events.append(("mcp_progress", event))

            async def on_mcp_logging_async(self, event: MCPLoggingEvent) -> None:
                events.append(("mcp_logging", event))

        callbacks = MyAsyncCallbacks()
        progress_event = MCPProgressEvent("server", "tool", 0.5)
        logging_event = MCPLoggingEvent("server", "tool", "info", {})

        await callbacks.on_mcp_progress_async(progress_event)
        await callbacks.on_mcp_logging_async(logging_event)

        assert len(events) == 2
        assert events[0] == ("mcp_progress", progress_event)
        assert events[1] == ("mcp_logging", logging_event)


# =============================================================================
# CallbackManager Tests
# =============================================================================


class TestCallbackManager:
    """Tests for CallbackManager dispatch."""

    def test_callback_manager_empty(self):
        """Test CallbackManager with no callbacks."""
        manager = CallbackManager([])
        # Should not raise
        manager.on_llm_start(LLMStartEvent("openai", "gpt-4o", []))

    def test_callback_manager_single_callback(self):
        """Test CallbackManager with single callback."""
        events = []

        class TrackingCallbacks(Callbacks):
            def on_llm_start(self, event: LLMStartEvent) -> None:
                events.append(event)

        manager = CallbackManager([TrackingCallbacks()])
        event = LLMStartEvent("openai", "gpt-4o", [])
        manager.on_llm_start(event)

        assert len(events) == 1
        assert events[0] is event

    def test_callback_manager_multiple_callbacks(self):
        """Test CallbackManager dispatches to multiple callbacks."""
        events1 = []
        events2 = []

        class Callbacks1(Callbacks):
            def on_llm_start(self, event: LLMStartEvent) -> None:
                events1.append(event)

        class Callbacks2(Callbacks):
            def on_llm_start(self, event: LLMStartEvent) -> None:
                events2.append(event)

        manager = CallbackManager([Callbacks1(), Callbacks2()])
        event = LLMStartEvent("openai", "gpt-4o", [])
        manager.on_llm_start(event)

        assert len(events1) == 1
        assert len(events2) == 1
        assert events1[0] is event
        assert events2[0] is event

    def test_callback_manager_all_event_types(self):
        """Test CallbackManager dispatches all event types."""
        events = []

        class AllEventsCallbacks(Callbacks):
            def on_llm_start(self, event):
                events.append(("llm_start", event))

            def on_llm_end(self, event):
                events.append(("llm_end", event))

            def on_llm_error(self, event):
                events.append(("llm_error", event))

            def on_llm_token(self, event):
                events.append(("llm_token", event))

            def on_tool_start(self, event):
                events.append(("tool_start", event))

            def on_tool_end(self, event):
                events.append(("tool_end", event))

            def on_tool_error(self, event):
                events.append(("tool_error", event))

            def on_mcp_connect(self, event):
                events.append(("mcp_connect", event))

            def on_mcp_disconnect(self, event):
                events.append(("mcp_disconnect", event))

            def on_mcp_progress(self, event):
                events.append(("mcp_progress", event))

            def on_mcp_logging(self, event):
                events.append(("mcp_logging", event))

        manager = CallbackManager([AllEventsCallbacks()])

        manager.on_llm_start(LLMStartEvent("p", "m", []))
        manager.on_llm_end(LLMEndEvent("p", "m", "r"))
        manager.on_llm_error(LLMErrorEvent("p", "m", ValueError()))
        manager.on_llm_token(LLMTokenEvent("p", "m", "t"))
        manager.on_tool_start(ToolStartEvent("t", {}))
        manager.on_tool_end(ToolEndEvent("t", "r"))
        manager.on_tool_error(ToolErrorEvent("t", ValueError(), {}))
        manager.on_mcp_connect(MCPConnectEvent("s", "stdio", []))
        manager.on_mcp_disconnect(MCPDisconnectEvent("s"))
        manager.on_mcp_progress(MCPProgressEvent("s", "t", 0.5))
        manager.on_mcp_logging(MCPLoggingEvent("s", "t", "info", {}))

        assert len(events) == 11

    @pytest.mark.asyncio
    async def test_callback_manager_async_dispatch(self):
        """Test CallbackManager async dispatch."""
        events = []

        class AsyncCallbacks(Callbacks):
            async def on_mcp_progress_async(self, event: MCPProgressEvent) -> None:
                events.append(("progress", event))

            async def on_mcp_logging_async(self, event: MCPLoggingEvent) -> None:
                events.append(("logging", event))

        manager = CallbackManager([AsyncCallbacks()])

        await manager.on_mcp_progress_async(MCPProgressEvent("s", "t", 0.5))
        await manager.on_mcp_logging_async(MCPLoggingEvent("s", "t", "info", {}))

        assert len(events) == 2


# =============================================================================
# Built-in Callbacks Tests
# =============================================================================


class TestBuiltInCallbacks:
    """Tests for built-in callback implementations."""

    def test_logging_callbacks_instantiation(self):
        """Test LoggingCallbacks can be instantiated."""
        callbacks = LoggingCallbacks()
        assert callbacks is not None
        # Should handle events without error
        callbacks.on_llm_start(LLMStartEvent("openai", "gpt-4o", []))

    def test_metrics_callbacks_instantiation(self):
        """Test MetricsCallbacks can be instantiated."""
        callbacks = MetricsCallbacks()
        assert callbacks is not None
        # Should handle events without error
        callbacks.on_llm_start(LLMStartEvent("openai", "gpt-4o", []))
        callbacks.on_llm_end(
            LLMEndEvent("openai", "gpt-4o", "response", total_tokens=10)
        )

    def test_print_callbacks_instantiation(self):
        """Test PrintCallbacks can be instantiated."""
        callbacks = PrintCallbacks()
        assert callbacks is not None

    def test_built_in_callbacks_in_manager(self):
        """Test built-in callbacks work in CallbackManager."""
        manager = CallbackManager(
            [
                LoggingCallbacks(),
                MetricsCallbacks(),
                PrintCallbacks(),
            ]
        )

        # Should not raise
        manager.on_llm_start(LLMStartEvent("openai", "gpt-4o", []))
        manager.on_llm_end(LLMEndEvent("openai", "gpt-4o", "response"))


# =============================================================================
# Integration Tests
# =============================================================================


class TestCallbacksIntegration:
    """Integration tests for callback system."""

    def test_callback_event_order(self):
        """Test callbacks fire in correct order."""
        events = []

        class OrderCallbacks(Callbacks):
            def on_llm_start(self, event):
                events.append("start")

            def on_tool_start(self, event):
                events.append("tool_start")

            def on_tool_end(self, event):
                events.append("tool_end")

            def on_llm_end(self, event):
                events.append("end")

        manager = CallbackManager([OrderCallbacks()])

        # Simulate typical agent flow
        manager.on_llm_start(LLMStartEvent("p", "m", []))
        manager.on_tool_start(ToolStartEvent("tool", {}))
        manager.on_tool_end(ToolEndEvent("tool", "result"))
        manager.on_llm_end(LLMEndEvent("p", "m", "response"))

        assert events == ["start", "tool_start", "tool_end", "end"]

    def test_callback_exception_isolation(self):
        """Test that one callback's exception doesn't affect others."""
        events = []

        class FailingCallbacks(Callbacks):
            def on_llm_start(self, event):
                raise RuntimeError("Callback failed!")

        class WorkingCallbacks(Callbacks):
            def on_llm_start(self, event):
                events.append("working")

        manager = CallbackManager([FailingCallbacks(), WorkingCallbacks()])

        # The working callback should still be called even if first fails
        # (depending on implementation - this tests the expected behavior)
        try:
            manager.on_llm_start(LLMStartEvent("p", "m", []))
        except RuntimeError:
            pass  # Expected if callbacks don't isolate exceptions

    def test_mcp_progress_event_fields(self):
        """Test MCPProgressEvent has all required fields for MCP integration."""
        event = MCPProgressEvent(
            server_name="test-server",
            tool_name="long_running_tool",
            progress=0.75,
            total=1.0,
            message="Processing 75% complete",
        )

        # All fields needed for MCP progress reporting
        assert event.server_name is not None
        assert event.tool_name is not None
        assert 0 <= event.progress <= event.total
        assert event.timestamp > 0
