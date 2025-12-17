"""Tests for MCPClient unified callback integration.

This module tests that the MCPClient class properly uses the unified
callback system for progress and logging events.
"""

import pytest

from ai_infra.callbacks import (
    CallbackManager,
    Callbacks,
    MCPLoggingEvent,
    MCPProgressEvent,
)
from ai_infra.mcp import MCPClient


class TestMCPClientCallbacksParameter:
    """Tests for MCPClient accepting unified callbacks parameter."""

    def test_mcp_client_accepts_callbacks(self):
        """Test MCPClient accepts Callbacks instance."""
        callbacks = Callbacks()
        mcp = MCPClient(
            [{"transport": "stdio", "command": "echo", "args": ["test"]}],
            callbacks=callbacks,
        )
        assert mcp._callbacks is not None

    def test_mcp_client_accepts_callback_manager(self):
        """Test MCPClient accepts CallbackManager instance."""
        manager = CallbackManager([Callbacks()])
        mcp = MCPClient(
            [{"transport": "stdio", "command": "echo", "args": ["test"]}],
            callbacks=manager,
        )
        assert mcp._callbacks is not None

    def test_mcp_client_without_callbacks(self):
        """Test MCPClient works without callbacks."""
        mcp = MCPClient(
            [{"transport": "stdio", "command": "echo", "args": ["test"]}],
        )
        assert mcp._callbacks is None

    def test_mcp_client_normalizes_single_callback(self):
        """Test MCPClient normalizes single Callbacks to CallbackManager."""

        class MyCallbacks(Callbacks):
            pass

        mcp = MCPClient(
            [{"transport": "stdio", "command": "echo", "args": ["test"]}],
            callbacks=MyCallbacks(),
        )
        # Should be normalized to CallbackManager
        assert mcp._callbacks is not None
        assert isinstance(mcp._callbacks, CallbackManager)

    def test_mcp_client_rejects_invalid_callbacks(self):
        """Test MCPClient rejects invalid callbacks type."""
        with pytest.raises(ValueError, match="Invalid callbacks type"):
            MCPClient(
                [{"transport": "stdio", "command": "echo", "args": ["test"]}],
                callbacks="not a callback",  # type: ignore
            )


class TestMCPClientCallbackEvents:
    """Tests for MCPClient firing callback events."""

    @pytest.fixture
    def tracking_callbacks(self):
        """Create callbacks that track MCP events."""
        events = []

        class TrackingCallbacks(Callbacks):
            def on_mcp_progress(self, event: MCPProgressEvent) -> None:
                events.append(("mcp_progress", event))

            async def on_mcp_progress_async(self, event: MCPProgressEvent) -> None:
                events.append(("mcp_progress_async", event))

            def on_mcp_logging(self, event: MCPLoggingEvent) -> None:
                events.append(("mcp_logging", event))

            async def on_mcp_logging_async(self, event: MCPLoggingEvent) -> None:
                events.append(("mcp_logging_async", event))

        return TrackingCallbacks(), events

    def test_mcp_client_callback_instantiation(self, tracking_callbacks):
        """Test MCPClient can be created with tracking callbacks."""
        callbacks, events = tracking_callbacks
        mcp = MCPClient(
            [{"transport": "stdio", "command": "echo", "args": ["test"]}],
            callbacks=callbacks,
        )
        assert mcp._callbacks is not None

    def test_mcp_client_with_multiple_callbacks(self):
        """Test MCPClient with CallbackManager containing multiple callbacks."""
        events1 = []
        events2 = []

        class Callbacks1(Callbacks):
            def on_mcp_progress(self, event):
                events1.append(event)

        class Callbacks2(Callbacks):
            def on_mcp_progress(self, event):
                events2.append(event)

        manager = CallbackManager([Callbacks1(), Callbacks2()])
        mcp = MCPClient(
            [{"transport": "stdio", "command": "echo", "args": ["test"]}],
            callbacks=manager,
        )

        # Both callbacks should be registered
        assert len(mcp._callbacks._callbacks) == 2


class TestMCPClientCallbackIntegration:
    """Integration tests for MCPClient callbacks."""

    def test_callback_manager_passed_correctly(self):
        """Test that CallbackManager is correctly stored on MCPClient."""

        class TestCallbacks(Callbacks):
            async def on_mcp_progress_async(self, event):
                pass

        callbacks = TestCallbacks()
        mcp = MCPClient(
            [{"transport": "stdio", "command": "echo", "args": ["test"]}],
            callbacks=callbacks,
        )

        # Verify the callbacks were normalized to CallbackManager
        assert isinstance(mcp._callbacks, CallbackManager)
        assert len(mcp._callbacks._callbacks) == 1
        assert isinstance(mcp._callbacks._callbacks[0], TestCallbacks)

    def test_multiple_callback_handlers(self):
        """Test MCPClient with multiple callback handlers."""
        handler1_events = []
        handler2_events = []

        class Handler1(Callbacks):
            async def on_mcp_progress_async(self, event):
                handler1_events.append(event)

        class Handler2(Callbacks):
            async def on_mcp_progress_async(self, event):
                handler2_events.append(event)

        manager = CallbackManager([Handler1(), Handler2()])
        mcp = MCPClient(
            [{"transport": "stdio", "command": "echo", "args": ["test"]}],
            callbacks=manager,
        )

        assert len(mcp._callbacks._callbacks) == 2


class TestMCPProgressEvent:
    """Tests for MCPProgressEvent dataclass."""

    def test_mcp_progress_event_creation(self):
        """Test creating MCPProgressEvent with all fields."""
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
        assert event.timestamp > 0

    def test_mcp_progress_event_minimal(self):
        """Test creating MCPProgressEvent with minimal fields."""
        event = MCPProgressEvent(
            server_name="test",
            tool_name=None,
            progress=0.5,
        )
        assert event.server_name == "test"
        assert event.tool_name is None
        assert event.total is None
        assert event.message is None

    def test_mcp_progress_event_percentage(self):
        """Test MCPProgressEvent progress as percentage."""
        event = MCPProgressEvent(
            server_name="test",
            tool_name="task",
            progress=75.0,
            total=100.0,
        )
        # Calculate percentage
        percentage = (event.progress / event.total) * 100
        assert percentage == 75.0


class TestMCPLoggingEvent:
    """Tests for MCPLoggingEvent dataclass."""

    def test_mcp_logging_event_creation(self):
        """Test creating MCPLoggingEvent with all fields."""
        event = MCPLoggingEvent(
            server_name="filesystem",
            tool_name="write_file",
            level="info",
            data={"bytes_written": 1024},
            logger_name="fs.write",
        )
        assert event.server_name == "filesystem"
        assert event.tool_name == "write_file"
        assert event.level == "info"
        assert event.data == {"bytes_written": 1024}
        assert event.logger_name == "fs.write"
        assert event.timestamp > 0

    def test_mcp_logging_event_minimal(self):
        """Test creating MCPLoggingEvent with minimal fields."""
        event = MCPLoggingEvent(
            server_name="test",
            tool_name=None,
            level="debug",
            data="Simple message",
        )
        assert event.server_name == "test"
        assert event.level == "debug"
        assert event.logger_name is None

    def test_mcp_logging_event_levels(self):
        """Test MCPLoggingEvent with different log levels."""
        levels = ["debug", "info", "warning", "error"]
        for level in levels:
            event = MCPLoggingEvent(
                server_name="test",
                tool_name="task",
                level=level,
                data={},
            )
            assert event.level == level
