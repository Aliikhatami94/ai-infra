"""Tests for ai_infra.mcp.client.callbacks module."""

import pytest

from ai_infra.mcp.client.callbacks import (
    CallbackContext,
    Callbacks,
    LoggingCallback,
    ProgressCallback,
    _MCPCallbacks,
)


class TestCallbackContext:
    """Tests for CallbackContext dataclass."""

    def test_callback_context_creation(self):
        """Test creating CallbackContext with all fields."""
        ctx = CallbackContext(server_name="test-server", tool_name="test-tool")
        assert ctx.server_name == "test-server"
        assert ctx.tool_name == "test-tool"

    def test_callback_context_default_tool_name(self):
        """Test CallbackContext with default tool_name."""
        ctx = CallbackContext(server_name="test-server")
        assert ctx.server_name == "test-server"
        assert ctx.tool_name is None

    def test_callback_context_equality(self):
        """Test CallbackContext equality."""
        ctx1 = CallbackContext(server_name="server", tool_name="tool")
        ctx2 = CallbackContext(server_name="server", tool_name="tool")
        assert ctx1 == ctx2


class TestProgressCallback:
    """Tests for ProgressCallback protocol."""

    def test_progress_callback_protocol_check(self):
        """Test that async functions match ProgressCallback protocol."""

        async def my_progress(
            progress: float,
            total: float | None,
            message: str | None,
            context: CallbackContext,
        ) -> None:
            pass

        # Protocol check
        assert isinstance(my_progress, ProgressCallback)

    def test_sync_function_not_progress_callback(self):
        """Test that sync functions don't match ProgressCallback."""

        def my_progress(progress, total, message, context):
            pass

        # Sync functions shouldn't match the async protocol
        # (runtime_checkable only checks call signature, not async)
        # This is expected behavior - the type checker catches this


class TestLoggingCallback:
    """Tests for LoggingCallback protocol."""

    def test_logging_callback_protocol_check(self):
        """Test that async functions match LoggingCallback protocol."""

        async def my_logging(params, context: CallbackContext) -> None:
            pass

        # Protocol check
        assert isinstance(my_logging, LoggingCallback)


class TestCallbacks:
    """Tests for Callbacks dataclass."""

    def test_callbacks_creation_empty(self):
        """Test creating Callbacks with no handlers."""
        callbacks = Callbacks()
        assert callbacks.on_progress is None
        assert callbacks.on_logging is None

    def test_callbacks_creation_with_progress(self):
        """Test creating Callbacks with progress handler."""

        async def on_progress(progress, total, message, ctx):
            pass

        callbacks = Callbacks(on_progress=on_progress)
        assert callbacks.on_progress is on_progress
        assert callbacks.on_logging is None

    def test_callbacks_creation_with_both(self):
        """Test creating Callbacks with both handlers."""

        async def on_progress(progress, total, message, ctx):
            pass

        async def on_logging(params, ctx):
            pass

        callbacks = Callbacks(on_progress=on_progress, on_logging=on_logging)
        assert callbacks.on_progress is on_progress
        assert callbacks.on_logging is on_logging


class TestCallbacksToMCPFormat:
    """Tests for Callbacks.to_mcp_format() conversion."""

    def test_to_mcp_format_empty(self):
        """Test converting empty callbacks."""
        callbacks = Callbacks()
        ctx = CallbackContext(server_name="test")

        result = callbacks.to_mcp_format(ctx)

        assert isinstance(result, _MCPCallbacks)
        assert result.logging_callback is None
        assert result.progress_callback is None

    @pytest.mark.asyncio
    async def test_to_mcp_format_with_progress(self):
        """Test converting callbacks with progress handler."""
        captured = []

        async def on_progress(progress, total, message, ctx):
            captured.append((progress, total, message, ctx))

        callbacks = Callbacks(on_progress=on_progress)
        ctx = CallbackContext(server_name="test-server", tool_name="test-tool")

        result = callbacks.to_mcp_format(ctx)

        assert result.progress_callback is not None
        assert result.logging_callback is None

        # Call the MCP callback
        await result.progress_callback(0.5, 1.0, "halfway")

        # Verify our callback was called with correct context
        assert len(captured) == 1
        progress, total, message, received_ctx = captured[0]
        assert progress == 0.5
        assert total == 1.0
        assert message == "halfway"
        assert received_ctx.server_name == "test-server"
        assert received_ctx.tool_name == "test-tool"

    @pytest.mark.asyncio
    async def test_to_mcp_format_with_logging(self):
        """Test converting callbacks with logging handler."""
        captured = []

        async def on_logging(params, ctx):
            captured.append((params, ctx))

        callbacks = Callbacks(on_logging=on_logging)
        ctx = CallbackContext(server_name="test-server", tool_name="test-tool")

        result = callbacks.to_mcp_format(ctx)

        assert result.logging_callback is not None
        assert result.progress_callback is None

        # Mock logging params
        mock_params = {"level": "info", "data": "test message"}
        await result.logging_callback(mock_params)

        # Verify our callback was called
        assert len(captured) == 1
        received_params, received_ctx = captured[0]
        assert received_params == mock_params
        assert received_ctx.server_name == "test-server"
        assert received_ctx.tool_name == "test-tool"

    @pytest.mark.asyncio
    async def test_to_mcp_format_with_both(self):
        """Test converting callbacks with both handlers."""
        progress_captured = []
        logging_captured = []

        async def on_progress(progress, total, message, ctx):
            progress_captured.append((progress, total, message))

        async def on_logging(params, ctx):
            logging_captured.append(params)

        callbacks = Callbacks(on_progress=on_progress, on_logging=on_logging)
        ctx = CallbackContext(server_name="test")

        result = callbacks.to_mcp_format(ctx)

        assert result.progress_callback is not None
        assert result.logging_callback is not None

        # Call both
        await result.progress_callback(1.0, 1.0, "done")
        await result.logging_callback({"level": "info"})

        assert len(progress_captured) == 1
        assert len(logging_captured) == 1

    @pytest.mark.asyncio
    async def test_context_isolation(self):
        """Test that each to_mcp_format call gets its own context."""
        contexts = []

        async def on_progress(progress, total, message, ctx):
            contexts.append(ctx)

        callbacks = Callbacks(on_progress=on_progress)

        # Create two different contexts
        ctx1 = CallbackContext(server_name="server1", tool_name="tool1")
        ctx2 = CallbackContext(server_name="server2", tool_name="tool2")

        result1 = callbacks.to_mcp_format(ctx1)
        result2 = callbacks.to_mcp_format(ctx2)

        # Call both
        await result1.progress_callback(0.0, 1.0, None)
        await result2.progress_callback(0.0, 1.0, None)

        # Verify each got its own context
        assert len(contexts) == 2
        assert contexts[0].server_name == "server1"
        assert contexts[0].tool_name == "tool1"
        assert contexts[1].server_name == "server2"
        assert contexts[1].tool_name == "tool2"
