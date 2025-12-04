"""Callbacks for MCP client operations.

This module provides callback infrastructure for receiving progress updates
and logging notifications from MCP tool execution.

Example:
    ```python
    from ai_infra.mcp import MCPClient, Callbacks

    async def on_progress(progress, total, message, ctx):
        print(f"[{ctx.server_name}/{ctx.tool_name}] {progress}/{total}: {message}")

    mcp = MCPClient([config], callbacks=Callbacks(on_progress=on_progress))
    tools = await mcp.list_tools()
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mcp.types import LoggingMessageNotificationParams


@dataclass
class CallbackContext:
    """Context passed to callbacks for routing and logging.

    Attributes:
        server_name: Name of the MCP server handling the operation.
        tool_name: Name of the tool being executed (None during discovery).
    """

    server_name: str
    tool_name: str | None = None


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress callbacks.

    Called when an MCP tool reports progress during execution.
    Useful for long-running operations that want to report incremental progress.
    """

    async def __call__(
        self,
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ) -> None:
        """Handle progress update from MCP tool.

        Args:
            progress: Current progress value.
            total: Total expected value (None if unknown).
            message: Optional progress message.
            context: Callback context with server and tool info.
        """
        ...


@runtime_checkable
class LoggingCallback(Protocol):
    """Protocol for logging callbacks.

    Called when an MCP server sends log messages during tool execution.
    Useful for debugging and observability.
    """

    async def __call__(
        self,
        params: "LoggingMessageNotificationParams",
        context: CallbackContext,
    ) -> None:
        """Handle log message from MCP server.

        Args:
            params: MCP logging parameters (level, data, logger name).
            context: Callback context with server and tool info.
        """
        ...


@dataclass
class _MCPCallbacks:
    """Internal MCP SDK-compatible callbacks. For internal use only."""

    logging_callback: object | None = None  # LoggingFnT
    progress_callback: object | None = None  # ProgressFnT


@dataclass
class Callbacks:
    """Callbacks for MCP client operations.

    Provides hooks for receiving progress updates and logging notifications
    from MCP tool execution.

    Attributes:
        on_progress: Called when a tool reports progress.
        on_logging: Called when the server sends log messages.

    Example:
        ```python
        async def on_progress(progress, total, message, ctx):
            print(f"[{ctx.tool_name}] {progress:.0%}")

        async def on_logging(params, ctx):
            print(f"[{ctx.server_name}] {params.level}: {params.data}")

        callbacks = Callbacks(
            on_progress=on_progress,
            on_logging=on_logging,
        )
        mcp = MCPClient([config], callbacks=callbacks)
        ```
    """

    on_progress: ProgressCallback | None = None
    on_logging: LoggingCallback | None = None

    def to_mcp_format(self, context: CallbackContext) -> _MCPCallbacks:
        """Convert to MCP SDK-compatible callbacks.

        Creates wrapper functions that inject the CallbackContext into
        the user-provided callbacks.

        Args:
            context: The callback context to inject.

        Returns:
            MCP SDK-compatible callback structure.
        """
        _mcp_logging_callback = None
        _mcp_progress_callback = None

        if self.on_logging is not None:
            on_logging = self.on_logging

            async def _logging_cb(
                params: "LoggingMessageNotificationParams",
            ) -> None:
                await on_logging(params, context)

            _mcp_logging_callback = _logging_cb

        if self.on_progress is not None:
            on_progress = self.on_progress

            async def _progress_cb(
                progress: float,
                total: float | None,
                message: str | None,
            ) -> None:
                await on_progress(progress, total, message, context)

            _mcp_progress_callback = _progress_cb

        return _MCPCallbacks(
            logging_callback=_mcp_logging_callback,
            progress_callback=_mcp_progress_callback,
        )
