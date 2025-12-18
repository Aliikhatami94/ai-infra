"""Callback wrapping utilities for Agent tools.

This module provides functions to wrap tools with callback instrumentation
for observability (start/end/error events).
"""

from __future__ import annotations

import functools
import inspect
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_infra.callbacks import CallbackManager

__all__ = [
    "wrap_tool_with_callbacks",
]


def wrap_tool_with_callbacks(tool: Any, callbacks: CallbackManager) -> Any:
    """Wrap a tool to fire callback events on start/end/error.

    For BaseTool subclasses, we wrap the _run/_arun methods directly.
    For plain functions, we wrap the function itself.

    Args:
        tool: The tool to wrap (can be a function or LangChain tool)
        callbacks: CallbackManager to dispatch events to

    Returns:
        Wrapped tool that fires callback events
    """
    from langchain_core.tools import BaseTool

    from ai_infra.callbacks import ToolEndEvent, ToolErrorEvent, ToolStartEvent

    tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

    # For BaseTool subclasses, wrap _run/_arun methods directly
    # This preserves the tool structure that LangGraph expects
    if isinstance(tool, BaseTool):
        original_run = tool._run
        original_arun = tool._arun if hasattr(tool, "_arun") else None

        def wrapped_run(*args: Any, **kwargs: Any) -> Any:
            callbacks.on_tool_start(ToolStartEvent(tool_name=tool_name, arguments=kwargs))
            start_time = time.time()
            try:
                result = original_run(*args, **kwargs)
                callbacks.on_tool_end(
                    ToolEndEvent(
                        tool_name=tool_name,
                        result=result,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
                return result
            except Exception as e:
                callbacks.on_tool_error(
                    ToolErrorEvent(
                        tool_name=tool_name,
                        error=e,
                        arguments=kwargs,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
                raise

        async def wrapped_arun(*args: Any, **kwargs: Any) -> Any:
            await callbacks.on_tool_start_async(
                ToolStartEvent(tool_name=tool_name, arguments=kwargs)
            )
            start_time = time.time()
            try:
                if original_arun:
                    result = await original_arun(*args, **kwargs)
                else:
                    # Fallback to sync run if no async version
                    result = original_run(*args, **kwargs)
                await callbacks.on_tool_end_async(
                    ToolEndEvent(
                        tool_name=tool_name,
                        result=result,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
                return result
            except Exception as e:
                await callbacks.on_tool_error_async(
                    ToolErrorEvent(
                        tool_name=tool_name,
                        error=e,
                        arguments=kwargs,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
                raise

        # Monkey-patch the methods
        tool._run = wrapped_run  # type: ignore[method-assign]
        tool._arun = wrapped_arun  # type: ignore[method-assign]
        return tool

    # For plain functions, wrap them
    func = getattr(tool, "func", tool)
    is_async = inspect.iscoroutinefunction(func)

    if is_async:
        # Async tool wrapper
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            callbacks.on_tool_start(ToolStartEvent(tool_name=tool_name, arguments=kwargs))
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                callbacks.on_tool_end(
                    ToolEndEvent(
                        tool_name=tool_name,
                        result=result,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
                return result
            except Exception as e:
                callbacks.on_tool_error(
                    ToolErrorEvent(
                        tool_name=tool_name,
                        error=e,
                        arguments=kwargs,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
                raise

        # Preserve tool attributes
        if hasattr(tool, "name"):
            async_wrapper.name = tool.name  # type: ignore[attr-defined]
        if hasattr(tool, "description"):
            async_wrapper.description = tool.description  # type: ignore[attr-defined]
        if hasattr(tool, "args_schema"):
            async_wrapper.args_schema = tool.args_schema  # type: ignore[attr-defined]

        # If it's a LangChain tool, wrap properly
        if hasattr(tool, "func"):
            tool.func = async_wrapper
            return tool
        return async_wrapper
    else:
        # Sync tool wrapper
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            callbacks.on_tool_start(ToolStartEvent(tool_name=tool_name, arguments=kwargs))
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                callbacks.on_tool_end(
                    ToolEndEvent(
                        tool_name=tool_name,
                        result=result,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
                return result
            except Exception as e:
                callbacks.on_tool_error(
                    ToolErrorEvent(
                        tool_name=tool_name,
                        error=e,
                        arguments=kwargs,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
                raise

        # Preserve tool attributes
        if hasattr(tool, "name"):
            sync_wrapper.name = tool.name  # type: ignore[attr-defined]
        if hasattr(tool, "description"):
            sync_wrapper.description = tool.description  # type: ignore[attr-defined]
        if hasattr(tool, "args_schema"):
            sync_wrapper.args_schema = tool.args_schema  # type: ignore[attr-defined]

        # If it's a LangChain tool, wrap properly
        if hasattr(tool, "func"):
            tool.func = sync_wrapper
            return tool
        return sync_wrapper
