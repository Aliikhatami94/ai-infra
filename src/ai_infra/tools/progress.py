"""
Tool Progress Streaming: Real-time updates from long-running tools.

This module provides the @progress decorator that allows tools to send
progress updates during execution, which can be consumed via agent streaming.

Example:
    ```python
    from ai_infra.tools import progress

    @progress
    async def analyze_dataset(file: str, stream) -> dict:
        '''Analyze a large dataset with progress updates.'''
        df = load_data(file)

        for i in range(10):
            await stream.update(f"Processing chunk {i+1}/10", percent=(i+1)*10)
            process_chunk(df, i)

        return {"status": "complete", "rows": len(df)}

    # Use with agent
    agent = Agent(tools=[analyze_dataset])

    # Progress visible via streaming
    async for event in agent.astream("Analyze sales.csv"):
        if event.type == "progress":
            print(f"[{event.tool}] {event.message} ({event.percent}%)")
    ```
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool


@dataclass
class ProgressEvent:
    """
    A progress update event from a tool.

    Attributes:
        type: Event type (always "progress")
        tool: Name of the tool sending the update
        message: Human-readable status message
        percent: Optional progress percentage (0-100)
        data: Optional structured data with additional info
    """

    type: str = "progress"
    tool: str = ""
    message: str = ""
    percent: int | None = None
    data: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "type": self.type,
            "tool": self.tool,
            "message": self.message,
        }
        if self.percent is not None:
            result["percent"] = self.percent
        if self.data is not None:
            result["data"] = self.data
        return result


class ProgressStream:
    """
    Stream for sending progress updates from tools.

    This class is injected into progress-enabled tools as the `stream` parameter.
    Tools can call `await stream.update()` to send progress updates.

    Example:
        ```python
        @progress
        async def my_tool(input: str, stream) -> str:
            await stream.update("Starting...", percent=0)
            # ... do work ...
            await stream.update("Halfway there!", percent=50)
            # ... more work ...
            await stream.update("Done!", percent=100)
            return "result"
        ```
    """

    def __init__(
        self,
        tool_name: str,
        callback: Callable[[ProgressEvent], Any] | None = None,
    ):
        """
        Initialize a progress stream.

        Args:
            tool_name: Name of the tool this stream belongs to
            callback: Function to call with progress events.
                Can be sync or async.
        """
        self.tool_name = tool_name
        self._callback = callback
        self._events: list[ProgressEvent] = []

    async def update(
        self,
        message: str,
        percent: int | None = None,
        data: Any | None = None,
    ) -> None:
        """
        Send a progress update.

        Args:
            message: Human-readable status message
            percent: Optional progress percentage (0-100)
            data: Optional structured data with additional info
        """
        event = ProgressEvent(
            type="progress",
            tool=self.tool_name,
            message=message,
            percent=percent,
            data=data,
        )
        self._events.append(event)

        if self._callback:
            result = self._callback(event)
            # Await if callback is async
            if hasattr(result, "__await__"):
                await result

    @property
    def events(self) -> list[ProgressEvent]:
        """Get all recorded events."""
        return self._events.copy()


async def _noop_callback(event: ProgressEvent) -> None:
    """No-op callback for when no progress handler is set."""
    pass


def progress(fn: Callable[..., Any]) -> StructuredTool:
    """
    Decorator to enable progress streaming from a tool.

    The decorated function receives a `stream` parameter that can be used
    to send progress updates during execution.

    The decorator marks the function with `_progress_enabled = True` so the
    Agent can detect it and set up progress handling.

    Args:
        fn: Async function to decorate. Must be an async function.

    Returns:
        StructuredTool with progress streaming support

    Example:
        ```python
        @progress
        async def process_files(pattern: str, stream) -> dict:
            '''Process files matching pattern.'''
            files = glob(pattern)

            for i, file in enumerate(files):
                await stream.update(
                    f"Processing {file}",
                    percent=int((i + 1) / len(files) * 100),
                )
                process_file(file)

            return {"processed": len(files)}
        ```

    Note:
        - The `stream` parameter is injected by the Agent when executing the tool
        - If called directly without a progress callback, updates are silently recorded
        - The function must be async (use `async def`)
    """
    import asyncio
    import inspect
    from typing import get_type_hints

    from langchain_core.tools import StructuredTool
    from pydantic import Field, create_model

    if not asyncio.iscoroutinefunction(fn):
        raise TypeError(
            f"@progress decorator requires an async function, "
            f"but {fn.__name__} is not async. Use 'async def' instead."
        )

    # Build args_schema excluding 'stream' parameter for LangChain compatibility
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    # Build fields for all parameters except 'stream'
    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name == "stream":
            continue  # Skip stream parameter
        if param_name.startswith("_"):
            continue  # Skip private parameters

        param_type = hints.get(param_name, str)

        # Skip ProgressStream type
        if param_type is ProgressStream or (
            hasattr(param_type, "__name__") and param_type.__name__ == "ProgressStream"
        ):
            continue

        if param.default is inspect.Parameter.empty:
            fields[param_name] = (
                param_type,
                Field(..., description=f"The {param_name}"),
            )
        else:
            fields[param_name] = (
                param_type,
                Field(default=param.default, description=f"The {param_name}"),
            )

    # Create args schema model
    args_schema = create_model(f"{fn.__name__.title().replace('_', '')}Args", **fields)

    # Create the wrapper that injects stream
    @wraps(fn)
    async def async_wrapper(**kwargs: Any) -> Any:
        # Get progress callback from kwargs (injected by agent) or use noop
        callback = kwargs.pop("_progress_callback", None)
        stream = ProgressStream(fn.__name__, callback or _noop_callback)

        # Call the function with stream injected
        return await fn(stream=stream, **kwargs)

    # Create sync wrapper for StructuredTool
    # This handles being called from a thread pool without an event loop
    def sync_wrapper(**kwargs: Any) -> Any:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context - need to use thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, async_wrapper(**kwargs))
                return future.result()
        else:
            # No running loop - we can just run it
            return asyncio.run(async_wrapper(**kwargs))

    # Create StructuredTool with explicit args_schema
    tool = StructuredTool.from_function(
        coroutine=async_wrapper,
        func=sync_wrapper,
        name=fn.__name__,
        description=fn.__doc__ or f"Execute {fn.__name__}",
        args_schema=args_schema,
    )

    # Mark as progress-enabled for Agent detection
    tool._progress_enabled = True  # type: ignore
    tool._original_fn = fn  # type: ignore
    tool._async_wrapper = async_wrapper  # type: ignore

    return tool


def is_progress_enabled(tool: Any) -> bool:
    """
    Check if a tool has progress streaming enabled.

    Args:
        tool: Tool function to check

    Returns:
        True if the tool supports progress streaming
    """
    return getattr(tool, "_progress_enabled", False)
