"""Streaming support for the Executor Graph.

Phase 1.6.2: Provides streaming output for real-time progress display.
Phase 0.1 Consolidation: Formatters now imported from ai_infra.utils.formatters.

This module enables:
- Streaming state updates during graph execution
- Real-time CLI progress display
- Configurable output formatters

Usage:
    from ai_infra.executor.streaming import (
        StreamingConfig,
        ExecutorStreamEvent,
        stream_to_console,
        create_streaming_executor,
    )

    # Stream with console output
    async for event in executor.astream_events():
        stream_to_console(event)

    # Or use the convenience wrapper
    executor = create_streaming_executor(
        agent,
        "ROADMAP.md",
        streaming_config=StreamingConfig.verbose(),
    )
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TextIO

from ai_infra.logging import get_logger
from ai_infra.utils.formatters import (
    JsonFormatter,
    MinimalFormatter,
    OutputFormat,
    PlainFormatter,
    StreamFormatter,
)

if TYPE_CHECKING:
    from ai_infra.executor.state import ExecutorGraphState
    from ai_infra.executor.todolist import TodoItem

logger = get_logger("executor.streaming")


# =============================================================================
# Event Types
# =============================================================================


class StreamEventType(str, Enum):
    """Types of streaming events."""

    # Graph lifecycle
    RUN_START = "run_start"
    RUN_END = "run_end"

    # Node lifecycle
    NODE_START = "node_start"
    NODE_END = "node_end"
    NODE_ERROR = "node_error"

    # State transitions
    STATE_UPDATE = "state_update"

    # Task lifecycle
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    TASK_SKIPPED = "task_skipped"

    # Progress indicators
    PROGRESS = "progress"

    # HITL events
    INTERRUPT = "interrupt"
    RESUME = "resume"

    # Phase 1.2: LLM token streaming events
    # These forward events from ai_infra.llm.streaming.StreamEvent
    LLM_THINKING = "llm_thinking"  # Agent started processing
    LLM_TOKEN = "llm_token"  # Text token from LLM
    LLM_TOOL_START = "llm_tool_start"  # Tool execution started
    LLM_TOOL_END = "llm_tool_end"  # Tool execution completed
    LLM_DONE = "llm_done"  # LLM response complete
    LLM_ERROR = "llm_error"  # Error in LLM streaming


@dataclass
class ExecutorStreamEvent:
    """A streaming event from the executor graph.

    Attributes:
        event_type: The type of event.
        node_name: Name of the current node (if applicable).
        timestamp: When the event occurred.
        data: Event-specific data.
        state_snapshot: Partial state snapshot (if enabled).
        duration_ms: Duration for completed events.
        task: Current task (if applicable).
        message: Human-readable message.
    """

    event_type: StreamEventType
    node_name: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = field(default_factory=dict)
    state_snapshot: dict[str, Any] | None = None
    duration_ms: float | None = None
    task: dict[str, Any] | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "node_name": self.node_name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "state_snapshot": self.state_snapshot,
            "duration_ms": self.duration_ms,
            "task": self.task,
            "message": self.message,
        }


# =============================================================================
# Streaming Configuration
# =============================================================================


@dataclass
class StreamingConfig:
    """Configuration for streaming output.

    Attributes:
        enabled: Whether streaming is enabled.
        include_state_snapshot: Include partial state in events.
        output_format: Format for output.
        show_node_transitions: Show node start/end events.
        show_task_progress: Show task-level progress.
        show_timing: Include timing information.
        output_stream: Where to write output (default: stdout).
        colors_enabled: Use ANSI colors in output.
        progress_interval_ms: Minimum interval between progress updates.
        stream_tokens: Forward LLM tokens (Phase 1.2).
        show_llm_thinking: Show LLM thinking events (Phase 1.2).
        show_llm_tools: Show LLM tool start/end events (Phase 1.2).
        token_visibility: Visibility level for token events (Phase 1.2).
            - "minimal": Only tokens
            - "standard": + tool names and timing (default)
            - "detailed": + tool arguments
            - "debug": + tool result previews
    """

    enabled: bool = True
    include_state_snapshot: bool = False
    output_format: OutputFormat = OutputFormat.PLAIN
    show_node_transitions: bool = True
    show_task_progress: bool = True
    show_timing: bool = True
    output_stream: TextIO | None = None
    colors_enabled: bool = True
    progress_interval_ms: float = 500.0
    # Phase 1.2: Token streaming options
    stream_tokens: bool = False  # Off by default for backward compatibility
    show_llm_thinking: bool = True
    show_llm_tools: bool = True
    token_visibility: Literal["minimal", "standard", "detailed", "debug"] = "standard"

    @classmethod
    def verbose(cls) -> StreamingConfig:
        """Verbose streaming configuration with all details."""
        return cls(
            enabled=True,
            include_state_snapshot=True,
            output_format=OutputFormat.PLAIN,
            show_node_transitions=True,
            show_task_progress=True,
            show_timing=True,
            colors_enabled=True,
            stream_tokens=True,
            show_llm_thinking=True,
            show_llm_tools=True,
            token_visibility="detailed",
        )

    @classmethod
    def minimal(cls) -> StreamingConfig:
        """Minimal streaming with only task progress."""
        return cls(
            enabled=True,
            include_state_snapshot=False,
            output_format=OutputFormat.MINIMAL,
            show_node_transitions=False,
            show_task_progress=True,
            show_timing=False,
            colors_enabled=True,
            stream_tokens=False,
        )

    @classmethod
    def json_output(cls) -> StreamingConfig:
        """JSON streaming for programmatic consumption."""
        return cls(
            enabled=True,
            include_state_snapshot=True,
            output_format=OutputFormat.JSON,
            show_node_transitions=True,
            show_task_progress=True,
            show_timing=True,
            colors_enabled=False,
            stream_tokens=True,
            token_visibility="detailed",
        )

    @classmethod
    def disabled(cls) -> StreamingConfig:
        """Disabled streaming."""
        return cls(enabled=False)

    @classmethod
    def tokens_only(cls) -> StreamingConfig:
        """Token streaming only - for real-time output display (Phase 1.2)."""
        return cls(
            enabled=True,
            include_state_snapshot=False,
            output_format=OutputFormat.PLAIN,
            show_node_transitions=False,
            show_task_progress=True,
            show_timing=False,
            colors_enabled=True,
            stream_tokens=True,
            show_llm_thinking=False,
            show_llm_tools=False,
            token_visibility="minimal",
        )


# =============================================================================
# Stream Output Helpers
# =============================================================================
# Note: PlainFormatter, MinimalFormatter, JsonFormatter, StreamFormatter, and
# OutputFormat are imported from ai_infra.utils.formatters for shared use.


def get_formatter(config: StreamingConfig) -> StreamFormatter:
    """Get formatter based on configuration.

    Args:
        config: Streaming configuration.

    Returns:
        Appropriate formatter instance.
    """
    if config.output_format == OutputFormat.JSON:
        return JsonFormatter()
    elif config.output_format == OutputFormat.MINIMAL:
        return MinimalFormatter(colors_enabled=config.colors_enabled)
    else:
        return PlainFormatter(
            colors_enabled=config.colors_enabled,
            show_timing=config.show_timing,
        )


def stream_to_console(
    event: ExecutorStreamEvent,
    config: StreamingConfig | None = None,
    formatter: StreamFormatter | None = None,
) -> None:
    """Stream an event to the console.

    Args:
        event: The event to output.
        config: Optional streaming configuration.
        formatter: Optional custom formatter.
    """
    if config is None:
        config = StreamingConfig()

    if not config.enabled:
        return

    if formatter is None:
        formatter = get_formatter(config)

    output = formatter.format(event)
    if not output:
        return

    stream = config.output_stream or sys.stdout
    print(output, file=stream, flush=True)


# =============================================================================
# Event Builders
# =============================================================================


def create_run_start_event(
    roadmap_path: str,
    total_tasks: int = 0,
) -> ExecutorStreamEvent:
    """Create a run start event.

    Args:
        roadmap_path: Path to the roadmap.
        total_tasks: Total number of tasks.

    Returns:
        ExecutorStreamEvent for run start.
    """
    return ExecutorStreamEvent(
        event_type=StreamEventType.RUN_START,
        message=f"Starting executor run (ROADMAP: {roadmap_path})",
        data={
            "roadmap_path": roadmap_path,
            "total_tasks": total_tasks,
        },
    )


def create_run_end_event(
    completed: int,
    failed: int,
    skipped: int,
    duration_ms: float,
) -> ExecutorStreamEvent:
    """Create a run end event.

    Args:
        completed: Number of completed tasks.
        failed: Number of failed tasks.
        skipped: Number of skipped tasks.
        duration_ms: Total duration.

    Returns:
        ExecutorStreamEvent for run end.
    """
    return ExecutorStreamEvent(
        event_type=StreamEventType.RUN_END,
        message=f"Executor run complete: {completed} completed, {failed} failed",
        data={
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
        },
        duration_ms=duration_ms,
    )


def create_node_start_event(
    node_name: str,
    state: ExecutorGraphState | None = None,
    include_state: bool = False,
) -> ExecutorStreamEvent:
    """Create a node start event.

    Args:
        node_name: Name of the node.
        state: Current state (optional).
        include_state: Whether to include state snapshot.

    Returns:
        ExecutorStreamEvent for node start.
    """
    state_snapshot = None
    if include_state and state:
        state_snapshot = _create_state_snapshot(state)

    return ExecutorStreamEvent(
        event_type=StreamEventType.NODE_START,
        node_name=node_name,
        message=f"Executing node: {node_name}",
        state_snapshot=state_snapshot,
    )


def create_node_end_event(
    node_name: str,
    duration_ms: float,
    state: ExecutorGraphState | None = None,
    include_state: bool = False,
) -> ExecutorStreamEvent:
    """Create a node end event.

    Args:
        node_name: Name of the node.
        duration_ms: Execution duration.
        state: Updated state (optional).
        include_state: Whether to include state snapshot.

    Returns:
        ExecutorStreamEvent for node end.
    """
    state_snapshot = None
    if include_state and state:
        state_snapshot = _create_state_snapshot(state)

    return ExecutorStreamEvent(
        event_type=StreamEventType.NODE_END,
        node_name=node_name,
        message=f"Completed node: {node_name}",
        duration_ms=duration_ms,
        state_snapshot=state_snapshot,
    )


def create_node_error_event(
    node_name: str,
    error: Exception,
    duration_ms: float,
) -> ExecutorStreamEvent:
    """Create a node error event.

    Args:
        node_name: Name of the node.
        error: The exception that occurred.
        duration_ms: Duration until error.

    Returns:
        ExecutorStreamEvent for node error.
    """
    return ExecutorStreamEvent(
        event_type=StreamEventType.NODE_ERROR,
        node_name=node_name,
        message=f"Node error: {node_name} - {error}",
        duration_ms=duration_ms,
        data={
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )


def create_task_start_event(
    task: TodoItem | dict[str, Any],
    task_number: int = 0,
    total_tasks: int = 0,
) -> ExecutorStreamEvent:
    """Create a task start event.

    Args:
        task: The task being started.
        task_number: Current task number.
        total_tasks: Total number of tasks.

    Returns:
        ExecutorStreamEvent for task start.
    """
    task_dict = task if isinstance(task, dict) else _task_to_dict(task)
    title = task_dict.get("title", "Task")

    progress = ""
    if total_tasks > 0:
        progress = f" [{task_number}/{total_tasks}]"

    return ExecutorStreamEvent(
        event_type=StreamEventType.TASK_START,
        message=f"Starting task{progress}: {title}",
        task=task_dict,
        data={
            "task_number": task_number,
            "total_tasks": total_tasks,
        },
    )


def create_task_complete_event(
    task: TodoItem | dict[str, Any],
    duration_ms: float,
    files_modified: int = 0,
) -> ExecutorStreamEvent:
    """Create a task complete event.

    Args:
        task: The completed task.
        duration_ms: Task duration.
        files_modified: Number of files modified.

    Returns:
        ExecutorStreamEvent for task complete.
    """
    task_dict = task if isinstance(task, dict) else _task_to_dict(task)
    title = task_dict.get("title", "Task")

    return ExecutorStreamEvent(
        event_type=StreamEventType.TASK_COMPLETE,
        message=f"Completed: {title}",
        task=task_dict,
        duration_ms=duration_ms,
        data={
            "files_modified": files_modified,
        },
    )


def create_task_failed_event(
    task: TodoItem | dict[str, Any],
    error: str,
    duration_ms: float,
) -> ExecutorStreamEvent:
    """Create a task failed event.

    Args:
        task: The failed task.
        error: Error message.
        duration_ms: Duration until failure.

    Returns:
        ExecutorStreamEvent for task failed.
    """
    task_dict = task if isinstance(task, dict) else _task_to_dict(task)
    title = task_dict.get("title", "Task")

    return ExecutorStreamEvent(
        event_type=StreamEventType.TASK_FAILED,
        message=f"Failed: {title} - {error}",
        task=task_dict,
        duration_ms=duration_ms,
        data={
            "error": error,
        },
    )


def create_interrupt_event(
    node_name: str,
    task: TodoItem | dict[str, Any] | None = None,
) -> ExecutorStreamEvent:
    """Create an interrupt event.

    Args:
        node_name: Node where interrupt occurred.
        task: Current task (if applicable).

    Returns:
        ExecutorStreamEvent for interrupt.
    """
    task_dict = None
    if task:
        task_dict = task if isinstance(task, dict) else _task_to_dict(task)

    return ExecutorStreamEvent(
        event_type=StreamEventType.INTERRUPT,
        node_name=node_name,
        message=f"Execution paused at {node_name} - awaiting human decision",
        task=task_dict,
    )


def create_resume_event(
    decision: str,
    node_name: str | None = None,
) -> ExecutorStreamEvent:
    """Create a resume event.

    Args:
        decision: The decision made (approve/reject/etc).
        node_name: Node where resuming.

    Returns:
        ExecutorStreamEvent for resume.
    """
    return ExecutorStreamEvent(
        event_type=StreamEventType.RESUME,
        node_name=node_name,
        message=f"Resuming execution with decision: {decision}",
        data={
            "decision": decision,
        },
    )


# =============================================================================
# Phase 1.2: LLM Token Event Builders
# =============================================================================


def create_llm_thinking_event(
    model: str | None = None,
    node_name: str = "execute_task",
) -> ExecutorStreamEvent:
    """Create an LLM thinking event.

    Phase 1.2: Indicates the LLM has started processing.

    Args:
        model: The model name being used.
        node_name: The node executing the LLM call.

    Returns:
        ExecutorStreamEvent for LLM thinking.
    """
    return ExecutorStreamEvent(
        event_type=StreamEventType.LLM_THINKING,
        node_name=node_name,
        message="Agent is thinking...",
        data={
            "model": model,
        },
    )


def create_llm_token_event(
    content: str,
    node_name: str = "execute_task",
) -> ExecutorStreamEvent:
    """Create an LLM token event.

    Phase 1.2: Represents a single text token from the LLM response.

    Args:
        content: The token content.
        node_name: The node executing the LLM call.

    Returns:
        ExecutorStreamEvent for LLM token.
    """
    return ExecutorStreamEvent(
        event_type=StreamEventType.LLM_TOKEN,
        node_name=node_name,
        message=content,  # Token content in message for easy access
        data={
            "content": content,
        },
    )


def create_llm_tool_start_event(
    tool: str,
    tool_id: str | None = None,
    arguments: dict[str, Any] | None = None,
    node_name: str = "execute_task",
) -> ExecutorStreamEvent:
    """Create an LLM tool start event.

    Phase 1.2: Indicates a tool is being called by the LLM.

    Args:
        tool: The tool name.
        tool_id: The tool call ID for correlation.
        arguments: Tool arguments (visibility dependent).
        node_name: The node executing the LLM call.

    Returns:
        ExecutorStreamEvent for LLM tool start.
    """
    data: dict[str, Any] = {"tool": tool}
    if tool_id:
        data["tool_id"] = tool_id
    if arguments:
        data["arguments"] = arguments

    return ExecutorStreamEvent(
        event_type=StreamEventType.LLM_TOOL_START,
        node_name=node_name,
        message=f"Calling tool: {tool}",
        data=data,
    )


def create_llm_tool_end_event(
    tool: str,
    tool_id: str | None = None,
    latency_ms: float | None = None,
    result: Any | None = None,
    preview: str | None = None,
    node_name: str = "execute_task",
) -> ExecutorStreamEvent:
    """Create an LLM tool end event.

    Phase 1.2: Indicates a tool call has completed.

    Args:
        tool: The tool name.
        tool_id: The tool call ID for correlation.
        latency_ms: Tool execution time in milliseconds.
        result: Tool result (visibility dependent).
        preview: Truncated preview of result (debug visibility).
        node_name: The node executing the LLM call.

    Returns:
        ExecutorStreamEvent for LLM tool end.
    """
    data: dict[str, Any] = {"tool": tool}
    if tool_id:
        data["tool_id"] = tool_id
    if latency_ms is not None:
        data["latency_ms"] = latency_ms
    if result is not None:
        data["result"] = result
    if preview is not None:
        data["preview"] = preview

    message = f"Tool {tool} completed"
    if latency_ms is not None:
        message += f" ({latency_ms:.1f}ms)"

    return ExecutorStreamEvent(
        event_type=StreamEventType.LLM_TOOL_END,
        node_name=node_name,
        message=message,
        duration_ms=latency_ms,
        data=data,
    )


def create_llm_done_event(
    tools_called: int = 0,
    node_name: str = "execute_task",
) -> ExecutorStreamEvent:
    """Create an LLM done event.

    Phase 1.2: Indicates the LLM response is complete.

    Args:
        tools_called: Total number of tools called.
        node_name: The node executing the LLM call.

    Returns:
        ExecutorStreamEvent for LLM done.
    """
    return ExecutorStreamEvent(
        event_type=StreamEventType.LLM_DONE,
        node_name=node_name,
        message="Agent response complete",
        data={
            "tools_called": tools_called,
        },
    )


def create_llm_error_event(
    error: str,
    node_name: str = "execute_task",
) -> ExecutorStreamEvent:
    """Create an LLM error event.

    Phase 1.2: Indicates an error occurred during LLM streaming.

    Args:
        error: The error message.
        node_name: The node executing the LLM call.

    Returns:
        ExecutorStreamEvent for LLM error.
    """
    return ExecutorStreamEvent(
        event_type=StreamEventType.LLM_ERROR,
        node_name=node_name,
        message=f"LLM error: {error}",
        data={
            "error": error,
        },
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _create_state_snapshot(state: ExecutorGraphState) -> dict[str, Any]:
    """Create a minimal state snapshot for streaming.

    Only includes relevant fields to avoid large payloads.

    Args:
        state: Current executor state.

    Returns:
        Minimal state snapshot.
    """
    return {
        "completed_count": state.get("completed_count", 0),
        "retry_count": state.get("retry_count", 0),
        "should_continue": state.get("should_continue", True),
        "error": state.get("error") is not None,
        "current_task_id": (
            str(state.get("current_task").id) if state.get("current_task") else None
        ),
    }


def _task_to_dict(task: TodoItem) -> dict[str, Any]:
    """Convert TodoItem to dictionary.

    Args:
        task: TodoItem to convert.

    Returns:
        Dictionary representation.
    """
    return {
        "id": str(task.id) if hasattr(task, "id") else None,
        "title": task.title if hasattr(task, "title") else str(task),
        "status": task.status.value if hasattr(task, "status") else None,
    }


# =============================================================================
# Streaming Executor Wrapper
# =============================================================================


class StreamingExecutorMixin:
    """Mixin to add streaming capabilities to ExecutorGraph.

    This mixin adds the astream_events method that yields
    ExecutorStreamEvent instances for real-time progress display.
    """

    async def astream_events(
        self,
        initial_state: Any = None,
        config: dict[str, Any] | None = None,
        streaming_config: StreamingConfig | None = None,
    ):
        """Stream execution events from the graph.

        Yields ExecutorStreamEvent instances for each significant
        event during execution.

        Args:
            initial_state: Optional initial state.
            config: Optional LangGraph config.
            streaming_config: Streaming configuration.

        Yields:
            ExecutorStreamEvent instances.
        """
        if streaming_config is None:
            streaming_config = StreamingConfig()

        state = initial_state or self.get_initial_state()  # type: ignore
        config = config or {}

        # Emit run start
        total_tasks = len(state.get("todos", []))
        yield create_run_start_event(
            roadmap_path=state.get("roadmap_path", "ROADMAP.md"),
            total_tasks=total_tasks,
        )

        run_start = time.time()
        completed = 0
        failed = 0
        current_task = None

        # Stream from underlying graph
        async for event in self.graph.astream(state, config=config):  # type: ignore
            for node_name, node_state in event.items():
                node_start = time.time()

                # Emit node start
                if streaming_config.show_node_transitions:
                    yield create_node_start_event(
                        node_name=node_name,
                        state=node_state,
                        include_state=streaming_config.include_state_snapshot,
                    )

                # Track task progress
                if node_name == "pick_task":
                    current_task = node_state.get("current_task")
                    if current_task and streaming_config.show_task_progress:
                        yield create_task_start_event(
                            task=current_task,
                            task_number=node_state.get("completed_count", 0) + 1,
                            total_tasks=len(node_state.get("todos", [])),
                        )

                elif node_name == "checkpoint":
                    if current_task and streaming_config.show_task_progress:
                        task_duration = (time.time() - node_start) * 1000
                        yield create_task_complete_event(
                            task=current_task,
                            duration_ms=task_duration,
                            files_modified=len(node_state.get("files_modified", [])),
                        )
                        completed += 1
                        current_task = None

                elif node_name == "handle_failure":
                    if current_task and streaming_config.show_task_progress:
                        error = node_state.get("error", {})
                        error_msg = (
                            error.get("message", "Unknown error") if error else "Unknown error"
                        )
                        yield create_task_failed_event(
                            task=current_task,
                            error=error_msg,
                            duration_ms=(time.time() - node_start) * 1000,
                        )
                        failed += 1

                # Emit node end
                if streaming_config.show_node_transitions:
                    node_duration = (time.time() - node_start) * 1000
                    yield create_node_end_event(
                        node_name=node_name,
                        duration_ms=node_duration,
                        state=node_state,
                        include_state=streaming_config.include_state_snapshot,
                    )

        # Emit run end
        run_duration = (time.time() - run_start) * 1000
        yield create_run_end_event(
            completed=completed,
            failed=failed,
            skipped=0,
            duration_ms=run_duration,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Event types
    "StreamEventType",
    "ExecutorStreamEvent",
    # Configuration
    "StreamingConfig",
    "OutputFormat",
    # Formatters
    "StreamFormatter",
    "PlainFormatter",
    "MinimalFormatter",
    "JsonFormatter",
    "get_formatter",
    # Output helpers
    "stream_to_console",
    # Event builders
    "create_run_start_event",
    "create_run_end_event",
    "create_node_start_event",
    "create_node_end_event",
    "create_node_error_event",
    "create_task_start_event",
    "create_task_complete_event",
    "create_task_failed_event",
    "create_interrupt_event",
    "create_resume_event",
    # Mixin
    "StreamingExecutorMixin",
]
