"""Shared formatters for console and streaming output.

This module provides reusable formatters for:
- Console output (plain text, colored, JSON)
- Stream events (ExecutorStreamEvent, StreamEvent)
- Progress display

These formatters are used by:
- executor/streaming.py for task progress
- CLI commands for output formatting
- MCP servers for structured responses

Example:
    from ai_infra.utils.formatters import ConsoleFormatter, OutputFormat

    formatter = ConsoleFormatter(colors_enabled=True)
    print(formatter.format(event, format=OutputFormat.PLAIN))
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, TextIO

if TYPE_CHECKING:
    pass


# =============================================================================
# Output Format Enum
# =============================================================================


class OutputFormat(str, Enum):
    """Output format for streaming and console output."""

    PLAIN = "plain"
    RICH = "rich"
    JSON = "json"
    MINIMAL = "minimal"


# =============================================================================
# ANSI Colors
# =============================================================================


@dataclass(frozen=True)
class ANSIColors:
    """ANSI color codes for terminal output."""

    reset: str = "\033[0m"
    bold: str = "\033[1m"
    dim: str = "\033[2m"
    green: str = "\033[32m"
    yellow: str = "\033[33m"
    red: str = "\033[31m"
    blue: str = "\033[34m"
    cyan: str = "\033[36m"
    magenta: str = "\033[35m"
    white: str = "\033[37m"


COLORS = ANSIColors()


def colorize(text: str, color: str, enabled: bool = True) -> str:
    """Apply ANSI color to text.

    Args:
        text: Text to colorize.
        color: Color name (green, red, blue, etc.).
        enabled: Whether colors are enabled.

    Returns:
        Colorized text if enabled, otherwise plain text.
    """
    if not enabled:
        return text
    color_code = getattr(COLORS, color, "")
    if not color_code:
        return text
    return f"{color_code}{text}{COLORS.reset}"


def _get_event_type(event: Any) -> str | None:
    """Extract event type from various event formats.

    Handles:
    - Core StreamEvent with `type` attribute
    - Executor ExecutorStreamEvent with `event_type` attribute (enum)
    - Dict events with `type` or `event_type` keys

    Args:
        event: Event object or dict.

    Returns:
        Event type as string, or None if not found.
    """
    # Try event_type first (executor format - may be enum)
    if hasattr(event, "event_type"):
        et = event.event_type
        return et.value if hasattr(et, "value") else str(et)

    # Try type (core format)
    if hasattr(event, "type"):
        return str(event.type)

    # Try dict keys
    if isinstance(event, dict):
        if "event_type" in event:
            et = event["event_type"]
            return et.value if hasattr(et, "value") else str(et)
        if "type" in event:
            return str(event["type"])

    return None


# =============================================================================
# Stream Formatter Protocol
# =============================================================================


class StreamFormatter(Protocol):
    """Protocol for stream event formatters.

    Formatters must implement the `format` method which takes
    an event and returns a formatted string.
    """

    def format(self, event: Any) -> str:
        """Format an event for output.

        Args:
            event: Event to format (StreamEvent, ExecutorStreamEvent, dict).

        Returns:
            Formatted string representation.
        """
        ...


# =============================================================================
# Plain Text Formatter
# =============================================================================


class PlainFormatter:
    """Plain text formatter with optional colors.

    Provides clear, readable output with ANSI colors for terminals.
    Supports both core StreamEvent and ExecutorStreamEvent.

    Example:
        formatter = PlainFormatter(colors_enabled=True)
        output = formatter.format(event)
        print(output)
    """

    # Symbol mapping for event types
    SYMBOLS: dict[str, str] = {
        # Core types
        "thinking": "[~]",
        "token": "",
        "tool_start": "[>]",
        "tool_end": "[<]",
        "done": "[*]",
        "error": "[!]",
        # Executor types
        "run_start": ">>>",
        "run_end": "<<<",
        "node_start": "-->",
        "node_end": "<--",
        "node_error": "[!]",
        "task_start": "[*]",
        "task_complete": "[+]",
        "task_failed": "[x]",
        "task_skipped": "[-]",
        "progress": "...",
        "interrupt": "[?]",
        "resume": "[>]",
        "state_update": "[~]",
    }

    # Color mapping for event types
    COLOR_MAP: dict[str, str] = {
        # Core types
        "thinking": "cyan",
        "token": "white",
        "tool_start": "blue",
        "tool_end": "blue",
        "done": "green",
        "error": "red",
        # Executor types
        "run_start": "cyan",
        "run_end": "cyan",
        "node_start": "blue",
        "node_end": "blue",
        "node_error": "red",
        "task_start": "yellow",
        "task_complete": "green",
        "task_failed": "red",
        "task_skipped": "yellow",
        "interrupt": "magenta",
        "resume": "magenta",
    }

    def __init__(
        self,
        colors_enabled: bool = True,
        show_timing: bool = True,
        show_timestamps: bool = False,
    ):
        """Initialize formatter.

        Args:
            colors_enabled: Whether to use ANSI colors.
            show_timing: Whether to show duration information.
            show_timestamps: Whether to show timestamps.
        """
        self.colors_enabled = colors_enabled
        self.show_timing = show_timing
        self.show_timestamps = show_timestamps

    def format(self, event: Any) -> str:
        """Format an event as plain text.

        Args:
            event: Event to format (must have 'type' or 'event_type' attribute or key).

        Returns:
            Formatted string.
        """
        # Get event type - handle both core (type) and executor (event_type)
        event_type = _get_event_type(event)
        if event_type is None:
            return str(event)

        # Handle token events specially (no symbol, just content)
        if event_type == "token":
            content = getattr(event, "content", None) or event.get("content", "")
            return str(content)

        # Build output parts
        parts: list[str] = []

        # Symbol
        symbol = self.SYMBOLS.get(str(event_type), "[?]")
        color = self.COLOR_MAP.get(str(event_type), "reset")
        parts.append(colorize(symbol, color, self.colors_enabled))

        # Node name (for executor events)
        node_name = getattr(event, "node_name", None)
        if node_name and event_type in ("node_start", "node_end", "node_error"):
            parts.append(colorize(node_name, "bold", self.colors_enabled))

        # Message
        message = getattr(event, "message", None)
        if message:
            parts.append(message)

        # Tool info (for core events)
        tool = getattr(event, "tool", None)
        if tool and event_type in ("tool_start", "tool_end"):
            parts.append(f"tool: {tool}")

        # Timing
        if self.show_timing:
            duration = getattr(event, "duration_ms", None) or getattr(event, "latency_ms", None)
            if duration is not None:
                timing = f"({duration:.0f}ms)"
                parts.append(colorize(timing, "dim", self.colors_enabled))

        return " ".join(parts)


# =============================================================================
# Minimal Formatter
# =============================================================================


class MinimalFormatter:
    """Minimal formatter showing only essential information.

    Best for clean progress display without verbose details.
    Only shows task start/complete/failed and final summary.

    Example:
        formatter = MinimalFormatter()
        output = formatter.format(event)
    """

    def __init__(self, colors_enabled: bool = True):
        """Initialize formatter.

        Args:
            colors_enabled: Whether to use ANSI colors.
        """
        self.colors_enabled = colors_enabled

    def format(self, event: Any) -> str:
        """Format an event minimally.

        Args:
            event: Event to format.

        Returns:
            Formatted string, or empty string for non-essential events.
        """
        # Get event type - handle both core (type) and executor (event_type)
        event_type = _get_event_type(event)
        if event_type is None:
            return ""

        # Handle specific event types
        if event_type == "task_start":
            task = getattr(event, "task", None) or event.get("task", {})
            title = task.get("title", "Task") if isinstance(task, dict) else str(task)
            return f"Starting: {title}"

        elif event_type == "task_complete":
            task = getattr(event, "task", None) or event.get("task", {})
            title = task.get("title", "Task") if isinstance(task, dict) else str(task)
            return colorize(f"Completed: {title}", "green", self.colors_enabled)

        elif event_type == "task_failed":
            task = getattr(event, "task", None) or event.get("task", {})
            title = task.get("title", "Task") if isinstance(task, dict) else str(task)
            return colorize(f"Failed: {title}", "red", self.colors_enabled)

        elif event_type in ("run_end", "done"):
            data = getattr(event, "data", None) or event
            if isinstance(data, dict):
                completed = data.get("completed", 0)
                failed = data.get("failed", 0)
                return f"Done: {completed} completed, {failed} failed"
            return "Done"

        elif event_type == "error":
            error = getattr(event, "error", None) or event.get("error", "Unknown error")
            return colorize(f"Error: {error}", "red", self.colors_enabled)

        # Skip other event types for minimal output
        return ""


# =============================================================================
# JSON Formatter
# =============================================================================


class JsonFormatter:
    """JSON formatter for programmatic consumption.

    Outputs events as single-line JSON objects for parsing
    by external tools, log aggregators, or frontends.

    Example:
        formatter = JsonFormatter()
        output = formatter.format(event)  # {"type": "task_start", ...}
    """

    def __init__(self, indent: int | None = None):
        """Initialize formatter.

        Args:
            indent: JSON indentation (None for compact single-line).
        """
        self.indent = indent

    def format(self, event: Any) -> str:
        """Format an event as JSON.

        Args:
            event: Event to format.

        Returns:
            JSON string representation.
        """
        # Convert to dict
        if hasattr(event, "to_dict"):
            data = event.to_dict()
        elif isinstance(event, dict):
            data = event
        else:
            data = {"value": str(event)}

        return json.dumps(data, indent=self.indent, default=str)


# =============================================================================
# Console Output Helpers
# =============================================================================


def get_formatter(
    format: OutputFormat,
    colors_enabled: bool = True,
    show_timing: bool = True,
) -> StreamFormatter:
    """Get formatter instance based on format type.

    Args:
        format: Output format to use.
        colors_enabled: Whether to use colors (for plain/minimal).
        show_timing: Whether to show timing (for plain).

    Returns:
        Appropriate formatter instance.
    """
    if format == OutputFormat.JSON:
        return JsonFormatter()
    elif format == OutputFormat.MINIMAL:
        return MinimalFormatter(colors_enabled=colors_enabled)
    else:
        return PlainFormatter(
            colors_enabled=colors_enabled,
            show_timing=show_timing,
        )


def stream_to_console(
    event: Any,
    format: OutputFormat = OutputFormat.PLAIN,
    formatter: StreamFormatter | None = None,
    stream: TextIO | None = None,
    colors_enabled: bool = True,
) -> None:
    """Stream an event to the console.

    Args:
        event: Event to output.
        format: Output format to use.
        formatter: Custom formatter (overrides format).
        stream: Output stream (default: stdout).
        colors_enabled: Whether to use colors.
    """
    if formatter is None:
        formatter = get_formatter(format, colors_enabled=colors_enabled)

    output = formatter.format(event)
    if not output:
        return

    target = stream or sys.stdout
    print(output, file=target, flush=True)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "OutputFormat",
    # Colors
    "ANSIColors",
    "COLORS",
    "colorize",
    # Protocol
    "StreamFormatter",
    # Formatters
    "PlainFormatter",
    "MinimalFormatter",
    "JsonFormatter",
    # Helpers
    "get_formatter",
    "stream_to_console",
]
