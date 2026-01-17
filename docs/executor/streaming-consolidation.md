# Streaming Consolidation: Executor → Core

> **Phase 0.1 of EXECUTOR_1.md**
> **Goal**: Reduce `executor/streaming.py` from 880 lines to ~200 lines by leveraging `ai_infra.llm.streaming`

---

## Executive Summary

| Metric | Before | After | Status |
|--------|--------|-------|--------|
| `executor/streaming.py` | 880 lines | 731 lines | Reduced ~150 lines |
| `utils/formatters.py` | N/A | ~460 lines | Shared formatters |
| Duplicated code | ~150 lines | 0 | Eliminated |

**Approach**: Formatters moved to shared `utils/formatters.py`, imported by `streaming.py`.

---

## Mapping: Executor Streaming → Core Streaming

### Event Types

| executor/streaming.py | ai_infra.llm.streaming | Action |
|-----------------------|------------------------|--------|
| `StreamEventType.RUN_START` | N/A | **Keep** (executor-specific) |
| `StreamEventType.RUN_END` | `StreamEvent(type="done")` | **Map** to core "done" |
| `StreamEventType.NODE_START` | N/A | **Keep** (executor-specific) |
| `StreamEventType.NODE_END` | N/A | **Keep** (executor-specific) |
| `StreamEventType.NODE_ERROR` | `StreamEvent(type="error")` | **Extend** core error |
| `StreamEventType.STATE_UPDATE` | N/A | **Keep** (executor-specific) |
| `StreamEventType.TASK_START` | N/A | **Keep** (executor-specific) |
| `StreamEventType.TASK_COMPLETE` | N/A | **Keep** (executor-specific) |
| `StreamEventType.TASK_FAILED` | N/A | **Keep** (executor-specific) |
| `StreamEventType.TASK_SKIPPED` | N/A | **Keep** (executor-specific) |
| `StreamEventType.PROGRESS` | N/A | **Keep** (executor-specific) |
| `StreamEventType.INTERRUPT` | N/A | **Keep** (executor-specific, HITL) |
| `StreamEventType.RESUME` | N/A | **Keep** (executor-specific, HITL) |

**Conclusion**: Most executor event types are unique to task execution workflow.
Core streaming focuses on LLM token/tool events, executor adds graph/task lifecycle.

### Data Classes

| executor/streaming.py | ai_infra.llm.streaming | Action |
|-----------------------|------------------------|--------|
| `ExecutorStreamEvent` | `StreamEvent` | **Extend** - add executor fields |
| `StreamingConfig` | `StreamConfig` | **Extend** - add executor settings |
| `OutputFormat` (enum) | N/A | **Move** to `utils/formatters.py` |

### Formatters

| executor/streaming.py | ai_infra Location | Action |
|-----------------------|-------------------|--------|
| `PlainFormatter` (80 lines) | N/A | **Move** to `utils/formatters.py` |
| `MinimalFormatter` (30 lines) | N/A | **Move** to `utils/formatters.py` |
| `JsonFormatter` (15 lines) | N/A | **Move** to `utils/formatters.py` |
| `StreamFormatter` (Protocol) | N/A | **Move** to `utils/formatters.py` |

### Functions

| executor/streaming.py | Action | Notes |
|-----------------------|--------|-------|
| `get_formatter()` | **Move** to `utils/formatters.py` | Generic utility |
| `stream_to_console()` | **Move** to `utils/formatters.py` | Generic utility |
| `create_run_start_event()` | **Keep** thin wrapper | Executor-specific |
| `create_run_end_event()` | **Keep** thin wrapper | Executor-specific |
| `create_node_start_event()` | **Keep** thin wrapper | Executor-specific |
| `create_node_end_event()` | **Keep** thin wrapper | Executor-specific |
| `create_node_error_event()` | **Keep** thin wrapper | Executor-specific |
| `create_task_start_event()` | **Keep** thin wrapper | Executor-specific |
| `create_task_complete_event()` | **Keep** thin wrapper | Executor-specific |
| `create_task_failed_event()` | **Keep** thin wrapper | Executor-specific |
| `create_interrupt_event()` | **Keep** thin wrapper | Executor-specific (HITL) |
| `create_resume_event()` | **Keep** thin wrapper | Executor-specific (HITL) |
| `_create_state_snapshot()` | **Keep** | Helper for state serialization |
| `_task_to_dict()` | **Keep** | Helper for task serialization |

### Classes

| executor/streaming.py | Action | Notes |
|-----------------------|--------|-------|
| `StreamingExecutorMixin` | **Keep** | Executor-specific graph wrapper |

---

## New Architecture

### 1. Core Extensions (executor/streaming.py ~200 lines)

```python
"""Executor streaming - extends core ai_infra.llm.streaming."""

from ai_infra.llm.streaming import StreamEvent, StreamConfig
from ai_infra.utils.formatters import ConsoleFormatter, OutputFormat

# Executor-specific event types (extends core types)
ExecutorEventType = Literal[
    # Core types (from StreamEvent)
    "token", "thinking", "tool_start", "tool_end", "done", "error",
    # Executor-specific types
    "run_start", "run_end", "node_start", "node_end", "node_error",
    "task_start", "task_complete", "task_failed", "task_skipped",
    "progress", "interrupt", "resume", "state_update",
]

@dataclass
class ExecutorStreamEvent(StreamEvent):
    """Executor-specific streaming event extending core StreamEvent."""

    # Executor-specific fields
    node_name: str | None = None
    task: dict[str, Any] | None = None
    state_snapshot: dict[str, Any] | None = None
    duration_ms: float | None = None
    message: str = ""

    @classmethod
    def from_core_event(cls, event: StreamEvent, **executor_data) -> "ExecutorStreamEvent":
        """Wrap a core StreamEvent with executor context."""
        return cls(
            type=event.type,
            content=event.content,
            tool=event.tool,
            tool_id=event.tool_id,
            **executor_data,
        )

@dataclass
class ExecutorStreamConfig(StreamConfig):
    """Executor-specific streaming configuration."""

    # Executor additions
    include_state_snapshot: bool = False
    show_node_transitions: bool = True
    show_task_progress: bool = True
    output_format: OutputFormat = OutputFormat.PLAIN
    colors_enabled: bool = True
    progress_interval_ms: float = 500.0
```

### 2. Shared Formatters (utils/formatters.py ~150 lines)

```python
"""Shared formatters for console output."""

from enum import Enum
from typing import Protocol, Any

class OutputFormat(str, Enum):
    PLAIN = "plain"
    RICH = "rich"
    JSON = "json"
    MINIMAL = "minimal"

class StreamFormatter(Protocol):
    def format(self, event: Any) -> str: ...

class ConsoleFormatter:
    """Unified console formatter with color support."""

    def __init__(self, colors_enabled: bool = True):
        self.colors_enabled = colors_enabled

    def format_plain(self, event: Any) -> str: ...
    def format_minimal(self, event: Any) -> str: ...
    def format_json(self, event: Any) -> str: ...

def stream_to_console(event: Any, format: OutputFormat = OutputFormat.PLAIN) -> None:
    """Stream event to console with appropriate formatting."""
    ...
```

---

## Lines Saved

| Component | Before | After | Saved |
|-----------|--------|-------|-------|
| Event types enum | 25 | 15 | 10 |
| ExecutorStreamEvent | 40 | 25 | 15 |
| StreamingConfig | 60 | 30 | 30 |
| Formatters | 140 | 0 (moved) | 140 |
| Format helpers | 50 | 10 | 40 |
| Event builders | 200 | 150 | 50 |
| StreamingExecutorMixin | 100 | 100 | 0 |
| Helpers | 50 | 40 | 10 |
| **Total** | **880** | **~200** | **~680** |

The formatters (140 lines) move to shared `utils/formatters.py` where they can be reused by other modules.

---

## Migration Steps

1. **Create `ai_infra/utils/formatters.py`** - Move formatters, OutputFormat, stream_to_console
2. **Refactor `ExecutorStreamEvent`** - Extend `StreamEvent` from core
3. **Refactor `ExecutorStreamConfig`** - Extend `StreamConfig` from core
4. **Update event builders** - Use new ExecutorStreamEvent
5. **Update imports** - Ensure backwards compatibility via re-exports
6. **Delete redundant code** - Remove duplicated functionality
7. **Update tests** - Adapt to new structure

---

## Backwards Compatibility

All existing imports will continue to work:

```python
# These imports still work (via re-exports)
from ai_infra.executor.streaming import (
    StreamEventType,           # Now uses ExecutorEventType
    ExecutorStreamEvent,       # Now extends StreamEvent
    StreamingConfig,           # Now is ExecutorStreamConfig
    PlainFormatter,            # Re-exported from utils.formatters
    create_task_start_event,   # Still works, same signature
)
```
