# Callbacks and Hooks

ai-infra provides a flexible callback system for observability and extensibility.

## Callback Events

```python
from ai_infra import (
    Callbacks,
    CallbackManager,
    LLMStartEvent,
    LLMEndEvent,
    ToolStartEvent,
    ToolEndEvent,
    ToolErrorEvent,
)
```

## Event Types

### LLM Events

| Event | Description | Fields |
|-------|-------------|--------|
| `LLMStartEvent` | Before LLM call | provider, model, messages, timestamp |
| `LLMEndEvent` | After LLM call | response, duration_ms, token_usage |
| `LLMErrorEvent` | On LLM error | error, duration_ms |

### Tool Events

| Event | Description | Fields |
|-------|-------------|--------|
| `ToolStartEvent` | Before tool execution | tool_name, args, timestamp |
| `ToolEndEvent` | After tool execution | result, duration_ms |
| `ToolErrorEvent` | On tool error | error, duration_ms |

## Creating Callbacks

### Simple Callback

```python
from ai_infra import Callbacks

class LoggingCallbacks(Callbacks):
    """Log all LLM and tool events."""

    def on_llm_start(self, event: LLMStartEvent) -> None:
        print(f"LLM call started: {event.provider}/{event.model}")

    def on_llm_end(self, event: LLMEndEvent) -> None:
        print(f"LLM call completed in {event.duration_ms:.0f}ms")

    def on_tool_start(self, event: ToolStartEvent) -> None:
        print(f"Tool {event.tool_name} started")

    def on_tool_end(self, event: ToolEndEvent) -> None:
        print(f"Tool completed in {event.duration_ms:.0f}ms")
```

### Async Callbacks

```python
class AsyncMetricsCallbacks(Callbacks):
    """Send metrics to monitoring service."""

    async def on_llm_end(self, event: LLMEndEvent) -> None:
        await metrics_client.record(
            "llm_duration_ms",
            event.duration_ms,
            tags={"provider": event.provider, "model": event.model},
        )
```

## Using Callbacks

### With CallbackManager

```python
from ai_infra import CallbackManager

# Create callback manager
manager = CallbackManager()
manager.add(LoggingCallbacks())
manager.add(MetricsCallbacks())

# Emit events
await manager.emit(LLMStartEvent(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
))

# Later...
await manager.emit(LLMEndEvent(
    response=response,
    duration_ms=150.5,
))
```

### With LLM/Agent Logging Hooks

```python
from ai_infra.llm import LLM

llm = LLM()
llm.set_logging_hooks(
    on_request=lambda ctx: print(f"Request: {ctx.provider}"),
    on_response=lambda ctx: print(f"Response in {ctx.duration_ms}ms"),
    on_error=lambda ctx: print(f"Error: {ctx.error}"),
)
```

## Built-in Callbacks

### LoggingCallbacks

```python
from ai_infra import LoggingCallbacks

callbacks = LoggingCallbacks(
    log_level="INFO",
    include_messages=False,  # Don't log message content
)
```

### MetricsCallbacks

```python
from ai_infra import MetricsCallbacks

callbacks = MetricsCallbacks()
# Records: llm_call_count, llm_duration_ms, tool_call_count, etc.
```

## Context Propagation

Pass context through callback events:

```python
class TracingCallbacks(Callbacks):
    """Propagate trace context through events."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def on_llm_start(self, event: LLMStartEvent) -> None:
        span = self.tracer.start_span("llm_call")
        span.set_attribute("provider", event.provider)
        event.metadata["span"] = span

    def on_llm_end(self, event: LLMEndEvent) -> None:
        span = event.metadata.get("span")
        if span:
            span.set_attribute("duration_ms", event.duration_ms)
            span.end()
```
