# Callbacks

ai-infra provides a unified callback system for observing and responding
to events during LLM calls, tool execution, MCP operations, and graph runs.

## Quick Start

```python
from ai_infra import Callbacks, Agent, LLM, MCPClient

class MyCallbacks(Callbacks):
    def on_llm_start(self, event):
        print(f"🚀 Starting {event.provider}/{event.model}")

    def on_llm_end(self, event):
        print(f"✅ Done in {event.latency_ms:.0f}ms, {event.total_tokens} tokens")

    def on_tool_start(self, event):
        print(f"🔧 Running {event.tool_name}")

    def on_tool_end(self, event):
        print(f"✅ Tool done: {event.result[:50]}...")

# Use with Agent
agent = Agent(tools=[...], callbacks=MyCallbacks())
agent.run("Hello!")

# Use with LLM
llm = LLM(callbacks=MyCallbacks())
llm.chat("What is 2+2?")

# Use with MCPClient
mcp = MCPClient([config], callbacks=MyCallbacks())
```

## Events

### LLM Events

| Event | Description | Key Fields |
|-------|-------------|------------|
| `LLMStartEvent` | Fired when LLM call starts | `provider`, `model`, `messages`, `temperature` |
| `LLMEndEvent` | Fired when LLM call completes | `response`, `input_tokens`, `output_tokens`, `latency_ms` |
| `LLMErrorEvent` | Fired when LLM call fails | `error`, `error_type`, `latency_ms` |
| `LLMTokenEvent` | Fired for each streaming token | `token`, `index` |

### Tool Events

| Event | Description | Key Fields |
|-------|-------------|------------|
| `ToolStartEvent` | Fired when tool execution starts | `tool_name`, `arguments` |
| `ToolEndEvent` | Fired when tool execution completes | `tool_name`, `result`, `latency_ms` |
| `ToolErrorEvent` | Fired when tool execution fails | `tool_name`, `error`, `error_type` |

### MCP Events

| Event | Description | Key Fields |
|-------|-------------|------------|
| `MCPConnectEvent` | Fired when MCP server connects | `server_name`, `transport`, `tools_count` |
| `MCPDisconnectEvent` | Fired when MCP server disconnects | `server_name`, `reason` |
| `MCPProgressEvent` | Fired when MCP tool reports progress | `server_name`, `tool_name`, `progress`, `total`, `message` |
| `MCPLoggingEvent` | Fired when MCP server sends log | `server_name`, `level`, `data`, `logger_name` |

### Graph Events

| Event | Description | Key Fields |
|-------|-------------|------------|
| `GraphNodeStartEvent` | Fired when graph node starts | `node_id`, `node_type`, `inputs`, `step` |
| `GraphNodeEndEvent` | Fired when graph node completes | `node_id`, `outputs`, `latency_ms` |
| `GraphNodeErrorEvent` | Fired when graph node fails | `node_id`, `error` |

## Creating Custom Callbacks

Subclass `Callbacks` and override the methods you need:

```python
from ai_infra import Callbacks, LLMStartEvent, LLMEndEvent, ToolStartEvent

class MyCallbacks(Callbacks):
    def __init__(self):
        self.call_count = 0
        self.total_tokens = 0

    def on_llm_start(self, event: LLMStartEvent) -> None:
        self.call_count += 1
        print(f"Call #{self.call_count}: {event.provider}/{event.model}")

    def on_llm_end(self, event: LLMEndEvent) -> None:
        if event.total_tokens:
            self.total_tokens += event.total_tokens
        print(f"Response: {event.response[:100]}...")

    def on_tool_start(self, event: ToolStartEvent) -> None:
        print(f"Tool: {event.tool_name}({event.arguments})")
```

## Multiple Callbacks

Use `CallbackManager` to run multiple callback handlers:

```python
from ai_infra import CallbackManager, LoggingCallbacks, MetricsCallbacks

manager = CallbackManager([
    LoggingCallbacks(),  # Built-in: logs all events
    MetricsCallbacks(),  # Built-in: tracks metrics
    MyCallbacks(),       # Your custom callbacks
])

agent = Agent(tools=[...], callbacks=manager)
```

## Built-in Callbacks

ai-infra provides several built-in callback implementations:

### LoggingCallbacks

Logs all events to Python's logging system:

```python
from ai_infra import LoggingCallbacks, Agent

agent = Agent(tools=[...], callbacks=LoggingCallbacks())
```

### MetricsCallbacks

Tracks call counts, token usage, and latency metrics:

```python
from ai_infra import MetricsCallbacks

metrics = MetricsCallbacks()
agent = Agent(tools=[...], callbacks=metrics)

# After running...
print(f"Total calls: {metrics.total_calls}")
print(f"Total tokens: {metrics.total_tokens}")
```

### PrintCallbacks

Prints events to console (useful for debugging):

```python
from ai_infra import PrintCallbacks

agent = Agent(tools=[...], callbacks=PrintCallbacks())
```

## Async Callbacks

For async operations (like MCP progress), use async versions:

```python
class MyAsyncCallbacks(Callbacks):
    async def on_mcp_progress_async(self, event):
        # Called during async MCP operations
        await notify_user(f"Progress: {event.progress:.0%}")

    async def on_llm_token_async(self, event):
        # Called for streaming tokens in async context
        await stream_to_client(event.token)
```

## Event Order

During a typical agent run, events fire in this order:

1. `LLMStartEvent` - Agent begins thinking
2. `ToolStartEvent` - Agent decides to use a tool
3. `ToolEndEvent` - Tool execution completes
4. `LLMStartEvent` - Agent processes tool result
5. `LLMEndEvent` - Agent generates final response

For streaming responses:
1. `LLMStartEvent`
2. `LLMTokenEvent` (multiple times)
3. `LLMEndEvent`

## Use Cases

### Observability & Monitoring

```python
class ObservabilityCallbacks(Callbacks):
    def on_llm_start(self, event):
        span = tracer.start_span("llm_call")
        span.set_attribute("provider", event.provider)
        span.set_attribute("model", event.model)

    def on_llm_end(self, event):
        span.set_attribute("tokens", event.total_tokens)
        span.set_attribute("latency_ms", event.latency_ms)
        span.end()

    def on_llm_error(self, event):
        span.record_exception(event.error)
        span.end()
```

### Cost Tracking

```python
class CostCallbacks(Callbacks):
    COSTS = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    }

    def __init__(self):
        self.total_cost = 0.0

    def on_llm_end(self, event):
        if event.model in self.COSTS:
            cost = self.COSTS[event.model]
            self.total_cost += (
                (event.input_tokens or 0) * cost["input"] / 1000 +
                (event.output_tokens or 0) * cost["output"] / 1000
            )
```

### Progress Reporting

```python
class ProgressCallbacks(Callbacks):
    async def on_mcp_progress_async(self, event):
        if event.total:
            percentage = event.progress / event.total * 100
            print(f"[{event.server_name}] {percentage:.0f}% - {event.message}")
```

## API Reference

### Callbacks Base Class

```python
class Callbacks(ABC):
    # LLM events
    def on_llm_start(self, event: LLMStartEvent) -> None: ...
    def on_llm_end(self, event: LLMEndEvent) -> None: ...
    def on_llm_error(self, event: LLMErrorEvent) -> None: ...
    def on_llm_token(self, event: LLMTokenEvent) -> None: ...

    # Tool events
    def on_tool_start(self, event: ToolStartEvent) -> None: ...
    def on_tool_end(self, event: ToolEndEvent) -> None: ...
    def on_tool_error(self, event: ToolErrorEvent) -> None: ...

    # MCP events
    def on_mcp_connect(self, event: MCPConnectEvent) -> None: ...
    def on_mcp_disconnect(self, event: MCPDisconnectEvent) -> None: ...
    def on_mcp_progress(self, event: MCPProgressEvent) -> None: ...
    def on_mcp_logging(self, event: MCPLoggingEvent) -> None: ...

    # Graph events
    def on_graph_node_start(self, event: GraphNodeStartEvent) -> None: ...
    def on_graph_node_end(self, event: GraphNodeEndEvent) -> None: ...
    def on_graph_node_error(self, event: GraphNodeErrorEvent) -> None: ...

    # Async versions (for async operations)
    async def on_mcp_progress_async(self, event: MCPProgressEvent) -> None: ...
    async def on_mcp_logging_async(self, event: MCPLoggingEvent) -> None: ...
    async def on_llm_token_async(self, event: LLMTokenEvent) -> None: ...
```

### CallbackManager

```python
class CallbackManager:
    def __init__(self, callbacks: List[Callbacks]): ...

    # Dispatches events to all registered callbacks
    def on_llm_start(self, event: LLMStartEvent) -> None: ...
    # ... all other dispatch methods
```
