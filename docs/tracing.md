# Distributed Tracing

ai-infra provides distributed tracing support with LangSmith and OpenTelemetry exporters.

## Quick Start

```python
from ai_infra import Tracer, ConsoleExporter

# Create tracer with console output
tracer = Tracer(exporter=ConsoleExporter())

# Create a span
with tracer.span("my_operation") as span:
    span.set_attribute("key", "value")
    # Do work...
```

## Exporters

### Console Exporter

For development and debugging:

```python
from ai_infra import Tracer, ConsoleExporter

tracer = Tracer(exporter=ConsoleExporter())
```

### LangSmith Exporter

For LangSmith tracing:

```python
from ai_infra import Tracer, LangSmithExporter

exporter = LangSmithExporter(
    api_key="ls_...",
    project_name="my-project",
)
tracer = Tracer(exporter=exporter)
```

### OpenTelemetry Exporter

For OTEL-compatible backends:

```python
from ai_infra import Tracer, OpenTelemetryExporter

exporter = OpenTelemetryExporter(
    endpoint="http://localhost:4317",
    service_name="my-service",
)
tracer = Tracer(exporter=exporter)
```

## Span Operations

### Creating Spans

```python
# Context manager (recommended)
with tracer.span("operation") as span:
    span.set_attribute("key", "value")
    # Work...

# Async context manager
async with tracer.span("async_operation") as span:
    await do_async_work()

# Manual management
span = tracer.start_span("operation")
try:
    # Work...
finally:
    span.end()
```

### Span Attributes

```python
with tracer.span("llm_call") as span:
    span.set_attribute("provider", "openai")
    span.set_attribute("model", "gpt-4o")
    span.set_attribute("temperature", 0.7)
    span.set_attribute("tokens", 150)
```

### Span Status

```python
with tracer.span("operation") as span:
    try:
        result = do_work()
        span.set_status("ok")
    except Exception as e:
        span.set_status("error", str(e))
        raise
```

### Nested Spans

```python
with tracer.span("parent") as parent:
    with tracer.span("child1"):
        # Child inherits parent context
        pass

    with tracer.span("child2"):
        pass
```

## Integration with Callbacks

Use `TracingCallbacks` to automatically trace LLM calls:

```python
from ai_infra import Tracer, TracingCallbacks, CallbackManager

tracer = Tracer(exporter=LangSmithExporter(...))
callbacks = TracingCallbacks(tracer)

manager = CallbackManager()
manager.add(callbacks)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | LangSmith API key |
| `LANGCHAIN_PROJECT` | LangSmith project name |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTEL endpoint |
| `OTEL_SERVICE_NAME` | Service name for OTEL |

## Best Practices

1. **Use descriptive span names**: `llm_call`, `tool_execution`, `rag_retrieval`
2. **Add relevant attributes**: provider, model, tool_name, duration
3. **Set status on completion**: "ok" or "error" with message
4. **Use nested spans for complex operations**
5. **Sample in production**: Don't trace every request
