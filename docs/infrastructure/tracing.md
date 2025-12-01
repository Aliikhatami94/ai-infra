# Distributed Tracing

> OpenTelemetry-based tracing for AI operations.

## Quick Start

```python
from ai_infra import configure_tracing, LLM

configure_tracing(
    service_name="my-ai-service",
    endpoint="http://jaeger:4317",
)

llm = LLM(provider="openai")
response = await llm.generate("Hello")  # Automatically traced
```

---

## Overview

ai-infra includes built-in OpenTelemetry tracing:
- Automatic spans for LLM calls, agent steps, and tool executions
- Context propagation across async operations
- Integration with popular tracing backends (Jaeger, Zipkin, OTLP)
- Cost and token tracking as span attributes

---

## Configuration

### Basic Setup

```python
from ai_infra import configure_tracing

configure_tracing(
    service_name="my-service",
    endpoint="http://localhost:4317",  # OTLP endpoint
)
```

### Full Configuration

```python
configure_tracing(
    service_name="my-service",
    service_version="1.0.0",
    environment="production",
    endpoint="http://jaeger:4317",
    exporter="otlp",  # "otlp", "jaeger", "zipkin", "console"
    sample_rate=1.0,  # 1.0 = trace all requests
    propagators=["tracecontext", "baggage"],
)
```

### Backend-Specific Setup

#### Jaeger

```python
configure_tracing(
    service_name="my-service",
    exporter="jaeger",
    endpoint="http://jaeger:14268/api/traces",
)
```

#### Zipkin

```python
configure_tracing(
    service_name="my-service",
    exporter="zipkin",
    endpoint="http://zipkin:9411/api/v2/spans",
)
```

#### Console (Development)

```python
configure_tracing(
    service_name="my-service",
    exporter="console",  # Print spans to console
)
```

---

## Automatic Tracing

### LLM Calls

```python
from ai_infra import LLM

llm = LLM(provider="openai")
response = await llm.generate("Hello")

# Creates span:
# - name: "llm.generate"
# - attributes:
#   - ai.provider: "openai"
#   - ai.model: "gpt-4o"
#   - ai.prompt_tokens: 5
#   - ai.completion_tokens: 20
#   - ai.total_tokens: 25
#   - ai.cost: 0.00015
```

### Agent Execution

```python
from ai_infra import Agent

agent = Agent(persona=persona, tools=tools)
result = await agent.run("Do something")

# Creates spans:
# - agent.run (root span)
#   - agent.step.1 (child span)
#     - llm.generate (child span)
#   - agent.step.2
#     - tool.search (child span)
#     - llm.generate
```

### Tool Calls

```python
# Automatic span for tool execution
# - name: "tool.search"
# - attributes:
#   - tool.name: "search"
#   - tool.args: {...}
#   - tool.duration_ms: 150
```

---

## Manual Tracing

### Create Spans

```python
from ai_infra import get_tracer

tracer = get_tracer(__name__)

with tracer.start_span("my_operation") as span:
    span.set_attribute("custom_key", "value")
    result = do_something()
    span.set_attribute("result_count", len(result))
```

### Async Spans

```python
async with tracer.start_span("async_operation") as span:
    result = await async_operation()
```

### Add Events

```python
with tracer.start_span("processing") as span:
    span.add_event("step_1_complete", {"items_processed": 100})
    process_batch_1()

    span.add_event("step_2_complete", {"items_processed": 200})
    process_batch_2()
```

---

## Context Propagation

### Automatic Propagation

```python
# Context is automatically propagated in async operations
async def handler(request):
    # Span from incoming request is automatically parent
    response = await llm.generate(prompt)
    return response
```

### Manual Propagation

```python
from ai_infra import inject_context, extract_context

# Inject context into headers
headers = {}
inject_context(headers)

# Extract context from headers
ctx = extract_context(headers)
with tracer.start_span("child", context=ctx) as span:
    pass
```

---

## Span Attributes

### Standard Attributes

| Attribute | Description |
|-----------|-------------|
| `ai.provider` | LLM provider name |
| `ai.model` | Model name |
| `ai.prompt_tokens` | Input token count |
| `ai.completion_tokens` | Output token count |
| `ai.total_tokens` | Total token count |
| `ai.cost` | Estimated cost in USD |
| `ai.duration_ms` | Operation duration |
| `ai.stream` | Whether streaming was used |

### Custom Attributes

```python
with tracer.start_span("custom") as span:
    span.set_attribute("user.id", "user123")
    span.set_attribute("feature.name", "chat")
    span.set_attribute("request.size", 1024)
```

---

## Sampling

### Rate-Based Sampling

```python
configure_tracing(
    sample_rate=0.1,  # Trace 10% of requests
)
```

### Custom Sampler

```python
def custom_sampler(span_name, context, trace_id):
    # Always trace errors
    if span_name.startswith("error"):
        return True
    # Sample other traces
    return random.random() < 0.1

configure_tracing(
    sampler=custom_sampler,
)
```

---

## Error Tracking

```python
with tracer.start_span("operation") as span:
    try:
        result = risky_operation()
    except Exception as e:
        span.record_exception(e)
        span.set_status(StatusCode.ERROR, str(e))
        raise
```

---

## Integration with Logging

```python
from ai_infra import configure_tracing, configure_logging

configure_tracing(service_name="my-service")
configure_logging(level="INFO")

# Logs automatically include trace context
# {"message": "...", "trace_id": "abc123", "span_id": "def456"}
```

---

## Metrics

Tracing can export metrics:

```python
configure_tracing(
    service_name="my-service",
    enable_metrics=True,
    metrics_endpoint="http://prometheus:9090",
)

# Exports metrics:
# - ai_llm_requests_total
# - ai_llm_tokens_total
# - ai_llm_duration_seconds
# - ai_llm_cost_dollars
```

---

## Disable Tracing

```python
# Globally disable
configure_tracing(enabled=False)

# Disable for specific operations
llm = LLM(provider="openai", trace=False)
```

---

## Environment Variables

```bash
# Enable tracing
AI_INFRA_TRACING_ENABLED=true
AI_INFRA_TRACING_ENDPOINT=http://jaeger:4317

# Service info
AI_INFRA_SERVICE_NAME=my-service
AI_INFRA_SERVICE_VERSION=1.0.0
AI_INFRA_ENVIRONMENT=production

# Sampling
AI_INFRA_TRACING_SAMPLE_RATE=0.1
```

---

## See Also

- [Logging](logging.md) - Structured logging
- [Errors](errors.md) - Error handling
- [Callbacks](callbacks.md) - Execution hooks
