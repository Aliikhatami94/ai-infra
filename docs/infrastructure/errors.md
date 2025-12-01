# Error Handling

> Structured errors and exception handling for AI operations.

## Quick Start

```python
from ai_infra import LLMError, RateLimitError, ContentFilterError

try:
    response = await llm.generate(prompt)
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except ContentFilterError as e:
    print(f"Content blocked: {e.reason}")
except LLMError as e:
    print(f"LLM error: {e}")
```

---

## Error Hierarchy

```
AIInfraError (base)
├── LLMError
│   ├── ProviderError
│   ├── RateLimitError
│   ├── ContentFilterError
│   ├── ContextLengthError
│   └── AuthenticationError
├── AgentError
│   ├── AgentTimeoutError
│   ├── AgentMaxIterationsError
│   └── ToolExecutionError
├── EmbeddingError
├── ImageGenError
├── VectorStoreError
└── ValidationError
```

---

## LLM Errors

### RateLimitError

```python
from ai_infra import RateLimitError

try:
    response = await llm.generate(prompt)
except RateLimitError as e:
    print(f"Rate limited by: {e.provider}")
    print(f"Retry after: {e.retry_after} seconds")

    # Automatic retry with backoff
    await asyncio.sleep(e.retry_after or 60)
    response = await llm.generate(prompt)
```

### ContentFilterError

```python
from ai_infra import ContentFilterError

try:
    response = await llm.generate(prompt)
except ContentFilterError as e:
    print(f"Content blocked: {e.reason}")
    print(f"Category: {e.category}")
    # Handle content policy violation
```

### ContextLengthError

```python
from ai_infra import ContextLengthError

try:
    response = await llm.generate(long_prompt)
except ContextLengthError as e:
    print(f"Prompt too long: {e.token_count} tokens")
    print(f"Max allowed: {e.max_tokens}")

    # Truncate or summarize prompt
    shorter_prompt = truncate_prompt(long_prompt, e.max_tokens)
    response = await llm.generate(shorter_prompt)
```

### AuthenticationError

```python
from ai_infra import AuthenticationError

try:
    response = await llm.generate(prompt)
except AuthenticationError as e:
    print(f"Authentication failed for: {e.provider}")
    print(f"Reason: {e.reason}")
    # Check API key configuration
```

### ProviderError

```python
from ai_infra import ProviderError

try:
    response = await llm.generate(prompt)
except ProviderError as e:
    print(f"Provider error: {e.provider}")
    print(f"Status code: {e.status_code}")
    print(f"Message: {e.message}")
```

---

## Agent Errors

### AgentTimeoutError

```python
from ai_infra import AgentTimeoutError

try:
    result = await agent.run(timeout=300)
except AgentTimeoutError as e:
    print(f"Agent timed out after {e.elapsed}s")
    partial = e.partial_result
    print(f"Completed steps: {partial.steps_count}")
```

### AgentMaxIterationsError

```python
from ai_infra import AgentMaxIterationsError

try:
    result = await agent.run(max_iterations=50)
except AgentMaxIterationsError as e:
    print(f"Max iterations ({e.max_iterations}) reached")
    partial = e.partial_result
    print(f"Last action: {partial.last_action}")
```

### ToolExecutionError

```python
from ai_infra import ToolExecutionError

try:
    result = await agent.run()
except ToolExecutionError as e:
    print(f"Tool failed: {e.tool_name}")
    print(f"Error: {e.error}")
    print(f"Arguments: {e.tool_args}")
```

---

## Validation Errors

```python
from ai_infra import ValidationError

try:
    llm = create_llm(invalid_config)
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    for error in e.errors:
        print(f"  - {error.field}: {error.message}")
```

---

## Error Properties

All errors include common properties:

```python
try:
    response = await llm.generate(prompt)
except AIInfraError as e:
    # Common properties
    print(f"Error: {e.message}")
    print(f"Code: {e.code}")
    print(f"Provider: {e.provider}")

    # Context
    print(f"Request ID: {e.request_id}")
    print(f"Timestamp: {e.timestamp}")

    # Original exception
    if e.original_error:
        print(f"Original: {e.original_error}")
```

---

## Retry Logic

### Built-in Retry

```python
from ai_infra import LLM

llm = LLM(
    provider="openai",
    retry_config={
        "max_retries": 3,
        "initial_delay": 1.0,
        "max_delay": 60.0,
        "exponential_base": 2.0,
        "retry_on": [RateLimitError, ProviderError],
    },
)

# Automatic retries with exponential backoff
response = await llm.generate(prompt)
```

### Manual Retry

```python
from ai_infra import retry_with_backoff, RateLimitError

@retry_with_backoff(
    max_retries=3,
    retry_on=[RateLimitError],
)
async def generate_with_retry(prompt: str):
    return await llm.generate(prompt)
```

---

## Error Logging

Errors are automatically logged with context:

```python
import logging

logging.basicConfig(level=logging.INFO)

# Errors include structured context
try:
    response = await llm.generate(prompt)
except LLMError as e:
    # Automatically logged with:
    # - Error type and message
    # - Provider and model
    # - Request ID
    # - Stack trace
    pass
```

---

## Custom Error Handling

### Global Error Handler

```python
from ai_infra import set_error_handler

def custom_handler(error: AIInfraError):
    # Log to monitoring service
    monitoring.log_error(error)

    # Send alert for critical errors
    if error.is_critical:
        alerting.send_alert(error)

set_error_handler(custom_handler)
```

### Context Manager

```python
from ai_infra import error_context

async with error_context(
    on_rate_limit=lambda e: notify_user("Please wait..."),
    on_error=lambda e: log_error(e),
):
    response = await llm.generate(prompt)
```

---

## Error Codes

| Code | Error Type | Description |
|------|------------|-------------|
| `RATE_LIMIT` | RateLimitError | API rate limit exceeded |
| `CONTENT_FILTER` | ContentFilterError | Content policy violation |
| `CONTEXT_LENGTH` | ContextLengthError | Input too long |
| `AUTH_FAILED` | AuthenticationError | Invalid credentials |
| `PROVIDER_ERROR` | ProviderError | Provider-side error |
| `TIMEOUT` | AgentTimeoutError | Operation timed out |
| `MAX_ITERATIONS` | AgentMaxIterationsError | Iteration limit reached |
| `TOOL_ERROR` | ToolExecutionError | Tool execution failed |
| `VALIDATION` | ValidationError | Invalid input |

---

## See Also

- [Logging](logging.md) - Logging configuration
- [Tracing](tracing.md) - Distributed tracing
- [Callbacks](callbacks.md) - Execution hooks
