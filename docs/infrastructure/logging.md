# Logging

> Structured logging for AI applications.

## Quick Start

```python
from ai_infra import configure_logging, get_logger

configure_logging(level="INFO")
logger = get_logger(__name__)

logger.info("Processing request", request_id="abc123", user_id="user1")
```

---

## Overview

ai-infra provides structured logging that:
- Outputs JSON-formatted logs for production
- Includes context like request IDs and trace IDs
- Integrates with LLM calls and agent execution
- Supports multiple output targets

---

## Configuration

### Basic Setup

```python
from ai_infra import configure_logging

configure_logging(
    level="INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="json",  # "json" or "text"
)
```

### Advanced Configuration

```python
configure_logging(
    level="INFO",
    format="json",
    output="stdout",  # "stdout", "stderr", or file path
    include_timestamp=True,
    include_caller=True,  # Include file/line info
    context={
        "service": "my-service",
        "environment": "production",
    },
)
```

### From Environment

```python
# Reads from AI_INFRA_LOG_LEVEL, AI_INFRA_LOG_FORMAT
configure_logging_from_env()
```

---

## Using the Logger

### Get Logger

```python
from ai_infra import get_logger

logger = get_logger(__name__)
```

### Log Messages

```python
# Basic logging
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# With structured context
logger.info(
    "User action completed",
    user_id="user123",
    action="generate",
    duration_ms=150,
)
```

### Log Exceptions

```python
try:
    result = await risky_operation()
except Exception as e:
    logger.exception("Operation failed", error=str(e))
```

---

## Automatic LLM Logging

LLM calls are automatically logged:

```python
from ai_infra import LLM

llm = LLM(provider="openai")
response = await llm.generate("Hello")

# Automatically logs:
# {
#   "event": "llm.generate.start",
#   "provider": "openai",
#   "model": "gpt-4o",
#   "prompt_tokens": 5
# }
# {
#   "event": "llm.generate.complete",
#   "duration_ms": 450,
#   "total_tokens": 25,
#   "cost": 0.00015
# }
```

### Control Log Level

```python
llm = LLM(
    provider="openai",
    log_level="DEBUG",  # Log all details
)
```

### Disable Logging

```python
llm = LLM(
    provider="openai",
    log_requests=False,
)
```

---

## Agent Logging

Agent execution is logged step-by-step:

```python
from ai_infra import Agent

agent = Agent(persona=persona, tools=tools)
result = await agent.run("Do something")

# Logs:
# {"event": "agent.run.start", "goal": "Do something"}
# {"event": "agent.step", "step": 1, "action": "think"}
# {"event": "agent.tool_call", "tool": "search", "args": {...}}
# {"event": "agent.step", "step": 2, "action": "respond"}
# {"event": "agent.run.complete", "steps": 2, "duration_ms": 2500}
```

---

## Log Context

### Request Context

```python
from ai_infra import with_log_context

async with with_log_context(
    request_id="req-123",
    user_id="user-456",
):
    # All logs within this context include request_id and user_id
    logger.info("Processing started")
    response = await llm.generate(prompt)
    logger.info("Processing complete")
```

### Manual Context

```python
from ai_infra import add_log_context

add_log_context(
    session_id="sess-789",
    feature="chat",
)

# All subsequent logs include this context
logger.info("Message received")
```

---

## Output Formats

### JSON Format (Production)

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "my_module",
  "message": "Request processed",
  "request_id": "abc123",
  "duration_ms": 150,
  "caller": "handler.py:42"
}
```

### Text Format (Development)

```
2024-01-15 10:30:00 INFO [my_module] Request processed (request_id=abc123, duration_ms=150)
```

---

## Log Levels

| Level | Use Case |
|-------|----------|
| `DEBUG` | Detailed debugging information |
| `INFO` | General operational messages |
| `WARNING` | Unexpected situations that aren't errors |
| `ERROR` | Errors that should be investigated |
| `CRITICAL` | System-wide failures |

### Per-Module Levels

```python
configure_logging(
    level="INFO",
    module_levels={
        "ai_infra.llm": "DEBUG",  # Verbose LLM logging
        "ai_infra.tools": "WARNING",  # Only warnings from tools
    },
)
```

---

## Output Targets

### File Output

```python
configure_logging(
    output="/var/log/app/ai.log",
    rotate=True,
    max_size_mb=100,
    backup_count=5,
)
```

### Multiple Outputs

```python
configure_logging(
    outputs=[
        {"target": "stdout", "level": "INFO", "format": "text"},
        {"target": "/var/log/app.log", "level": "DEBUG", "format": "json"},
    ],
)
```

---

## Integration with Tracing

Logs are correlated with traces:

```python
from ai_infra import configure_logging, configure_tracing

configure_logging(level="INFO")
configure_tracing(service_name="my-service")

# Logs automatically include trace_id and span_id
# {
#   "message": "Processing request",
#   "trace_id": "abc123",
#   "span_id": "def456"
# }
```

---

## Sensitive Data

### Redact Sensitive Fields

```python
configure_logging(
    redact_fields=["api_key", "password", "secret"],
)

# api_key will be logged as "[REDACTED]"
logger.info("Config loaded", api_key="sk-123")
# Output: {"message": "Config loaded", "api_key": "[REDACTED]"}
```

### Disable Prompt Logging

```python
configure_logging(
    log_prompts=False,  # Don't log full prompts
    log_responses=False,  # Don't log full responses
)
```

---

## Performance

### Async Logging

```python
configure_logging(
    async_mode=True,  # Non-blocking logging
    queue_size=10000,
)
```

### Sampling

```python
configure_logging(
    sample_rate=0.1,  # Log 10% of DEBUG messages
    sample_levels=["DEBUG"],
)
```

---

## See Also

- [Tracing](tracing.md) - Distributed tracing
- [Errors](errors.md) - Error handling
- [Callbacks](callbacks.md) - Execution hooks
