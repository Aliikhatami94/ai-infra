# Structured Logging

ai-infra provides structured logging utilities for consistent observability.

## Quick Start

```python
from ai_infra import get_logger

logger = get_logger(__name__)

logger.info("Processing request", user_id="123", action="chat")
# Output: 2024-01-15 10:30:00 INFO [my_module] Processing request user_id=123 action=chat
```

## Loggers

### StructuredLogger

Base logger with structured output:

```python
from ai_infra import StructuredLogger

logger = StructuredLogger(
    name="my_app",
    level="INFO",
    format="json",  # or "human"
)

logger.info("Event", key1="value1", key2="value2")
```

### RequestLogger

For HTTP request/response logging:

```python
from ai_infra import RequestLogger

logger = RequestLogger()

logger.log_request(
    method="POST",
    url="/api/chat",
    headers={"Authorization": "Bearer ..."},
    body={"message": "Hello"},
)

logger.log_response(
    status_code=200,
    headers={},
    body={"response": "Hi there"},
    duration_ms=150.5,
)
```

### LLMLogger

For LLM-specific logging:

```python
from ai_infra import LLMLogger

logger = LLMLogger()

logger.log_call(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    response="Hi there!",
    duration_ms=150.5,
    token_usage={"input": 10, "output": 5},
)
```

## Formatters

### JSON Formatter

Structured JSON output for log aggregators:

```python
from ai_infra import JSONFormatter

formatter = JSONFormatter()
# Output: {"timestamp": "...", "level": "INFO", "message": "...", "data": {...}}
```

### Human Formatter

Readable output for development:

```python
from ai_infra import HumanFormatter

formatter = HumanFormatter(colorize=True)
# Output: 2024-01-15 10:30:00 INFO [module] Message key=value
```

## Integration with Python Logging

```python
import logging
from ai_infra import StructuredLogger, JSONFormatter

# Get standard logger
logger = logging.getLogger("my_app")

# Add structured handler
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

## Log Levels

| Level | Usage |
|-------|-------|
| DEBUG | Detailed debugging information |
| INFO | General operational messages |
| WARNING | Unexpected but handled situations |
| ERROR | Errors that need attention |
| CRITICAL | System failures |

## Context Managers

### Log Context

Add context to all logs within a block:

```python
from ai_infra import log_context

with log_context(request_id="abc123", user_id="user1"):
    logger.info("Processing")  # Includes request_id and user_id
    logger.info("Done")        # Also includes both
```

## Best Practices

1. **Use structured data**: Pass key-value pairs, not formatted strings
2. **Include request IDs**: For tracing across services
3. **Don't log secrets**: Filter sensitive data
4. **Use appropriate levels**: INFO for normal ops, ERROR for real problems
5. **Add context**: Include user_id, request_id, session_id where relevant
