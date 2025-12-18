# Error Handling Guide

This guide documents exception hierarchies and error handling patterns in ai-infra.

## Exception Hierarchy

```
Exception
└── AIInfraError (base for all ai-infra exceptions)
    ├── LLMError
    │   ├── ProviderNotFoundError
    │   ├── ModelNotFoundError
    │   ├── AuthenticationError
    │   ├── RateLimitError
    │   ├── ContextLengthExceededError
    │   └── ContentFilterError
    ├── AgentError
    │   ├── RecursionLimitExceededError
    │   ├── ToolExecutionError
    │   └── ToolNotFoundError
    ├── MCPError
    │   ├── ConnectionError
    │   ├── TimeoutError
    │   └── ToolCallError
    ├── RetrieverError
    │   ├── EmbeddingError
    │   └── IndexNotFoundError
    └── ValidationError
        ├── SchemaValidationError
        └── OutputParsingError
```

## Critical: AI Safety Error Handling

### Recursion Limit Errors

```python
from ai_infra.exceptions import RecursionLimitExceededError

try:
    result = await agent.run(prompt)
except RecursionLimitExceededError as e:
    logger.warning(f"Agent hit recursion limit: {e}")
    # Return partial results or graceful fallback
    return {"status": "partial", "message": "Task too complex, returning partial results"}
```

### Context Length Errors

```python
from ai_infra.exceptions import ContextLengthExceededError

try:
    response = await llm.chat(messages)
except ContextLengthExceededError:
    # Truncate history and retry
    messages = messages[-10:]  # Keep last 10
    response = await llm.chat(messages)
```

### Tool Execution Errors

```python
from ai_infra.exceptions import ToolExecutionError

try:
    result = await tool.run(args)
except ToolExecutionError as e:
    # Log the error but don't crash the agent
    logger.error(f"Tool {tool.name} failed: {e}")
    return {"error": str(e)}  # Return error to LLM
```

### MCP Timeout Errors

```python
import asyncio
from ai_infra.exceptions import MCPTimeoutError

try:
    result = await asyncio.wait_for(
        mcp_client.call_tool(name, args),
        timeout=60.0
    )
except asyncio.TimeoutError:
    raise MCPTimeoutError(f"Tool {name} timed out after 60s")
```

## Provider-Specific Errors

### OpenAI Errors

```python
from ai_infra.llm.exceptions import OpenAIError

try:
    response = await llm.chat(messages)
except OpenAIError as e:
    if "rate_limit" in str(e).lower():
        await asyncio.sleep(60)  # Wait and retry
        response = await llm.chat(messages)
    raise
```

### Anthropic Errors

```python
from ai_infra.llm.exceptions import AnthropicError

try:
    response = await llm.chat(messages)
except AnthropicError as e:
    if e.status_code == 529:  # Overloaded
        await asyncio.sleep(30)
        response = await llm.chat(messages)
    raise
```

## HTTP Status Code Mapping

| Exception | HTTP Status |
|-----------|-------------|
| AuthenticationError | 401 |
| RateLimitError | 429 |
| ContextLengthExceededError | 400 |
| ContentFilterError | 400 |
| RecursionLimitExceededError | 422 |
| ProviderNotFoundError | 500 |
| MCPTimeoutError | 504 |

## Best Practices

1. **Always set recursion limits** on agent loops
2. **Truncate tool results** before sending to LLM
3. **Add timeouts** to all MCP and provider calls
4. **Log token usage** for cost monitoring
5. **Validate LLM output** before using it
6. **Never use eval()** on LLM-generated code

## Retry Patterns

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(RateLimitError),
)
async def call_llm(messages):
    return await llm.chat(messages)
```
