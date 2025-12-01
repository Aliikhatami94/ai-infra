# Callbacks

> Hook into LLM and agent execution lifecycle.

## Quick Start

```python
from ai_infra import LLM, Callbacks

class MyCallbacks(Callbacks):
    async def on_llm_start(self, prompt, **kwargs):
        print(f"Starting LLM call with prompt: {prompt[:50]}...")

    async def on_llm_end(self, response, **kwargs):
        print(f"LLM complete: {response.token_count} tokens")

llm = LLM(provider="openai", callbacks=[MyCallbacks()])
response = await llm.generate("Hello")
```

---

## Overview

Callbacks let you hook into the execution lifecycle for:
- Monitoring and observability
- Cost tracking
- Rate limiting
- Custom logging
- Audit trails
- Progress reporting

---

## Callback Class

### Base Class

```python
from ai_infra import Callbacks

class MyCallbacks(Callbacks):
    async def on_llm_start(self, prompt, **kwargs):
        """Called before LLM inference."""
        pass

    async def on_llm_end(self, response, **kwargs):
        """Called after LLM inference completes."""
        pass

    async def on_llm_error(self, error, **kwargs):
        """Called when LLM inference fails."""
        pass
```

### All Callback Methods

```python
class FullCallbacks(Callbacks):
    # LLM callbacks
    async def on_llm_start(self, prompt, **kwargs): ...
    async def on_llm_end(self, response, **kwargs): ...
    async def on_llm_error(self, error, **kwargs): ...
    async def on_llm_stream_chunk(self, chunk, **kwargs): ...

    # Agent callbacks
    async def on_agent_start(self, goal, **kwargs): ...
    async def on_agent_end(self, result, **kwargs): ...
    async def on_agent_error(self, error, **kwargs): ...
    async def on_agent_step(self, step, **kwargs): ...

    # Tool callbacks
    async def on_tool_start(self, tool_name, args, **kwargs): ...
    async def on_tool_end(self, tool_name, result, **kwargs): ...
    async def on_tool_error(self, tool_name, error, **kwargs): ...

    # Chain callbacks
    async def on_chain_start(self, chain_name, **kwargs): ...
    async def on_chain_end(self, result, **kwargs): ...

    # Retrieval callbacks
    async def on_retrieval_start(self, query, **kwargs): ...
    async def on_retrieval_end(self, documents, **kwargs): ...
```

---

## Using Callbacks

### With LLM

```python
from ai_infra import LLM

llm = LLM(
    provider="openai",
    callbacks=[MyCallbacks(), AnotherCallback()],
)

response = await llm.generate("Hello")
```

### With Agent

```python
from ai_infra import Agent

agent = Agent(
    persona=persona,
    tools=tools,
    callbacks=[AgentMonitor()],
)

result = await agent.run("Do something")
```

### Global Callbacks

```python
from ai_infra import set_global_callbacks

set_global_callbacks([MetricsCallback(), LoggingCallback()])

# All LLM and agent calls will use these callbacks
```

---

## Callback Context

Callbacks receive context about the operation:

```python
class ContextAwareCallbacks(Callbacks):
    async def on_llm_start(self, prompt, **kwargs):
        provider = kwargs.get("provider")
        model = kwargs.get("model")
        request_id = kwargs.get("request_id")
        trace_id = kwargs.get("trace_id")

        print(f"[{request_id}] {provider}/{model}: {prompt[:50]}...")

    async def on_llm_end(self, response, **kwargs):
        tokens = kwargs.get("token_count")
        cost = kwargs.get("cost")
        duration_ms = kwargs.get("duration_ms")

        print(f"Completed in {duration_ms}ms, {tokens} tokens, ${cost:.4f}")
```

---

## Built-in Callbacks

### CostTracker

```python
from ai_infra.callbacks import CostTracker

tracker = CostTracker()

llm = LLM(provider="openai", callbacks=[tracker])
await llm.generate("Hello")
await llm.generate("World")

print(f"Total cost: ${tracker.total_cost:.4f}")
print(f"Total tokens: {tracker.total_tokens}")
print(f"Requests: {tracker.request_count}")
```

### TokenCounter

```python
from ai_infra.callbacks import TokenCounter

counter = TokenCounter()

llm = LLM(provider="openai", callbacks=[counter])
await llm.generate("Hello")

print(f"Prompt tokens: {counter.prompt_tokens}")
print(f"Completion tokens: {counter.completion_tokens}")
```

### StreamingProgress

```python
from ai_infra.callbacks import StreamingProgress

progress = StreamingProgress(
    on_chunk=lambda chunk: print(chunk, end="", flush=True),
    on_complete=lambda: print("\nDone!"),
)

llm = LLM(provider="openai", callbacks=[progress])
async for chunk in llm.stream("Tell me a story"):
    pass
```

---

## Example: Cost Limiting

```python
class CostLimiter(Callbacks):
    def __init__(self, max_cost: float):
        self.max_cost = max_cost
        self.total_cost = 0.0

    async def on_llm_end(self, response, **kwargs):
        cost = kwargs.get("cost", 0)
        self.total_cost += cost

        if self.total_cost >= self.max_cost:
            raise Exception(f"Cost limit ${self.max_cost} exceeded")

# Limit spending to $1.00
limiter = CostLimiter(max_cost=1.0)
llm = LLM(provider="openai", callbacks=[limiter])
```

---

## Example: Audit Logging

```python
class AuditCallback(Callbacks):
    def __init__(self, audit_log):
        self.audit_log = audit_log

    async def on_llm_start(self, prompt, **kwargs):
        self.audit_log.record({
            "event": "llm_request",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": kwargs.get("user_id"),
            "prompt_preview": prompt[:100],
            "provider": kwargs.get("provider"),
            "model": kwargs.get("model"),
        })

    async def on_tool_start(self, tool_name, args, **kwargs):
        self.audit_log.record({
            "event": "tool_call",
            "timestamp": datetime.utcnow().isoformat(),
            "tool": tool_name,
            "args": args,
        })
```

---

## Example: Rate Limiting

```python
import asyncio
from collections import deque
from time import time

class RateLimiter(Callbacks):
    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.timestamps = deque()

    async def on_llm_start(self, prompt, **kwargs):
        now = time()

        # Remove timestamps older than 1 minute
        while self.timestamps and self.timestamps[0] < now - 60:
            self.timestamps.popleft()

        # Wait if at limit
        if len(self.timestamps) >= self.rpm:
            wait_time = 60 - (now - self.timestamps[0])
            await asyncio.sleep(wait_time)

        self.timestamps.append(now)

# Limit to 60 requests per minute
limiter = RateLimiter(requests_per_minute=60)
llm = LLM(provider="openai", callbacks=[limiter])
```

---

## Example: Progress Reporting

```python
class ProgressReporter(Callbacks):
    def __init__(self, on_progress):
        self.on_progress = on_progress
        self.step = 0

    async def on_agent_step(self, step, **kwargs):
        self.step += 1
        self.on_progress({
            "step": self.step,
            "action": step.action,
            "status": "in_progress",
        })

    async def on_agent_end(self, result, **kwargs):
        self.on_progress({
            "step": self.step,
            "status": "complete",
            "result": result,
        })

# WebSocket progress reporting
reporter = ProgressReporter(
    on_progress=lambda p: websocket.send(json.dumps(p))
)
```

---

## Callback Chaining

Multiple callbacks are called in order:

```python
llm = LLM(
    provider="openai",
    callbacks=[
        LoggingCallback(),   # Called first
        CostTracker(),       # Called second
        AuditCallback(),     # Called third
    ],
)
```

---

## Error Handling in Callbacks

```python
class SafeCallback(Callbacks):
    async def on_llm_start(self, prompt, **kwargs):
        try:
            # Do something that might fail
            await external_service.log(prompt)
        except Exception as e:
            # Don't let callback errors break the main flow
            print(f"Callback error: {e}")
```

---

## See Also

- [Logging](logging.md) - Structured logging
- [Tracing](tracing.md) - Distributed tracing
- [Errors](errors.md) - Error handling
