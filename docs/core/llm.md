# LLM Class

> Provider-agnostic chat completions with zero-config initialization.

## Quick Start

```python
from ai_infra import LLM

# Zero-config: auto-detects from environment
llm = LLM()
response = llm.chat("What is 2+2?")
print(response)  # "4"
```

---

## Zero-Config Initialization

ai-infra automatically detects your provider from environment variables:

```python
# Just set OPENAI_API_KEY, then:
llm = LLM()  # Uses OpenAI with gpt-4o-mini

# Or set ANTHROPIC_API_KEY:
llm = LLM()  # Uses Anthropic with claude-3-5-haiku
```

**Detection order**: OpenAI -> Anthropic -> Google -> xAI

---

## Explicit Provider Selection

Override the auto-detection:

```python
from ai_infra import LLM

# Explicit provider
llm = LLM(provider="anthropic", model="claude-sonnet-4-20250514")

# Or pass provider per-call
llm = LLM()
response = llm.chat("Hello", provider="openai", model_name="gpt-4o")
```

---

## Chat Methods

### Synchronous Chat

```python
llm = LLM()

# Simple chat
response = llm.chat("What is Python?")

# With system prompt
response = llm.chat(
    "Explain recursion",
    system="You are a computer science teacher. Be concise."
)

# Multi-turn conversation
messages = [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Hello Alice!"},
    {"role": "user", "content": "What's my name?"},
]
response = llm.chat(messages)
```

### Asynchronous Chat

```python
import asyncio
from ai_infra import LLM

async def main():
    llm = LLM()
    response = await llm.achat("What is async programming?")
    print(response)

asyncio.run(main())
```

### Streaming

```python
llm = LLM()

# Stream tokens as they arrive
for token in llm.stream_tokens("Tell me a story"):
    print(token, end="", flush=True)
```

---

## Structured Output

Get responses as Pydantic models:

```python
from pydantic import BaseModel
from ai_infra import LLM

class Person(BaseModel):
    name: str
    age: int
    occupation: str

llm = LLM()
person = llm.chat(
    "Extract: John is a 30-year-old software engineer",
    response_model=Person
)
print(person.name)  # "John"
print(person.age)   # 30
```

---

## Model Discovery

Discover available providers and models:

```python
from ai_infra import LLM

# List all supported providers
providers = LLM.list_providers()
# ['openai', 'anthropic', 'google_genai', 'xai']

# List only configured providers (have API keys)
configured = LLM.list_configured_providers()
# ['openai', 'anthropic']

# List models for a provider (fetches from API)
models = LLM.list_models("openai")
# ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', ...]

# Check if provider is configured
is_ready = LLM.is_provider_configured("openai")
# True
```

---

## Configuration Options

All LangChain kwargs are supported:

```python
llm = LLM(
    provider="openai",
    model="gpt-4o",

    # Common settings
    temperature=0.7,
    max_tokens=4096,
    top_p=0.9,

    # Infrastructure
    timeout=30,
    max_retries=3,

    # Provider-specific
    organization="org-xxx",  # OpenAI only
)
```

### Per-Call Overrides

```python
llm = LLM(temperature=0.5)

# Override for specific call
response = llm.chat(
    "Be creative",
    extra={"temperature": 0.9}
)
```

---

## Retry Configuration

Built-in exponential backoff:

```python
response = llm.chat(
    "Hello",
    extra={
        "retry": {
            "max_tries": 3,
            "base": 0.5,
            "jitter": 0.2
        }
    }
)
```

---

## Access Underlying Model

Power users can access the LangChain model directly:

```python
llm = LLM(provider="openai")

# Get the underlying LangChain ChatOpenAI instance
langchain_model = llm.get_model()

# Use LangChain directly
from langchain_core.messages import HumanMessage
result = langchain_model.invoke([HumanMessage(content="Hi")])
```

---

## Error Handling

ai-infra translates provider errors into clear exceptions:

```python
from ai_infra import LLM
from ai_infra.errors import (
    AIInfraError,
    AuthenticationError,
    RateLimitError,
    ContextLengthError,
    ModelNotFoundError,
)

llm = LLM()

try:
    response = llm.chat("Hello")
except AuthenticationError as e:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except ContextLengthError as e:
    print(f"Too many tokens: {e.token_count}")
except ModelNotFoundError as e:
    print(f"Model not found: {e.model}")
except AIInfraError as e:
    print(f"AI error: {e}")
```

---

## Logging Hooks

Monitor requests and responses:

```python
llm = LLM()

llm.set_logging_hooks(
    on_request=lambda ctx: print(f"Request to {ctx.provider}/{ctx.model_name}"),
    on_response=lambda ctx: print(f"Response in {ctx.duration_ms:.0f}ms"),
    on_error=lambda ctx: print(f"Error: {ctx.error}"),
)

response = llm.chat("Hello")
```

---

## See Also

- [Agent](agents.md) - Add tools to your LLM
- [Providers](providers.md) - Provider registry details
- [Getting Started](../getting-started.md) - Installation and setup
