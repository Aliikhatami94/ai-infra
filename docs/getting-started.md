# Getting Started with ai-infra

> Get from zero to AI-powered chat in under 5 minutes.

## Installation

```bash
pip install ai-infra
```

Or with Poetry:

```bash
poetry add ai-infra
```

---

## API Keys

ai-infra auto-detects providers from environment variables. Set at least one:

| Provider | Environment Variable | Alternative |
|----------|---------------------|-------------|
| OpenAI | `OPENAI_API_KEY` | - |
| Anthropic | `ANTHROPIC_API_KEY` | - |
| Google | `GEMINI_API_KEY` | `GOOGLE_API_KEY`, `GOOGLE_GENAI_API_KEY` |
| xAI | `XAI_API_KEY` | - |
| ElevenLabs | `ELEVENLABS_API_KEY` | `ELEVEN_API_KEY` |
| Deepgram | `DEEPGRAM_API_KEY` | - |
| Stability | `STABILITY_API_KEY` | - |
| Replicate | `REPLICATE_API_TOKEN` | - |
| Voyage | `VOYAGE_API_KEY` | - |
| Cohere | `COHERE_API_KEY` | `CO_API_KEY` |

**Recommended**: Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

ai-infra automatically loads `.env` files on import.

---

## Quick Start: Your First Chat

```python
from ai_infra import LLM

# Zero-config: auto-detects provider from env vars
llm = LLM()
response = llm.chat("What is 2 + 2?")
print(response)  # "4"
```

That's it! With `OPENAI_API_KEY` set, you have a working AI chat.

---

## Explicit Provider Selection

```python
from ai_infra import LLM

# Use a specific provider
llm = LLM()
response = llm.chat("Hello!", provider="anthropic", model_name="claude-sonnet-4-20250514")
print(response)
```

---

## Streaming Responses

```python
from ai_infra import LLM

llm = LLM()
for token in llm.stream_tokens("Tell me a story"):
    print(token, end="", flush=True)
```

---

## Structured Output

Get typed responses with Pydantic:

```python
from pydantic import BaseModel
from ai_infra import LLM

class Answer(BaseModel):
    value: int
    explanation: str

llm = LLM()
result = llm.chat(
    "What is 2 + 2?",
    response_model=Answer
)
print(result.value)        # 4
print(result.explanation)  # "Two plus two equals four"
```

---

## Agents with Tools

Create agents that can use tools:

```python
from ai_infra import Agent

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

agent = Agent(tools=[get_weather])
result = agent.run("What's the weather in San Francisco?")
print(result)  # Uses the tool and responds
```

---

## Discover Providers & Models

```python
from ai_infra import list_providers, list_providers_for_capability, ProviderCapability

# List all registered providers
print(list_providers())
# ['openai', 'anthropic', 'google_genai', 'xai', 'elevenlabs', ...]

# List providers for a specific capability
chat_providers = list_providers_for_capability(ProviderCapability.CHAT)
print(chat_providers)  # ['openai', 'anthropic', 'google_genai', 'xai']

# List available models (dynamic discovery)
from ai_infra import LLM
models = LLM.list_models(provider="openai")
print(models)  # ['gpt-4o', 'gpt-4o-mini', ...]
```

---

## Default Models

When you don't specify a model, ai-infra uses sensible defaults:

| Provider | Default Model |
|----------|--------------|
| OpenAI | `gpt-4o-mini` |
| Anthropic | `claude-3-5-haiku-latest` |
| Google | `gemini-2.0-flash` |
| xAI | `grok-3-mini` |

---

## Provider Auto-Detection Order

When no provider is specified, ai-infra checks for API keys in this order:

1. `OPENAI_API_KEY` → OpenAI
2. `ANTHROPIC_API_KEY` → Anthropic
3. `GEMINI_API_KEY` / `GOOGLE_API_KEY` → Google
4. `XAI_API_KEY` → xAI

The first provider with a configured API key is used.

---

## Feature Overview

| Feature | Description | Example |
|---------|-------------|---------|
| **LLM** | Chat with any provider | `LLM().chat("Hi")` |
| **Agent** | Tool-calling agents | `Agent(tools=[...]).run()` |
| **Graph** | LangGraph workflows | `Graph().add_node()` |
| **TTS** | Text-to-speech | `TTS().speak("Hello")` |
| **STT** | Speech-to-text | `STT().transcribe(audio)` |
| **Realtime** | Voice conversations | `RealtimeVoice().connect()` |
| **Embeddings** | Text embeddings | `Embeddings().embed("text")` |
| **Retriever** | RAG pipelines | `Retriever().add().search()` |
| **ImageGen** | Image generation | `ImageGen().generate("cat")` |
| **MCP** | Tool servers | `MCPServer().serve()` |

---

## Next Steps

- **[LLM Documentation](core/llm.md)** - Deep dive into chat completions
- **[Agent Documentation](core/agents.md)** - Build tool-calling agents
- **[Provider Registry](core/providers.md)** - Understand provider configuration
- **[Examples](/examples/)** - Runnable code examples

---

## Common Patterns

### Multi-Provider Comparison

```python
from ai_infra import LLM, list_providers_for_capability, ProviderCapability

llm = LLM()
prompt = "Explain quantum computing in one sentence."

for provider in list_providers_for_capability(ProviderCapability.CHAT):
    try:
        response = llm.chat(prompt, provider=provider)
        print(f"{provider}: {response}")
    except Exception as e:
        print(f"{provider}: Not configured")
```

### Async Chat

```python
import asyncio
from ai_infra import LLM

async def main():
    llm = LLM()
    response = await llm.achat("What is AI?")
    print(response)

asyncio.run(main())
```

### With Retry Logic

```python
from ai_infra import LLM

llm = LLM()
response = llm.chat(
    "Hello!",
    extra={"retry": {"max_tries": 3, "base": 0.5}}
)
```

---

## Troubleshooting

### "No API key found"

```
AIInfraError: No API key found. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, ...
```

**Solution**: Set at least one API key environment variable or pass `api_key=` to the constructor.

### Provider not available

```python
from ai_infra import is_provider_configured

if is_provider_configured("openai"):
    # Safe to use OpenAI
    ...
```

### Model not found

Use `LLM.list_models(provider)` to see available models for a provider.
