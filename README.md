<div align="center">

# ai-infra

[![CI](https://github.com/nfraxlab/ai-infra/actions/workflows/ci.yml/badge.svg)](https://github.com/nfraxlab/ai-infra/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ai-infra.svg)](https://pypi.org/project/ai-infra/)
[![Python](https://img.shields.io/pypi/pyversions/ai-infra.svg)](https://pypi.org/project/ai-infra/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/ai-infra.svg)](https://pypi.org/project/ai-infra/)
[![codecov](https://codecov.io/gh/nfraxlab/ai-infra/branch/main/graph/badge.svg)](https://codecov.io/gh/nfraxlab/ai-infra)

### Build AI applications in minutes, not months

**One unified SDK for LLMs, agents, RAG, voice, images, and MCP—across 10+ providers.**

[Documentation](docs/) · [Examples](examples/) · [PyPI](https://pypi.org/project/ai-infra/)

</div>

---

## Why ai-infra?

Building AI apps means juggling OpenAI, Anthropic, Google, embeddings, vector stores, tool calling, MCP servers... each with different APIs and gotchas.

**ai-infra** gives you one clean interface that works everywhere:

```python
from ai_infra import Agent

def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

agent = Agent(tools=[search_web])
result = agent.run("Find the latest news about AI")
# Works with OpenAI, Anthropic, Google—same code.
```

## Quick Install

```bash
pip install ai-infra
```

## What's Included

| Feature | What You Get | One-liner |
|---------|-------------|-----------|
| **LLM Chat** | Chat, streaming, structured output, retries | `LLM().chat("Hello")` |
| **Agents** | Tool calling, human-in-the-loop, deep mode | `Agent(tools=[...]).run(...)` |
| **RAG** | Embeddings, vector stores, retrieval | `Retriever().search(...)` |
| **MCP** | Client/server, OpenAPI→MCP, tool discovery | `MCPClient(url)` |
| **Voice** | Text-to-speech, speech-to-text, realtime | `TTS().speak(...)` |
| **Images** | DALL-E, Stability, Imagen generation | `ImageGen().generate(...)` |
| **Graph** | LangGraph workflows, typed state | `Graph().add_node(...)` |
| **Memory** | Context fitting, rolling summaries | `fit_context(messages, max_tokens=4000)` |
| **Workspace** | Sandboxed file operations for agents | `Workspace("./project")` |
| **Validation** | Prompt injection, PII detection | `validate_prompt(input)` |
| **Tracing** | OpenTelemetry distributed tracing | `configure_tracing(...)` |

## 30-Second Examples

### Chat with any LLM

```python
from ai_infra import LLM

llm = LLM()  # Uses OPENAI_API_KEY by default
response = llm.chat("Explain quantum computing in one sentence")
print(response)

# Switch providers instantly
llm = LLM(provider="anthropic", model="claude-sonnet-4-20250514")
response = llm.chat("Same question, different model")
```

### Build an Agent with Tools

```python
from ai_infra import Agent

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72F and sunny in {city}"

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Top results for: {query}"

agent = Agent(tools=[get_weather, search_web])
result = agent.run("What's the weather in Tokyo and find me restaurants there")
# Agent automatically calls both tools and synthesizes the answer
```

### RAG in 5 Lines

```python
from ai_infra import Retriever

retriever = Retriever()
retriever.add_file("company_docs.pdf")
retriever.add_file("product_manual.md")

results = retriever.search("How do I reset my password?")
print(results[0].content)
```

### Connect to MCP Servers

```python
from ai_infra import MCPClient

async with MCPClient("http://localhost:8080") as client:
    tools = await client.list_tools()
    result = await client.call_tool("search", {"query": "AI news"})
```

### Create an MCP Server

```python
from ai_infra import mcp_from_functions

def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Found: {query}"

mcp = mcp_from_functions(name="my-mcp", functions=[search_docs])
mcp.run(transport="stdio")
```

## Supported Providers

| Provider | Chat | Embeddings | TTS | STT | Images | Realtime |
|----------|:----:|:----------:|:---:|:---:|:------:|:--------:|
| **OpenAI** | Yes | Yes | Yes | Yes | Yes | Yes |
| **Anthropic** | Yes | - | - | - | - | - |
| **Google** | Yes | Yes | Yes | Yes | Yes | Yes |
| **xAI (Grok)** | Yes | - | - | - | - | - |
| **ElevenLabs** | - | - | Yes | - | - | - |
| **Deepgram** | - | - | - | Yes | - | - |
| **Stability AI** | - | - | - | - | Yes | - |
| **Replicate** | - | - | - | - | Yes | - |
| **Voyage AI** | - | Yes | - | - | - | - |
| **Cohere** | - | Yes | - | - | - | - |

## Setup

```bash
# Set your API keys (use whichever providers you need)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...

# That's it. ai-infra auto-detects available providers.
```

## Feature Highlights

### Deep Agent (Autonomous Mode)

For complex, multi-step tasks:

```python
from ai_infra import DeepAgent

agent = DeepAgent(
    goal="Analyze this codebase and generate documentation",
    tools=[read_file, write_file, search],
    max_iterations=50,
)

result = await agent.run()
print(result.output)
```

**Includes:** Planning, self-correction, progress tracking, human approval gates.

### MCP Client with Interceptors

Advanced MCP features:

```python
from ai_infra import MCPClient
from ai_infra.mcp import RetryInterceptor, CachingInterceptor, LoggingInterceptor

async with MCPClient(
    "http://localhost:8080",
    interceptors=[
        RetryInterceptor(max_retries=3),
        CachingInterceptor(ttl=300),
        LoggingInterceptor(),
    ]
) as client:
    # Automatic retries, caching, and logging for all tool calls
    result = await client.call_tool("expensive_operation", {...})
```

**Includes:** Callbacks, interceptors, prompts, resources, progress tracking.

### RAG with Multiple Backends

```python
from ai_infra import Retriever

# In-memory (development)
retriever = Retriever(backend="memory")

# SQLite (local persistence)
retriever = Retriever(backend="sqlite", path="./vectors.db")

# PostgreSQL with pgvector (production)
retriever = Retriever(backend="postgres", connection_string="...")

# Pinecone (managed cloud)
retriever = Retriever(backend="pinecone", index_name="my-index")
```

### Voice & Multimodal

```python
from ai_infra import TTS, STT

# Text to speech
tts = TTS(provider="elevenlabs")
audio = tts.speak("Hello, world!")

# Speech to text
stt = STT(provider="deepgram")
text = stt.transcribe("audio.mp3")
```

### Image Generation

```python
from ai_infra import ImageGen

gen = ImageGen(provider="openai")  # or "stability", "replicate"
image = gen.generate("A futuristic city at sunset")
image.save("city.png")
```

## CLI Tools

```bash
# Test MCP connections
ai-infra mcp test --url http://localhost:8080

# List MCP tools
ai-infra mcp tools --url http://localhost:8080

# Call an MCP tool
ai-infra mcp call --url http://localhost:8080 --tool search --args '{"query": "test"}'

# Server info
ai-infra mcp info --url http://localhost:8080
```

## Documentation

| Section | Description |
|---------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, API keys, first example |
| **Core** | |
| [LLM](docs/core/llm.md) | Chat, streaming, structured output |
| [Agent](docs/core/agents.md) | Tool calling, human-in-the-loop |
| [Graph](docs/core/graph.md) | LangGraph workflows |
| **RAG & Embeddings** | |
| [Retriever](docs/embeddings/retriever.md) | Vector search, file loading |
| [Embeddings](docs/embeddings/embeddings.md) | Text embeddings |
| **MCP** | |
| [Client](docs/mcp/client.md) | Connect to MCP servers |
| [Server](docs/mcp/server.md) | Create MCP servers |
| **Multimodal** | |
| [TTS](docs/multimodal/tts.md) | Text-to-speech |
| [STT](docs/multimodal/stt.md) | Speech-to-text |
| [Vision](docs/multimodal/vision.md) | Image understanding |
| **Advanced** | |
| [Deep Agent](docs/features/deep-agent.md) | Autonomous agents |
| [Personas](docs/features/personas.md) | Agent personalities |
| [Workspace](docs/features/workspace.md) | Sandboxed file operations |
| [Memory](docs/memory.md) | Context management, rolling summaries |
| [Streaming](docs/streaming.md) | Typed streaming events |
| **Infrastructure** | |
| [Validation](docs/infrastructure/validation.md) | Prompt/response validation |
| [Tracing](docs/infrastructure/tracing.md) | OpenTelemetry tracing |
| [Callbacks](docs/callbacks.md) | Execution hooks |
| [CLI Reference](docs/cli.md) | Command-line tools |

## Running Examples

```bash
git clone https://github.com/nfraxlab/ai-infra.git
cd ai-infra
poetry install

# Chat
poetry run python -c "from ai_infra import LLM; print(LLM().chat('Hello!'))"

# Agent
poetry run python examples/agents/01_basic_tools.py

# See more examples
ls examples/
```

## Related Packages

ai-infra is part of the **nfrax** infrastructure suite:

| Package | Purpose |
|---------|---------|
| **[ai-infra](https://github.com/nfraxlab/ai-infra)** | AI/LLM infrastructure (agents, tools, RAG, MCP) |
| **[svc-infra](https://github.com/nfraxlab/svc-infra)** | Backend infrastructure (auth, billing, jobs, webhooks) |
| **[fin-infra](https://github.com/nfraxlab/fin-infra)** | Financial infrastructure (banking, portfolio, insights) |

## License

MIT License - use it for anything.

---

<div align="center">

**Built by [nfraxlab](https://github.com/nfraxlab)**

[Star us on GitHub](https://github.com/nfraxlab/ai-infra) | [View on PyPI](https://pypi.org/project/ai-infra/)

</div>
