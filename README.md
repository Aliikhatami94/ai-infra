# ai-infra

> **Production-ready Python SDK for building AI applications with LLMs, agents, and multimodal capabilities.**

ai-infra provides clean interfaces for chat, agents, embeddings, voice, and image generation across 10+ providers—all with zero-config defaults.

## ✨ Features

- **LLM**: Chat, structured output, streaming, retries, multi-turn conversations
- **Agents**: Tool calling, human-in-the-loop, provider fallbacks, autonomous deep mode
- **Graph**: LangGraph workflows with typed state and conditional branching
- **Embeddings & RAG**: Vector storage, document retrieval, multiple backends
- **Multimodal**: Text-to-speech, speech-to-text, vision, realtime voice
- **Image Generation**: DALL-E, Imagen, Stability AI, Replicate
- **MCP**: Model Context Protocol client/server, OpenAPI→MCP conversion

## 🚀 Quick Start

**5 lines to your first chat:**

```python
from ai_infra import LLM

llm = LLM()  # Auto-detects configured provider
response = llm.chat("What is the capital of France?")
print(response)
```

**With tools (agent):**

```python
from ai_infra import Agent

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

agent = Agent(tools=[get_weather])
result = agent.run("What's the weather in Tokyo?")
print(result)
```

## 📦 Installation

**Python**: 3.11 – 3.13

```bash
# Using pip
pip install ai-infra

# Using Poetry (development)
poetry install
poetry shell
```

## 🔑 Provider Setup

Set API keys for the providers you want to use:

```bash
# Required: At least one chat provider
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...

# Optional: Specialized providers
export ELEVENLABS_API_KEY=...     # TTS
export DEEPGRAM_API_KEY=...       # STT
export STABILITY_API_KEY=...      # Image generation
export REPLICATE_API_TOKEN=...    # Image generation
export VOYAGE_API_KEY=...         # Embeddings
export COHERE_API_KEY=...         # Embeddings
```

## 🔌 Supported Providers

| Provider | Chat | Embeddings | TTS | STT | ImageGen | Realtime |
|----------|:----:|:----------:|:---:|:---:|:--------:|:--------:|
| OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Anthropic | ✅ | - | - | - | - | - |
| Google | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| xAI | ✅ | - | - | - | - | - |
| ElevenLabs | - | - | ✅ | - | - | - |
| Deepgram | - | - | - | ✅ | - | - |
| Stability | - | - | - | - | ✅ | - |
| Replicate | - | - | - | - | ✅ | - |
| Voyage | - | ✅ | - | - | - | - |
| Cohere | - | ✅ | - | - | - | - |

## 📚 Documentation

Full documentation is in the [`docs/`](docs/) folder:

| Section | Description |
|---------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, API keys, first example |
| [Core Modules](docs/core/) | LLM, Agent, Graph, Providers |
| [Multimodal](docs/multimodal/) | TTS, STT, Vision, Realtime Voice |
| [Embeddings & RAG](docs/embeddings/) | Embeddings, VectorStore, Retriever |
| [Tools](docs/tools/) | Schema tools, progress streaming |
| [MCP](docs/mcp/) | Model Context Protocol client/server |
| [Advanced Features](docs/features/) | Personas, Replay, Workspace, Deep Agent |
| [Image Generation](docs/imagegen/) | DALL-E, Imagen, Stability, Replicate |
| [Infrastructure](docs/infrastructure/) | Errors, Logging, Tracing, Callbacks |
| [CLI Reference](docs/cli.md) | Command-line interface |

## 📁 Module Overview

| Module | Description |
|--------|-------------|
| `ai_infra.llm` | LLM chat, agents, structured output, streaming |
| `ai_infra.graph` | LangGraph workflows with typed state |
| `ai_infra.mcp` | MCP client/server, OpenAPI→MCP conversion |
| `ai_infra.embeddings` | Text embeddings across providers |
| `ai_infra.retriever` | RAG with multiple vector store backends |
| `ai_infra.imagegen` | Image generation (DALL-E, Stability, etc.) |
| `ai_infra.providers` | Centralized provider registry |

## 🧪 Examples

See the [`examples/`](examples/) folder for runnable scripts:

```bash
# LLM chat
python -c "from ai_infra.llm.examples.02_llm_chat_basic import main; main()"

# Agent with tools
python -c "from ai_infra.llm.examples.01_agent_basic import main; main()"

# Graph workflow
python -c "from ai_infra.graph.examples.01_graph_basic import main; main()"

# MCP client
python -m ai_infra.mcp.examples.01_mcps
```

## 🛠️ Development

```bash
# Install dev dependencies
poetry install

# Run tests
pytest -q

# Lint
ruff check src tests

# Type check
mypy src

# Format
ruff format
```

## 📄 License

MIT
