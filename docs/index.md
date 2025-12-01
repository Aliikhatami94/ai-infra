# ai-infra Documentation

> **ai-infra** is a production-ready Python SDK for building AI applications with LLMs, agents, and multimodal capabilities.

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started.md) | Installation, API keys, first example |
| [Core Modules](#core-modules) | LLM, Agent, Graph, Providers |
| [Multimodal](#multimodal) | TTS, STT, Vision, Realtime Voice |
| [Embeddings & RAG](#embeddings--rag) | Embeddings, VectorStore, Retriever |
| [MCP](#mcp) | Model Context Protocol client/server |
| [Advanced Features](#advanced-features) | Personas, Replay, Workspace, Deep Agent |
| [Image Generation](#image-generation) | DALL-E, Imagen, Stability, Replicate |
| [Infrastructure](#infrastructure) | Errors, Logging, Tracing, Callbacks |
| [CLI Reference](cli.md) | Command-line interface |

---

## Start Here

**New to ai-infra?** Start with [Getting Started](getting-started.md) for installation and your first chat in 5 lines.

**Looking for examples?** Check the [`/examples`](../examples/) folder for runnable scripts.

---

## Core Modules

The foundation of ai-infra - chat, agents, and workflows.

| Module | Description | Doc |
|--------|-------------|-----|
| **LLM** | Chat completions with any provider | [llm.md](core/llm.md) |
| **Agent** | Tool-calling agents with HITL support | [agents.md](core/agents.md) |
| **Graph** | LangGraph workflows and state machines | [graph.md](core/graph.md) |
| **Providers** | Centralized provider registry | [providers.md](core/providers.md) |

---

## Multimodal

Voice, speech, and vision capabilities.

| Module | Description | Doc |
|--------|-------------|-----|
| **TTS** | Text-to-speech (OpenAI, ElevenLabs, Google) | [tts.md](multimodal/tts.md) |
| **STT** | Speech-to-text (Whisper, Deepgram, Google) | [stt.md](multimodal/stt.md) |
| **Vision** | Image understanding in chat | [vision.md](multimodal/vision.md) |
| **Realtime** | Speech-to-speech voice API | [realtime.md](multimodal/realtime.md) |

---

## Embeddings & RAG

Vector embeddings and retrieval-augmented generation.

| Module | Description | Doc |
|--------|-------------|-----|
| **Embeddings** | Text embeddings (OpenAI, Voyage, Cohere) | [embeddings.md](embeddings/embeddings.md) |
| **VectorStore** | Document storage and search | [vectorstore.md](embeddings/vectorstore.md) |
| **Retriever** | RAG with multiple backends | [retriever.md](embeddings/retriever.md) |

---

## Tools

Tool utilities for agents.

| Module | Description | Doc |
|--------|-------------|-----|
| **Schema Tools** | Auto-generate CRUD tools from models | [schema-tools.md](tools/schema-tools.md) |
| **Progress** | Stream progress from long-running tools | [progress.md](tools/progress.md) |

---

## MCP

Model Context Protocol - connect to and expose AI tools.

| Module | Description | Doc |
|--------|-------------|-----|
| **MCPClient** | Connect to MCP servers | [client.md](mcp/client.md) |
| **MCPServer** | Expose tools as MCP server | [server.md](mcp/server.md) |
| **OpenAPI** | Convert OpenAPI specs to MCP | [openapi.md](mcp/openapi.md) |

---

## Advanced Features

Power-user capabilities.

| Feature | Description | Doc |
|---------|-------------|-----|
| **Personas** | YAML-driven agent behavior | [personas.md](features/personas.md) |
| **Replay** | Debug workflows with "what-if" | [replay.md](features/replay.md) |
| **Workspace** | Unified file operations | [workspace.md](features/workspace.md) |
| **Deep Agent** | Autonomous multi-step agents | [deep-agent.md](features/deep-agent.md) |

---

## Image Generation

Generate images with multiple providers.

| Module | Description | Doc |
|--------|-------------|-----|
| **ImageGen** | DALL-E, Imagen, Stability, Replicate | [imagegen.md](imagegen/imagegen.md) |

---

## Infrastructure

Cross-cutting concerns for production apps.

| Module | Description | Doc |
|--------|-------------|-----|
| **Errors** | Exception handling | [errors.md](infrastructure/errors.md) |
| **Logging** | Logging configuration | [logging.md](infrastructure/logging.md) |
| **Tracing** | Observability and tracing | [tracing.md](infrastructure/tracing.md) |
| **Callbacks** | Event hooks | [callbacks.md](infrastructure/callbacks.md) |
| **Validation** | Input validation | [validation.md](infrastructure/validation.md) |

---

## Supported Providers

ai-infra supports **10 providers** across **6 capabilities**:

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

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-org/ai-infra/issues)
- **Examples**: [`/examples`](../examples/) folder
- **Source**: [`/src/ai_infra`](../src/ai_infra/)
