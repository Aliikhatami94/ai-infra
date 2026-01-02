# ai-infra Examples

Runnable examples demonstrating **all** ai-infra capabilities for AI/LLM application development.

##  Quick Setup

```bash
cd examples
pip install -e ..  # Install ai-infra in development mode
# or: poetry install

# Set your API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

##  API Keys

Different examples require different API keys. Set the ones you need:

| Provider | Environment Variable | Get API Key |
|----------|---------------------|-------------|
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| Anthropic | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| Google | `GOOGLE_API_KEY` | [ai.google.dev](https://ai.google.dev) |
| xAI | `XAI_API_KEY` | [x.ai](https://x.ai) |
| Mistral | `MISTRAL_API_KEY` | [console.mistral.ai](https://console.mistral.ai) |
| Deepseek | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |

You can also create a `.env` file in the examples directory:

```bash
# examples/.env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

##  Example Categories

###  Chat (`chat/`)
Basic LLM interactions - the foundation of ai-infra.

| Example | Description |
|---------|-------------|
| [01_basic.py](chat/01_basic.py) | Simple chat completion |
| [02_streaming.py](chat/02_streaming.py) | Streaming token responses |
| [03_structured_output.py](chat/03_structured_output.py) | JSON/Pydantic structured output |
| [04_multi_provider.py](chat/04_multi_provider.py) | Switch between providers |
| [05_vision.py](chat/05_vision.py) | Vision/multimodal with images |
| [06_conversation.py](chat/06_conversation.py) | Multi-turn conversation |

###  Agents (`agents/`)
Autonomous agents with tool calling and HITL.

| Example | Description |
|---------|-------------|
| [01_basic_tools.py](agents/01_basic_tools.py) | Agent with simple tools |
| [02_structured_tools.py](agents/02_structured_tools.py) | Tools with Pydantic schemas |
| [03_hitl.py](agents/03_hitl.py) | Human-in-the-loop approval |
| [04_fallbacks.py](agents/04_fallbacks.py) | Provider fallback chain |
| [05_mcp_tools.py](agents/05_mcp_tools.py) | Agent using MCP tools |

###  Graph (`graph/`)
LangGraph workflows for complex orchestration.

| Example | Description |
|---------|-------------|
| [01_basic_workflow.py](graph/01_basic_workflow.py) | Simple state machine |
| [02_conditional.py](graph/02_conditional.py) | Conditional branching |
| [03_with_agent.py](graph/03_with_agent.py) | Graph containing agent nodes |
| [04_parallel.py](graph/04_parallel.py) | Parallel execution |

###  MCP (`mcp/`)
Model Context Protocol for tool discovery.

| Example | Description |
|---------|-------------|
| [01_client_basic.py](mcp/01_client_basic.py) | Connect to MCP server |
| [02_server_basic.py](mcp/02_server_basic.py) | Create simple MCP server |
| [03_openapi_to_mcp.py](mcp/03_openapi_to_mcp.py) | Generate MCP from OpenAPI |
| [04_agent_with_mcp.py](mcp/04_agent_with_mcp.py) | Agent using MCP tools |

### 🧮 Embeddings (`embeddings/`)
Vector embeddings for semantic search.

| Example | Description |
|---------|-------------|
| [01_basic.py](embeddings/01_basic.py) | Basic embeddings |
| [02_batch.py](embeddings/02_batch.py) | Batch embeddings with different providers |

### 📚 Retriever (`retriever/`)
RAG (Retrieval-Augmented Generation) pipelines.

| Example | Description |
|---------|-------------|
| [01_basic.py](retriever/01_basic.py) | Add text and search |
| [02_files.py](retriever/02_files.py) | Load PDFs, DOCX, and other files |
| [03_directory.py](retriever/03_directory.py) | Load entire directory with patterns |
| [04_sqlite.py](retriever/04_sqlite.py) | SQLite backend for local persistence |
| [05_postgres.py](retriever/05_postgres.py) | PostgreSQL + pgvector for production |
| [06_chroma.py](retriever/06_chroma.py) | Chroma backend |
| [07_pinecone.py](retriever/07_pinecone.py) | Pinecone managed cloud |
| [08_advanced_search.py](retriever/08_advanced_search.py) | min_score, detailed results |
| [09_with_agent.py](retriever/09_with_agent.py) | RAG as Agent tool |
| [10_async.py](retriever/10_async.py) | Async add and search |

###  Image Generation (`imagegen/`)
AI image generation.

| Example | Description |
|---------|-------------|
| [01_basic.py](imagegen/01_basic.py) | Basic image generation |

### 🎙 Realtime Voice (`realtime/`)
Voice conversations with streaming audio.

| Example | Description |
|---------|-------------|
| [01_basic_voice.py](realtime/01_basic_voice.py) | Basic voice conversation |
| [02_with_tools.py](realtime/02_with_tools.py) | Voice with tool calling |
| [03_explicit_provider.py](realtime/03_explicit_provider.py) | Provider selection |
| [04_model_discovery.py](realtime/04_model_discovery.py) | List providers/models |
| [05_fastapi_integration.py](realtime/05_fastapi_integration.py) | WebSocket integration |

### 🧠 Memory (`memory/`)
Conversation memory and context management.

| Example | Description |
|---------|-------------|
| [01_trim_messages.py](memory/01_trim_messages.py) | Trim messages by count or token limit |
| [02_summarization.py](memory/02_summarization.py) | Auto-summarize long conversations |
| [03_memory_store.py](memory/03_memory_store.py) | Long-term memory with semantic search |
| [04_conversation_memory.py](memory/04_conversation_memory.py) | RAG over conversation history |
| [05_agent_with_memory.py](memory/05_agent_with_memory.py) | Agent with full memory stack |
| [06_production_setup.py](memory/06_production_setup.py) | PostgreSQL backend for production |

## > Running Examples

Each example is a standalone Python script:

```bash
# Run a single example
python chat/01_basic.py

# Run with specific provider
OPENAI_API_KEY=sk-... python chat/01_basic.py

# Run async examples with asyncio
python chat/02_streaming.py
```

##  Testing Examples

To verify all examples work with your configuration:

```bash
# Test all chat examples
for f in chat/*.py; do python "$f"; done

# Test specific category
python -m pytest examples/ -v  # If examples have test mode
```

##  Example Template

All examples follow this structure:

```python
#!/usr/bin/env python
"""Example title and description.

This example demonstrates:
- Feature 1
- Feature 2
- Feature 3

Required API Keys:
- OPENAI_API_KEY (or your preferred provider)
"""

from ai_infra import LLM

def main():
    # Example code here
    llm = LLM()
    response = llm.chat("Hello!")
    print(response.content)

if __name__ == "__main__":
    main()
```

##  Related Documentation

- [ai-infra README](../README.md) - Main package documentation
- [API Reference](https://nfrax.com/ai-infra) - Full API documentation
- [LangChain Docs](https://python.langchain.com) - LangChain integration
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/) - LangGraph workflows
