# Migration Guide

This guide documents breaking changes and migration paths between versions of ai-infra.

## Version Compatibility

| ai-infra | Python | LangChain | LangGraph | Notes |
|----------|--------|-----------|-----------|-------|
| 0.1.x | 3.11+ | >=1.0.0 | >=1.0.0 | Current stable |
| 0.2.x (planned) | 3.11+ | >=1.0.0 | >=1.0.0 | API simplification |

## Migrating to 1.0.0

### Retriever Persistence Format (Breaking)

**v1.0.0 replaces pickle serialization with JSON + numpy for security.**

The Retriever now saves to a directory with `state.json` + `embeddings.npy` instead of a single `retriever.pkl` file.

#### Automatic Migration

Legacy pickle files are automatically detected and loaded with a deprecation warning:

```python
# Old pickle files still work but emit a warning
retriever = Retriever.load("./my_retriever.pkl")
# DeprecationWarning: Loading from legacy pickle format. Run Retriever.migrate() to convert.
```

#### Explicit Migration

Convert legacy files to the new format:

```python
from ai_infra import Retriever

# Migrate and keep the old pickle file as backup
Retriever.migrate("./my_retriever.pkl")

# Migrate and remove the old pickle file
Retriever.migrate("./my_retriever.pkl", remove_pickle=True)
```

After migration, the directory structure is:
```
my_retriever/
    state.json      # Metadata, texts, config
    embeddings.npy  # Embedding vectors (numpy binary)
```

#### New Save/Load API

```python
# Saving (creates directory)
retriever.save("./my_retriever")

# Loading (from directory)
retriever = Retriever.load("./my_retriever")
```

### Why This Change?

Pickle deserialization is a code execution risk (Bandit B301). The new format:
- Uses JSON for metadata (human-readable, safe)
- Uses numpy binary for embeddings (efficient, safe)
- Prevents arbitrary code execution from malicious files

## Migrating to 0.1.x

### From Direct LangChain

If you're migrating from direct LangChain usage:

#### LLM Chat

```python
# Before (direct LangChain)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", api_key="...")
response = llm.invoke([HumanMessage(content="Hello")])

# After (ai-infra)
from ai_infra.llm import CoreLLM

llm = CoreLLM(provider="openai", model="gpt-4")
response = await llm.achat(
    messages=[{"role": "user", "content": "Hello"}]
)
```

#### Structured Output

```python
# Before (direct LangChain)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Answer(BaseModel):
    text: str
    confidence: float

llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(Answer)
result = structured_llm.invoke("What is 2+2?")

# After (ai-infra)
from ai_infra.llm import CoreLLM

llm = CoreLLM(provider="openai", model="gpt-4")
result = await llm.achat(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    output_schema=Answer,
)
```

#### Agents

```python
# Before (direct LangGraph)
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, tools)  # No limit!

# After (ai-infra)
from ai_infra.llm import CoreLLM
from ai_infra.agent import CoreAgent

llm = CoreLLM(provider="openai", model="gpt-4")
agent = CoreAgent(
    llm=llm,
    tools=tools,
    recursion_limit=50,  # REQUIRED
)
```

### From Direct OpenAI SDK

```python
# Before (direct OpenAI)
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
)

# After (ai-infra)
from ai_infra.llm import CoreLLM

llm = CoreLLM(provider="openai", model="gpt-4")
response = await llm.achat(
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Critical Migration Notes

### Recursion Limits

**All agents MUST have recursion limits:**

```python
# Before (dangerous)
agent = create_react_agent(llm, tools)  # INFINITE LOOP POSSIBLE

# After (required)
agent = CoreAgent(llm, tools, recursion_limit=50)  # REQUIRED
```

### Timeouts

**All external calls need timeouts:**

```python
# Before (hangs forever)
response = await mcp_client.call_tool(name, args)

# After (explicit timeout)
response = await asyncio.wait_for(
    mcp_client.call_tool(name, args),
    timeout=60.0
)
```

### Tool Result Truncation

**Truncate before sending to LLM:**

```python
# Before (context explosion)
messages.append({"role": "tool", "content": result})

# After (truncated)
if len(result) > 10000:
    result = result[:10000] + "\n[TRUNCATED]"
messages.append({"role": "tool", "content": result})
```

## Planned Breaking Changes (0.2.x)

### Simplified Imports

```python
# 0.1.x
from ai_infra.llm import CoreLLM
from ai_infra.agent import CoreAgent

# 0.2.x (planned)
from ai_infra import LLM, Agent  # Simpler names
```

### Provider Auto-Detection

```python
# 0.1.x
llm = CoreLLM(provider="openai", model="gpt-4")

# 0.2.x (planned) - auto-detect from model name
llm = LLM(model="gpt-4")  # Infers OpenAI
llm = LLM(model="claude-3")  # Infers Anthropic
```

## Deprecation Notices

### 0.1.100+

- `CoreLLM` -> `LLM` (simpler name)
- `CoreAgent` -> `Agent` (simpler name)
- Explicit `provider` parameter may become optional

### 0.1.150+

- Sync methods deprecated (use async)
- Old callback patterns deprecated

## Getting Help

- Check the [error handling guide](error-handling.md) for exception changes
- Use the LangChain docs MCP for latest patterns
- Open an issue for migration questions
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines
