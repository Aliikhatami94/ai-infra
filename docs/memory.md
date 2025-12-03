# Memory Management

ai-infra provides comprehensive memory capabilities for building production agents:

- **Short-term memory**: Manage context window with trimming and summarization
- **Long-term memory**: Key-value store with semantic search across sessions
- **Conversation RAG**: Search and recall past conversations via agent tool

## Quick Start

```python
from ai_infra import Agent
from ai_infra.llm.memory import trim_messages, MemoryStore
from ai_infra.llm.tools.custom import ConversationMemory, create_memory_tool

# 1. Trim messages to fit context window
trimmed = trim_messages(messages, strategy="token", max_tokens=4000)

# 2. Store long-term memories
store = MemoryStore()
store.put(("user_123", "prefs"), "language", {"value": "Python"})

# 3. Agent with conversation recall
memory = ConversationMemory()
recall_tool = create_memory_tool(memory, user_id="user_123")
agent = Agent(tools=[recall_tool])
```

---

## Session vs Memory

| | **Session** | **Memory** |
|---|---|---|
| **Module** | `ai_infra.llm.session` | `ai_infra.llm.memory` |
| **Purpose** | Persist one conversation thread | Store/recall across ALL conversations |
| **Scope** | Single `session_id` | User-wide (`user_id` + `namespace`) |
| **Use case** | "Continue this chat tomorrow" | "What did we discuss last month?" |

```python
from ai_infra import Agent
from ai_infra.llm.session import memory as session_memory

# Session: same conversation across requests
agent = Agent(session=session_memory())
agent.run("Hi, I'm Bob", session_id="thread_123")
agent.run("What's my name?", session_id="thread_123")  # Knows "Bob"

# Memory: recall across ALL past conversations
recall_tool = create_memory_tool(conv_memory, user_id="user_123")
agent = Agent(tools=[recall_tool])
agent.run("How did we fix that auth bug last month?")  # Searches all history
```

---

## Short-Term Memory

Utilities for managing context window within a single conversation.

### trim_messages

Reduce message history to fit context limits.

```python
from ai_infra.llm.memory import trim_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="You are helpful."),
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there!"),
    HumanMessage(content="Tell me about Python"),
    AIMessage(content="Python is a programming language..."),
    # ... many more messages
]

# Strategy: "last" - Keep last N messages
trimmed = trim_messages(messages, strategy="last", max_messages=10)

# Strategy: "first" - Keep first N messages  
trimmed = trim_messages(messages, strategy="first", max_messages=10)

# Strategy: "token" - Fit within token limit
trimmed = trim_messages(messages, strategy="token", max_tokens=4000)

# Preserve system message (default: True)
trimmed = trim_messages(
    messages,
    strategy="last",
    max_messages=5,
    preserve_system=True,  # System message always kept
)
```

### count_tokens

Count tokens in messages for context management.

```python
from ai_infra.llm.memory import count_tokens, count_tokens_approximate, get_context_limit

# Approximate count (fast, no dependencies)
count = count_tokens_approximate(messages)

# Exact count with tiktoken (requires tiktoken package)
count = count_tokens(messages, model="gpt-4")

# Get context limit for a model
limit = get_context_limit("gpt-4")  # 128000
limit = get_context_limit("claude-3-opus")  # 200000
```

### summarize_messages

Compress old messages into a summary to preserve context while freeing tokens.

```python
from ai_infra.llm.memory import summarize_messages, asummarize_messages

# Summarize old messages, keep last 5
result = summarize_messages(
    messages,
    keep_last=5,
    llm_provider="openai",  # LLM for summarization
    llm_model="gpt-4o-mini",  # Fast, cheap model
)

# Result contains:
# - result.summary: str - The summary text
# - result.messages: List - Summary + kept messages
# - result.original_count: int - Original message count
# - result.summarized_count: int - How many were summarized

# Async version
result = await asummarize_messages(messages, keep_last=5)
```

### SummarizationMiddleware

Auto-summarize when context gets too long (for use with Agent middleware).

```python
from ai_infra.llm.memory import SummarizationMiddleware

middleware = SummarizationMiddleware(
    trigger_tokens=8000,  # Summarize when exceeds this
    # OR
    trigger_messages=50,  # Summarize when exceeds this count
    keep_last=10,  # Always keep last N messages
)

# Check if summarization needed
if middleware.should_summarize(messages):
    result = middleware.process(messages)
```

---

## Long-Term Memory Store

Key-value store with semantic search for persistent memories across sessions.

### Basic Usage

```python
from ai_infra.llm.memory import MemoryStore, MemoryItem

# Create store (in-memory for dev)
store = MemoryStore()

# Store a memory
store.put(
    namespace=("user_123", "preferences"),
    key="language",
    value={"preference": "Python", "reason": "Data science work"},
    ttl=86400,  # Optional: expires in 24 hours
)

# Retrieve a memory
item = store.get(("user_123", "preferences"), "language")
if item:
    print(item.value)  # {"preference": "Python", ...}

# Delete a memory
store.delete(("user_123", "preferences"), "language")

# List all memories in namespace
items = store.list(("user_123", "preferences"))
```

### Semantic Search

Search memories by meaning (requires embeddings).

```python
from ai_infra.llm.memory import MemoryStore

# Create store with embeddings for search
store = MemoryStore(
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)

# Store some memories
store.put(("user_123", "facts"), "prog_lang", {
    "value": "User prefers Python for data science"
})
store.put(("user_123", "facts"), "editor", {
    "value": "User uses VS Code with Vim keybindings"
})
store.put(("user_123", "facts"), "os", {
    "value": "User runs macOS on M2 MacBook Pro"
})

# Search by meaning
results = store.search(
    namespace=("user_123", "facts"),
    query="What programming tools does the user like?",
    limit=3,
)

for item, score in results:
    print(f"{item.key}: {item.value} (score: {score:.3f})")
```

### Convenience Methods

```python
# User-scoped memories
store.put_user_memory("user_123", "theme", {"value": "dark"})
item = store.get_user_memory("user_123", "theme")

# Session-scoped memories
store.put_session_memory("session_456", "context", {"value": "debugging"})
item = store.get_session_memory("session_456", "context")

# Global memories
store.put_global_memory("app_version", {"value": "1.0.0"})
item = store.get_global_memory("app_version")
```

### Backends

```python
from ai_infra.llm.memory import MemoryStore

# In-memory (development, testing)
store = MemoryStore()

# SQLite (single-instance production)
store = MemoryStore.sqlite("./memories.db")

# PostgreSQL (multi-instance production)
store = MemoryStore.postgres(os.environ["DATABASE_URL"])
```

| Backend | Use Case | Persistence | Multi-Instance |
|---------|----------|-------------|----------------|
| `MemoryStore()` | Dev/testing | ❌ | ❌ |
| `MemoryStore.sqlite()` | Single server | ✅ | ❌ |
| `MemoryStore.postgres()` | Production | ✅ | ✅ |

---

## Conversation History RAG

Index and search across ALL past conversations with a user.

### ConversationMemory

```python
from ai_infra.llm.tools.custom import ConversationMemory
from langchain_core.messages import HumanMessage, AIMessage

# Create memory (in-memory for dev)
memory = ConversationMemory(
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)

# Index a completed conversation
memory.index_conversation(
    user_id="user_123",
    session_id="session_001",
    messages=[
        HumanMessage(content="I'm getting an auth error"),
        AIMessage(content="Let's check your JWT token..."),
        HumanMessage(content="It was expired!"),
        AIMessage(content="You can refresh it with..."),
    ],
    metadata={"topic": "authentication", "date": "2025-01-15"},
)

# Search across all conversations
results = memory.search(
    user_id="user_123",
    query="how did we fix the authentication issue?",
    limit=3,
)

for result in results:
    print(f"Session: {result.chunk.session_id}")
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.chunk.text[:200]}...")
```

### Chunking Options

Long conversations are split into chunks for better retrieval:

```python
memory = ConversationMemory(
    chunk_size=10,         # Messages per chunk (default: 10)
    chunk_overlap=2,       # Overlap between chunks (default: 2)
    include_summary=True,  # Add summary to each chunk (default: False)
)
```

### Backends

```python
# In-memory (development)
memory = ConversationMemory()

# SQLite (single-instance)
memory = ConversationMemory(backend="sqlite", path="./conversations.db")

# PostgreSQL (production)
memory = ConversationMemory(
    backend="postgres",
    connection_string=os.environ["DATABASE_URL"],
)
```

### Agent Integration

Create a tool that lets agents recall past conversations:

```python
from ai_infra import Agent
from ai_infra.llm.tools.custom import ConversationMemory, create_memory_tool

# Set up conversation memory
memory = ConversationMemory(
    backend="postgres",
    connection_string=os.environ["DATABASE_URL"],
    embedding_provider="openai",
)

# Create recall tool for a specific user
recall_tool = create_memory_tool(
    memory,
    user_id="user_123",  # Scoped to this user
    name="recall_past_conversations",
    description="Search through past conversations to find relevant context.",
    limit=3,
)

# Agent can now recall past conversations
agent = Agent(
    tools=[recall_tool],
    system="You are helpful. Use recall_past_conversations when the user references previous discussions.",
)

# Agent will use the tool to answer this
result = agent.run("How did we fix that caching issue last month?")
```

### Async Support

```python
# Async indexing
await memory.aindex_conversation(
    user_id="user_123",
    session_id="session_002",
    messages=[...],
)

# Async search
results = await memory.asearch(
    user_id="user_123",
    query="authentication",
)

# Async tool
recall_tool = create_memory_tool_async(memory, user_id="user_123")
```

---

## Production Setup

Complete example for production deployment:

```python
import os
from ai_infra import Agent
from ai_infra.llm.session import postgres as session_postgres
from ai_infra.llm.memory import MemoryStore, SummarizationMiddleware
from ai_infra.llm.tools.custom import ConversationMemory, create_memory_tool

DATABASE_URL = os.environ["DATABASE_URL"]

# 1. Session storage (short-term, per-thread)
session_storage = session_postgres(DATABASE_URL)

# 2. Memory store (long-term facts)
memory_store = MemoryStore.postgres(DATABASE_URL)

# 3. Conversation memory (long-term conversation history)
conv_memory = ConversationMemory(
    backend="postgres",
    connection_string=DATABASE_URL,
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)

# 4. Create recall tool
def get_recall_tool(user_id: str):
    return create_memory_tool(
        conv_memory,
        user_id=user_id,
        limit=5,
    )

# 5. Create agent
def create_agent_for_user(user_id: str) -> Agent:
    return Agent(
        tools=[get_recall_tool(user_id)],
        session=session_storage,
        system="You are a helpful assistant. Use recall_past_conversations when users reference previous discussions.",
    )

# Usage
agent = create_agent_for_user("user_123")

# Run with session (conversation persists)
result = agent.run(
    "How did we fix that bug last week?",
    session_id="thread_789",
)

# After conversation ends, index it for future recall
# (Call this when session ends or periodically)
conv_memory.index_conversation(
    user_id="user_123",
    session_id="thread_789",
    messages=result.messages,  # Get from session
    metadata={"date": "2025-01-20"},
)
```

---

## API Reference

### ai_infra.llm.memory

| Function/Class | Description |
|----------------|-------------|
| `trim_messages(messages, strategy, ...)` | Trim messages by count or tokens |
| `count_tokens(messages, model)` | Exact token count (requires tiktoken) |
| `count_tokens_approximate(messages)` | Fast approximate token count |
| `get_context_limit(model)` | Get context window size for model |
| `summarize_messages(messages, ...)` | Summarize old messages |
| `asummarize_messages(messages, ...)` | Async summarization |
| `SummarizationMiddleware` | Auto-summarize middleware |
| `MemoryStore` | Key-value store with semantic search |
| `MemoryItem` | Item returned from MemoryStore |

### ai_infra.llm.tools.custom

| Function/Class | Description |
|----------------|-------------|
| `ConversationMemory` | Index and search past conversations |
| `ConversationChunk` | Chunk of indexed conversation |
| `SearchResult` | Result from conversation search |
| `create_memory_tool(memory, user_id, ...)` | Create Agent-compatible recall tool |
| `create_memory_tool_async(memory, user_id, ...)` | Async version |
