"""Memory management for ai-infra agents.

This module provides:
1. Short-term memory utilities (trim_messages, summarize_messages)
2. Long-term memory store (MemoryStore with semantic search)
3. Conversation history RAG (ConversationMemory)

Short-term Memory (within a session):
    ```python
    from ai_infra.memory import trim_messages, summarize_messages

    # Trim to last 10 messages
    trimmed = trim_messages(messages, strategy="last", max_messages=10)

    # Trim to fit token limit
    trimmed = trim_messages(messages, strategy="token", max_tokens=4000)

    # Summarize old messages
    result = summarize_messages(messages, keep_last=5)
    ```

Long-term Memory (across sessions):
    ```python
    from ai_infra.memory import MemoryStore

    # In-memory (dev)
    store = MemoryStore()

    # SQLite (single-instance)
    store = MemoryStore.sqlite("./memories.db")

    # PostgreSQL (production)
    store = MemoryStore.postgres(os.environ["DATABASE_URL"])

    # Store and search memories
    store.put(("user_123", "preferences"), "lang", {"value": "Python"})
    results = store.search(("user_123", "preferences"), "programming language")
    ```

Conversation History RAG:
    ```python
    from ai_infra.memory import ConversationMemory, create_memory_tool

    memory = ConversationMemory(backend="sqlite", path="./conversations.db")

    # Index a conversation
    memory.index_conversation(
        user_id="user_123",
        session_id="session_456",
        messages=[...],
    )

    # Search past conversations
    results = memory.search(user_id="user_123", query="authentication bug")

    # Create tool for agent
    recall_tool = create_memory_tool(memory)
    agent = Agent(tools=[recall_tool])
    ```
"""

from ai_infra.llm.memory.store import MemoryItem, MemoryStore
from ai_infra.llm.memory.summarize import (
    SummarizationMiddleware,
    SummarizeResult,
    asummarize_messages,
    summarize_messages,
)
from ai_infra.llm.memory.tokens import count_tokens, count_tokens_approximate, get_context_limit
from ai_infra.llm.memory.trim import trim_messages

# Re-export ConversationMemory from tools/custom for convenience
from ai_infra.llm.tools.custom.memory import (
    ConversationChunk,
    ConversationMemory,
    SearchResult,
    create_memory_tool,
    create_memory_tool_async,
)

__all__ = [
    # Trim utilities (6.4.1)
    "trim_messages",
    # Token utilities (6.4.1)
    "count_tokens",
    "count_tokens_approximate",
    "get_context_limit",
    # Summarization (6.4.1)
    "summarize_messages",
    "asummarize_messages",
    "SummarizeResult",
    "SummarizationMiddleware",
    # Long-term memory store (6.4.2)
    "MemoryStore",
    "MemoryItem",
    # Conversation memory (6.4.3)
    "ConversationMemory",
    "ConversationChunk",
    "SearchResult",
    "create_memory_tool",
    "create_memory_tool_async",
]
