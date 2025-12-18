"""Memory tool for agent conversation recall.

This module provides tools for agents to search and recall past conversations.
It includes:
1. ConversationMemory - Indexes and searches conversation history
2. create_memory_tool() - Creates an Agent-compatible tool for recall

Example:
    ```python
    from ai_infra import Agent
    from ai_infra.llm.tools.custom import ConversationMemory, create_memory_tool

    # Create conversation memory
    memory = ConversationMemory(
        backend="sqlite",
        path="./conversations.db",
        embedding_provider="openai",
    )

    # Index a past conversation
    memory.index_conversation(
        user_id="user_123",
        session_id="session_456",
        messages=[...],
        metadata={"topic": "debugging"},
    )

    # Create tool for agent
    recall_tool = create_memory_tool(memory)

    # Agent can now search past conversations
    agent = Agent(tools=[recall_tool])
    result = agent.run(
        "How did we fix the auth bug?",
        user_id="user_123",
    )
    ```
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ai_infra.llm.llm import LLM

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Type alias for messages
MessageLike = BaseMessage | dict[str, Any]


@dataclass
class ConversationChunk:
    """A chunk of conversation that has been indexed."""

    chunk_id: str
    """Unique identifier for this chunk."""

    user_id: str
    """User who participated in this conversation."""

    session_id: str
    """Session identifier."""

    messages: list[BaseMessage]
    """The messages in this chunk."""

    text: str
    """Text representation of the conversation (for embedding)."""

    summary: str | None = None
    """Optional summary of this chunk."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (topic, date, etc.)."""

    created_at: float = field(default_factory=time.time)
    """Unix timestamp when indexed."""

    score: float | None = None
    """Similarity score (populated during search)."""


@dataclass
class SearchResult:
    """A search result from conversation memory."""

    chunk: ConversationChunk
    """The conversation chunk that matched."""

    score: float
    """Similarity score (0-1)."""

    context: str
    """Formatted text of the conversation for display."""


class ConversationMemory:
    """Index and search past conversations with semantic search.

    Provides long-term memory by storing and searching conversation history
    across multiple sessions. Each conversation is chunked and embedded
    for semantic search.

    Supports three storage backends:
    - "memory": In-memory (dev/testing, not persisted)
    - "sqlite": SQLite file (single-instance production)
    - "postgres": PostgreSQL (multi-instance production)

    Example:
        ```python
        from ai_infra.llm.tools.custom import ConversationMemory

        # SQLite for single server
        memory = ConversationMemory(
            backend="sqlite",
            path="./conversations.db",
            embedding_provider="openai",
        )

        # Index a conversation
        memory.index_conversation(
            user_id="user_123",
            session_id="session_abc",
            messages=[
                HumanMessage(content="How do I debug Python?"),
                AIMessage(content="You can use pdb or breakpoints..."),
            ],
            metadata={"topic": "debugging"},
        )

        # Search past conversations
        results = memory.search(
            user_id="user_123",
            query="debugging python code",
            limit=5,
        )
        for r in results:
            print(f"Score: {r.score:.2f}")
            print(r.context)
        ```
    """

    def __init__(
        self,
        *,
        backend: Literal["memory", "sqlite", "postgres"] = "memory",
        path: str | None = None,
        connection_string: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        chunk_size: int = 10,
        chunk_overlap: int = 2,
        include_summary: bool = False,
    ):
        """Initialize ConversationMemory.

        Args:
            backend: Storage backend ("memory", "sqlite", "postgres")
            path: File path for SQLite backend
            connection_string: Connection string for PostgreSQL backend
            embedding_provider: Provider for embeddings (e.g., "openai", "huggingface")
            embedding_model: Model for embeddings (e.g., "text-embedding-3-small")
            chunk_size: Number of messages per chunk (default: 10)
            chunk_overlap: Number of messages to overlap between chunks (default: 2)
            include_summary: Whether to generate summaries for each chunk (default: False)

        Example:
            ```python
            # In-memory for testing
            memory = ConversationMemory()

            # SQLite with OpenAI embeddings
            memory = ConversationMemory(
                backend="sqlite",
                path="./conversations.db",
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
            )

            # PostgreSQL for production
            memory = ConversationMemory(
                backend="postgres",
                connection_string=os.environ["DATABASE_URL"],
                embedding_provider="openai",
            )
            ```
        """
        self._backend_type = backend
        self._path = path
        self._connection_string = connection_string
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._include_summary = include_summary

        # Initialize storage using MemoryStore
        from ai_infra.llm.memory.store import MemoryStore

        if backend == "sqlite":
            if not path:
                raise ValueError("path is required for sqlite backend")
            self._store = MemoryStore.sqlite(
                path,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )
        elif backend == "postgres":
            if not connection_string:
                raise ValueError("connection_string is required for postgres backend")
            self._store = MemoryStore.postgres(
                connection_string,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )
        else:
            self._store = MemoryStore(
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )

        # LLM for summaries (lazy init)
        self._llm: LLM | None = None

    @classmethod
    def sqlite(
        cls,
        path: str,
        *,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        chunk_size: int = 10,
        chunk_overlap: int = 2,
        include_summary: bool = False,
    ) -> ConversationMemory:
        """Create a SQLite-backed ConversationMemory.

        Args:
            path: Path to SQLite database file
            embedding_provider: Provider for embeddings
            embedding_model: Model for embeddings
            chunk_size: Messages per chunk
            chunk_overlap: Overlap between chunks
            include_summary: Generate summaries

        Returns:
            ConversationMemory with SQLite backend
        """
        return cls(
            backend="sqlite",
            path=path,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_summary=include_summary,
        )

    @classmethod
    def postgres(
        cls,
        connection_string: str,
        *,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        chunk_size: int = 10,
        chunk_overlap: int = 2,
        include_summary: bool = False,
    ) -> ConversationMemory:
        """Create a PostgreSQL-backed ConversationMemory.

        Args:
            connection_string: PostgreSQL connection string
            embedding_provider: Provider for embeddings
            embedding_model: Model for embeddings
            chunk_size: Messages per chunk
            chunk_overlap: Overlap between chunks
            include_summary: Generate summaries

        Returns:
            ConversationMemory with PostgreSQL backend
        """
        return cls(
            backend="postgres",
            connection_string=connection_string,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_summary=include_summary,
        )

    def index_conversation(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[MessageLike],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Index a conversation for later search.

        Splits the conversation into chunks and stores them with embeddings
        for semantic search.

        Args:
            user_id: User identifier (for scoping searches)
            session_id: Session identifier (unique conversation ID)
            messages: The conversation messages to index
            metadata: Additional metadata (topic, date, etc.)

        Returns:
            List of chunk IDs that were created

        Example:
            ```python
            chunk_ids = memory.index_conversation(
                user_id="user_123",
                session_id="session_456",
                messages=[
                    HumanMessage(content="What is Python?"),
                    AIMessage(content="Python is a programming language..."),
                ],
                metadata={"topic": "programming", "date": "2025-01-15"},
            )
            print(f"Indexed {len(chunk_ids)} chunks")
            ```
        """
        # Normalize messages
        normalized = self._normalize_messages(messages)

        if not normalized:
            return []

        # Split into chunks
        chunks = self._create_chunks(
            user_id=user_id,
            session_id=session_id,
            messages=normalized,
            metadata=metadata or {},
        )

        # Store each chunk
        chunk_ids = []
        for chunk in chunks:
            self._store.put(
                namespace=(user_id, "conversations"),
                key=chunk.chunk_id,
                value={
                    "session_id": chunk.session_id,
                    "text": chunk.text,
                    "summary": chunk.summary,
                    "metadata": chunk.metadata,
                    "messages": [self._message_to_dict(m) for m in chunk.messages],
                    "created_at": chunk.created_at,
                },
            )
            chunk_ids.append(chunk.chunk_id)
            logger.debug(f"Indexed chunk {chunk.chunk_id} for user {user_id}")

        logger.info(f"Indexed {len(chunk_ids)} chunks for user={user_id}, session={session_id}")
        return chunk_ids

    async def aindex_conversation(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[MessageLike],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Async version of index_conversation.

        See index_conversation for full documentation.
        """
        # Normalize messages
        normalized = self._normalize_messages(messages)

        if not normalized:
            return []

        # Split into chunks
        chunks = self._create_chunks(
            user_id=user_id,
            session_id=session_id,
            messages=normalized,
            metadata=metadata or {},
        )

        # Store each chunk
        chunk_ids = []
        for chunk in chunks:
            await self._store.aput(  # type: ignore[attr-defined]
                namespace=(user_id, "conversations"),
                key=chunk.chunk_id,
                value={
                    "session_id": chunk.session_id,
                    "text": chunk.text,
                    "summary": chunk.summary,
                    "metadata": chunk.metadata,
                    "messages": [self._message_to_dict(m) for m in chunk.messages],
                    "created_at": chunk.created_at,
                },
            )
            chunk_ids.append(chunk.chunk_id)

        logger.info(f"Indexed {len(chunk_ids)} chunks for user={user_id}, session={session_id}")
        return chunk_ids

    def search(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        min_score: float | None = None,
        session_id: str | None = None,
    ) -> list[SearchResult]:
        """Search past conversations for relevant context.

        Args:
            user_id: User identifier (searches only this user's conversations)
            query: Search query
            limit: Maximum number of results (default: 5)
            min_score: Minimum similarity score (0-1, default: None)
            session_id: Optional filter to specific session

        Returns:
            List of SearchResult with matching conversation chunks

        Example:
            ```python
            results = memory.search(
                user_id="user_123",
                query="how to debug python",
                limit=3,
            )
            for r in results:
                print(f"Score: {r.score:.2f}")
                print(f"Context: {r.context}")
            ```
        """
        # Try semantic search if embeddings are configured
        if self._embedding_provider:
            items = self._store.search(
                namespace=(user_id, "conversations"),
                query=query,
                limit=limit,
            )
        else:
            # Fallback to listing all and filtering by text match
            # This is less accurate but works without embeddings
            all_items = self._store.list((user_id, "conversations"))
            query_lower = query.lower()

            # Simple text matching fallback
            scored_items = []
            for item in all_items:
                text = item.value.get("text", "").lower()
                # Simple relevance: count query words in text
                words = query_lower.split()
                matches = sum(1 for w in words if w in text)
                if matches > 0:
                    # Assign a pseudo-score based on word matches
                    item.score = matches / len(words) if words else 0
                    scored_items.append(item)

            # Sort by score and limit
            scored_items.sort(key=lambda x: x.score or 0, reverse=True)
            items = scored_items[:limit]

        # Convert to SearchResult
        results = []
        for item in items:
            # Apply min_score filter
            if min_score and (item.score is None or item.score < min_score):
                continue

            # Apply session_id filter
            if session_id and item.value.get("session_id") != session_id:
                continue

            # Reconstruct messages
            messages = [self._dict_to_message(m) for m in item.value.get("messages", [])]

            chunk = ConversationChunk(
                chunk_id=item.key,
                user_id=user_id,
                session_id=item.value.get("session_id", ""),
                messages=messages,
                text=item.value.get("text", ""),
                summary=item.value.get("summary"),
                metadata=item.value.get("metadata", {}),
                created_at=item.value.get("created_at", 0),
                score=item.score,
            )

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=item.score or 0.0,
                    context=self._format_chunk_context(chunk),
                )
            )

        return results

    async def asearch(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        min_score: float | None = None,
        session_id: str | None = None,
    ) -> list[SearchResult]:
        """Async version of search.

        See search for full documentation.
        """
        # Try semantic search if embeddings are configured
        if self._embedding_provider:
            items = await self._store.asearch(  # type: ignore[attr-defined]
                namespace=(user_id, "conversations"),
                query=query,
                limit=limit,
            )
        else:
            # Fallback to listing all and filtering by text match
            # Note: MemoryStore.list() is sync only, so we call it directly
            all_items = self._store.list((user_id, "conversations"))
            query_lower = query.lower()

            scored_items = []
            for item in all_items:
                text = item.value.get("text", "").lower()
                words = query_lower.split()
                matches = sum(1 for w in words if w in text)
                if matches > 0:
                    item.score = matches / len(words) if words else 0
                    scored_items.append(item)

            scored_items.sort(key=lambda x: x.score or 0, reverse=True)
            items = scored_items[:limit]

        # Convert to SearchResult
        results = []
        for item in items:
            if min_score and (item.score is None or item.score < min_score):
                continue

            if session_id and item.value.get("session_id") != session_id:
                continue

            messages = [self._dict_to_message(m) for m in item.value.get("messages", [])]

            chunk = ConversationChunk(
                chunk_id=item.key,
                user_id=user_id,
                session_id=item.value.get("session_id", ""),
                messages=messages,
                text=item.value.get("text", ""),
                summary=item.value.get("summary"),
                metadata=item.value.get("metadata", {}),
                created_at=item.value.get("created_at", 0),
                score=item.score,
            )

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=item.score or 0.0,
                    context=self._format_chunk_context(chunk),
                )
            )

        return results

    def delete_conversation(self, user_id: str, session_id: str) -> int:
        """Delete all chunks for a specific conversation.

        Args:
            user_id: User identifier
            session_id: Session to delete

        Returns:
            Number of chunks deleted
        """
        # List all chunks for this user
        items = self._store.list((user_id, "conversations"))

        # Filter and delete chunks from this session
        deleted = 0
        for item in items:
            if item.value.get("session_id") == session_id:
                self._store.delete((user_id, "conversations"), item.key)
                deleted += 1

        logger.info(f"Deleted {deleted} chunks for session {session_id}")
        return deleted

    def delete_user_conversations(self, user_id: str) -> int:
        """Delete all conversations for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of chunks deleted
        """
        items = self._store.list((user_id, "conversations"))
        deleted = 0
        for item in items:
            self._store.delete((user_id, "conversations"), item.key)
            deleted += 1

        logger.info(f"Deleted {deleted} chunks for user {user_id}")
        return deleted

    def _normalize_messages(self, messages: Sequence[MessageLike]) -> list[BaseMessage]:
        """Convert various message formats to BaseMessage."""
        result = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                result.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("system", "System"):
                    result.append(SystemMessage(content=content))
                elif role in ("assistant", "Assistant", "ai", "AI"):
                    result.append(AIMessage(content=content))
                else:
                    result.append(HumanMessage(content=content))
        return result

    def _create_chunks(
        self,
        user_id: str,
        session_id: str,
        messages: list[BaseMessage],
        metadata: dict[str, Any],
    ) -> list[ConversationChunk]:
        """Split messages into chunks with overlap."""
        if not messages:
            return []

        # Filter out system messages for chunking (but keep for context)
        work_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        if not work_messages:
            return []

        chunks = []
        step = max(1, self._chunk_size - self._chunk_overlap)

        for i in range(0, len(work_messages), step):
            chunk_messages = work_messages[i : i + self._chunk_size]

            if not chunk_messages:
                continue

            # Generate chunk ID
            chunk_id = self._generate_chunk_id(session_id, i)

            # Format text for embedding
            text = self._format_messages_for_embedding(chunk_messages)

            # Generate summary if enabled
            summary = None
            if self._include_summary:
                summary = self._generate_summary(chunk_messages)

            chunks.append(
                ConversationChunk(
                    chunk_id=chunk_id,
                    user_id=user_id,
                    session_id=session_id,
                    messages=chunk_messages,
                    text=text,
                    summary=summary,
                    metadata=metadata,
                )
            )

        return chunks

    def _generate_chunk_id(self, session_id: str, index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{session_id}:{index}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _format_messages_for_embedding(self, messages: list[BaseMessage]) -> str:
        """Format messages as text for embedding."""
        lines = []
        for msg in messages:
            role = self._get_message_role(msg)
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _format_chunk_context(self, chunk: ConversationChunk) -> str:
        """Format chunk as readable context."""
        lines = []

        # Add metadata if present
        if chunk.metadata:
            meta_parts = []
            if "topic" in chunk.metadata:
                meta_parts.append(f"Topic: {chunk.metadata['topic']}")
            if "date" in chunk.metadata:
                meta_parts.append(f"Date: {chunk.metadata['date']}")
            if meta_parts:
                lines.append(f"[{', '.join(meta_parts)}]")
                lines.append("")

        # Add summary if present
        if chunk.summary:
            lines.append(f"Summary: {chunk.summary}")
            lines.append("")

        # Add messages
        for msg in chunk.messages:
            role = self._get_message_role(msg)
            lines.append(f"{role}: {msg.content}")

        return "\n".join(lines)

    def _get_message_role(self, msg: BaseMessage) -> str:
        """Get role string from message type."""
        if isinstance(msg, SystemMessage):
            return "System"
        elif isinstance(msg, AIMessage):
            return "Assistant"
        elif isinstance(msg, HumanMessage):
            return "Human"
        else:
            return "Unknown"

    def _message_to_dict(self, msg: BaseMessage) -> dict[str, Any]:
        """Convert message to dict for storage."""
        return {
            "role": self._get_message_role(msg).lower(),
            "content": msg.content,
        }

    def _dict_to_message(self, data: dict[str, Any]) -> BaseMessage:
        """Convert dict back to message."""
        role = data.get("role", "user")
        content = data.get("content", "")

        if role == "system":
            return SystemMessage(content=content)
        elif role in ("assistant", "ai"):
            return AIMessage(content=content)
        else:
            return HumanMessage(content=content)

    def _generate_summary(self, messages: list[BaseMessage]) -> str | None:
        """Generate summary of messages using LLM."""
        if not self._include_summary:
            return None

        try:
            if self._llm is None:
                from ai_infra import LLM

                self._llm = LLM()

            text = self._format_messages_for_embedding(messages)
            prompt = f"Summarize this conversation briefly in 1-2 sentences:\n\n{text}"
            response = self._llm.chat(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return None


# =============================================================================
# Tool Creation
# =============================================================================


class MemoryToolInput(BaseModel):
    """Input schema for memory search tool."""

    query: str = Field(
        description="Search query to find relevant past conversations. "
        "Be specific about what you're looking for."
    )


def create_memory_tool(
    memory: ConversationMemory,
    *,
    user_id: str | None = None,
    name: str = "recall_past_conversations",
    description: str = (
        "Search through past conversations with this user to find relevant "
        "context, solutions, or preferences discussed before. Use this when "
        "the user references something from a previous conversation."
    ),
    limit: int = 3,
    min_score: float | None = None,
    return_scores: bool = False,
    max_chars: int | None = None,
) -> StructuredTool:
    """Create an Agent-compatible tool for conversation recall.

    Creates a tool that agents can use to search past conversations.
    The user_id can be provided at tool creation time, or the tool
    will return an error if not provided.

    Args:
        memory: ConversationMemory instance
        user_id: User ID to scope searches to (required for multi-user scenarios)
        name: Tool name (default: "recall_past_conversations")
        description: Tool description for the agent
        limit: Maximum results to return (default: 3)
        min_score: Minimum similarity score (0-1)
        return_scores: Include similarity scores in output
        max_chars: Maximum output characters (truncates if exceeded)

    Returns:
        StructuredTool that can be used with Agent

    Example:
        ```python
        from ai_infra import Agent
        from ai_infra.llm.tools.custom import ConversationMemory, create_memory_tool

        memory = ConversationMemory(
            backend="sqlite",
            path="./conversations.db",
            embedding_provider="openai",
        )

        # Create tool for a specific user
        recall = create_memory_tool(memory, user_id="user_123", limit=5)

        # Add to agent
        agent = Agent(tools=[recall])

        # Agent can now search past conversations
        result = agent.run("How did we fix that bug last week?")
        ```
    """
    # Store reference for closure
    _memory = memory
    _user_id = user_id
    _limit = limit
    _min_score = min_score
    _return_scores = return_scores
    _max_chars = max_chars

    def search_conversations(query: str) -> str:
        """Search past conversations for relevant context."""
        if not _user_id:
            return (
                "Error: user_id not provided when creating the tool. Cannot search conversations."
            )

        # Search
        results = _memory.search(
            user_id=_user_id,
            query=query,
            limit=_limit,
            min_score=_min_score,
        )

        if not results:
            return "No relevant past conversations found."

        # Format results
        output_parts = []
        for i, r in enumerate(results, 1):
            parts = [f"### Result {i}"]
            if _return_scores:
                parts.append(f"**Score:** {r.score:.2f}")
            parts.append("")
            parts.append(r.context)
            output_parts.append("\n".join(parts))

        output = "\n\n---\n\n".join(output_parts)

        # Truncate if needed
        if _max_chars and len(output) > _max_chars:
            output = output[: _max_chars - 3] + "..."

        return output

    return StructuredTool.from_function(
        func=search_conversations,
        name=name,
        description=description,
        args_schema=MemoryToolInput,
    )


def create_memory_tool_async(
    memory: ConversationMemory,
    *,
    user_id: str | None = None,
    name: str = "recall_past_conversations",
    description: str = (
        "Search through past conversations with this user to find relevant "
        "context, solutions, or preferences discussed before. Use this when "
        "the user references something from a previous conversation."
    ),
    limit: int = 3,
    min_score: float | None = None,
    return_scores: bool = False,
    max_chars: int | None = None,
) -> StructuredTool:
    """Create an async Agent-compatible tool for conversation recall.

    Same as create_memory_tool but uses async search. Use this in async
    contexts like FastAPI endpoints.

    Args:
        memory: ConversationMemory instance
        user_id: User ID to scope searches to (required for multi-user scenarios)
        name: Tool name
        description: Tool description for the agent
        limit: Maximum results to return
        min_score: Minimum similarity score (0-1)
        return_scores: Include similarity scores in output
        max_chars: Maximum output characters

    Returns:
        StructuredTool with async execution
    """
    _memory = memory
    _user_id = user_id
    _limit = limit
    _min_score = min_score
    _return_scores = return_scores
    _max_chars = max_chars

    async def search_conversations_async(query: str) -> str:
        """Search past conversations for relevant context (async)."""
        if not _user_id:
            return (
                "Error: user_id not provided when creating the tool. Cannot search conversations."
            )

        results = await _memory.asearch(
            user_id=_user_id,
            query=query,
            limit=_limit,
            min_score=_min_score,
        )

        if not results:
            return "No relevant past conversations found."

        output_parts = []
        for i, r in enumerate(results, 1):
            parts = [f"### Result {i}"]
            if _return_scores:
                parts.append(f"**Score:** {r.score:.2f}")
            parts.append("")
            parts.append(r.context)
            output_parts.append("\n".join(parts))

        output = "\n\n---\n\n".join(output_parts)

        if _max_chars and len(output) > _max_chars:
            output = output[: _max_chars - 3] + "..."

        return output

    return StructuredTool.from_function(
        coroutine=search_conversations_async,
        name=name,
        description=description,
        args_schema=MemoryToolInput,
    )
