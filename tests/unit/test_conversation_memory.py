"""Unit tests for ConversationMemory and create_memory_tool.

Tests for Phase 6.4.3 - Conversation History RAG.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_infra.llm.tools.custom.memory import (
    ConversationChunk,
    ConversationMemory,
    SearchResult,
    create_memory_tool,
    create_memory_tool_async,
)

# =============================================================================
# ConversationMemory Tests
# =============================================================================


class TestConversationMemoryInit:
    """Tests for ConversationMemory initialization."""

    def test_default_init(self):
        """Test default in-memory initialization."""
        memory = ConversationMemory()
        assert memory._backend_type == "memory"
        assert memory._chunk_size == 10
        assert memory._chunk_overlap == 2
        assert memory._include_summary is False

    def test_sqlite_init(self):
        """Test SQLite backend initialization."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.db")
            memory = ConversationMemory(backend="sqlite", path=path)
            assert memory._backend_type == "sqlite"
            assert memory._path == path

    def test_sqlite_classmethod(self):
        """Test SQLite classmethod factory."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.db")
            memory = ConversationMemory.sqlite(path, chunk_size=5)
            assert memory._backend_type == "sqlite"
            assert memory._chunk_size == 5

    def test_sqlite_requires_path(self):
        """Test that sqlite backend requires path."""
        with pytest.raises(ValueError, match="path is required"):
            ConversationMemory(backend="sqlite")

    def test_postgres_requires_connection_string(self):
        """Test that postgres backend requires connection_string."""
        with pytest.raises(ValueError, match="connection_string is required"):
            ConversationMemory(backend="postgres")

    def test_custom_chunk_settings(self):
        """Test custom chunk size and overlap."""
        memory = ConversationMemory(
            chunk_size=20,
            chunk_overlap=5,
            include_summary=True,
        )
        assert memory._chunk_size == 20
        assert memory._chunk_overlap == 5
        assert memory._include_summary is True


class TestConversationMemoryIndexing:
    """Tests for conversation indexing."""

    def test_index_simple_conversation(self):
        """Test indexing a simple conversation."""
        memory = ConversationMemory()

        messages = [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
        ]

        chunk_ids = memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=messages,
        )

        assert len(chunk_ids) == 1
        assert len(chunk_ids[0]) == 16  # SHA256 hash truncated

    def test_index_with_metadata(self):
        """Test indexing with metadata."""
        memory = ConversationMemory()

        messages = [
            HumanMessage(content="Help with debugging"),
            AIMessage(content="Let me help you debug."),
        ]

        chunk_ids = memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=messages,
            metadata={"topic": "debugging", "date": "2025-01-15"},
        )

        assert len(chunk_ids) == 1

        # Verify metadata is stored
        items = memory._store.list(("user_123", "conversations"))
        assert len(items) == 1
        assert items[0].value["metadata"]["topic"] == "debugging"

    def test_index_long_conversation_creates_multiple_chunks(self):
        """Test that long conversations are split into chunks."""
        memory = ConversationMemory(chunk_size=3, chunk_overlap=1)

        # Create 10 messages (should create multiple chunks)
        messages = []
        for i in range(10):
            messages.append(HumanMessage(content=f"Question {i}"))
            messages.append(AIMessage(content=f"Answer {i}"))

        chunk_ids = memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=messages,
        )

        # With 20 messages, chunk_size=3, step=2: ceil(20/2) = 10 chunks
        assert len(chunk_ids) > 1

    def test_index_empty_conversation(self):
        """Test indexing empty conversation returns empty list."""
        memory = ConversationMemory()
        chunk_ids = memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=[],
        )
        assert chunk_ids == []

    def test_index_system_messages_excluded_from_chunks(self):
        """Test that system messages are excluded from chunking."""
        memory = ConversationMemory(chunk_size=10)

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        chunk_ids = memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=messages,
        )

        assert len(chunk_ids) == 1

        # Verify chunk doesn't include system message
        items = memory._store.list(("user_123", "conversations"))
        assert len(items) == 1
        stored_messages = items[0].value["messages"]
        assert len(stored_messages) == 2  # Human + AI, not System

    def test_index_dict_messages(self):
        """Test indexing dict-format messages."""
        memory = ConversationMemory()

        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ]

        chunk_ids = memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=messages,
        )

        assert len(chunk_ids) == 1


class TestConversationMemorySearch:
    """Tests for conversation search."""

    def test_search_returns_results(self):
        """Test basic search functionality."""
        memory = ConversationMemory()

        # Index a conversation
        messages = [
            HumanMessage(content="How do I debug Python code?"),
            AIMessage(content="You can use pdb or IDE breakpoints."),
        ]

        memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=messages,
        )

        # Search
        results = memory.search(
            user_id="user_123",
            query="python debugging",
            limit=5,
        )

        # Without embeddings, search returns all items
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)

    def test_search_empty_returns_empty(self):
        """Test search with no indexed conversations."""
        memory = ConversationMemory()

        results = memory.search(
            user_id="user_123",
            query="anything",
            limit=5,
        )

        assert results == []

    def test_search_respects_user_scope(self):
        """Test that search is scoped to user_id."""
        memory = ConversationMemory()

        # Index for user_123
        memory.index_conversation(
            user_id="user_123",
            session_id="session_1",
            messages=[HumanMessage(content="User 123 question")],
        )

        # Index for user_456
        memory.index_conversation(
            user_id="user_456",
            session_id="session_2",
            messages=[HumanMessage(content="User 456 question")],
        )

        # Search for user_123 only
        results = memory.search(
            user_id="user_123",
            query="question",
            limit=10,
        )

        # Should only find user_123's conversation
        for r in results:
            assert r.chunk.user_id == "user_123"

    def test_search_result_includes_context(self):
        """Test that search results include formatted context."""
        memory = ConversationMemory()

        messages = [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
        ]

        memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=messages,
            metadata={"topic": "programming"},
        )

        results = memory.search(
            user_id="user_123",
            query="python",
            limit=1,
        )

        assert len(results) >= 1
        assert "Python" in results[0].context
        assert "Human:" in results[0].context or "Assistant:" in results[0].context


class TestConversationMemoryDelete:
    """Tests for conversation deletion."""

    def test_delete_conversation(self):
        """Test deleting a specific conversation."""
        memory = ConversationMemory()

        # Index two conversations
        memory.index_conversation(
            user_id="user_123",
            session_id="session_1",
            messages=[HumanMessage(content="First conversation")],
        )
        memory.index_conversation(
            user_id="user_123",
            session_id="session_2",
            messages=[HumanMessage(content="Second conversation")],
        )

        # Delete first session
        deleted = memory.delete_conversation("user_123", "session_1")

        assert deleted >= 1

        # Verify only second session remains
        items = memory._store.list(("user_123", "conversations"))
        for item in items:
            assert item.value.get("session_id") == "session_2"

    def test_delete_user_conversations(self):
        """Test deleting all conversations for a user."""
        memory = ConversationMemory()

        # Index conversations
        memory.index_conversation(
            user_id="user_123",
            session_id="session_1",
            messages=[HumanMessage(content="Conversation 1")],
        )
        memory.index_conversation(
            user_id="user_123",
            session_id="session_2",
            messages=[HumanMessage(content="Conversation 2")],
        )

        # Delete all
        deleted = memory.delete_user_conversations("user_123")

        assert deleted >= 2

        # Verify nothing remains
        items = memory._store.list(("user_123", "conversations"))
        assert len(items) == 0


class TestConversationMemoryChunking:
    """Tests for conversation chunking logic."""

    def test_chunk_overlap(self):
        """Test that chunks overlap correctly."""
        memory = ConversationMemory(chunk_size=4, chunk_overlap=2)

        # Create 8 messages
        messages = [HumanMessage(content=f"Message {i}") for i in range(8)]

        chunk_ids = memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=messages,
        )

        # With 8 messages, chunk_size=4, step=2: positions 0, 2, 4, 6
        assert len(chunk_ids) == 4

    def test_single_chunk_when_small(self):
        """Test single chunk for small conversations."""
        memory = ConversationMemory(chunk_size=10)

        messages = [HumanMessage(content="Short message")]

        chunk_ids = memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=messages,
        )

        assert len(chunk_ids) == 1


# =============================================================================
# create_memory_tool Tests
# =============================================================================


class TestCreateMemoryTool:
    """Tests for create_memory_tool factory."""

    def test_creates_structured_tool(self):
        """Test that create_memory_tool returns a StructuredTool."""
        from langchain_core.tools import StructuredTool

        memory = ConversationMemory()
        tool = create_memory_tool(memory)

        assert isinstance(tool, StructuredTool)
        assert tool.name == "recall_past_conversations"

    def test_custom_name_and_description(self):
        """Test custom tool name and description."""
        memory = ConversationMemory()
        tool = create_memory_tool(
            memory,
            name="search_history",
            description="Search conversation history",
        )

        assert tool.name == "search_history"
        assert tool.description == "Search conversation history"

    def test_tool_requires_user_id(self):
        """Test that tool returns error without user_id."""
        memory = ConversationMemory()
        tool = create_memory_tool(memory)  # No user_id provided

        # Call without user_id
        result = tool.func("test query")
        assert "Error" in result
        assert "user_id" in result

    def test_tool_searches_with_user_id(self):
        """Test that tool searches with provided user_id."""
        memory = ConversationMemory()

        # Index a conversation
        memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=[HumanMessage(content="Test message")],
        )

        # Create tool with user_id
        tool = create_memory_tool(memory, user_id="user_123")

        # Call the tool
        result = tool.func("test")

        # Should return results (not "No relevant" since we indexed something)
        assert "Error" not in result

    def test_tool_no_results_message(self):
        """Test message when no results found."""
        memory = ConversationMemory()
        tool = create_memory_tool(memory, user_id="user_with_no_history")

        result = tool.func("query")

        assert "No relevant past conversations found" in result

    def test_tool_respects_limit(self):
        """Test that limit parameter is respected."""
        memory = ConversationMemory()

        # Index multiple conversations
        for i in range(5):
            memory.index_conversation(
                user_id="user_123",
                session_id=f"session_{i}",
                messages=[HumanMessage(content=f"Message {i}")],
            )

        tool = create_memory_tool(memory, user_id="user_123", limit=2)

        result = tool.func("message")

        # Result should have at most 2 results
        # Count "### Result" headers
        result_count = result.count("### Result")
        assert result_count <= 2

    def test_tool_with_scores(self):
        """Test return_scores parameter."""
        memory = ConversationMemory()

        memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=[HumanMessage(content="Test")],
        )

        tool = create_memory_tool(memory, user_id="user_123", return_scores=True)

        result = tool.func("test")

        # Should include score in output
        assert "Score:" in result

    def test_tool_max_chars_truncation(self):
        """Test max_chars truncation."""
        memory = ConversationMemory()

        # Index a conversation with content that matches the query
        memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=[HumanMessage(content="test " + "A" * 1000)],
        )

        tool = create_memory_tool(memory, user_id="user_123", max_chars=100)

        result = tool.func("test")

        # If we got results, they should be truncated
        if "No relevant" not in result:
            assert len(result) <= 100
            assert result.endswith("...")


class TestCreateMemoryToolAsync:
    """Tests for async memory tool."""

    def test_creates_async_tool(self):
        """Test that create_memory_tool_async returns async tool."""
        from langchain_core.tools import StructuredTool

        memory = ConversationMemory()
        tool = create_memory_tool_async(memory)

        assert isinstance(tool, StructuredTool)
        assert tool.coroutine is not None

    @pytest.mark.asyncio
    async def test_async_tool_searches(self):
        """Test async tool functionality."""
        memory = ConversationMemory()

        # Index a conversation
        memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=[HumanMessage(content="Async test message")],
        )

        tool = create_memory_tool_async(memory, user_id="user_123")

        result = await tool.coroutine("test")

        assert "Error" not in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestConversationMemoryIntegration:
    """Integration tests for ConversationMemory."""

    def test_full_workflow(self):
        """Test complete workflow: index -> search -> delete."""
        memory = ConversationMemory()

        # Index
        chunk_ids = memory.index_conversation(
            user_id="user_123",
            session_id="session_456",
            messages=[
                HumanMessage(content="How do I fix a Python import error?"),
                AIMessage(content="Check your PYTHONPATH and module location."),
            ],
            metadata={"topic": "python", "date": "2025-01-15"},
        )

        assert len(chunk_ids) >= 1

        # Search
        results = memory.search(
            user_id="user_123",
            query="python import error fix",
            limit=5,
        )

        assert len(results) >= 1
        assert "import" in results[0].context.lower() or "python" in results[0].context.lower()

        # Delete
        deleted = memory.delete_conversation("user_123", "session_456")
        assert deleted >= 1

        # Verify deleted
        results_after = memory.search(
            user_id="user_123",
            query="python",
            limit=5,
        )
        assert len(results_after) == 0

    def test_multi_user_isolation(self):
        """Test that conversations are isolated by user."""
        memory = ConversationMemory()

        # Index for different users
        memory.index_conversation(
            user_id="alice",
            session_id="session_1",
            messages=[HumanMessage(content="Alice's secret question")],
        )

        memory.index_conversation(
            user_id="bob",
            session_id="session_2",
            messages=[HumanMessage(content="Bob's secret question")],
        )

        # Alice's search shouldn't find Bob's content
        alice_results = memory.search(user_id="alice", query="secret", limit=10)
        for r in alice_results:
            assert r.chunk.user_id == "alice"

        # Bob's search shouldn't find Alice's content
        bob_results = memory.search(user_id="bob", query="secret", limit=10)
        for r in bob_results:
            assert r.chunk.user_id == "bob"
