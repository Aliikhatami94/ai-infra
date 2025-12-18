"""Unit tests for ai_infra.memory module.

Tests for Phase 6.4.1 (Short-term memory) and 6.4.2 (Long-term memory store).
Note: trim_messages, summarize_messages etc are now internal to fit_context,
but we test them to ensure the underlying functionality works.
"""

import time
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Public API imports
from ai_infra import MemoryStore, count_tokens, count_tokens_approximate
from ai_infra.llm.memory.summarize import (
    SummarizationMiddleware,
    SummarizeResult,
    summarize_messages,
)
from ai_infra.llm.memory.tokens import get_context_limit

# Internal imports for testing underlying functionality
from ai_infra.llm.memory.trim import trim_messages

# =============================================================================
# Tests for trim_messages (6.4.1)
# =============================================================================


class TestTrimMessages:
    """Tests for trim_messages utility."""

    def test_trim_last_basic(self):
        """Test keeping last N messages."""
        messages = [
            HumanMessage(content="msg1"),
            AIMessage(content="msg2"),
            HumanMessage(content="msg3"),
            AIMessage(content="msg4"),
        ]
        result = trim_messages(messages, strategy="last", max_messages=2)
        assert len(result) == 2
        assert result[0].content == "msg3"
        assert result[1].content == "msg4"

    def test_trim_first_basic(self):
        """Test keeping first N messages."""
        messages = [
            HumanMessage(content="msg1"),
            AIMessage(content="msg2"),
            HumanMessage(content="msg3"),
            AIMessage(content="msg4"),
        ]
        result = trim_messages(messages, strategy="first", max_messages=2)
        assert len(result) == 2
        assert result[0].content == "msg1"
        assert result[1].content == "msg2"

    def test_preserve_system_message(self):
        """Test that system message is preserved by default."""
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="msg1"),
            AIMessage(content="msg2"),
            HumanMessage(content="msg3"),
        ]
        result = trim_messages(messages, strategy="last", max_messages=1)
        assert len(result) == 2  # system + 1 kept
        assert isinstance(result[0], SystemMessage)
        assert result[1].content == "msg3"

    def test_no_preserve_system(self):
        """Test disabling system message preservation."""
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="msg1"),
            AIMessage(content="msg2"),
        ]
        result = trim_messages(messages, strategy="last", max_messages=1, preserve_system=False)
        assert len(result) == 1
        assert result[0].content == "msg2"

    def test_trim_by_token_strategy(self):
        """Test trimming by token count."""
        messages = [
            HumanMessage(content="This is a long message with many words"),
            AIMessage(content="Short"),
            HumanMessage(content="Another message"),
        ]
        # Approximate tokens: each message ~10-15 tokens
        result = trim_messages(messages, strategy="token", max_tokens=20)
        # Should keep some recent messages
        assert len(result) <= len(messages)

    def test_empty_messages(self):
        """Test with empty message list."""
        result = trim_messages([], strategy="last", max_messages=5)
        assert result == []

    def test_max_messages_zero(self):
        """Test with max_messages=0."""
        messages = [HumanMessage(content="msg")]
        result = trim_messages(messages, strategy="last", max_messages=0)
        assert result == []

    def test_dict_messages(self):
        """Test with dict-format messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = trim_messages(messages, strategy="last", max_messages=1)
        assert len(result) == 1
        assert result[0].content == "Hi"

    def test_invalid_strategy_raises(self):
        """Test that invalid strategy raises ValueError."""
        messages = [HumanMessage(content="msg")]
        with pytest.raises(ValueError, match="Unknown strategy"):
            trim_messages(messages, strategy="invalid", max_messages=1)  # type: ignore

    def test_missing_max_messages_raises(self):
        """Test that missing max_messages raises for last/first strategy."""
        messages = [HumanMessage(content="msg")]
        with pytest.raises(ValueError, match="max_messages required"):
            trim_messages(messages, strategy="last")

    def test_missing_max_tokens_raises(self):
        """Test that missing max_tokens raises for token strategy."""
        messages = [HumanMessage(content="msg")]
        with pytest.raises(ValueError, match="max_tokens required"):
            trim_messages(messages, strategy="token")


# =============================================================================
# Tests for count_tokens (6.4.1)
# =============================================================================


class TestCountTokens:
    """Tests for token counting utilities."""

    def test_count_tokens_approximate_basic(self):
        """Test approximate token counting."""
        messages = [HumanMessage(content="Hello, world!")]
        tokens = count_tokens_approximate(messages)
        # "Hello, world!" is 13 chars, ~3-4 tokens + 4 overhead
        assert 5 <= tokens <= 10

    def test_count_tokens_approximate_empty(self):
        """Test with empty messages."""
        tokens = count_tokens_approximate([])
        assert tokens == 0

    def test_count_tokens_approximate_string(self):
        """Test with raw string input."""
        tokens = count_tokens_approximate(["Hello, world!"])
        # Raw string, no message overhead
        assert 3 <= tokens <= 5

    def test_count_tokens_with_model(self):
        """Test count_tokens with model specification."""
        messages = [HumanMessage(content="Hello")]
        # Should work regardless of tiktoken availability
        tokens = count_tokens(messages, model="gpt-4o")
        assert tokens > 0

    def test_get_context_limit_openai(self):
        """Test context limits for OpenAI models."""
        assert get_context_limit("gpt-4o") == 128_000
        assert get_context_limit("gpt-4-turbo") == 128_000
        assert get_context_limit("gpt-4") == 8_192
        assert get_context_limit("gpt-3.5-turbo") == 4_096

    def test_get_context_limit_anthropic(self):
        """Test context limits for Anthropic models."""
        assert get_context_limit("claude-3-opus") == 200_000
        assert get_context_limit("claude-sonnet-4-20250514") == 200_000

    def test_get_context_limit_google(self):
        """Test context limits for Google models."""
        assert get_context_limit("gemini-2.0-flash") == 1_000_000

    def test_get_context_limit_unknown(self):
        """Test default context limit for unknown models."""
        assert get_context_limit("unknown-model") == 4_096


# =============================================================================
# Tests for summarize_messages (6.4.1)
# =============================================================================


class TestSummarizeMessages:
    """Tests for summarize_messages utility."""

    def test_summarize_returns_result(self):
        """Test that summarize_messages returns SummarizeResult."""
        messages = [
            HumanMessage(content="msg1"),
            AIMessage(content="msg2"),
            HumanMessage(content="msg3"),
            AIMessage(content="msg4"),
            HumanMessage(content="msg5"),
            AIMessage(content="msg6"),
        ]

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Summary of the conversation"

        result = summarize_messages(messages, keep_last=2, llm=mock_llm)

        assert isinstance(result, SummarizeResult)
        assert result.summary == "Summary of the conversation"
        assert result.summarized_count == 4
        assert result.kept_count == 2
        assert result.original_count == 6

    def test_summarize_preserves_system(self):
        """Test that system message is preserved."""
        messages = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="msg1"),
            AIMessage(content="msg2"),
            HumanMessage(content="msg3"),
            AIMessage(content="msg4"),
        ]

        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Summary"

        result = summarize_messages(messages, keep_last=1, llm=mock_llm)

        # Should have: system + summary + 1 kept
        assert len(result.messages) == 3
        assert isinstance(result.messages[0], SystemMessage)
        assert result.messages[0].content == "system prompt"

    def test_summarize_not_enough_messages(self):
        """Test that no summarization happens if not enough messages."""
        messages = [
            HumanMessage(content="msg1"),
            AIMessage(content="msg2"),
        ]

        mock_llm = MagicMock()
        result = summarize_messages(messages, keep_last=5, llm=mock_llm)

        # LLM should not be called
        mock_llm.chat.assert_not_called()
        assert result.summary == ""
        assert result.summarized_count == 0

    def test_summarize_empty_messages(self):
        """Test with empty message list."""
        result = summarize_messages([], keep_last=5)
        assert result.messages == []
        assert result.original_count == 0


# =============================================================================
# Tests for SummarizationMiddleware (6.4.1)
# =============================================================================


class TestSummarizationMiddleware:
    """Tests for SummarizationMiddleware."""

    def test_should_summarize_by_token(self):
        """Test token-based trigger."""
        middleware = SummarizationMiddleware(trigger_tokens=50)

        # Short messages - should not trigger
        short_msgs = [HumanMessage(content="Hi")]
        assert not middleware.should_summarize(short_msgs)

        # Long messages - should trigger
        long_msgs = [HumanMessage(content="x" * 500)]
        assert middleware.should_summarize(long_msgs)

    def test_should_summarize_by_message_count(self):
        """Test message count trigger."""
        middleware = SummarizationMiddleware(trigger_messages=3)

        # Few messages - should not trigger
        few = [HumanMessage(content="1"), HumanMessage(content="2")]
        assert not middleware.should_summarize(few)

        # Many messages - should trigger
        many = [HumanMessage(content=str(i)) for i in range(5)]
        assert middleware.should_summarize(many)

    def test_process_no_summarization_needed(self):
        """Test process when no summarization needed."""
        middleware = SummarizationMiddleware(trigger_messages=10)
        messages = [HumanMessage(content="msg1"), AIMessage(content="msg2")]

        result = middleware.process(messages)
        assert result == messages  # Unchanged

    def test_process_with_summarization(self):
        """Test process when summarization is triggered."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Summary of conversation"

        middleware = SummarizationMiddleware(
            trigger_messages=3,
            keep_messages=1,
            llm=mock_llm,
        )

        messages = [
            HumanMessage(content="msg1"),
            AIMessage(content="msg2"),
            HumanMessage(content="msg3"),
            AIMessage(content="msg4"),
        ]

        result = middleware.process(messages)

        # Should have summary + kept messages
        assert len(result) < len(messages)
        mock_llm.chat.assert_called_once()


# =============================================================================
# Tests for MemoryStore (6.4.2)
# =============================================================================


class TestMemoryStore:
    """Tests for MemoryStore class."""

    def test_put_and_get(self):
        """Test basic put and get operations."""
        store = MemoryStore()

        store.put(
            namespace=("user_1", "prefs"),
            key="language",
            value={"preference": "Python"},
        )

        item = store.get(("user_1", "prefs"), "language")
        assert item is not None
        assert item.value["preference"] == "Python"

    def test_get_nonexistent(self):
        """Test getting nonexistent key."""
        store = MemoryStore()
        item = store.get(("user_1", "prefs"), "nonexistent")
        assert item is None

    def test_delete(self):
        """Test delete operation."""
        store = MemoryStore()

        store.put(("user_1", "prefs"), "key1", {"value": "1"})
        assert store.get(("user_1", "prefs"), "key1") is not None

        deleted = store.delete(("user_1", "prefs"), "key1")
        assert deleted is True
        assert store.get(("user_1", "prefs"), "key1") is None

    def test_delete_nonexistent(self):
        """Test deleting nonexistent key."""
        store = MemoryStore()
        deleted = store.delete(("user_1", "prefs"), "nonexistent")
        assert deleted is False

    def test_list(self):
        """Test listing items in namespace."""
        store = MemoryStore()

        store.put(("user_1", "prefs"), "key1", {"value": "1"})
        store.put(("user_1", "prefs"), "key2", {"value": "2"})
        store.put(("user_2", "prefs"), "key3", {"value": "3"})

        items = store.list(("user_1", "prefs"))
        assert len(items) == 2

        keys = {item.key for item in items}
        assert keys == {"key1", "key2"}

    def test_list_with_limit(self):
        """Test listing with limit."""
        store = MemoryStore()

        for i in range(5):
            store.put(("user_1", "prefs"), f"key{i}", {"value": str(i)})

        items = store.list(("user_1", "prefs"), limit=2)
        assert len(items) == 2

    def test_namespace_isolation(self):
        """Test that namespaces are isolated."""
        store = MemoryStore()

        store.put(("user_1", "prefs"), "key", {"value": "user1"})
        store.put(("user_2", "prefs"), "key", {"value": "user2"})

        item1 = store.get(("user_1", "prefs"), "key")
        item2 = store.get(("user_2", "prefs"), "key")

        assert item1.value["value"] == "user1"
        assert item2.value["value"] == "user2"

    def test_string_namespace(self):
        """Test that string namespace is converted to tuple."""
        store = MemoryStore()

        store.put("simple", "key", {"value": "test"})
        item = store.get("simple", "key")

        assert item is not None
        assert item.namespace == ("simple",)

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        store = MemoryStore()

        # Store with very short TTL
        store.put(("user_1", "temp"), "key", {"value": "temp"}, ttl=1)

        # Manually set expires_at to the past for testing
        # Access the internal backend to manipulate the item
        ns_key = "user_1/temp"
        item, embedding = store._backend._data[ns_key]["key"]
        item.expires_at = time.time() - 1  # Already expired

        # Should be expired
        result = store.get(("user_1", "temp"), "key")
        assert result is None

    def test_convenience_methods(self):
        """Test user and app convenience methods."""
        store = MemoryStore()

        # User memory
        store.put_user_memory("user_123", "pref", {"value": "test"})
        item = store.get_user_memory("user_123", "pref")
        assert item.value["value"] == "test"

        # App memory
        store.put_app_memory("config", {"setting": "value"})
        item = store.get_app_memory("config")
        assert item.value["setting"] == "value"


class TestMemoryStoreSearch:
    """Tests for MemoryStore semantic search."""

    def test_search_requires_embeddings(self):
        """Test that search without embeddings raises error."""
        store = MemoryStore()

        store.put(("user_1", "prefs"), "key", {"value": "test"})

        with pytest.raises(ValueError, match="Semantic search requires embedding_provider"):
            store.search(("user_1", "prefs"), "test query")

    def test_search_with_mock_embeddings(self):
        """Test search with mocked embeddings."""
        # Create store and mock embeddings
        store = MemoryStore()

        mock_embeddings = MagicMock()
        # Return simple embeddings for testing
        mock_embeddings.embed.side_effect = [
            [1.0, 0.0, 0.0],  # First put
            [0.9, 0.1, 0.0],  # Second put
            [0.8, 0.2, 0.0],  # Query
        ]
        store._embeddings = mock_embeddings

        store.put(("user_1", "prefs"), "key1", {"topic": "Python programming"})
        store.put(("user_1", "prefs"), "key2", {"topic": "JavaScript development"})

        results = store.search(("user_1", "prefs"), "programming", limit=2)

        assert len(results) <= 2
        # Results should have scores
        for item in results:
            assert item.score is not None


class TestMemoryStoreSQLite:
    """Tests for SQLite backend."""

    def test_sqlite_basic_operations(self, tmp_path):
        """Test basic operations with SQLite backend."""
        db_path = str(tmp_path / "test.db")
        store = MemoryStore.sqlite(db_path)

        store.put(("user_1", "prefs"), "key", {"value": "test"})
        item = store.get(("user_1", "prefs"), "key")

        assert item is not None
        assert item.value["value"] == "test"

    def test_sqlite_persistence(self, tmp_path):
        """Test that SQLite persists data."""
        db_path = str(tmp_path / "test.db")

        # First store
        store1 = MemoryStore.sqlite(db_path)
        store1.put(("user_1", "prefs"), "key", {"value": "persisted"})

        # New store instance with same path
        store2 = MemoryStore.sqlite(db_path)
        item = store2.get(("user_1", "prefs"), "key")

        assert item is not None
        assert item.value["value"] == "persisted"

    def test_sqlite_list(self, tmp_path):
        """Test listing with SQLite backend."""
        db_path = str(tmp_path / "test.db")
        store = MemoryStore.sqlite(db_path)

        store.put(("user_1", "prefs"), "key1", {"value": "1"})
        store.put(("user_1", "prefs"), "key2", {"value": "2"})

        items = store.list(("user_1", "prefs"))
        assert len(items) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryIntegration:
    """Integration tests for memory module."""

    def test_trim_then_summarize(self):
        """Test using trim and summarize together."""
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="Long conversation message 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Long conversation message 2"),
            AIMessage(content="Response 2"),
            HumanMessage(content="Long conversation message 3"),
            AIMessage(content="Response 3"),
        ]

        # First trim
        trimmed = trim_messages(messages, strategy="last", max_messages=3)
        assert len(trimmed) == 4  # system + 3 kept

        # Check token count
        tokens = count_tokens_approximate(trimmed)
        assert tokens > 0
