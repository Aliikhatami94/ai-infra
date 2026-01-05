"""Tests for semantic cache module.

Tests the SemanticCache class, cache backends, and key generation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from ai_infra.cache import (
    CacheBackend,
    CacheEntry,
    CacheHit,
    CacheKeyGenerator,
    CacheMiss,
    CacheStats,
    SemanticCache,
)
from ai_infra.cache.backends.memory import MemoryCacheBackend

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings instance."""
    mock = MagicMock()
    # Simple mock embedding - just returns a predictable vector
    mock.embed.side_effect = lambda text: [hash(text) % 100 / 100.0] * 10

    # Async mock needs to return an awaitable
    async def async_embed(text):
        return [hash(text) % 100 / 100.0] * 10

    mock.aembed = async_embed
    return mock


@pytest.fixture
def memory_backend():
    """Create a memory backend for testing."""
    return MemoryCacheBackend(max_entries=100)


@pytest.fixture
def sample_entry():
    """Create a sample cache entry."""
    return CacheEntry(
        key="What is Python?",
        value="Python is a programming language.",
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
    )


# -----------------------------------------------------------------------------
# CacheEntry Tests
# -----------------------------------------------------------------------------


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_entry(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            key="test",
            value="response",
            embedding=[0.1, 0.2, 0.3],
        )
        assert entry.key == "test"
        assert entry.value == "response"
        assert entry.embedding == [0.1, 0.2, 0.3]
        assert entry.hit_count == 0
        assert entry.is_expired is False

    def test_entry_expiry(self):
        """Test entry expiration check."""
        # Not expired
        entry = CacheEntry(
            key="test",
            value="response",
            embedding=[0.1],
            expires_at=time.time() + 3600,
        )
        assert entry.is_expired is False

        # Expired
        entry = CacheEntry(
            key="test",
            value="response",
            embedding=[0.1],
            expires_at=time.time() - 1,
        )
        assert entry.is_expired is True

        # No expiry
        entry = CacheEntry(
            key="test",
            value="response",
            embedding=[0.1],
            expires_at=None,
        )
        assert entry.is_expired is False

    def test_entry_serialization(self):
        """Test entry to/from dict."""
        entry = CacheEntry(
            key="test",
            value="response",
            embedding=[0.1, 0.2],
            metadata={"model": "gpt-4"},
        )
        data = entry.to_dict()
        restored = CacheEntry.from_dict(data)

        assert restored.key == entry.key
        assert restored.value == entry.value
        assert restored.embedding == entry.embedding
        assert restored.metadata == entry.metadata


# -----------------------------------------------------------------------------
# CacheKeyGenerator Tests
# -----------------------------------------------------------------------------


class TestCacheKeyGenerator:
    """Tests for CacheKeyGenerator."""

    def test_generate_basic(self):
        """Test basic key generation."""
        gen = CacheKeyGenerator()
        key = gen.generate("What is Python?")
        assert key == "What is Python?"

    def test_generate_normalizes_whitespace(self):
        """Test whitespace normalization."""
        gen = CacheKeyGenerator(normalize_whitespace=True)
        key = gen.generate("  What   is   Python?  ")
        assert key == "What is Python?"

    def test_generate_lowercase(self):
        """Test lowercase conversion."""
        gen = CacheKeyGenerator(lowercase=True)
        key = gen.generate("What Is PYTHON?")
        assert key == "what is python?"

    def test_generate_from_messages(self):
        """Test key generation from messages."""
        gen = CacheKeyGenerator()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        key = gen.generate_from_messages(messages)
        # Should include user messages but not system by default
        assert "Hello" in key
        assert "How are you?" in key
        assert "Be helpful" not in key

    def test_generate_from_messages_with_system(self):
        """Test key generation including system message."""
        gen = CacheKeyGenerator()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        key = gen.generate_from_messages(messages, include_system=True)
        assert "Be helpful" in key
        assert "Hello" in key

    def test_generate_hash(self):
        """Test hash generation."""
        gen = CacheKeyGenerator()
        hash1 = gen.generate_hash("Hello")
        hash2 = gen.generate_hash("Hello")
        hash3 = gen.generate_hash("World")

        assert hash1 == hash2  # Same input = same hash
        assert hash1 != hash3  # Different input = different hash
        assert len(hash1) == 64  # SHA-256 is 64 hex chars

    def test_generate_with_context(self):
        """Test key generation with context."""
        gen = CacheKeyGenerator()
        key = gen.generate_with_context(
            "Hello",
            model="gpt-4",
            temperature=0.7,
        )
        assert "Hello" in key
        assert "gpt-4" in key
        assert "0.7" in key

    def test_extract_last_user_message(self):
        """Test extracting last user message."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        last = CacheKeyGenerator.extract_last_user_message(messages)
        assert last == "Second"

    def test_extract_last_user_message_empty(self):
        """Test extracting from empty messages."""
        last = CacheKeyGenerator.extract_last_user_message([])
        assert last is None


# -----------------------------------------------------------------------------
# MemoryCacheBackend Tests
# -----------------------------------------------------------------------------


class TestMemoryCacheBackend:
    """Tests for MemoryCacheBackend."""

    def test_set_and_get(self, memory_backend, sample_entry):
        """Test basic set and get operations."""
        memory_backend.set(sample_entry)

        # Get with exact embedding
        result = memory_backend.get(sample_entry.embedding, threshold=0.99)
        assert result is not None
        assert result.key == sample_entry.key
        assert result.value == sample_entry.value

    def test_get_no_match(self, memory_backend, sample_entry):
        """Test get with no matching entry."""
        memory_backend.set(sample_entry)

        # Get with very different embedding
        result = memory_backend.get([0.9, 0.9, 0.9, 0.9, 0.9], threshold=0.99)
        assert result is None

    def test_get_below_threshold(self, memory_backend, sample_entry):
        """Test get with similarity below threshold."""
        memory_backend.set(sample_entry)

        # Get with opposite direction embedding (negative cosine similarity)
        result = memory_backend.get([-0.1, -0.2, -0.3, -0.4, -0.5], threshold=0.99)
        assert result is None

    def test_delete(self, memory_backend, sample_entry):
        """Test delete operation."""
        memory_backend.set(sample_entry)
        assert len(memory_backend) == 1

        deleted = memory_backend.delete(sample_entry.key)
        assert deleted is True
        assert len(memory_backend) == 0

        # Delete non-existent
        deleted = memory_backend.delete("nonexistent")
        assert deleted is False

    def test_clear(self, memory_backend):
        """Test clear operation."""
        for i in range(5):
            memory_backend.set(
                CacheEntry(
                    key=f"key{i}",
                    value=f"value{i}",
                    embedding=[float(i)] * 5,
                )
            )

        assert len(memory_backend) == 5
        count = memory_backend.clear()
        assert count == 5
        assert len(memory_backend) == 0

    def test_lru_eviction(self):
        """Test LRU eviction when max_entries exceeded."""
        backend = MemoryCacheBackend(max_entries=3)

        # Add 3 entries
        for i in range(3):
            backend.set(
                CacheEntry(
                    key=f"key{i}",
                    value=f"value{i}",
                    embedding=[float(i)] * 5,
                )
            )

        assert len(backend) == 3

        # Add 4th entry - should evict oldest (key0)
        backend.set(
            CacheEntry(
                key="key3",
                value="value3",
                embedding=[3.0] * 5,
            )
        )

        assert len(backend) == 3
        assert "key0" not in backend
        assert "key3" in backend

    def test_evict_expired(self):
        """Test eviction of expired entries."""
        backend = MemoryCacheBackend()

        # Add expired entry
        backend.set(
            CacheEntry(
                key="expired",
                value="old",
                embedding=[0.1] * 5,
                expires_at=time.time() - 1,
            )
        )

        # Add valid entry
        backend.set(
            CacheEntry(
                key="valid",
                value="new",
                embedding=[0.2] * 5,
                expires_at=time.time() + 3600,
            )
        )

        assert len(backend) == 2
        evicted = backend.evict_expired()
        assert evicted == 1
        assert len(backend) == 1
        assert "expired" not in backend
        assert "valid" in backend

    def test_stats(self, memory_backend, sample_entry):
        """Test statistics gathering."""
        memory_backend.set(sample_entry)
        stats = memory_backend.stats()

        assert stats.entries == 1
        assert stats.size_bytes > 0

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Identical vectors
        sim = CacheBackend.cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 0.001

        # Orthogonal vectors
        sim = CacheBackend.cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(sim - 0.0) < 0.001

        # Similar vectors
        sim = CacheBackend.cosine_similarity([1, 1, 0], [1, 0.9, 0])
        assert sim > 0.99


# -----------------------------------------------------------------------------
# SemanticCache Tests
# -----------------------------------------------------------------------------


class TestSemanticCache:
    """Tests for SemanticCache class."""

    def test_create_with_defaults(self, mock_embeddings):
        """Test creating cache with default settings."""
        cache = SemanticCache(backend="memory", embeddings=mock_embeddings)
        assert cache.similarity_threshold == 0.95
        assert cache.ttl is None

    def test_create_with_custom_settings(self, mock_embeddings):
        """Test creating cache with custom settings."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.90,
            ttl=3600,
            max_entries=1000,
            embeddings=mock_embeddings,
        )
        assert cache.similarity_threshold == 0.90
        assert cache.ttl == 3600
        assert cache.max_entries == 1000

    def test_invalid_threshold(self, mock_embeddings):
        """Test validation of similarity threshold."""
        with pytest.raises(ValueError, match="similarity_threshold"):
            SemanticCache(
                backend="memory",
                similarity_threshold=1.5,
                embeddings=mock_embeddings,
            )

        with pytest.raises(ValueError, match="similarity_threshold"):
            SemanticCache(
                backend="memory",
                similarity_threshold=-0.1,
                embeddings=mock_embeddings,
            )

    def test_set_and_get(self, mock_embeddings):
        """Test basic set and get operations."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.5,
            embeddings=mock_embeddings,
        )

        # Set a value
        cache.set("What is Python?", "Python is a programming language.")

        # Get with same query (should hit)
        result = cache.get("What is Python?")
        assert result == "Python is a programming language."

    def test_cache_miss(self, mock_embeddings):
        """Test cache miss."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.99,
            embeddings=mock_embeddings,
        )

        # Get without setting
        result = cache.get("What is Python?")
        assert result is None

    def test_lookup_returns_details(self, mock_embeddings):
        """Test lookup returns CacheHit/CacheMiss with details."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.5,
            embeddings=mock_embeddings,
        )

        # Miss
        result = cache.lookup("test query")
        assert isinstance(result, CacheMiss)
        assert result.reason == "not_found"

        # Set and hit
        cache.set("test query", "test response")
        result = cache.lookup("test query")
        assert isinstance(result, CacheHit)
        assert result.value == "test response"
        assert result.similarity > 0

    def test_delete(self, mock_embeddings):
        """Test delete operation."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.5,
            embeddings=mock_embeddings,
        )

        cache.set("test", "value")
        assert cache.get("test") == "value"

        deleted = cache.delete("test")
        assert deleted is True
        assert cache.get("test") is None

    def test_clear(self, mock_embeddings):
        """Test clear operation."""
        cache = SemanticCache(
            backend="memory",
            embeddings=mock_embeddings,
        )

        cache.set("test1", "value1")
        cache.set("test2", "value2")

        count = cache.clear()
        assert count == 2

        stats = cache.stats()
        assert stats.entries == 0

    def test_stats(self, mock_embeddings):
        """Test statistics gathering."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.5,
            embeddings=mock_embeddings,
        )

        cache.set("test", "value")
        cache.get("test")  # Hit

        stats = cache.stats()
        assert stats.hits >= 1
        assert stats.entries == 1

    def test_hit_rate(self, mock_embeddings):
        """Test hit rate calculation."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.5,
            embeddings=mock_embeddings,
        )

        assert cache.hit_rate == 0.0  # No lookups yet

        cache.set("test", "value")
        cache.get("test")  # Hit

        # Hit rate should be > 0 after a hit
        assert cache.hit_rate > 0

    def test_repr(self, mock_embeddings):
        """Test string representation."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.95,
            embeddings=mock_embeddings,
        )
        repr_str = repr(cache)
        assert "SemanticCache" in repr_str
        assert "0.95" in repr_str


# -----------------------------------------------------------------------------
# Async Tests
# -----------------------------------------------------------------------------


class TestSemanticCacheAsync:
    """Tests for async SemanticCache operations."""

    @pytest.mark.asyncio
    async def test_aget(self, mock_embeddings):
        """Test async get operation."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.5,
            embeddings=mock_embeddings,
        )

        cache.set("test", "value")  # Use sync set
        result = await cache.aget("test")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_aset(self, mock_embeddings):
        """Test async set operation."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.5,
            embeddings=mock_embeddings,
        )

        await cache.aset("test", "value")
        result = cache.get("test")  # Use sync get
        assert result == "value"

    @pytest.mark.asyncio
    async def test_alookup(self, mock_embeddings):
        """Test async lookup operation."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.5,
            embeddings=mock_embeddings,
        )

        # Miss
        result = await cache.alookup("test")
        assert isinstance(result, CacheMiss)

        # Set and hit
        await cache.aset("test", "value")
        result = await cache.alookup("test")
        assert isinstance(result, CacheHit)
        assert result.value == "value"

    @pytest.mark.asyncio
    async def test_adelete(self, mock_embeddings):
        """Test async delete operation."""
        cache = SemanticCache(
            backend="memory",
            similarity_threshold=0.5,
            embeddings=mock_embeddings,
        )

        await cache.aset("test", "value")
        deleted = await cache.adelete("test")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_aclear(self, mock_embeddings):
        """Test async clear operation."""
        cache = SemanticCache(
            backend="memory",
            embeddings=mock_embeddings,
        )

        await cache.aset("test1", "value1")
        await cache.aset("test2", "value2")

        count = await cache.aclear()
        assert count == 2


# -----------------------------------------------------------------------------
# CacheStats Tests
# -----------------------------------------------------------------------------


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == 0.7

    def test_hit_rate_no_lookups(self):
        """Test hit rate with no lookups."""
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_defaults(self):
        """Test default values."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.entries == 0
        assert stats.size_bytes == 0
