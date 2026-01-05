"""Base classes for cache backends.

This module provides the abstract base classes and data structures for
implementing cache backends.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base exception for cache errors."""

    pass


@dataclass
class CacheEntry:
    """A cached response with metadata.

    Attributes:
        key: Original query/prompt that was cached.
        value: The cached response.
        embedding: Vector embedding of the key for similarity search.
        created_at: Unix timestamp when entry was created.
        expires_at: Unix timestamp when entry expires (None = never).
        metadata: Additional metadata about the cached response.
        hit_count: Number of times this entry has been retrieved.
        last_accessed: Unix timestamp of last access.

    Example:
        >>> entry = CacheEntry(
        ...     key="What is the capital of France?",
        ...     value="Paris is the capital of France.",
        ...     embedding=[0.1, 0.2, ...],
        ... )
    """

    key: str
    value: str
    embedding: list[float]
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    hit_count: int = 0
    last_accessed: float | None = None

    @property
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
            "hit_count": self.hit_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """Create entry from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            embedding=data["embedding"],
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            metadata=data.get("metadata", {}),
            hit_count=data.get("hit_count", 0),
            last_accessed=data.get("last_accessed"),
        )


@dataclass
class CacheHit:
    """Result of a successful cache lookup.

    Attributes:
        entry: The matched cache entry.
        similarity: Similarity score between query and cached key (0.0 to 1.0).
        latency_ms: Time taken for the lookup in milliseconds.
    """

    entry: CacheEntry
    similarity: float
    latency_ms: float = 0.0

    @property
    def value(self) -> str:
        """Get the cached value."""
        return self.entry.value


@dataclass
class CacheMiss:
    """Result of a cache lookup that found no match.

    Attributes:
        query: The query that was searched.
        reason: Why the cache missed.
        latency_ms: Time taken for the lookup in milliseconds.
    """

    query: str
    reason: Literal["not_found", "below_threshold", "expired", "error"] = "not_found"
    latency_ms: float = 0.0


@dataclass
class CacheStats:
    """Statistics about cache usage.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        entries: Total entries in cache.
        size_bytes: Approximate size of cache in bytes.
        avg_similarity: Average similarity score of hits.
        avg_latency_ms: Average lookup latency in milliseconds.
    """

    hits: int = 0
    misses: int = 0
    entries: int = 0
    size_bytes: int = 0
    avg_similarity: float = 0.0
    avg_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class CacheBackend(ABC):
    """Abstract base class for cache backend implementations.

    Cache backends are responsible for storing and retrieving cached entries.
    They must implement vector similarity search for semantic matching.

    Subclasses must implement:
        - get(): Find semantically similar cached entry
        - set(): Store a new cache entry
        - delete(): Remove an entry by key
        - clear(): Remove all entries
        - stats(): Return cache statistics

    Example - Implementing a custom backend:
        >>> class MyBackend(CacheBackend):
        ...     def __init__(self):
        ...         self._cache = {}
        ...
        ...     def get(
        ...         self,
        ...         embedding: list[float],
        ...         threshold: float,
        ...     ) -> CacheEntry | None:
        ...         # Find similar entry
        ...         ...
        ...
        ...     def set(self, entry: CacheEntry) -> None:
        ...         self._cache[entry.key] = entry
    """

    @abstractmethod
    def get(
        self,
        embedding: list[float],
        threshold: float = 0.95,
    ) -> CacheEntry | None:
        """Find a cached entry with semantic similarity above threshold.

        Args:
            embedding: Vector embedding of the query.
            threshold: Minimum cosine similarity (0.0 to 1.0).

        Returns:
            Matching CacheEntry if found, None otherwise.
        """
        pass

    @abstractmethod
    async def aget(
        self,
        embedding: list[float],
        threshold: float = 0.95,
    ) -> CacheEntry | None:
        """Async version of get()."""
        pass

    @abstractmethod
    def set(self, entry: CacheEntry) -> None:
        """Store a cache entry.

        Args:
            entry: The CacheEntry to store.
        """
        pass

    @abstractmethod
    async def aset(self, entry: CacheEntry) -> None:
        """Async version of set()."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a cache entry by its original key.

        Args:
            key: The original query/prompt key.

        Returns:
            True if entry was deleted, False if not found.
        """
        pass

    @abstractmethod
    async def adelete(self, key: str) -> bool:
        """Async version of delete()."""
        pass

    @abstractmethod
    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns:
            Number of entries removed.
        """
        pass

    @abstractmethod
    async def aclear(self) -> int:
        """Async version of clear()."""
        pass

    @abstractmethod
    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with current cache state.
        """
        pass

    def evict_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries evicted.
        """
        # Default implementation does nothing
        # Backends should override if they support TTL
        return 0

    async def aevict_expired(self) -> int:
        """Async version of evict_expired()."""
        return self.evict_expired()

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity (0.0 to 1.0).
        """
        if len(a) != len(b):
            raise ValueError(f"Vectors must have same length: {len(a)} != {len(b)}")

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def hash_key(key: str) -> str:
        """Generate a hash for a key string.

        Args:
            key: The key to hash.

        Returns:
            SHA-256 hash of the key.
        """
        return hashlib.sha256(key.encode()).hexdigest()
