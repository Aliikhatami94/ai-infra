"""In-memory cache backend with LRU eviction.

This is the simplest and fastest cache backend, suitable for development
and single-process applications. Data is lost when the process exits.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from ai_infra.cache.base import CacheBackend, CacheEntry, CacheStats

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction.

    Stores cache entries in memory with O(n) vector similarity search.
    Supports LRU eviction when max_entries is reached and TTL-based expiry.

    Features:
        - Thread-safe operations
        - LRU eviction when max_entries reached
        - TTL support with automatic expiry checks
        - O(n) brute-force similarity search (suitable for < 10k entries)

    Example:
        >>> from ai_infra.cache.backends import MemoryCacheBackend
        >>>
        >>> backend = MemoryCacheBackend(max_entries=1000)
        >>> backend.set(CacheEntry(
        ...     key="test",
        ...     value="response",
        ...     embedding=[0.1, 0.2, 0.3],
        ... ))
        >>> entry = backend.get([0.1, 0.2, 0.3], threshold=0.95)

    Note:
        For larger caches (> 10k entries), consider using SQLite or Redis
        backends which use optimized vector indexes.
    """

    def __init__(
        self,
        max_entries: int | None = None,
    ) -> None:
        """Initialize the in-memory cache backend.

        Args:
            max_entries: Maximum number of entries to store. When exceeded,
                the least recently used entries are evicted. None = unlimited.
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._total_hits = 0
        self._total_similarity = 0.0

    def get(
        self,
        embedding: list[float],
        threshold: float = 0.95,
    ) -> CacheEntry | None:
        """Find a cached entry with semantic similarity above threshold.

        Performs brute-force comparison against all entries. Thread-safe.

        Args:
            embedding: Vector embedding of the query.
            threshold: Minimum cosine similarity (0.0 to 1.0).

        Returns:
            Best matching CacheEntry if similarity >= threshold, None otherwise.
        """
        with self._lock:
            best_entry: CacheEntry | None = None
            best_similarity = threshold  # Start with threshold as minimum

            for entry in self._cache.values():
                # Skip expired entries
                if entry.is_expired:
                    continue

                similarity = self.cosine_similarity(embedding, entry.embedding)
                if similarity >= best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            if best_entry is not None:
                # Update LRU order - move to end
                self._cache.move_to_end(best_entry.key)

                # Update hit statistics
                best_entry.hit_count += 1
                best_entry.last_accessed = time.time()

                self._total_hits += 1
                self._total_similarity += best_similarity

            return best_entry

    async def aget(
        self,
        embedding: list[float],
        threshold: float = 0.95,
    ) -> CacheEntry | None:
        """Async version of get(). Memory backend is sync, so just wraps get()."""
        return self.get(embedding, threshold)

    def set(self, entry: CacheEntry) -> None:
        """Store a cache entry.

        If max_entries is set and exceeded, evicts least recently used entries.

        Args:
            entry: The CacheEntry to store.
        """
        with self._lock:
            # If entry exists, update it
            if entry.key in self._cache:
                self._cache.move_to_end(entry.key)

            self._cache[entry.key] = entry

            # Evict if necessary
            if self._max_entries is not None:
                while len(self._cache) > self._max_entries:
                    # Pop oldest (first) item
                    evicted_key, evicted_entry = self._cache.popitem(last=False)
                    logger.debug(f"Evicted cache entry: {evicted_key[:50]}...")

    async def aset(self, entry: CacheEntry) -> None:
        """Async version of set()."""
        self.set(entry)

    def delete(self, key: str) -> bool:
        """Delete a cache entry by key.

        Args:
            key: The cache key to delete.

        Returns:
            True if entry was deleted, False if not found.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def adelete(self, key: str) -> bool:
        """Async version of delete()."""
        return self.delete(key)

    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._total_hits = 0
            self._total_similarity = 0.0
            return count

    async def aclear(self) -> int:
        """Async version of clear()."""
        return self.clear()

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with current cache state.
        """
        with self._lock:
            # Estimate size in bytes
            size_bytes = 0
            for entry in self._cache.values():
                size_bytes += len(entry.key.encode())
                size_bytes += len(entry.value.encode())
                size_bytes += len(entry.embedding) * 8  # float64

            avg_similarity = 0.0
            if self._total_hits > 0:
                avg_similarity = self._total_similarity / self._total_hits

            return CacheStats(
                entries=len(self._cache),
                size_bytes=size_bytes,
                avg_similarity=avg_similarity,
            )

    def evict_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries evicted.
        """
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    async def aevict_expired(self) -> int:
        """Async version of evict_expired()."""
        return self.evict_expired()

    def __len__(self) -> int:
        """Return number of entries in cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        with self._lock:
            return key in self._cache

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MemoryCacheBackend(entries={len(self)}, max={self._max_entries})"
