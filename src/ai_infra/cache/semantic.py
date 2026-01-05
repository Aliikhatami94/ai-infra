"""Semantic cache for LLM responses.

This module provides the main SemanticCache class that integrates with
LLM providers to cache responses based on semantic similarity.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

from ai_infra.cache.base import (
    CacheBackend,
    CacheEntry,
    CacheError,
    CacheHit,
    CacheMiss,
    CacheStats,
)
from ai_infra.cache.key import CacheKeyGenerator

if TYPE_CHECKING:
    from ai_infra.embeddings import Embeddings

logger = logging.getLogger(__name__)


class SemanticCache:
    """Semantic cache for LLM responses.

    Caches LLM responses based on semantic similarity rather than exact
    string matching. Uses vector embeddings to find similar queries and
    returns cached responses when the similarity exceeds a threshold.

    Features:
        - Semantic matching: Find cached responses for similar queries
        - Multiple backends: Memory, SQLite, Redis, PostgreSQL
        - TTL support: Automatic expiration of cached entries
        - LRU eviction: Automatic cleanup when cache is full
        - Statistics: Track hit rate, latency, and cache size
        - Async support: Full async API for non-blocking operations

    Example - Basic usage:
        >>> from ai_infra.cache import SemanticCache
        >>>
        >>> cache = SemanticCache(
        ...     backend="memory",
        ...     similarity_threshold=0.95,
        ... )
        >>>
        >>> # Check cache
        >>> response = cache.get("What is the capital of France?")
        >>> if response is None:
        ...     response = "Paris is the capital."
        ...     cache.set("What is the capital of France?", response)

    Example - With LLM:
        >>> from ai_infra import LLM
        >>> from ai_infra.cache import SemanticCache
        >>>
        >>> cache = SemanticCache(backend="memory")
        >>> llm = LLM(cache=cache)
        >>>
        >>> # First call - fetches from LLM
        >>> response1 = llm.chat("What is France's capital?")
        >>>
        >>> # Second call - returns cached (semantically similar)
        >>> response2 = llm.chat("What's the capital of France?")

    Example - Persistent cache:
        >>> cache = SemanticCache(
        ...     backend="sqlite",
        ...     path="./cache.db",
        ...     ttl=3600,  # 1 hour
        ...     max_entries=10000,
        ... )
    """

    def __init__(
        self,
        backend: Literal["memory", "sqlite", "redis", "postgres"] | CacheBackend = "memory",
        similarity_threshold: float = 0.95,
        ttl: int | None = None,
        max_entries: int | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embeddings: Embeddings | None = None,
        key_generator: CacheKeyGenerator | None = None,
        **backend_kwargs: Any,
    ) -> None:
        """Initialize the semantic cache.

        Args:
            backend: Cache backend to use. Can be a string ("memory", "sqlite",
                "redis", "postgres") or a CacheBackend instance.
            similarity_threshold: Minimum cosine similarity for cache hit (0.0-1.0).
                Higher values require closer semantic match. Default 0.95.
            ttl: Time-to-live in seconds for cache entries. None = no expiry.
            max_entries: Maximum number of entries to cache. None = unlimited.
            embedding_provider: Provider for embeddings (e.g., "openai", "huggingface").
                Uses ai-infra Embeddings auto-detection if not specified.
            embedding_model: Model for embeddings. Uses provider default if not specified.
            embeddings: Pre-configured Embeddings instance. If provided, embedding_provider
                and embedding_model are ignored.
            key_generator: Custom cache key generator. Uses default if not specified.
            **backend_kwargs: Additional arguments passed to the backend constructor.
                - For "sqlite": path (str) - path to database file
                - For "redis": url (str) - Redis connection URL
                - For "postgres": url (str) - PostgreSQL connection URL

        Raises:
            ValueError: If similarity_threshold is not between 0.0 and 1.0.
            CacheError: If backend initialization fails.

        Example:
            >>> # Memory backend (default)
            >>> cache = SemanticCache()
            >>>
            >>> # SQLite backend with custom settings
            >>> cache = SemanticCache(
            ...     backend="sqlite",
            ...     path="./cache.db",
            ...     similarity_threshold=0.90,
            ...     ttl=3600,
            ... )
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}"
            )

        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.max_entries = max_entries

        # Initialize key generator
        self._key_generator = key_generator or CacheKeyGenerator()

        # Initialize embeddings
        self._embeddings = embeddings
        if self._embeddings is None:
            from ai_infra.embeddings import Embeddings as EmbeddingsClass

            self._embeddings = EmbeddingsClass(
                provider=embedding_provider,
                model=embedding_model,
            )

        # Initialize backend
        if isinstance(backend, CacheBackend):
            self._backend = backend
        else:
            self._backend = self._create_backend(
                backend,
                max_entries=max_entries,
                **backend_kwargs,
            )

        # Statistics
        self._hits = 0
        self._misses = 0
        self._total_latency = 0.0

        logger.debug(
            f"Initialized SemanticCache with backend={backend}, "
            f"threshold={similarity_threshold}, ttl={ttl}"
        )

    def _create_backend(
        self,
        backend_type: str,
        **kwargs: Any,
    ) -> CacheBackend:
        """Create a cache backend instance.

        Args:
            backend_type: Type of backend to create.
            **kwargs: Backend-specific arguments.

        Returns:
            Initialized CacheBackend instance.

        Raises:
            CacheError: If backend creation fails.
        """
        try:
            if backend_type == "memory":
                from ai_infra.cache.backends.memory import MemoryCacheBackend

                return MemoryCacheBackend(
                    max_entries=kwargs.get("max_entries"),
                )
            elif backend_type == "sqlite":
                from ai_infra.cache.backends.sqlite import SQLiteCacheBackend

                return SQLiteCacheBackend(
                    path=kwargs.get("path", "./cache.db"),
                    max_entries=kwargs.get("max_entries"),
                )
            elif backend_type == "redis":
                from ai_infra.cache.backends.redis import RedisCacheBackend

                return RedisCacheBackend(
                    url=kwargs.get("url", "redis://localhost:6379"),
                    index_name=kwargs.get("index_name", "ai_cache"),
                    max_entries=kwargs.get("max_entries"),
                )
            elif backend_type == "postgres":
                from ai_infra.cache.backends.postgres import PostgresCacheBackend

                return PostgresCacheBackend(
                    url=kwargs.get("url"),
                    table_name=kwargs.get("table_name", "ai_cache"),
                    max_entries=kwargs.get("max_entries"),
                )
            else:
                raise CacheError(f"Unknown backend type: {backend_type}")
        except ImportError as e:
            raise CacheError(
                f"Failed to import backend '{backend_type}': {e}. "
                f"Make sure required dependencies are installed."
            ) from e
        except Exception as e:
            raise CacheError(f"Failed to create backend '{backend_type}': {e}") from e

    def get(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Look up a cached response for a query.

        Args:
            query: The query to look up.
            context: Optional context for key generation.

        Returns:
            Cached response if found and above threshold, None otherwise.

        Example:
            >>> cache = SemanticCache()
            >>> response = cache.get("What is the capital of France?")
            >>> if response is None:
            ...     # Cache miss - need to generate response
            ...     pass
        """
        result = self.lookup(query, context)
        if isinstance(result, CacheHit):
            return result.value
        return None

    async def aget(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Async version of get()."""
        result = await self.alookup(query, context)
        if isinstance(result, CacheHit):
            return result.value
        return None

    def lookup(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> CacheHit | CacheMiss:
        """Look up a cached response with full result details.

        Args:
            query: The query to look up.
            context: Optional context for key generation.

        Returns:
            CacheHit with entry and similarity, or CacheMiss with reason.

        Example:
            >>> result = cache.lookup("What is Python?")
            >>> if isinstance(result, CacheHit):
            ...     print(f"Hit! Similarity: {result.similarity:.2f}")
            ...     print(result.value)
            >>> else:
            ...     print(f"Miss: {result.reason}")
        """
        start_time = time.time()

        try:
            # Generate cache key
            key = self._key_generator.generate(query)

            # Get embedding for the query
            assert self._embeddings is not None, "Embeddings not initialized"
            embedding = self._embeddings.embed(key)

            # Look up in backend
            entry = self._backend.get(embedding, self.similarity_threshold)

            latency_ms = (time.time() - start_time) * 1000

            if entry is None:
                self._misses += 1
                self._total_latency += latency_ms
                return CacheMiss(query=query, reason="not_found", latency_ms=latency_ms)

            if entry.is_expired:
                self._misses += 1
                self._total_latency += latency_ms
                # Optionally delete expired entry
                self._backend.delete(entry.key)
                return CacheMiss(query=query, reason="expired", latency_ms=latency_ms)

            # Calculate similarity for the result
            similarity = CacheBackend.cosine_similarity(embedding, entry.embedding)

            self._hits += 1
            self._total_latency += latency_ms

            logger.debug(f"Cache hit for '{query[:50]}...' (similarity={similarity:.3f})")

            return CacheHit(entry=entry, similarity=similarity, latency_ms=latency_ms)

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._misses += 1
            self._total_latency += latency_ms
            logger.warning(f"Cache lookup failed: {e}")
            return CacheMiss(query=query, reason="error", latency_ms=latency_ms)

    async def alookup(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> CacheHit | CacheMiss:
        """Async version of lookup()."""
        start_time = time.time()

        try:
            # Generate cache key
            key = self._key_generator.generate(query)

            # Get embedding for the query
            assert self._embeddings is not None, "Embeddings not initialized"
            embedding = await self._embeddings.aembed(key)

            # Look up in backend
            entry = await self._backend.aget(embedding, self.similarity_threshold)

            latency_ms = (time.time() - start_time) * 1000

            if entry is None:
                self._misses += 1
                self._total_latency += latency_ms
                return CacheMiss(query=query, reason="not_found", latency_ms=latency_ms)

            if entry.is_expired:
                self._misses += 1
                self._total_latency += latency_ms
                await self._backend.adelete(entry.key)
                return CacheMiss(query=query, reason="expired", latency_ms=latency_ms)

            similarity = CacheBackend.cosine_similarity(embedding, entry.embedding)

            self._hits += 1
            self._total_latency += latency_ms

            logger.debug(f"Cache hit for '{query[:50]}...' (similarity={similarity:.3f})")

            return CacheHit(entry=entry, similarity=similarity, latency_ms=latency_ms)

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._misses += 1
            self._total_latency += latency_ms
            logger.warning(f"Async cache lookup failed: {e}")
            return CacheMiss(query=query, reason="error", latency_ms=latency_ms)

    def set(
        self,
        query: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Cache a response for a query.

        Args:
            query: The original query.
            response: The response to cache.
            metadata: Optional metadata to store with the entry.

        Example:
            >>> cache.set(
            ...     "What is Python?",
            ...     "Python is a programming language.",
            ...     metadata={"model": "gpt-4"},
            ... )
        """
        try:
            # Generate cache key
            key = self._key_generator.generate(query)

            # Get embedding
            assert self._embeddings is not None, "Embeddings not initialized"
            embedding = self._embeddings.embed(key)

            # Calculate expiry
            expires_at = None
            if self.ttl is not None:
                expires_at = time.time() + self.ttl

            # Create entry
            entry = CacheEntry(
                key=key,
                value=response,
                embedding=embedding,
                expires_at=expires_at,
                metadata=metadata or {},
            )

            # Store in backend
            self._backend.set(entry)

            logger.debug(f"Cached response for '{query[:50]}...'")

        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    async def aset(
        self,
        query: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Async version of set()."""
        try:
            key = self._key_generator.generate(query)
            assert self._embeddings is not None, "Embeddings not initialized"
            embedding = await self._embeddings.aembed(key)

            expires_at = None
            if self.ttl is not None:
                expires_at = time.time() + self.ttl

            entry = CacheEntry(
                key=key,
                value=response,
                embedding=embedding,
                expires_at=expires_at,
                metadata=metadata or {},
            )

            await self._backend.aset(entry)

            logger.debug(f"Cached response for '{query[:50]}...'")

        except Exception as e:
            logger.warning(f"Failed to async cache response: {e}")

    def delete(self, query: str) -> bool:
        """Delete a cached entry.

        Args:
            query: The query to delete from cache.

        Returns:
            True if entry was deleted, False if not found.
        """
        key = self._key_generator.generate(query)
        return self._backend.delete(key)

    async def adelete(self, query: str) -> bool:
        """Async version of delete()."""
        key = self._key_generator.generate(query)
        return await self._backend.adelete(key)

    def clear(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries removed.
        """
        count = self._backend.clear()
        self._hits = 0
        self._misses = 0
        self._total_latency = 0.0
        logger.info(f"Cleared {count} entries from cache")
        return count

    async def aclear(self) -> int:
        """Async version of clear()."""
        count = await self._backend.aclear()
        self._hits = 0
        self._misses = 0
        self._total_latency = 0.0
        logger.info(f"Cleared {count} entries from cache")
        return count

    def evict_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries evicted.
        """
        count = self._backend.evict_expired()
        if count > 0:
            logger.info(f"Evicted {count} expired entries from cache")
        return count

    async def aevict_expired(self) -> int:
        """Async version of evict_expired()."""
        count = await self._backend.aevict_expired()
        if count > 0:
            logger.info(f"Evicted {count} expired entries from cache")
        return count

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with hits, misses, hit rate, etc.

        Example:
            >>> stats = cache.stats()
            >>> print(f"Hit rate: {stats.hit_rate:.1%}")
            >>> print(f"Entries: {stats.entries}")
        """
        backend_stats = self._backend.stats()

        # Merge with our tracking
        backend_stats.hits = self._hits
        backend_stats.misses = self._misses

        total_lookups = self._hits + self._misses
        if total_lookups > 0:
            backend_stats.avg_latency_ms = self._total_latency / total_lookups

        return backend_stats

    @property
    def hit_rate(self) -> float:
        """Get the cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def __repr__(self) -> str:
        """Return string representation."""
        stats = self.stats()
        return (
            f"SemanticCache(threshold={self.similarity_threshold}, "
            f"entries={stats.entries}, hit_rate={self.hit_rate:.1%})"
        )
