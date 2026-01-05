"""Cache backend implementations.

This module provides various cache backend implementations for different
storage needs:

- MemoryCacheBackend: In-memory cache with LRU eviction
- SQLiteCacheBackend: SQLite with vector search (requires sqlite-vec)
- RedisCacheBackend: Redis with vector search (requires Redis Stack)
- PostgresCacheBackend: PostgreSQL with pgvector

Example:
    >>> from ai_infra.cache.backends import MemoryCacheBackend
    >>>
    >>> backend = MemoryCacheBackend(max_entries=1000)
"""

from ai_infra.cache.backends.memory import MemoryCacheBackend

__all__ = [
    "MemoryCacheBackend",
]

# Optional backends - imported on demand
# from ai_infra.cache.backends.sqlite import SQLiteCacheBackend
# from ai_infra.cache.backends.redis import RedisCacheBackend
# from ai_infra.cache.backends.postgres import PostgresCacheBackend
