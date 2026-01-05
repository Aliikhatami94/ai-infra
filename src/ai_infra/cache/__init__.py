"""Semantic cache module for ai-infra.

This module provides semantic caching for LLM responses, reducing costs and
latency by caching responses based on semantic similarity rather than exact
string matching.

Key Components:
- SemanticCache: Main cache class with pluggable backends
- CacheBackend: Abstract base for cache storage implementations
- CacheEntry: Cached response with metadata

Example - Basic usage:
    >>> from ai_infra.cache import SemanticCache
    >>>
    >>> cache = SemanticCache(
    ...     backend="memory",
    ...     similarity_threshold=0.95,
    ... )
    >>>
    >>> # Manual usage
    >>> response = cache.get("What is the capital of France?")
    >>> if response is None:
    ...     response = "Paris is the capital of France."
    ...     cache.set("What is the capital of France?", response)

Example - With LLM integration:
    >>> from ai_infra import LLM
    >>> from ai_infra.cache import SemanticCache
    >>>
    >>> cache = SemanticCache(backend="memory")
    >>> llm = LLM(cache=cache)
    >>>
    >>> # First call - cache miss
    >>> response1 = llm.chat("What is France's capital?")
    >>>
    >>> # Second call - cache hit (semantically similar)
    >>> response2 = llm.chat("What's the capital of France?")

Example - SQLite backend for persistence:
    >>> cache = SemanticCache(
    ...     backend="sqlite",
    ...     path="./cache.db",
    ...     ttl=3600,  # 1 hour expiry
    ... )

For full documentation on semantic caching, see:
https://docs.nfrax.dev/ai-infra/cache/
"""

from __future__ import annotations

from ai_infra.cache.base import (
    CacheBackend,
    CacheEntry,
    CacheError,
    CacheHit,
    CacheMiss,
    CacheStats,
)
from ai_infra.cache.key import CacheKeyGenerator
from ai_infra.cache.semantic import SemanticCache

__all__ = [
    # Main class
    "SemanticCache",
    # Base classes
    "CacheBackend",
    "CacheEntry",
    "CacheHit",
    "CacheMiss",
    "CacheStats",
    "CacheError",
    # Key generation
    "CacheKeyGenerator",
]
