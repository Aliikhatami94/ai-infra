"""Tool loading utilities for MCP (Model Context Protocol).

This module provides helpers for loading and caching MCP tools with
automatic locking for thread-safety.

Example:
    from ai_infra import Agent, load_mcp_tools_cached

    # Load tools (cached automatically)
    tools = await load_mcp_tools_cached("http://localhost:8000/mcp/docs/mcp")
    agent = Agent(tools=tools)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Global cache for loaded MCP tools
_cached_tools: dict[str, list[Any]] = {}

# Locks per cache key for thread-safe loading
_locks: dict[str, asyncio.Lock] = {}


async def load_mcp_tools_cached(
    url: str,
    *,
    transport: Literal["stdio", "streamable_http", "sse"] = "streamable_http",
    cache_key: str | None = None,
    force_refresh: bool = False,
) -> list[Any]:
    """Load MCP tools with automatic caching.

    Tools are cached by URL to avoid repeated network calls. Thread-safe
    using asyncio locks per cache key.

    Args:
        url: MCP server URL
        transport: Transport type (default: streamable_http)
        cache_key: Custom cache key (default: url)
        force_refresh: Force reload even if cached

    Returns:
        List of tools from MCP server

    Example:
        ```python
        # First call loads from server
        tools = await load_mcp_tools_cached(
            "http://localhost:8000/mcp/docs/mcp"
        )

        # Subsequent calls return cached tools
        tools = await load_mcp_tools_cached(
            "http://localhost:8000/mcp/docs/mcp"
        )

        # Force refresh to reload
        tools = await load_mcp_tools_cached(
            "http://localhost:8000/mcp/docs/mcp",
            force_refresh=True
        )

        # Use with Agent
        agent = Agent(tools=tools)
        result = await agent.arun("Search documentation")
        ```
    """
    key = cache_key or url

    # Create lock for this key if it doesn't exist
    # Note: This dict access itself is not thread-safe, but asyncio Lock
    # creation is idempotent and this is acceptable for the use case
    if key not in _locks:
        _locks[key] = asyncio.Lock()

    # Acquire lock for all cache operations to prevent race conditions
    # Previous implementation had check-before-lock which could return
    # stale data while another coroutine was updating
    async with _locks[key]:
        # Check cache inside lock to prevent race conditions
        if not force_refresh and key in _cached_tools:
            logger.debug(f"Using cached MCP tools for {url}")
            return _cached_tools[key]

        # Load tools from MCP server
        try:
            from ai_infra.mcp.client import MCPClient
            from ai_infra.mcp.client.models import McpServerConfig

            logger.info(f"Loading MCP tools from {url}")

            # Create client config
            config = McpServerConfig(transport=transport, url=url)

            # Create client and load tools
            client = MCPClient([config])
            tools = await client.list_tools()

            # Cache the tools
            _cached_tools[key] = tools

            # Log tool names for observability
            tool_names = [getattr(t, "name", str(t)) for t in tools]
            logger.info(f"Loaded {len(tools)} MCP tools from {url}: {tool_names}")

            return tools

        except Exception as e:
            logger.error(f"Failed to load MCP tools from {url}: {e}")
            raise


def clear_mcp_cache(url: str | None = None) -> None:
    """Clear MCP tool cache.

    Args:
        url: Specific URL to clear, or None to clear all

    Example:
        ```python
        # Clear specific URL
        clear_mcp_cache("http://localhost:8000/mcp/docs/mcp")

        # Clear all cached tools
        clear_mcp_cache()
        ```
    """
    if url:
        _cached_tools.pop(url, None)
        logger.debug(f"Cleared MCP cache for {url}")
    else:
        _cached_tools.clear()
        logger.debug("Cleared all MCP cache")


def get_cached_tools(url: str) -> list[Any] | None:
    """Get cached tools without loading.

    Args:
        url: MCP server URL

    Returns:
        Cached tools, or None if not cached

    Example:
        ```python
        # Check if tools are cached
        tools = get_cached_tools("http://localhost:8000/mcp/docs/mcp")
        if tools is None:
            print("Not cached yet")
        ```
    """
    return _cached_tools.get(url)


def is_cached(url: str) -> bool:
    """Check if tools are cached for a URL.

    Args:
        url: MCP server URL

    Returns:
        True if tools are cached

    Example:
        ```python
        if is_cached("http://localhost:8000/mcp/docs/mcp"):
            print("Tools are cached")
        ```
    """
    return url in _cached_tools


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics.

    Returns:
        Dict with cache info (urls, total_tools, etc.)

    Example:
        ```python
        stats = get_cache_stats()
        print(f"Cached URLs: {stats['cached_urls']}")
        print(f"Total tools: {stats['total_tools']}")
        ```
    """
    return {
        "cached_urls": list(_cached_tools.keys()),
        "cache_size": len(_cached_tools),
        "total_tools": sum(len(tools) for tools in _cached_tools.values()),
    }


__all__ = [
    "clear_mcp_cache",
    "get_cache_stats",
    "get_cached_tools",
    "is_cached",
    "load_mcp_tools_cached",
]
