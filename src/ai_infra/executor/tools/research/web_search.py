"""Web search tool for research capability.

Phase 3.4.1 of EXECUTOR_1.md: Provides web search functionality
for agents to find information when stuck.

Supports multiple search backends:
- Brave Search API (default, fast)
- DuckDuckGo (fallback, no API key required)

Example:
    ```python
    from ai_infra.executor.tools.research import web_search

    # Simple search
    results = await web_search.ainvoke({
        "query": "FastAPI authentication best practices",
        "max_results": 5,
    })

    for result in results:
        print(f"{result['title']}: {result['url']}")
    ```
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_infra.logging import get_logger

__all__ = [
    "WebSearchResult",
    "WebSearchTool",
    "create_web_search_tool",
    "web_search",
]

logger = get_logger("executor.tools.web_search")


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class WebSearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(description="Search query. Be specific and include context.")
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return (1-10).",
        ge=1,
        le=10,
    )


# =============================================================================
# Search Backends
# =============================================================================


async def _search_brave(
    query: str,
    max_results: int,
    api_key: str,
) -> list[WebSearchResult]:
    """Search using Brave Search API.

    Args:
        query: Search query.
        max_results: Max results to return.
        api_key: Brave Search API key.

    Returns:
        List of search results.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for web search. Install with: pip install httpx")

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }
    params = {
        "q": query,
        "count": max_results,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

    results: list[WebSearchResult] = []
    for item in data.get("web", {}).get("results", []):
        results.append(
            WebSearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            )
        )

    return results[:max_results]


async def _search_duckduckgo(
    query: str,
    max_results: int,
) -> list[WebSearchResult]:
    """Search using DuckDuckGo (no API key required).

    Args:
        query: Search query.
        max_results: Max results to return.

    Returns:
        List of search results.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError(
            "duckduckgo-search is required for fallback. "
            "Install with: pip install duckduckgo-search"
        )

    results: list[WebSearchResult] = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                )
            )

    return results[:max_results]


# =============================================================================
# Web Search Tool
# =============================================================================


class WebSearchTool:
    """Web search tool with configurable backend.

    Supports Brave Search API (fast, requires key) and
    DuckDuckGo (fallback, no key required).

    Example:
        ```python
        tool = WebSearchTool(api_key="brave-api-key")
        results = await tool.search("FastAPI auth", max_results=5)
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        backend: str = "auto",
    ) -> None:
        """Initialize the web search tool.

        Args:
            api_key: Brave Search API key (optional).
            backend: "brave", "duckduckgo", or "auto" (default).
        """
        self._api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY")
        self._backend = backend

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """Search the web.

        Args:
            query: Search query.
            max_results: Maximum results to return.

        Returns:
            List of WebSearchResult objects.
        """
        # Determine backend
        backend = self._backend
        if backend == "auto":
            backend = "brave" if self._api_key else "duckduckgo"

        logger.debug(f"Web search: query='{query}', backend={backend}")

        if backend == "brave":
            if not self._api_key:
                raise ValueError(
                    "Brave Search API key required. Set BRAVE_SEARCH_API_KEY "
                    "or pass api_key parameter."
                )
            return await _search_brave(query, max_results, self._api_key)

        elif backend == "duckduckgo":
            return await _search_duckduckgo(query, max_results)

        else:
            raise ValueError(f"Unknown search backend: {backend}")


# =============================================================================
# LangChain Tool Creation
# =============================================================================


def create_web_search_tool(
    api_key: str | None = None,
    backend: str = "auto",
) -> StructuredTool:
    """Create a LangChain-compatible web search tool.

    Args:
        api_key: Brave Search API key (optional).
        backend: "brave", "duckduckgo", or "auto".

    Returns:
        StructuredTool for use with Agent.
    """
    tool_instance = WebSearchTool(api_key=api_key, backend=backend)

    async def _search(query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Search the web for information.

        Use when you need:
        - Documentation for a library
        - How to solve a specific problem
        - Current best practices
        - Error message explanations

        Args:
            query: Search query. Be specific and include context.
            max_results: Maximum results to return (1-10).

        Returns:
            List of search results with title, url, snippet.
        """
        results = await tool_instance.search(query, max_results)
        return [r.to_dict() for r in results]

    return StructuredTool.from_function(
        coroutine=_search,
        name="web_search",
        description=(
            "Search the web for information. Use when you need documentation, "
            "best practices, or solutions to problems. Returns title, URL, "
            "and snippet for each result."
        ),
        args_schema=WebSearchInput,
    )


# Default tool instance (uses environment variable for API key)
web_search = create_web_search_tool()
