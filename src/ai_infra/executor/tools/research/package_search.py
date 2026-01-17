"""Package registry search tool for research capability.

Phase 3.4.3 of EXECUTOR_1.md: Provides package registry search
for discovering packages across PyPI, npm, and crates.io.

Example:
    ```python
    from ai_infra.executor.tools.research import search_packages

    # Search PyPI
    results = await search_packages.ainvoke({
        "query": "async http client",
        "registry": "pypi",
    })

    for pkg in results:
        print(f"{pkg['name']} v{pkg['version']}: {pkg['description']}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_infra.logging import get_logger

__all__ = [
    "PackageInfo",
    "PackageSearchTool",
    "create_package_search_tool",
    "search_packages",
]

logger = get_logger("executor.tools.package_search")


# Registry type
RegistryType = Literal["pypi", "npm", "crates"]


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class PackageInfo:
    """Information about a package."""

    name: str
    version: str
    description: str
    registry: str
    url: str | None = None
    downloads: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "registry": self.registry,
            "url": self.url,
            "downloads": self.downloads,
        }


class PackageSearchInput(BaseModel):
    """Input schema for package search."""

    query: str = Field(description="Search query for packages")
    registry: RegistryType = Field(
        default="pypi",
        description="Package registry to search: pypi, npm, or crates",
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return (1-20)",
        ge=1,
        le=20,
    )


# =============================================================================
# Registry Search Functions
# =============================================================================


async def _search_pypi(
    query: str,
    max_results: int,
) -> list[PackageInfo]:
    """Search PyPI for packages.

    Args:
        query: Search query.
        max_results: Maximum results.

    Returns:
        List of PackageInfo.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for package search. Install with: pip install httpx")

    # PyPI doesn't have a great search API, so we use the simple API
    # and supplement with pypi.org/search endpoint
    url = "https://pypi.org/search/"
    params = {"q": query}

    results: list[PackageInfo] = []

    try:
        async with httpx.AsyncClient() as client:
            # Try the warehouse search endpoint (returns HTML, so we parse minimally)
            # For a production implementation, consider using a proper PyPI search API
            response = await client.get(
                url,
                params=params,
                timeout=15.0,
                follow_redirects=True,
            )

            if response.status_code != 200:
                logger.warning(f"PyPI search returned status {response.status_code}")
                return results

            # Parse HTML to extract package names (basic parsing)
            html = response.text
            import re

            # Find package links in the search results
            pattern = r'href="/project/([^/]+)/"'
            matches = re.findall(pattern, html)

            # Get unique package names
            seen = set()
            package_names = []
            for name in matches:
                if name not in seen:
                    seen.add(name)
                    package_names.append(name)
                    if len(package_names) >= max_results:
                        break

            # Fetch details for each package
            for name in package_names[:max_results]:
                pkg_url = f"https://pypi.org/pypi/{name}/json"
                try:
                    pkg_response = await client.get(pkg_url, timeout=5.0)
                    if pkg_response.status_code == 200:
                        data = pkg_response.json()
                        info = data.get("info", {})
                        results.append(
                            PackageInfo(
                                name=info.get("name", name),
                                version=info.get("version", "unknown"),
                                description=info.get("summary", "")[:200],
                                registry="pypi",
                                url=f"https://pypi.org/project/{name}/",
                            )
                        )
                except Exception as e:
                    logger.debug(f"Failed to fetch {name}: {e}")

    except Exception as e:
        logger.warning(f"PyPI search failed: {e}")

    return results


async def _search_npm(
    query: str,
    max_results: int,
) -> list[PackageInfo]:
    """Search npm for packages.

    Args:
        query: Search query.
        max_results: Maximum results.

    Returns:
        List of PackageInfo.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for package search. Install with: pip install httpx")

    url = "https://registry.npmjs.org/-/v1/search"
    params = {
        "text": query,
        "size": max_results,
    }

    results: list[PackageInfo] = []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=15.0)

            if response.status_code != 200:
                logger.warning(f"npm search returned status {response.status_code}")
                return results

            data = response.json()

            for obj in data.get("objects", []):
                pkg = obj.get("package", {})
                results.append(
                    PackageInfo(
                        name=pkg.get("name", ""),
                        version=pkg.get("version", "unknown"),
                        description=pkg.get("description", "")[:200],
                        registry="npm",
                        url=f"https://www.npmjs.com/package/{pkg.get('name', '')}",
                    )
                )

    except Exception as e:
        logger.warning(f"npm search failed: {e}")

    return results[:max_results]


async def _search_crates(
    query: str,
    max_results: int,
) -> list[PackageInfo]:
    """Search crates.io for Rust packages.

    Args:
        query: Search query.
        max_results: Maximum results.

    Returns:
        List of PackageInfo.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for package search. Install with: pip install httpx")

    url = "https://crates.io/api/v1/crates"
    params = {
        "q": query,
        "per_page": max_results,
    }
    headers = {
        "User-Agent": "ai-infra-executor/1.0",
    }

    results: list[PackageInfo] = []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers, timeout=15.0)

            if response.status_code != 200:
                logger.warning(f"crates.io search returned status {response.status_code}")
                return results

            data = response.json()

            for crate in data.get("crates", []):
                results.append(
                    PackageInfo(
                        name=crate.get("name", ""),
                        version=crate.get("max_version", "unknown"),
                        description=crate.get("description", "")[:200],
                        registry="crates",
                        url=f"https://crates.io/crates/{crate.get('name', '')}",
                        downloads=crate.get("downloads"),
                    )
                )

    except Exception as e:
        logger.warning(f"crates.io search failed: {e}")

    return results[:max_results]


# =============================================================================
# Package Search Tool
# =============================================================================


class PackageSearchTool:
    """Package registry search tool.

    Searches PyPI, npm, and crates.io for packages.

    Example:
        ```python
        tool = PackageSearchTool()
        results = await tool.search("http client", registry="pypi")
        ```
    """

    def __init__(self) -> None:
        """Initialize the package search tool."""
        pass

    async def search(
        self,
        query: str,
        registry: RegistryType = "pypi",
        max_results: int = 10,
    ) -> list[PackageInfo]:
        """Search for packages.

        Args:
            query: Search query.
            registry: Which registry to search.
            max_results: Maximum results.

        Returns:
            List of PackageInfo objects.
        """
        logger.debug(f"Package search: query='{query}', registry={registry}")

        if registry == "pypi":
            return await _search_pypi(query, max_results)
        elif registry == "npm":
            return await _search_npm(query, max_results)
        elif registry == "crates":
            return await _search_crates(query, max_results)
        else:
            raise ValueError(f"Unknown registry: {registry}")


# =============================================================================
# LangChain Tool Creation
# =============================================================================


def create_package_search_tool() -> StructuredTool:
    """Create a LangChain-compatible package search tool.

    Returns:
        StructuredTool for use with Agent.
    """
    tool_instance = PackageSearchTool()

    async def _search(
        query: str,
        registry: RegistryType = "pypi",
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search package registries for libraries.

        Use when you need:
        - Find a library for a specific purpose
        - Compare packages with similar functionality
        - Check if a package exists
        - Find the latest version of a package

        Args:
            query: Search query (e.g., "async http client")
            registry: Registry to search - "pypi" (Python), "npm" (JavaScript), "crates" (Rust)
            max_results: Maximum results to return (1-20)

        Returns:
            List of packages with name, version, description, and URL.
        """
        results = await tool_instance.search(query, registry, max_results)
        return [r.to_dict() for r in results]

    return StructuredTool.from_function(
        coroutine=_search,
        name="search_packages",
        description=(
            "Search package registries (PyPI, npm, crates.io) for libraries. "
            "Returns package name, version, description, and registry URL. "
            "Use to find libraries for specific purposes."
        ),
        args_schema=PackageSearchInput,
    )


# Default tool instance
search_packages = create_package_search_tool()
