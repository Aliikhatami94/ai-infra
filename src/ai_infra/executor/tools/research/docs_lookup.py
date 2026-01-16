"""Documentation lookup tool for research capability.

Phase 3.4.2 of EXECUTOR_1.md: Provides documentation lookup
for packages and libraries.

Uses multiple sources:
- PyPI package metadata and readthedocs
- npm package documentation
- Built-in documentation databases

Example:
    ```python
    from ai_infra.executor.tools.research import lookup_docs

    # Look up FastAPI docs
    result = await lookup_docs.ainvoke({
        "package": "fastapi",
        "topic": "authentication",
    })

    print(result["content"])
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_infra.logging import get_logger

__all__ = [
    "DocsLookupResult",
    "DocsLookupTool",
    "create_docs_lookup_tool",
    "lookup_docs",
]

logger = get_logger("executor.tools.docs_lookup")


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class DocsLookupResult:
    """Result from documentation lookup."""

    package: str
    topic: str | None
    content: str
    source: str
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "package": self.package,
            "topic": self.topic,
            "content": self.content,
            "source": self.source,
            "url": self.url,
        }


class DocsLookupInput(BaseModel):
    """Input schema for documentation lookup."""

    package: str = Field(description="Package name (e.g., 'fastapi', 'react', 'tokio')")
    topic: str | None = Field(
        default=None,
        description="Specific topic to search for (e.g., 'authentication', 'hooks')",
    )


# =============================================================================
# Documentation Sources
# =============================================================================

# Well-known documentation URLs for popular packages
_KNOWN_DOCS: dict[str, str] = {
    # Python
    "fastapi": "https://fastapi.tiangolo.com",
    "pydantic": "https://docs.pydantic.dev",
    "sqlalchemy": "https://docs.sqlalchemy.org",
    "pytest": "https://docs.pytest.org",
    "django": "https://docs.djangoproject.com",
    "flask": "https://flask.palletsprojects.com",
    "requests": "https://requests.readthedocs.io",
    "httpx": "https://www.python-httpx.org",
    "aiohttp": "https://docs.aiohttp.org",
    "numpy": "https://numpy.org/doc",
    "pandas": "https://pandas.pydata.org/docs",
    "langchain": "https://python.langchain.com/docs",
    "langgraph": "https://langchain-ai.github.io/langgraph",
    # JavaScript/TypeScript
    "react": "https://react.dev",
    "nextjs": "https://nextjs.org/docs",
    "vue": "https://vuejs.org/guide",
    "express": "https://expressjs.com",
    "typescript": "https://www.typescriptlang.org/docs",
    # Rust
    "tokio": "https://docs.rs/tokio",
    "serde": "https://serde.rs",
    "actix-web": "https://actix.rs/docs",
}


async def _fetch_pypi_info(package: str) -> dict[str, Any] | None:
    """Fetch package info from PyPI.

    Args:
        package: Package name.

    Returns:
        Package info dict or None if not found.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed, skipping PyPI lookup")
        return None

    url = f"https://pypi.org/pypi/{package}/json"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.debug(f"PyPI lookup failed for {package}: {e}")

    return None


async def _fetch_npm_info(package: str) -> dict[str, Any] | None:
    """Fetch package info from npm.

    Args:
        package: Package name.

    Returns:
        Package info dict or None if not found.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed, skipping npm lookup")
        return None

    url = f"https://registry.npmjs.org/{package}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.debug(f"npm lookup failed for {package}: {e}")

    return None


async def _search_readthedocs(
    package: str,
    topic: str | None,
) -> str | None:
    """Search ReadTheDocs for package documentation.

    Args:
        package: Package name.
        topic: Optional topic to search.

    Returns:
        Documentation content or None.
    """
    try:
        import httpx
    except ImportError:
        return None

    # Try common readthedocs patterns
    patterns = [
        f"https://{package}.readthedocs.io/en/latest/",
        f"https://{package.replace('-', '_')}.readthedocs.io/en/latest/",
        f"https://{package.replace('_', '-')}.readthedocs.io/en/latest/",
    ]

    async with httpx.AsyncClient() as client:
        for base_url in patterns:
            try:
                url = base_url
                if topic:
                    url = f"{base_url}search.html?q={topic}"

                response = await client.get(url, timeout=10.0, follow_redirects=True)
                if response.status_code == 200:
                    return f"Documentation available at: {base_url}"
            except Exception:
                continue

    return None


# =============================================================================
# Documentation Lookup Tool
# =============================================================================


class DocsLookupTool:
    """Documentation lookup tool.

    Searches multiple sources for package documentation.

    Example:
        ```python
        tool = DocsLookupTool()
        result = await tool.lookup("fastapi", topic="authentication")
        print(result.content)
        ```
    """

    def __init__(self) -> None:
        """Initialize the documentation lookup tool."""
        pass

    async def lookup(
        self,
        package: str,
        topic: str | None = None,
    ) -> DocsLookupResult:
        """Look up documentation for a package.

        Args:
            package: Package name.
            topic: Optional specific topic.

        Returns:
            DocsLookupResult with documentation content.
        """
        logger.debug(f"Looking up docs: package={package}, topic={topic}")

        package_lower = package.lower()
        content_parts: list[str] = []
        source = "unknown"
        url = None

        # Check known docs
        if package_lower in _KNOWN_DOCS:
            url = _KNOWN_DOCS[package_lower]
            content_parts.append(f"Official documentation: {url}")
            if topic:
                content_parts.append(f"Search for '{topic}' in the documentation.")
            source = "known_docs"

        # Try PyPI
        pypi_info = await _fetch_pypi_info(package)
        if pypi_info:
            info = pypi_info.get("info", {})
            source = "pypi"

            # Add package summary
            if info.get("summary"):
                content_parts.append(f"\n## Summary\n{info['summary']}")

            # Add description (truncated)
            if info.get("description"):
                desc = info["description"]
                if len(desc) > 2000:
                    desc = desc[:2000] + "...\n[truncated]"
                content_parts.append(f"\n## Description\n{desc}")

            # Add project URLs
            urls = info.get("project_urls", {})
            if urls:
                url_lines = ["\n## Links"]
                for name, link in urls.items():
                    url_lines.append(f"- {name}: {link}")
                content_parts.append("\n".join(url_lines))

                # Set primary URL
                if not url:
                    url = urls.get("Documentation") or urls.get("Homepage")

        # Try npm if no PyPI info
        if not pypi_info:
            npm_info = await _fetch_npm_info(package)
            if npm_info:
                source = "npm"

                # Add description
                if npm_info.get("description"):
                    content_parts.append(f"\n## Summary\n{npm_info['description']}")

                # Add readme (truncated)
                if npm_info.get("readme"):
                    readme = npm_info["readme"]
                    if len(readme) > 2000:
                        readme = readme[:2000] + "...\n[truncated]"
                    content_parts.append(f"\n## README\n{readme}")

                # Set URL
                if not url and npm_info.get("homepage"):
                    url = npm_info["homepage"]

        # Try ReadTheDocs
        rtd_result = await _search_readthedocs(package, topic)
        if rtd_result:
            content_parts.append(f"\n## ReadTheDocs\n{rtd_result}")
            if source == "unknown":
                source = "readthedocs"

        # Build final content
        if not content_parts:
            content = (
                f"No documentation found for '{package}'. "
                "Try searching the web or checking the package repository."
            )
        else:
            content = "\n".join(content_parts)

        return DocsLookupResult(
            package=package,
            topic=topic,
            content=content,
            source=source,
            url=url,
        )


# =============================================================================
# LangChain Tool Creation
# =============================================================================


def create_docs_lookup_tool() -> StructuredTool:
    """Create a LangChain-compatible documentation lookup tool.

    Returns:
        StructuredTool for use with Agent.
    """
    tool_instance = DocsLookupTool()

    async def _lookup(
        package: str,
        topic: str | None = None,
    ) -> dict[str, Any]:
        """Look up documentation for a package.

        Use when you need:
        - Package installation instructions
        - API reference for a library
        - Usage examples
        - Configuration options

        Args:
            package: Package name (e.g., "fastapi", "react", "tokio")
            topic: Specific topic to search for (e.g., "authentication")

        Returns:
            Documentation content with source and URL.
        """
        result = await tool_instance.lookup(package, topic)
        return result.to_dict()

    return StructuredTool.from_function(
        coroutine=_lookup,
        name="lookup_docs",
        description=(
            "Look up documentation for a package or library. "
            "Returns package description, README content, and documentation links. "
            "Supports Python (PyPI), JavaScript (npm), and common packages."
        ),
        args_schema=DocsLookupInput,
    )


# Default tool instance
lookup_docs = create_docs_lookup_tool()
