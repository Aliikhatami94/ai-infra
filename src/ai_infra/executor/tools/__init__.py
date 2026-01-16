"""Research tools for the Executor.

Phase 3.4 of EXECUTOR_1.md: Provides research capabilities for agents
to search for information when stuck.

Research tools include:
- web_search: Search the web for information
- lookup_docs: Look up package documentation
- search_packages: Search package registries (PyPI, npm, crates)

Example:
    ```python
    from ai_infra.executor.tools.research import (
        web_search,
        lookup_docs,
        search_packages,
    )

    # Use with Agent
    agent = Agent(tools=[web_search, lookup_docs, search_packages])
    ```
"""

from ai_infra.executor.tools.research.docs_lookup import (
    DocsLookupResult,
    DocsLookupTool,
    create_docs_lookup_tool,
    lookup_docs,
)
from ai_infra.executor.tools.research.package_search import (
    PackageInfo,
    PackageSearchTool,
    create_package_search_tool,
    search_packages,
)
from ai_infra.executor.tools.research.web_search import (
    WebSearchResult,
    WebSearchTool,
    create_web_search_tool,
    web_search,
)

__all__ = [
    # Web search
    "WebSearchResult",
    "WebSearchTool",
    "create_web_search_tool",
    "web_search",
    # Docs lookup
    "DocsLookupResult",
    "DocsLookupTool",
    "create_docs_lookup_tool",
    "lookup_docs",
    # Package search
    "PackageInfo",
    "PackageSearchTool",
    "create_package_search_tool",
    "search_packages",
]
