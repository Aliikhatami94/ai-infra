"""Create Agent-compatible tools from Retriever instances.

This module provides utilities to wrap a Retriever as a tool that can be
used with ai-infra's Agent class for autonomous document search.

Example:
    ```python
    from ai_infra import Agent, Retriever, create_retriever_tool

    # Create retriever with documents
    retriever = Retriever(backend="sqlite", path="./docs.db")
    retriever.add("./company_docs/")

    # Create tool from retriever
    search_docs = create_retriever_tool(
        retriever=retriever,
        name="search_company_docs",
        description="Search company documentation for policies and procedures",
    )

    # Add to agent
    agent = Agent(tools=[search_docs])

    # Agent autonomously uses retrieval when needed
    result = agent.run("What's our refund policy?")
    ```
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ai_infra.retriever.retriever import Retriever


class RetrieverToolInput(BaseModel):
    """Input schema for retriever tool."""

    query: str = Field(description="The search query to find relevant documents")


# Output format type
OutputFormat = Literal["text", "markdown", "json"]


def _format_results(
    results: list,
    return_scores: bool,
    format: OutputFormat,
    max_chars: int | None,
) -> str:
    """Format retriever results based on specified options.

    Args:
        results: List of SearchResult objects.
        return_scores: Whether to include similarity scores.
        format: Output format ("text", "markdown", "json").
        max_chars: Maximum characters for output (None = no limit).

    Returns:
        Formatted string representation of results.
    """
    if not results:
        return "No relevant documents found."

    if format == "json":
        # JSON format
        output = []
        for r in results:
            item = {"text": r.text}
            if return_scores and r.score is not None:
                item["score"] = round(r.score, 4)
            if r.source:
                item["source"] = r.source
            output.append(item)
        formatted = json.dumps(output, indent=2)

    elif format == "markdown":
        # Markdown format with headers
        formatted_items = []
        for i, r in enumerate(results, 1):
            parts = [f"### Result {i}"]
            if return_scores and r.score is not None:
                parts.append(f"**Score:** {r.score:.2f}")
            if r.source:
                parts.append(f"**Source:** {r.source}")
            parts.append("")
            parts.append(r.text)
            formatted_items.append("\n".join(parts))
        formatted = "\n\n---\n\n".join(formatted_items)

    else:
        # Default text format
        if return_scores:
            formatted_items = []
            for i, r in enumerate(results, 1):
                score_str = f" (score: {r.score:.2f})" if r.score is not None else ""
                source_str = f" [from: {r.source}]" if r.source else ""
                formatted_items.append(f"{i}. {r.text}{score_str}{source_str}")
            formatted = "\n\n".join(formatted_items)
        else:
            formatted = "\n\n---\n\n".join(r.text for r in results)

    # Apply max_chars truncation if specified
    if max_chars and len(formatted) > max_chars:
        formatted = formatted[: max_chars - 3] + "..."

    return formatted


def create_retriever_tool(
    retriever: Retriever,
    name: str = "search_documents",
    description: str = "Search documents for relevant information",
    k: int = 5,
    min_score: float | None = None,
    filter: dict[str, Any] | None = None,
    return_scores: bool = False,
    max_chars: int | None = None,
    format: OutputFormat = "text",
    structured: bool = False,
) -> StructuredTool:
    """Create an Agent-compatible tool from a Retriever instance.

    This wraps a Retriever's search functionality as a LangChain StructuredTool
    that can be used with ai-infra's Agent class.

    Args:
        retriever: The Retriever instance to wrap.
        name: Tool name (used by the agent to identify the tool).
        description: Tool description (helps the agent understand when to use it).
        k: Number of results to return (default: 5).
        min_score: Minimum similarity score threshold (0-1). Results below this
            score are filtered out. Default: None (no filtering).
        filter: Metadata filter dict for search. Used to filter results by
            metadata fields (e.g., {"type": "docs", "package": "svc-infra"}).
            Default: None (no metadata filtering).
        return_scores: If True, include similarity scores in the output.
            Default: False (just return the text).
        max_chars: Maximum characters in the output. Longer results are truncated
            with "...". Default: None (no limit).
        format: Output format. Options:
            - "text" (default): Plain text with results separated by ---
            - "markdown": Formatted with headers and metadata
            - "json": JSON array with text, score, and source fields
        structured: If True, returns a dictionary with structured results
            that can be parsed by frontends. The dictionary contains:
            - "results": List of SearchResult.to_dict() objects
            - "query": The original search query
            - "count": Number of results returned
            If False (default), returns formatted text for LLM consumption.

    Returns:
        A StructuredTool that can be passed to Agent(tools=[...]).

    Example:
        ```python
        from ai_infra import Retriever, create_retriever_tool

        retriever = Retriever()
        retriever.add("./docs/")

        # Basic tool (text output for LLM)
        tool = create_retriever_tool(
            retriever=retriever,
            name="search_docs",
            description="Search documentation for answers",
            k=3,
            min_score=0.7,
        )

        # Structured output for API/frontend with filtering
        tool = create_retriever_tool(
            retriever=retriever,
            name="search_docs",
            description="Search documentation",
            structured=True,  # Returns dict with results, query, count
            filter={"type": "docs"},  # Only search docs, not examples
        )
        ```
    """

    def search_documents(query: str) -> str | dict[str, Any]:
        """Search the retriever for relevant documents."""
        results = retriever.search(query, k=k, min_score=min_score, filter=filter, detailed=True)

        if structured:
            return {
                "results": [r.to_dict() for r in results],
                "query": query,
                "count": len(results),
            }
        else:
            return _format_results(results, return_scores, format, max_chars)

    return StructuredTool.from_function(
        func=search_documents,
        name=name,
        description=description,
        args_schema=RetrieverToolInput,
    )


def create_retriever_tool_async(
    retriever: Retriever,
    name: str = "search_documents",
    description: str = "Search documents for relevant information",
    k: int = 5,
    min_score: float | None = None,
    filter: dict[str, Any] | None = None,
    return_scores: bool = False,
    max_chars: int | None = None,
    format: OutputFormat = "text",
    structured: bool = False,
) -> StructuredTool:
    """Create an async Agent-compatible tool from a Retriever instance.

    Same as create_retriever_tool but uses async search for better performance
    in async contexts (e.g., FastAPI endpoints).

    Args:
        retriever: The Retriever instance to wrap.
        name: Tool name (used by the agent to identify the tool).
        description: Tool description (helps the agent understand when to use it).
        k: Number of results to return (default: 5).
        min_score: Minimum similarity score threshold (0-1).
        filter: Metadata filter dict for search. Used to filter results by
            metadata fields (e.g., {"type": "docs", "package": "svc-infra"}).
            Default: None (no metadata filtering).
        return_scores: If True, include similarity scores in the output.
        max_chars: Maximum characters in the output. Longer results are truncated.
        format: Output format ("text", "markdown", "json").
        structured: If True, returns a dictionary with structured results
            that can be parsed by frontends. The dictionary contains:
            - "results": List of SearchResult.to_dict() objects
            - "query": The original search query
            - "count": Number of results returned
            If False (default), returns formatted text for LLM consumption.

    Returns:
        A StructuredTool with async execution.

    Example:
        ```python
        # Structured output for API endpoints with filtering
        tool = create_retriever_tool_async(
            retriever=retriever,
            name="search_docs",
            description="Search documentation",
            structured=True,
            filter={"type": "docs"},  # Only search docs, not examples
        )
        ```
    """

    async def search_documents_async(query: str) -> str | dict[str, Any]:
        """Search the retriever for relevant documents (async)."""
        results = await retriever.asearch(
            query, k=k, min_score=min_score, filter=filter, detailed=True
        )

        if structured:
            from ai_infra.retriever.models import SearchResult

            detailed_results = cast("list[SearchResult]", results)
            return {
                "results": [r.to_dict() for r in detailed_results],
                "query": query,
                "count": len(results),
            }
        else:
            return _format_results(results, return_scores, format, max_chars)

    return StructuredTool.from_function(
        coroutine=search_documents_async,
        name=name,
        description=description,
        args_schema=RetrieverToolInput,
    )
