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

from typing import TYPE_CHECKING, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ai_infra.retriever.retriever import Retriever


class RetrieverToolInput(BaseModel):
    """Input schema for retriever tool."""

    query: str = Field(description="The search query to find relevant documents")


def create_retriever_tool(
    retriever: "Retriever",
    name: str = "search_documents",
    description: str = "Search documents for relevant information",
    k: int = 5,
    min_score: Optional[float] = None,
    return_scores: bool = False,
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
        return_scores: If True, include similarity scores in the output.
            Default: False (just return the text).

    Returns:
        A StructuredTool that can be passed to Agent(tools=[...]).

    Example:
        ```python
        from ai_infra import Retriever, create_retriever_tool

        retriever = Retriever()
        retriever.add("./docs/")

        tool = create_retriever_tool(
            retriever=retriever,
            name="search_docs",
            description="Search documentation for answers",
            k=3,
            min_score=0.7,
        )
        ```
    """

    def search_documents(query: str) -> str:
        """Search the retriever for relevant documents."""
        results = retriever.search(query, k=k, min_score=min_score, detailed=True)

        if not results:
            return "No relevant documents found."

        if return_scores:
            # Format with scores
            formatted = []
            for i, r in enumerate(results, 1):
                score_str = f" (score: {r.score:.2f})" if r.score is not None else ""
                source_str = f" [from: {r.source}]" if r.source else ""
                formatted.append(f"{i}. {r.text}{score_str}{source_str}")
            return "\n\n".join(formatted)
        else:
            # Just return the text chunks
            return "\n\n---\n\n".join(r.text for r in results)

    return StructuredTool.from_function(
        func=search_documents,
        name=name,
        description=description,
        args_schema=RetrieverToolInput,
    )


def create_retriever_tool_async(
    retriever: "Retriever",
    name: str = "search_documents",
    description: str = "Search documents for relevant information",
    k: int = 5,
    min_score: Optional[float] = None,
    return_scores: bool = False,
) -> StructuredTool:
    """Create an async Agent-compatible tool from a Retriever instance.

    Same as create_retriever_tool but uses async search for better performance
    in async contexts.

    Args:
        retriever: The Retriever instance to wrap.
        name: Tool name (used by the agent to identify the tool).
        description: Tool description (helps the agent understand when to use it).
        k: Number of results to return (default: 5).
        min_score: Minimum similarity score threshold (0-1).
        return_scores: If True, include similarity scores in the output.

    Returns:
        A StructuredTool with async execution.
    """

    async def search_documents_async(query: str) -> str:
        """Search the retriever for relevant documents (async)."""
        results = await retriever.asearch(query, k=k, min_score=min_score, detailed=True)

        if not results:
            return "No relevant documents found."

        if return_scores:
            formatted = []
            for i, r in enumerate(results, 1):
                score_str = f" (score: {r.score:.2f})" if r.score is not None else ""
                source_str = f" [from: {r.source}]" if r.source else ""
                formatted.append(f"{i}. {r.text}{score_str}{source_str}")
            return "\n\n".join(formatted)
        else:
            return "\n\n---\n\n".join(r.text for r in results)

    return StructuredTool.from_function(
        coroutine=search_documents_async,
        name=name,
        description=description,
        args_schema=RetrieverToolInput,
    )
