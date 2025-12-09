"""Unit tests for the RAG tool (create_retriever_tool).

Tests cover:
- Tool creation with various parameters
- Tool execution returning search results
- Async tool variant
- Score filtering and formatting
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import StructuredTool

from ai_infra.llm.tools.custom.retriever import (
    RetrieverToolInput,
    create_retriever_tool,
    create_retriever_tool_async,
)
from ai_infra.retriever.models import SearchResult

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_retriever() -> MagicMock:
    """Create a mock Retriever with sample search results."""
    retriever = MagicMock()

    # Default search results
    retriever.search.return_value = [
        SearchResult(text="First document content", score=0.95, source="doc1.txt"),
        SearchResult(text="Second document content", score=0.85, source="doc2.txt"),
        SearchResult(text="Third document content", score=0.75, source="doc3.txt"),
    ]

    # Async version
    retriever.asearch = AsyncMock(
        return_value=[
            SearchResult(text="Async first document", score=0.92, source="async1.txt"),
            SearchResult(text="Async second document", score=0.82, source="async2.txt"),
        ]
    )

    return retriever


@pytest.fixture
def empty_mock_retriever() -> MagicMock:
    """Create a mock Retriever that returns no results."""
    retriever = MagicMock()
    retriever.search.return_value = []
    retriever.asearch = AsyncMock(return_value=[])
    return retriever


# =============================================================================
# Tool Creation Tests
# =============================================================================


class TestCreateRetrieverTool:
    """Tests for create_retriever_tool function."""

    def test_creates_structured_tool(self, mock_retriever: MagicMock) -> None:
        """Test that create_retriever_tool returns a StructuredTool."""
        tool = create_retriever_tool(mock_retriever)
        assert isinstance(tool, StructuredTool)

    def test_default_name_and_description(self, mock_retriever: MagicMock) -> None:
        """Test default tool name and description."""
        tool = create_retriever_tool(mock_retriever)
        assert tool.name == "search_documents"
        assert tool.description == "Search documents for relevant information"

    def test_custom_name_and_description(self, mock_retriever: MagicMock) -> None:
        """Test custom tool name and description."""
        tool = create_retriever_tool(
            mock_retriever,
            name="search_policies",
            description="Search company policies and guidelines",
        )
        assert tool.name == "search_policies"
        assert tool.description == "Search company policies and guidelines"

    def test_has_correct_args_schema(self, mock_retriever: MagicMock) -> None:
        """Test that tool has the correct input schema."""
        tool = create_retriever_tool(mock_retriever)
        assert tool.args_schema == RetrieverToolInput

    def test_input_schema_has_query_field(self) -> None:
        """Test that RetrieverToolInput has a query field."""
        schema = RetrieverToolInput.model_json_schema()
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"


# =============================================================================
# Tool Execution Tests
# =============================================================================


class TestToolExecution:
    """Tests for tool execution."""

    def test_returns_search_results(self, mock_retriever: MagicMock) -> None:
        """Test that tool returns search results as formatted string."""
        tool = create_retriever_tool(mock_retriever)
        result = tool.invoke({"query": "test query"})

        assert isinstance(result, str)
        assert "First document content" in result
        assert "Second document content" in result
        assert "Third document content" in result

    def test_calls_retriever_search(self, mock_retriever: MagicMock) -> None:
        """Test that tool calls retriever.search with correct parameters."""
        tool = create_retriever_tool(mock_retriever, k=3, min_score=0.8)
        tool.invoke({"query": "test query"})

        mock_retriever.search.assert_called_once_with(
            "test query", k=3, min_score=0.8, filter=None, detailed=True
        )

    def test_returns_no_results_message(self, empty_mock_retriever: MagicMock) -> None:
        """Test that tool returns appropriate message when no results found."""
        tool = create_retriever_tool(empty_mock_retriever)
        result = tool.invoke({"query": "nonexistent topic"})

        assert result == "No relevant documents found."

    def test_formats_with_scores(self, mock_retriever: MagicMock) -> None:
        """Test that return_scores=True includes scores in output."""
        tool = create_retriever_tool(mock_retriever, return_scores=True)
        result = tool.invoke({"query": "test query"})

        assert "(score: 0.95)" in result
        assert "(score: 0.85)" in result
        assert "[from: doc1.txt]" in result

    def test_formats_without_scores(self, mock_retriever: MagicMock) -> None:
        """Test that return_scores=False excludes scores from output."""
        tool = create_retriever_tool(mock_retriever, return_scores=False)
        result = tool.invoke({"query": "test query"})

        assert "(score:" not in result
        assert "[from:" not in result
        # Results are separated by ---
        assert "---" in result

    def test_results_separated_by_divider(self, mock_retriever: MagicMock) -> None:
        """Test that results are properly separated."""
        tool = create_retriever_tool(mock_retriever, return_scores=False)
        result = tool.invoke({"query": "test query"})

        # Default format uses --- as separator
        parts = result.split("---")
        assert len(parts) == 3  # 3 results

    def test_results_numbered_with_scores(self, mock_retriever: MagicMock) -> None:
        """Test that results are numbered when showing scores."""
        tool = create_retriever_tool(mock_retriever, return_scores=True)
        result = tool.invoke({"query": "test query"})

        assert "1." in result
        assert "2." in result
        assert "3." in result


# =============================================================================
# Async Tool Tests
# =============================================================================


class TestCreateRetrieverToolAsync:
    """Tests for create_retriever_tool_async function."""

    def test_creates_structured_tool(self, mock_retriever: MagicMock) -> None:
        """Test that create_retriever_tool_async returns a StructuredTool."""
        tool = create_retriever_tool_async(mock_retriever)
        assert isinstance(tool, StructuredTool)

    def test_custom_name_and_description(self, mock_retriever: MagicMock) -> None:
        """Test custom tool name and description for async variant."""
        tool = create_retriever_tool_async(
            mock_retriever,
            name="async_search",
            description="Async document search",
        )
        assert tool.name == "async_search"
        assert tool.description == "Async document search"

    @pytest.mark.asyncio
    async def test_async_returns_search_results(self, mock_retriever: MagicMock) -> None:
        """Test that async tool returns search results."""
        tool = create_retriever_tool_async(mock_retriever)
        result = await tool.ainvoke({"query": "async test"})

        assert isinstance(result, str)
        assert "Async first document" in result
        assert "Async second document" in result

    @pytest.mark.asyncio
    async def test_async_calls_asearch(self, mock_retriever: MagicMock) -> None:
        """Test that async tool calls retriever.asearch."""
        tool = create_retriever_tool_async(mock_retriever, k=2, min_score=0.9)
        await tool.ainvoke({"query": "async test"})

        mock_retriever.asearch.assert_called_once_with(
            "async test", k=2, min_score=0.9, filter=None, detailed=True
        )

    @pytest.mark.asyncio
    async def test_async_returns_no_results_message(self, empty_mock_retriever: MagicMock) -> None:
        """Test that async tool returns appropriate message when no results."""
        tool = create_retriever_tool_async(empty_mock_retriever)
        result = await tool.ainvoke({"query": "nonexistent"})

        assert result == "No relevant documents found."

    @pytest.mark.asyncio
    async def test_async_formats_with_scores(self, mock_retriever: MagicMock) -> None:
        """Test that async tool includes scores when requested."""
        tool = create_retriever_tool_async(mock_retriever, return_scores=True)
        result = await tool.ainvoke({"query": "test"})

        assert "(score: 0.92)" in result
        assert "[from: async1.txt]" in result


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_handles_none_score(self) -> None:
        """Test handling of results with None score."""
        retriever = MagicMock()
        retriever.search.return_value = [
            SearchResult(text="No score content", score=None, source="noscore.txt"),
        ]

        tool = create_retriever_tool(retriever, return_scores=True)
        result = tool.invoke({"query": "test"})

        # Should not show score when it's None
        assert "No score content" in result
        assert "(score:" not in result

    def test_handles_none_source(self) -> None:
        """Test handling of results with None source."""
        retriever = MagicMock()
        retriever.search.return_value = [
            SearchResult(text="No source content", score=0.9, source=None),
        ]

        tool = create_retriever_tool(retriever, return_scores=True)
        result = tool.invoke({"query": "test"})

        # Should not show source when it's None
        assert "No source content" in result
        assert "[from:" not in result

    def test_handles_empty_source(self) -> None:
        """Test handling of results with empty source string."""
        retriever = MagicMock()
        retriever.search.return_value = [
            SearchResult(text="Empty source content", score=0.9, source=""),
        ]

        tool = create_retriever_tool(retriever, return_scores=True)
        result = tool.invoke({"query": "test"})

        # Should not show source when it's empty
        assert "Empty source content" in result
        assert "[from:" not in result

    def test_single_result(self) -> None:
        """Test handling of single search result."""
        retriever = MagicMock()
        retriever.search.return_value = [
            SearchResult(text="Only result", score=0.99, source="only.txt"),
        ]

        tool = create_retriever_tool(retriever, return_scores=False)
        result = tool.invoke({"query": "test"})

        assert result == "Only result"
        assert "---" not in result  # No separator for single result

    def test_k_parameter_passed_to_retriever(self, mock_retriever: MagicMock) -> None:
        """Test that k parameter is correctly passed to retriever."""
        tool = create_retriever_tool(mock_retriever, k=10)
        tool.invoke({"query": "test"})

        mock_retriever.search.assert_called_with(
            "test", k=10, min_score=None, filter=None, detailed=True
        )

    def test_min_score_parameter_passed_to_retriever(self, mock_retriever: MagicMock) -> None:
        """Test that min_score parameter is correctly passed to retriever."""
        tool = create_retriever_tool(mock_retriever, min_score=0.75)
        tool.invoke({"query": "test"})

        mock_retriever.search.assert_called_with(
            "test", k=5, min_score=0.75, filter=None, detailed=True
        )

    def test_filter_parameter_passed_to_retriever(self, mock_retriever: MagicMock) -> None:
        """Test that filter parameter is correctly passed to retriever."""
        filter_dict = {"type": "docs", "package": "svc-infra"}
        tool = create_retriever_tool(mock_retriever, filter=filter_dict)
        tool.invoke({"query": "test"})

        mock_retriever.search.assert_called_with(
            "test", k=5, min_score=None, filter=filter_dict, detailed=True
        )


# =============================================================================
# Formatting Options Tests
# =============================================================================


class TestFormattingOptions:
    """Tests for new formatting options (max_chars, format)."""

    def test_max_chars_truncates_output(self, mock_retriever: MagicMock) -> None:
        """Test that max_chars truncates long output."""
        tool = create_retriever_tool(mock_retriever, max_chars=30)
        result = tool.invoke({"query": "test"})

        assert len(result) <= 30
        assert result.endswith("...")

    def test_max_chars_none_no_truncation(self, mock_retriever: MagicMock) -> None:
        """Test that max_chars=None doesn't truncate."""
        tool = create_retriever_tool(mock_retriever, max_chars=None)
        result = tool.invoke({"query": "test"})

        # All content should be present
        assert "First document content" in result
        assert "Second document content" in result
        assert "Third document content" in result

    def test_format_text_default(self, mock_retriever: MagicMock) -> None:
        """Test default text format."""
        tool = create_retriever_tool(mock_retriever, format="text")
        result = tool.invoke({"query": "test"})

        # Text format uses --- separator
        assert "---" in result
        assert "First document content" in result

    def test_format_markdown(self, mock_retriever: MagicMock) -> None:
        """Test markdown format."""
        tool = create_retriever_tool(mock_retriever, format="markdown", return_scores=True)
        result = tool.invoke({"query": "test"})

        # Markdown format uses headers
        assert "### Result 1" in result
        assert "**Score:**" in result
        assert "**Source:**" in result
        assert "---" in result  # Separator between results

    def test_format_json(self, mock_retriever: MagicMock) -> None:
        """Test JSON format."""
        import json

        tool = create_retriever_tool(mock_retriever, format="json", return_scores=True)
        result = tool.invoke({"query": "test"})

        # Should be valid JSON
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["text"] == "First document content"
        assert data[0]["score"] == pytest.approx(0.95, rel=1e-3)
        assert data[0]["source"] == "doc1.txt"

    def test_format_json_without_scores(self, mock_retriever: MagicMock) -> None:
        """Test JSON format without scores."""
        import json

        tool = create_retriever_tool(mock_retriever, format="json", return_scores=False)
        result = tool.invoke({"query": "test"})

        data = json.loads(result)
        assert "text" in data[0]
        assert "score" not in data[0]
        assert "source" in data[0]

    def test_format_markdown_without_scores(self, mock_retriever: MagicMock) -> None:
        """Test markdown format without scores."""
        tool = create_retriever_tool(mock_retriever, format="markdown", return_scores=False)
        result = tool.invoke({"query": "test"})

        assert "### Result 1" in result
        assert "**Score:**" not in result
        assert "**Source:**" in result

    @pytest.mark.asyncio
    async def test_async_format_json(self, mock_retriever: MagicMock) -> None:
        """Test JSON format works with async tool."""
        import json

        tool = create_retriever_tool_async(mock_retriever, format="json", return_scores=True)
        result = await tool.ainvoke({"query": "test"})

        data = json.loads(result)
        assert isinstance(data, list)
        assert data[0]["text"] == "Async first document"

    @pytest.mark.asyncio
    async def test_async_max_chars(self, mock_retriever: MagicMock) -> None:
        """Test max_chars works with async tool."""
        tool = create_retriever_tool_async(mock_retriever, max_chars=25)
        result = await tool.ainvoke({"query": "test"})

        assert len(result) <= 25
        assert result.endswith("...")
