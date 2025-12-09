"""Unit tests for Retriever Phase 6.9 enhancements.

Tests cover:
- Environment auto-configuration (6.9.1)
- Remote content loading (6.9.2)
- SearchResult enhancements (6.9.3)
- Structured tool results (6.9.4)
- StreamEvent structured result support (6.9.5)
- Module exports (6.9.6)
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.retriever.models import Chunk, SearchResult
from ai_infra.retriever.retriever import (
    KNOWN_EMBEDDING_DIMENSIONS,
    Retriever,
    _get_embedding_dimension,
)

# =============================================================================
# 6.9.1 Environment Auto-Configuration Tests
# =============================================================================


class TestKnownEmbeddingDimensions:
    """Tests for the KNOWN_EMBEDDING_DIMENSIONS constant."""

    def test_contains_openai_models(self) -> None:
        """Test OpenAI models are in the dimension map."""
        assert "text-embedding-3-small" in KNOWN_EMBEDDING_DIMENSIONS
        assert "text-embedding-3-large" in KNOWN_EMBEDDING_DIMENSIONS
        assert "text-embedding-ada-002" in KNOWN_EMBEDDING_DIMENSIONS

    def test_contains_huggingface_models(self) -> None:
        """Test HuggingFace models are in the dimension map."""
        # HuggingFace models use full org/model format
        assert "sentence-transformers/all-MiniLM-L6-v2" in KNOWN_EMBEDDING_DIMENSIONS
        assert "sentence-transformers/all-mpnet-base-v2" in KNOWN_EMBEDDING_DIMENSIONS
        assert "BAAI/bge-small-en-v1.5" in KNOWN_EMBEDDING_DIMENSIONS

    def test_contains_bge_models(self) -> None:
        """Test BGE models are in the dimension map."""
        assert "BAAI/bge-small-en-v1.5" in KNOWN_EMBEDDING_DIMENSIONS
        assert "BAAI/bge-base-en-v1.5" in KNOWN_EMBEDDING_DIMENSIONS
        assert "BAAI/bge-large-en-v1.5" in KNOWN_EMBEDDING_DIMENSIONS

    def test_dimensions_are_positive_integers(self) -> None:
        """Test all dimensions are positive integers."""
        for model, dim in KNOWN_EMBEDDING_DIMENSIONS.items():
            assert isinstance(dim, int), f"{model} dimension is not int"
            assert dim > 0, f"{model} dimension is not positive"


class TestGetEmbeddingDimension:
    """Tests for _get_embedding_dimension helper."""

    def test_returns_known_dimension_for_openai(self) -> None:
        """Test returns dimension for known OpenAI models."""
        assert _get_embedding_dimension("openai", "text-embedding-3-small") == 1536
        assert _get_embedding_dimension("openai", "text-embedding-3-large") == 3072

    def test_returns_known_dimension_for_huggingface(self) -> None:
        """Test returns dimension for known HuggingFace models."""
        assert (
            _get_embedding_dimension("huggingface", "sentence-transformers/all-MiniLM-L6-v2") == 384
        )

    def test_raises_for_unknown_model(self) -> None:
        """Test raises ValueError for unknown models."""
        with pytest.raises(ValueError, match="Unknown embedding dimension"):
            _get_embedding_dimension("openai", "unknown-model-xyz")


class TestRetrieverAutoConfig:
    """Tests for Retriever auto-configuration."""

    def test_auto_configure_default_true(self) -> None:
        """Test auto_configure defaults to True."""
        r = Retriever()
        # Should not raise - auto_configure is True by default
        assert r is not None

    def test_auto_configure_false_skips_detection(self) -> None:
        """Test auto_configure=False skips environment detection."""
        # With auto_configure=False, should use explicit params only
        r = Retriever(auto_configure=False)
        assert r is not None

    def test_explicit_params_override_auto_config(self) -> None:
        """Test explicit parameters override auto-configured values."""
        # Explicit backend should override any DATABASE_URL detection
        r = Retriever(backend="memory", auto_configure=True)
        assert r._backend is not None


# =============================================================================
# 6.9.2 Remote Content Loading Tests
# =============================================================================


class TestAddFromGitHub:
    """Tests for add_from_github() method."""

    @pytest.mark.asyncio
    async def test_add_from_github_exists(self) -> None:
        """Test add_from_github method exists."""
        r = Retriever()
        assert hasattr(r, "add_from_github")
        assert callable(r.add_from_github)

    def test_sync_wrapper_exists(self) -> None:
        """Test sync wrapper add_from_github_sync exists."""
        r = Retriever()
        assert hasattr(r, "add_from_github_sync")
        assert callable(r.add_from_github_sync)


class TestAddFromUrl:
    """Tests for add_from_url() method."""

    @pytest.mark.asyncio
    async def test_add_from_url_exists(self) -> None:
        """Test add_from_url method exists."""
        r = Retriever()
        assert hasattr(r, "add_from_url")
        assert callable(r.add_from_url)

    def test_sync_wrapper_exists(self) -> None:
        """Test sync wrapper add_from_url_sync exists."""
        r = Retriever()
        assert hasattr(r, "add_from_url_sync")
        assert callable(r.add_from_url_sync)


class TestAddFromLoader:
    """Tests for add_from_loader() generic method."""

    @pytest.mark.asyncio
    async def test_add_from_loader_exists(self) -> None:
        """Test add_from_loader method exists."""
        r = Retriever()
        assert hasattr(r, "add_from_loader")
        assert callable(r.add_from_loader)

    def test_sync_wrapper_exists(self) -> None:
        """Test sync wrapper add_from_loader_sync exists."""
        r = Retriever()
        assert hasattr(r, "add_from_loader_sync")
        assert callable(r.add_from_loader_sync)

    @pytest.mark.asyncio
    async def test_accepts_metadata_param(self) -> None:
        """Test add_from_loader accepts metadata parameter."""
        r = Retriever()

        # Mock loader with proper content attribute
        mock_content = MagicMock()
        mock_content.content = "Test content from loader"
        mock_content.metadata = {"source": "test.md"}

        mock_loader = MagicMock()
        mock_loader.load = AsyncMock(return_value=[mock_content])

        # Should work with metadata parameter
        ids = await r.add_from_loader(mock_loader, metadata={"package": "test"})
        assert isinstance(ids, list)


# =============================================================================
# 6.9.3 SearchResult Enhancements Tests
# =============================================================================


class TestSearchResultToDict:
    """Tests for SearchResult.to_dict() method."""

    def test_returns_json_serializable_dict(self) -> None:
        """Test to_dict returns a JSON-serializable dictionary."""
        result = SearchResult(
            text="Test content",
            score=0.95432,
            metadata={"package": "svc-infra", "path": "docs/auth.md"},
            source="auth.md",
            page=None,
            chunk_index=0,
        )

        d = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(d)
        assert json_str is not None

    def test_rounds_score_to_4_decimals(self) -> None:
        """Test score is rounded to 4 decimal places."""
        result = SearchResult(
            text="Test",
            score=0.123456789,
            metadata={},
        )

        d = result.to_dict()

        assert d["score"] == 0.1235  # Rounded to 4 decimals

    def test_includes_all_fields(self) -> None:
        """Test to_dict includes all expected fields."""
        result = SearchResult(
            text="Test content",
            score=0.9,
            metadata={"key": "value"},
            source="test.md",
            page=5,
            chunk_index=2,
        )

        d = result.to_dict()

        assert "text" in d
        assert "score" in d
        assert "metadata" in d
        assert "source" in d
        assert "page" in d
        assert "chunk_index" in d

    def test_handles_none_values(self) -> None:
        """Test to_dict handles None values correctly."""
        result = SearchResult(
            text="Test",
            score=0.8,
            metadata={},
            source=None,
            page=None,
            chunk_index=None,
        )

        d = result.to_dict()

        # Should include None values (not omit them)
        assert d["source"] is None
        assert d["page"] is None
        assert d["chunk_index"] is None


class TestSearchResultConvenienceProperties:
    """Tests for SearchResult convenience properties."""

    def test_package_property(self) -> None:
        """Test package property extracts from metadata."""
        result = SearchResult(
            text="Test",
            score=0.9,
            metadata={"package": "svc-infra"},
        )

        assert result.package == "svc-infra"

    def test_package_property_returns_none_if_missing(self) -> None:
        """Test package property returns None if not in metadata."""
        result = SearchResult(text="Test", score=0.9, metadata={})
        assert result.package is None

    def test_repo_property(self) -> None:
        """Test repo property extracts from metadata."""
        result = SearchResult(
            text="Test",
            score=0.9,
            metadata={"repo": "nfraxio/svc-infra"},
        )

        assert result.repo == "nfraxio/svc-infra"

    def test_path_property(self) -> None:
        """Test path property extracts from metadata."""
        result = SearchResult(
            text="Test",
            score=0.9,
            metadata={"path": "docs/auth.md"},
        )

        assert result.path == "docs/auth.md"

    def test_content_type_property(self) -> None:
        """Test content_type property extracts 'type' from metadata."""
        # Note: content_type looks for "type" key, not "content_type"
        result = SearchResult(
            text="Test",
            score=0.9,
            metadata={"type": "docs"},
        )

        assert result.content_type == "docs"


# =============================================================================
# 6.9.4 Structured Tool Results Tests
# =============================================================================


class TestStructuredToolResults:
    """Tests for create_retriever_tool structured output."""

    def test_structured_false_returns_string(self) -> None:
        """Test structured=False returns formatted string."""
        from ai_infra.llm.tools.custom.retriever import create_retriever_tool

        r = Retriever()
        r.add_text("Paris is the capital of France.")

        tool = create_retriever_tool(r, structured=False)
        result = tool.invoke({"query": "capital"})

        assert isinstance(result, str)
        assert "Paris" in result

    def test_structured_true_returns_dict(self) -> None:
        """Test structured=True returns dictionary."""
        from ai_infra.llm.tools.custom.retriever import create_retriever_tool

        r = Retriever()
        r.add_text("Paris is the capital of France.")

        tool = create_retriever_tool(r, structured=True)
        result = tool.invoke({"query": "capital"})

        assert isinstance(result, dict)
        assert "results" in result
        assert "query" in result
        assert "count" in result

    def test_structured_result_is_json_serializable(self) -> None:
        """Test structured result can be JSON serialized."""
        from ai_infra.llm.tools.custom.retriever import create_retriever_tool

        r = Retriever()
        r.add_text("Test content about authentication.")

        tool = create_retriever_tool(r, structured=True)
        result = tool.invoke({"query": "auth"})

        # Should serialize without error
        json_str = json.dumps(result)
        assert json_str is not None

    def test_structured_result_uses_to_dict(self) -> None:
        """Test structured result uses SearchResult.to_dict()."""
        from ai_infra.llm.tools.custom.retriever import create_retriever_tool

        r = Retriever()
        r.add_text("Test content", metadata={"package": "test-pkg"})

        tool = create_retriever_tool(r, structured=True)
        result = tool.invoke({"query": "test"})

        # Results should have to_dict format
        first_result = result["results"][0]
        assert "text" in first_result
        assert "score" in first_result
        assert "metadata" in first_result

    def test_structured_default_is_false(self) -> None:
        """Test structured parameter defaults to False."""
        from ai_infra.llm.tools.custom.retriever import create_retriever_tool

        r = Retriever()
        r.add_text("Test content")

        # No structured param = default False = string output
        tool = create_retriever_tool(r)
        result = tool.invoke({"query": "test"})

        assert isinstance(result, str)


class TestStructuredToolResultsAsync:
    """Tests for create_retriever_tool_async structured output."""

    @pytest.mark.asyncio
    async def test_async_structured_true_returns_dict(self) -> None:
        """Test async structured=True returns dictionary."""
        from ai_infra.llm.tools.custom.retriever import create_retriever_tool_async

        r = Retriever()
        r.add_text("Berlin is the capital of Germany.")

        tool = create_retriever_tool_async(r, structured=True)
        result = await tool.ainvoke({"query": "capital"})

        assert isinstance(result, dict)
        assert "results" in result
        assert "query" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_async_structured_false_returns_string(self) -> None:
        """Test async structured=False returns string."""
        from ai_infra.llm.tools.custom.retriever import create_retriever_tool_async

        r = Retriever()
        r.add_text("Berlin is the capital of Germany.")

        tool = create_retriever_tool_async(r, structured=False)
        result = await tool.ainvoke({"query": "capital"})

        assert isinstance(result, str)
        assert "Berlin" in result


# =============================================================================
# 6.9.5 StreamEvent Structured Result Support Tests
# =============================================================================


class TestStreamEventStructuredResults:
    """Tests for StreamEvent structured result handling."""

    def test_result_structured_default_false(self) -> None:
        """Test result_structured defaults to False."""
        from ai_infra.llm.streaming import StreamEvent

        event = StreamEvent(type="tool_end", tool="search")
        assert event.result_structured is False

    def test_to_dict_with_text_result(self) -> None:
        """Test to_dict includes result for text results."""
        from ai_infra.llm.streaming import StreamEvent

        event = StreamEvent(
            type="tool_end",
            tool="search",
            result="Some text result",
            result_structured=False,
        )

        d = event.to_dict()

        assert "result" in d
        assert d["result"] == "Some text result"
        assert "structured_result" not in d

    def test_to_dict_with_structured_result(self) -> None:
        """Test to_dict uses structured_result key for structured results."""
        from ai_infra.llm.streaming import StreamEvent

        structured_data = {"results": [{"text": "Test"}], "query": "q", "count": 1}
        event = StreamEvent(
            type="tool_end",
            tool="search",
            result=structured_data,
            result_structured=True,
        )

        d = event.to_dict()

        assert "structured_result" in d
        assert d["structured_result"] == structured_data
        assert d["result_structured"] is True
        assert "result" not in d  # Should NOT have 'result' key

    def test_to_dict_structured_is_json_serializable(self) -> None:
        """Test to_dict output with structured result is JSON serializable."""
        from ai_infra.llm.streaming import StreamEvent

        event = StreamEvent(
            type="tool_end",
            tool="search",
            result={"results": [], "query": "test", "count": 0},
            result_structured=True,
        )

        d = event.to_dict()
        json_str = json.dumps(d)
        assert json_str is not None


# =============================================================================
# 6.9.6 Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_retriever_module_exports_chunk(self) -> None:
        """Test ai_infra.retriever exports Chunk."""
        from ai_infra.retriever import Chunk

        assert Chunk is not None

    def test_retriever_module_exports_search_result(self) -> None:
        """Test ai_infra.retriever exports SearchResult."""
        from ai_infra.retriever import SearchResult

        assert SearchResult is not None

    def test_top_level_exports_retriever_search_result(self) -> None:
        """Test ai_infra exports RetrieverSearchResult alias."""
        from ai_infra import RetrieverSearchResult

        assert RetrieverSearchResult is not None

    def test_top_level_exports_retriever_chunk(self) -> None:
        """Test ai_infra exports RetrieverChunk alias."""
        from ai_infra import RetrieverChunk

        assert RetrieverChunk is not None

    def test_retriever_search_result_is_correct_class(self) -> None:
        """Test RetrieverSearchResult is from retriever.models."""
        from ai_infra import RetrieverSearchResult
        from ai_infra.retriever.models import SearchResult

        assert RetrieverSearchResult is SearchResult

    def test_embeddings_search_result_still_available(self) -> None:
        """Test embeddings SearchResult is still at top level."""
        from ai_infra import SearchResult
        from ai_infra.embeddings.vectorstore import SearchResult as EmbeddingsSearchResult

        assert SearchResult is EmbeddingsSearchResult

    def test_two_search_results_are_different(self) -> None:
        """Test embeddings and retriever SearchResult are different classes."""
        from ai_infra import RetrieverSearchResult, SearchResult

        assert SearchResult is not RetrieverSearchResult
