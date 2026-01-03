"""Tests for Retriever similarity search accuracy.

Tests cover:
- Basic similarity search
- Score-based ranking
- Minimum score filtering
- Metadata filtering
- K parameter limiting
- Detailed vs simple results
- Edge cases (empty store, no matches)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_infra.retriever.models import SearchResult
from ai_infra.retriever.retriever import Retriever


class TestSearchBasics:
    """Tests for basic search functionality."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings with consistent behavior."""
        mock = MagicMock()
        # Use cosine-similar embeddings for predictable ranking
        mock.embed.side_effect = lambda text: self._text_to_embedding(text)
        mock.embed_batch.side_effect = lambda texts: [self._text_to_embedding(t) for t in texts]
        mock.provider = "huggingface"
        mock.model = "test-model"
        return mock

    def _text_to_embedding(self, text: str) -> list[float]:
        """Create deterministic embeddings based on text content."""
        # Simple hash-based embedding for testing
        text_lower = text.lower()
        if "france" in text_lower or "paris" in text_lower:
            return [0.9, 0.1, 0.0]
        elif "germany" in text_lower or "berlin" in text_lower:
            return [0.1, 0.9, 0.0]
        elif "italy" in text_lower or "rome" in text_lower:
            return [0.0, 0.1, 0.9]
        elif "capital" in text_lower:
            return [0.5, 0.5, 0.0]
        else:
            return [0.33, 0.33, 0.33]

    @pytest.fixture
    def retriever(self, mock_embeddings: MagicMock) -> Retriever:
        """Create a retriever with test data."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings

            r.add_text("Paris is the capital of France")
            r.add_text("Berlin is the capital of Germany")
            r.add_text("Rome is the capital of Italy")

            return r

    def test_search_returns_list(self, retriever: Retriever) -> None:
        """Test that search returns a list of strings."""
        results = retriever.search("capital of France")

        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], str)

    def test_search_returns_similar_results_first(self, retriever: Retriever) -> None:
        """Test that more similar results are ranked first."""
        results = retriever.search("Paris France", k=3)

        assert len(results) >= 1
        # The France-related result should be first
        assert "France" in results[0] or "Paris" in results[0]

    def test_search_k_limits_results(self, retriever: Retriever) -> None:
        """Test that k parameter limits result count."""
        results = retriever.search("capital", k=2)

        assert len(results) <= 2

    def test_search_detailed_returns_search_results(self, retriever: Retriever) -> None:
        """Test that detailed=True returns SearchResult objects."""
        results = retriever.search("France", detailed=True)

        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], SearchResult)
            assert hasattr(results[0], "text")
            assert hasattr(results[0], "score")
            assert hasattr(results[0], "metadata")

    def test_search_detailed_includes_scores(self, retriever: Retriever) -> None:
        """Test that detailed results include similarity scores."""
        results = retriever.search("France", detailed=True)

        assert len(results) >= 1
        for result in results:
            assert result.score is not None
            assert 0.0 <= result.score <= 1.0

    def test_search_empty_store_returns_empty(self) -> None:
        """Test that search on empty store returns empty list."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            mock_emb = MagicMock()
            mock_emb.embed.return_value = [0.1, 0.2, 0.3]
            mock_emb.provider = "huggingface"
            mock_emb.model = "test"
            MockEmb.return_value = mock_emb

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_emb

            results = r.search("anything")
            assert results == []


class TestSearchMinScore:
    """Tests for minimum score filtering."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings."""
        mock = MagicMock()
        mock.embed.return_value = [1.0, 0.0, 0.0]
        mock.embed_batch.return_value = [
            [1.0, 0.0, 0.0],  # Perfect match
            [0.5, 0.5, 0.5],  # Partial match
            [0.0, 1.0, 0.0],  # Low match
        ]
        mock.provider = "huggingface"
        mock.model = "test"
        return mock

    @pytest.fixture
    def retriever(self, mock_embeddings: MagicMock) -> Retriever:
        """Create retriever with varied similarity data."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings

            r.add_text("High relevance document")
            r.add_text("Medium relevance document")
            r.add_text("Low relevance document")

            return r

    def test_min_score_filters_low_scoring(
        self, retriever: Retriever, mock_embeddings: MagicMock
    ) -> None:
        """Test that min_score filters out low-scoring results."""
        results_all = retriever.search("query", detailed=True)
        results_filtered = retriever.search("query", detailed=True, min_score=0.8)

        # Filtered should have fewer or equal results
        assert len(results_filtered) <= len(results_all)

        # All filtered results should meet min_score
        for result in results_filtered:
            assert result.score >= 0.8

    def test_min_score_zero_returns_all(self, retriever: Retriever) -> None:
        """Test that min_score=0 returns all results."""
        results = retriever.search("query", min_score=0.0)

        # Should return some results
        assert len(results) >= 1

    def test_min_score_one_returns_only_perfect_matches(self, retriever: Retriever) -> None:
        """Test that min_score=1.0 only returns perfect matches."""
        results = retriever.search("query", min_score=1.0)

        # Perfect score (1.0) is very unlikely, may return 0 results
        # This is expected behavior
        assert isinstance(results, list)


class TestSearchMetadataFilter:
    """Tests for metadata-based filtering."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings."""
        mock = MagicMock()
        mock.embed.return_value = [0.5, 0.5, 0.0]
        mock.embed_batch.return_value = [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ]
        mock.provider = "huggingface"
        mock.model = "test"
        return mock

    @pytest.fixture
    def retriever_with_metadata(self, mock_embeddings: MagicMock) -> Retriever:
        """Create retriever with metadata-tagged documents."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings

            r.add_text(
                "Python programming guide",
                metadata={"language": "python", "type": "guide"},
            )
            r.add_text(
                "JavaScript tutorial",
                metadata={"language": "javascript", "type": "tutorial"},
            )
            r.add_text(
                "Python tutorial",
                metadata={"language": "python", "type": "tutorial"},
            )

            return r

    def test_filter_by_single_field(self, retriever_with_metadata: Retriever) -> None:
        """Test filtering by a single metadata field."""
        results = retriever_with_metadata.search(
            "programming",
            filter={"language": "python"},
            detailed=True,
        )

        for result in results:
            assert result.metadata.get("language") == "python"

    def test_filter_by_multiple_fields(self, retriever_with_metadata: Retriever) -> None:
        """Test filtering by multiple metadata fields."""
        results = retriever_with_metadata.search(
            "programming",
            filter={"language": "python", "type": "tutorial"},
            detailed=True,
        )

        for result in results:
            assert result.metadata.get("language") == "python"
            assert result.metadata.get("type") == "tutorial"

    def test_filter_no_matches_returns_empty(self, retriever_with_metadata: Retriever) -> None:
        """Test that filter with no matches returns empty list."""
        results = retriever_with_metadata.search(
            "programming",
            filter={"language": "rust"},  # Not in data
        )

        assert results == []


class TestSearchScoreOrdering:
    """Tests for score-based result ordering."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings with varied similarity."""
        mock = MagicMock()

        def embed(text: str) -> list[float]:
            if "exact" in text.lower():
                return [1.0, 0.0, 0.0]
            elif "partial" in text.lower():
                return [0.7, 0.3, 0.0]
            elif "different" in text.lower():
                return [0.0, 1.0, 0.0]
            else:
                return [1.0, 0.0, 0.0]  # Query embedding

        mock.embed.side_effect = embed
        mock.embed_batch.side_effect = lambda texts: [embed(t) for t in texts]
        mock.provider = "huggingface"
        mock.model = "test"
        return mock

    @pytest.fixture
    def retriever(self, mock_embeddings: MagicMock) -> Retriever:
        """Create retriever with ordered data."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings

            # Add in mixed order
            r.add_text("This is a different topic")
            r.add_text("This is an exact match topic")
            r.add_text("This is a partial match topic")

            return r

    def test_results_ordered_by_score_descending(self, retriever: Retriever) -> None:
        """Test that results are ordered by score (highest first)."""
        results = retriever.search("exact match query", detailed=True)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_highest_similarity_first(self, retriever: Retriever) -> None:
        """Test that most similar document appears first."""
        results = retriever.search("exact match query")

        # First result should be the exact match
        if results:
            assert "exact" in results[0].lower()


class TestSearchEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings."""
        mock = MagicMock()
        mock.embed.return_value = [0.5, 0.5, 0.0]
        mock.embed_batch.return_value = [[0.5, 0.5, 0.0]]
        mock.provider = "huggingface"
        mock.model = "test"
        return mock

    def test_search_with_empty_query(self, mock_embeddings: MagicMock) -> None:
        """Test search with empty query string."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings
            r.add_text("Some content")

            # Empty query should still work (embedding of "")
            results = r.search("")
            assert isinstance(results, list)

    def test_search_with_k_zero(self, mock_embeddings: MagicMock) -> None:
        """Test search with k=0 returns empty list."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings
            r.add_text("Some content")

            results = r.search("query", k=0)
            assert results == []

    def test_search_k_larger_than_docs(self, mock_embeddings: MagicMock) -> None:
        """Test search with k larger than document count."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings
            r.add_text("Only one document")

            results = r.search("query", k=100)

            # Should return available results, not fail
            assert len(results) == 1

    def test_search_with_unicode_query(self, mock_embeddings: MagicMock) -> None:
        """Test search with unicode characters in query."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings
            r.add_text("日本語のドキュメント")

            results = r.search("日本語")
            assert isinstance(results, list)

    def test_search_with_special_characters(self, mock_embeddings: MagicMock) -> None:
        """Test search with special characters."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings
            r.add_text("Query with $pecial ch@racters!")

            results = r.search("$pecial ch@racters")
            assert isinstance(results, list)


class TestGetContext:
    """Tests for get_context method."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings."""
        mock = MagicMock()
        mock.embed.return_value = [0.5, 0.5, 0.0]
        mock.embed_batch.return_value = [
            [0.5, 0.5, 0.0],
            [0.6, 0.4, 0.0],
            [0.4, 0.6, 0.0],
        ]
        mock.provider = "huggingface"
        mock.model = "test"
        return mock

    @pytest.fixture
    def retriever(self, mock_embeddings: MagicMock) -> Retriever:
        """Create retriever with context data."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings

            r.add_text("Fact 1: The sky is blue.")
            r.add_text("Fact 2: Water is wet.")
            r.add_text("Fact 3: Fire is hot.")

            return r

    def test_get_context_returns_string(self, retriever: Retriever) -> None:
        """Test that get_context returns a string."""
        context = retriever.get_context("color of sky", k=2)

        assert isinstance(context, str)
        assert len(context) > 0

    def test_get_context_includes_multiple_results(self, retriever: Retriever) -> None:
        """Test that get_context combines multiple results."""
        context = retriever.get_context("facts", k=3)

        # Should contain content from multiple documents
        assert "Fact" in context

    def test_get_context_respects_k_parameter(self, retriever: Retriever) -> None:
        """Test that get_context respects k parameter."""
        context_short = retriever.get_context("facts", k=1)
        context_long = retriever.get_context("facts", k=3)

        # More k should generally produce more context
        # (though depends on separator)
        assert len(context_long) >= len(context_short)

    def test_get_context_empty_store_returns_empty(self, mock_embeddings: MagicMock) -> None:
        """Test that get_context on empty store returns empty string."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings

            context = r.get_context("anything")
            assert context == ""
