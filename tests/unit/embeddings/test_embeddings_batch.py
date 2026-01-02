"""Tests for Embeddings batch operations.

Tests cover:
- Batch embedding of multiple texts
- Async batch embedding
- Batch size handling
- Concurrency control
- Empty list handling
- Large batch processing
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestEmbedBatch:
    """Tests for Embeddings.embed_batch() method."""

    @pytest.fixture
    def mock_lc_embeddings(self) -> MagicMock:
        """Create mock LangChain embeddings."""
        mock = MagicMock()
        mock.embed_documents.side_effect = lambda texts: [
            [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)] for i in range(len(texts))
        ]
        mock.embed_query.return_value = [0.1, 0.2, 0.3]
        return mock

    @pytest.fixture
    def embeddings(self, mock_lc_embeddings: MagicMock) -> Any:
        """Create Embeddings instance with mocked backend."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=mock_lc_embeddings):
            with patch.object(Embeddings, "_get_available_provider", return_value="huggingface"):
                emb = Embeddings(provider="huggingface")
                emb._lc_embeddings = mock_lc_embeddings
                return emb

    def test_embed_batch_returns_list_of_vectors(self, embeddings: Any) -> None:
        """Test that embed_batch returns a list of embedding vectors."""
        texts = ["Hello", "World", "Test"]
        vectors = embeddings.embed_batch(texts)

        assert isinstance(vectors, list)
        assert len(vectors) == 3
        for vector in vectors:
            assert isinstance(vector, list)
            assert all(isinstance(v, float) for v in vector)

    def test_embed_batch_empty_list_returns_empty(self, embeddings: Any) -> None:
        """Test that embed_batch with empty list returns empty list."""
        vectors = embeddings.embed_batch([])

        assert vectors == []

    def test_embed_batch_single_item(self, embeddings: Any) -> None:
        """Test embed_batch with single item."""
        vectors = embeddings.embed_batch(["Single item"])

        assert len(vectors) == 1
        assert isinstance(vectors[0], list)

    def test_embed_batch_preserves_order(
        self, embeddings: Any, mock_lc_embeddings: MagicMock
    ) -> None:
        """Test that embed_batch preserves input order."""
        texts = ["First", "Second", "Third"]
        embeddings.embed_batch(texts)

        # Verify the call was made with texts in order
        mock_lc_embeddings.embed_documents.assert_called()
        called_texts = mock_lc_embeddings.embed_documents.call_args[0][0]
        assert called_texts == texts

    def test_embed_batch_respects_batch_size(
        self, embeddings: Any, mock_lc_embeddings: MagicMock
    ) -> None:
        """Test that embed_batch respects batch_size parameter."""
        # Create more texts than batch size
        texts = [f"Text {i}" for i in range(150)]

        vectors = embeddings.embed_batch(texts, batch_size=100)

        assert len(vectors) == 150
        # Should have been called twice (100 + 50)
        assert mock_lc_embeddings.embed_documents.call_count == 2

    def test_embed_batch_small_batch_size(
        self, embeddings: Any, mock_lc_embeddings: MagicMock
    ) -> None:
        """Test embed_batch with small batch size."""
        texts = ["A", "B", "C", "D", "E"]

        vectors = embeddings.embed_batch(texts, batch_size=2)

        assert len(vectors) == 5
        # Should be called 3 times (2, 2, 1)
        assert mock_lc_embeddings.embed_documents.call_count == 3


class TestEmbedBatchAsync:
    """Tests for async Embeddings.aembed_batch() method."""

    @pytest.fixture
    def mock_lc_embeddings_async(self) -> MagicMock:
        """Create mock LangChain embeddings with async support."""
        mock = MagicMock()
        mock.aembed_documents = AsyncMock(
            side_effect=lambda texts: [
                [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)] for i in range(len(texts))
            ]
        )
        mock.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock.embed_documents.side_effect = lambda texts: [
            [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)] for i in range(len(texts))
        ]
        return mock

    @pytest.fixture
    def embeddings_async(self, mock_lc_embeddings_async: MagicMock) -> Any:
        """Create Embeddings instance with async mocked backend."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=mock_lc_embeddings_async):
            with patch.object(Embeddings, "_get_available_provider", return_value="huggingface"):
                emb = Embeddings(provider="huggingface")
                emb._lc_embeddings = mock_lc_embeddings_async
                return emb

    @pytest.mark.asyncio
    async def test_aembed_batch_returns_list_of_vectors(self, embeddings_async: Any) -> None:
        """Test that aembed_batch returns a list of embedding vectors."""
        texts = ["Async", "Batch", "Test"]
        vectors = await embeddings_async.aembed_batch(texts)

        assert isinstance(vectors, list)
        assert len(vectors) == 3

    @pytest.mark.asyncio
    async def test_aembed_batch_empty_list_returns_empty(self, embeddings_async: Any) -> None:
        """Test that aembed_batch with empty list returns empty list."""
        vectors = await embeddings_async.aembed_batch([])

        assert vectors == []

    @pytest.mark.asyncio
    async def test_aembed_batch_respects_batch_size(
        self, embeddings_async: Any, mock_lc_embeddings_async: MagicMock
    ) -> None:
        """Test that aembed_batch respects batch_size parameter."""
        texts = [f"Text {i}" for i in range(150)]

        vectors = await embeddings_async.aembed_batch(texts, batch_size=100)

        assert len(vectors) == 150
        # Should have been called twice (100 + 50)
        assert mock_lc_embeddings_async.aembed_documents.call_count == 2

    @pytest.mark.asyncio
    async def test_aembed_batch_respects_concurrency(
        self, embeddings_async: Any, mock_lc_embeddings_async: MagicMock
    ) -> None:
        """Test that aembed_batch respects max_concurrency parameter."""
        texts = [f"Text {i}" for i in range(500)]

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0

        original_aembed = mock_lc_embeddings_async.aembed_documents

        async def tracked_aembed(texts: list[str]) -> list[list[float]]:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)  # Simulate API delay
            concurrent_count -= 1
            return await original_aembed(texts)

        mock_lc_embeddings_async.aembed_documents = tracked_aembed

        await embeddings_async.aembed_batch(texts, batch_size=100, max_concurrency=2)

        # Should never exceed max_concurrency
        assert max_concurrent <= 2


class TestEmbedBatchLargeInputs:
    """Tests for handling large batch inputs."""

    @pytest.fixture
    def mock_lc_embeddings(self) -> MagicMock:
        """Create mock LangChain embeddings."""
        mock = MagicMock()
        mock.embed_documents.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
        return mock

    @pytest.fixture
    def embeddings(self, mock_lc_embeddings: MagicMock) -> Any:
        """Create Embeddings instance."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=mock_lc_embeddings):
            with patch.object(Embeddings, "_get_available_provider", return_value="huggingface"):
                emb = Embeddings(provider="huggingface")
                emb._lc_embeddings = mock_lc_embeddings
                return emb

    def test_embed_batch_thousand_texts(self, embeddings: Any) -> None:
        """Test embedding a large batch of texts."""
        texts = [f"Text number {i}" for i in range(1000)]

        vectors = embeddings.embed_batch(texts, batch_size=100)

        assert len(vectors) == 1000

    def test_embed_batch_long_texts(self, embeddings: Any) -> None:
        """Test embedding texts with long content."""
        long_text = "Word " * 1000  # ~5000 characters
        texts = [long_text for _ in range(10)]

        vectors = embeddings.embed_batch(texts)

        assert len(vectors) == 10


class TestEmbedBatchErrorHandling:
    """Tests for batch embedding error handling."""

    @pytest.fixture
    def mock_lc_embeddings_failing(self) -> MagicMock:
        """Create mock that fails on certain inputs."""
        mock = MagicMock()
        call_count = [0]

        def embed_with_failure(texts: list[str]) -> list[list[float]]:
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second batch
                raise RuntimeError("API rate limit exceeded")
            return [[0.1, 0.2, 0.3] for _ in texts]

        mock.embed_documents.side_effect = embed_with_failure
        return mock

    def test_embed_batch_propagates_errors(self, mock_lc_embeddings_failing: MagicMock) -> None:
        """Test that errors from underlying embeddings are propagated."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=mock_lc_embeddings_failing):
            with patch.object(Embeddings, "_get_available_provider", return_value="huggingface"):
                emb = Embeddings(provider="huggingface")
                emb._lc_embeddings = mock_lc_embeddings_failing

                texts = [f"Text {i}" for i in range(200)]

                with pytest.raises(RuntimeError, match="rate limit"):
                    emb.embed_batch(texts, batch_size=100)


class TestSimilarity:
    """Tests for similarity calculation."""

    @pytest.fixture
    def mock_lc_embeddings(self) -> MagicMock:
        """Create mock LangChain embeddings."""
        mock = MagicMock()

        def embed_query(text: str) -> list[float]:
            if "hello" in text.lower():
                return [1.0, 0.0, 0.0]
            elif "hi" in text.lower():
                return [0.9, 0.1, 0.0]
            elif "goodbye" in text.lower():
                return [0.0, 1.0, 0.0]
            else:
                return [0.5, 0.5, 0.0]

        mock.embed_query.side_effect = embed_query
        return mock

    @pytest.fixture
    def embeddings(self, mock_lc_embeddings: MagicMock) -> Any:
        """Create Embeddings instance."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=mock_lc_embeddings):
            with patch.object(Embeddings, "_get_available_provider", return_value="huggingface"):
                emb = Embeddings(provider="huggingface")
                emb._lc_embeddings = mock_lc_embeddings
                return emb

    def test_similarity_returns_float(self, embeddings: Any) -> None:
        """Test that similarity returns a float."""
        score = embeddings.similarity("hello", "hi")

        assert isinstance(score, float)

    def test_similarity_identical_texts_high(self, embeddings: Any) -> None:
        """Test that identical texts have high similarity."""
        score = embeddings.similarity("hello world", "hello world")

        assert score > 0.9

    def test_similarity_similar_texts(self, embeddings: Any) -> None:
        """Test that similar texts have moderate-high similarity."""
        score = embeddings.similarity("hello", "hi")

        # Should be similar but not identical
        assert 0.5 < score < 1.0

    def test_similarity_different_texts_low(self, embeddings: Any) -> None:
        """Test that different texts have lower similarity."""
        score = embeddings.similarity("hello", "goodbye")

        # Should be lower than similar texts
        assert score < 0.5

    def test_similarity_range(self, embeddings: Any) -> None:
        """Test that similarity is in valid range [-1, 1]."""
        score = embeddings.similarity("random", "text")

        assert -1.0 <= score <= 1.0


class TestAsyncSimilarity:
    """Tests for async similarity calculation."""

    @pytest.fixture
    def mock_lc_embeddings_async(self) -> MagicMock:
        """Create mock LangChain embeddings with async support."""
        mock = MagicMock()
        mock.aembed_query = AsyncMock(
            side_effect=lambda text: (
                [1.0, 0.0, 0.0] if "hello" in text.lower() else [0.0, 1.0, 0.0]
            )
        )
        return mock

    @pytest.fixture
    def embeddings_async(self, mock_lc_embeddings_async: MagicMock) -> Any:
        """Create Embeddings instance with async support."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=mock_lc_embeddings_async):
            with patch.object(Embeddings, "_get_available_provider", return_value="huggingface"):
                emb = Embeddings(provider="huggingface")
                emb._lc_embeddings = mock_lc_embeddings_async
                return emb

    @pytest.mark.asyncio
    async def test_asimilarity_returns_float(self, embeddings_async: Any) -> None:
        """Test that asimilarity returns a float."""
        score = await embeddings_async.asimilarity("hello", "hi")

        assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_asimilarity_concurrent_embeds(
        self, embeddings_async: Any, mock_lc_embeddings_async: MagicMock
    ) -> None:
        """Test that asimilarity embeds both texts concurrently."""
        await embeddings_async.asimilarity("hello", "goodbye")

        # Both texts should be embedded
        assert mock_lc_embeddings_async.aembed_query.call_count == 2


class TestEmbedSingle:
    """Tests for single text embedding."""

    @pytest.fixture
    def mock_lc_embeddings(self) -> MagicMock:
        """Create mock LangChain embeddings."""
        mock = MagicMock()
        mock.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        return mock

    @pytest.fixture
    def embeddings(self, mock_lc_embeddings: MagicMock) -> Any:
        """Create Embeddings instance."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=mock_lc_embeddings):
            with patch.object(Embeddings, "_get_available_provider", return_value="huggingface"):
                emb = Embeddings(provider="huggingface")
                emb._lc_embeddings = mock_lc_embeddings
                return emb

    def test_embed_returns_list_of_floats(self, embeddings: Any) -> None:
        """Test that embed returns a list of floats."""
        vector = embeddings.embed("Hello world")

        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)

    def test_embed_empty_string(self, embeddings: Any) -> None:
        """Test embed with empty string."""
        vector = embeddings.embed("")

        assert isinstance(vector, list)

    def test_embed_unicode(self, embeddings: Any) -> None:
        """Test embed with unicode text."""
        vector = embeddings.embed("こんにちは世界")

        assert isinstance(vector, list)


class TestAsyncEmbedSingle:
    """Tests for async single text embedding."""

    @pytest.fixture
    def mock_lc_embeddings_async(self) -> MagicMock:
        """Create mock LangChain embeddings with async support."""
        mock = MagicMock()
        mock.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        return mock

    @pytest.fixture
    def embeddings_async(self, mock_lc_embeddings_async: MagicMock) -> Any:
        """Create Embeddings instance with async support."""
        from ai_infra.embeddings import Embeddings

        with patch.object(Embeddings, "_init_embeddings", return_value=mock_lc_embeddings_async):
            with patch.object(Embeddings, "_get_available_provider", return_value="huggingface"):
                emb = Embeddings(provider="huggingface")
                emb._lc_embeddings = mock_lc_embeddings_async
                return emb

    @pytest.mark.asyncio
    async def test_aembed_returns_list_of_floats(self, embeddings_async: Any) -> None:
        """Test that aembed returns a list of floats."""
        vector = await embeddings_async.aembed("Hello world")

        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)
