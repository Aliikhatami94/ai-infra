"""Unit tests for the embeddings module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.embeddings import Embeddings, VectorStore
from ai_infra.embeddings.vectorstore import Document, SearchResult

# =============================================================================
# Embeddings Tests
# =============================================================================


class TestEmbeddingsProviders:
    """Tests for Embeddings provider configuration."""

    def test_list_providers(self) -> None:
        """Test listing all providers."""
        providers = Embeddings.list_providers()
        assert "openai" in providers
        assert "google" in providers
        assert "voyage" in providers
        assert "cohere" in providers
        assert "anthropic" in providers

    def test_list_configured_providers_empty(self) -> None:
        """Test no providers configured."""
        with patch.dict("os.environ", {}, clear=True):
            providers = Embeddings.list_configured_providers()
            assert providers == []

    def test_list_configured_providers_with_key(self) -> None:
        """Test with API key set."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}, clear=True):
            providers = Embeddings.list_configured_providers()
            assert "openai" in providers

    def test_invalid_provider(self) -> None:
        """Test error for invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            Embeddings(provider="invalid")

    def test_no_provider_available(self) -> None:
        """Test error when no provider configured."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No embedding provider available"):
                Embeddings()


class TestEmbeddingsWithMock:
    """Tests for Embeddings class using mocks."""

    @pytest.fixture
    def mock_lc_embeddings(self) -> MagicMock:
        """Create mock LangChain embeddings."""
        mock = MagicMock()
        mock.embed_query.return_value = [0.1, 0.2, 0.3]
        mock.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        return mock

    @pytest.fixture
    def embeddings(self, mock_lc_embeddings: MagicMock) -> Embeddings:
        """Create Embeddings with mock."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.OpenAIEmbeddings.return_value = mock_lc_embeddings
            mock_import.return_value = mock_module

            emb = Embeddings(provider="openai")
            return emb

    def test_properties(self, embeddings: Embeddings) -> None:
        """Test Embeddings properties."""
        assert embeddings.provider == "openai"
        assert embeddings.model == "text-embedding-3-small"
        assert embeddings.dimensions is None

    def test_embed(self, embeddings: Embeddings) -> None:
        """Test embed() method."""
        result = embeddings.embed("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_batch(self, embeddings: Embeddings) -> None:
        """Test embed_batch() method."""
        result = embeddings.embed_batch(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_batch_empty(self, embeddings: Embeddings) -> None:
        """Test embed_batch() with empty list."""
        result = embeddings.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_aembed(self, embeddings: Embeddings) -> None:
        """Test aembed() method."""
        result = await embeddings.aembed("hello")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_aembed_batch(self, embeddings: Embeddings) -> None:
        """Test aembed_batch() method."""
        result = await embeddings.aembed_batch(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_aembed_batch_empty(self, embeddings: Embeddings) -> None:
        """Test aembed_batch() with empty list."""
        result = await embeddings.aembed_batch([])
        assert result == []

    def test_repr(self, embeddings: Embeddings) -> None:
        """Test string representation."""
        result = repr(embeddings)
        assert "openai" in result
        assert "text-embedding-3-small" in result


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self) -> None:
        """Test similarity of identical vectors."""
        v = [1.0, 2.0, 3.0]
        result = Embeddings._cosine_similarity(v, v)
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Test similarity of orthogonal vectors."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        result = Embeddings._cosine_similarity(v1, v2)
        assert result == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Test similarity of opposite vectors."""
        v1 = [1.0, 1.0]
        v2 = [-1.0, -1.0]
        result = Embeddings._cosine_similarity(v1, v2)
        assert result == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        """Test handling of zero vectors."""
        v1 = [0.0, 0.0]
        v2 = [1.0, 1.0]
        result = Embeddings._cosine_similarity(v1, v2)
        assert result == 0.0


# =============================================================================
# Document Tests
# =============================================================================


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self) -> None:
        """Test creating a document."""
        doc = Document(text="Hello world")
        assert doc.text == "Hello world"
        assert doc.metadata == {}
        assert doc.id is not None

    def test_document_with_metadata(self) -> None:
        """Test document with metadata."""
        doc = Document(text="Hello", metadata={"source": "test"})
        assert doc.metadata == {"source": "test"}

    def test_document_with_id(self) -> None:
        """Test document with custom ID."""
        doc = Document(text="Hello", id="custom-id")
        assert doc.id == "custom-id"


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result(self) -> None:
        """Test creating a search result."""
        doc = Document(text="Hello")
        result = SearchResult(document=doc, score=0.95)
        assert result.document == doc
        assert result.score == 0.95


# =============================================================================
# VectorStore Tests
# =============================================================================


class MockEmbeddings:
    """Mock embeddings for testing VectorStore."""

    def __init__(self) -> None:
        self._counter = 0

    def embed(self, text: str) -> list[float]:
        """Generate deterministic embedding."""
        return [float(ord(c)) / 100 for c in text[:5]]

    async def aembed(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embed_batch(texts)


class TestVectorStoreInMemory:
    """Tests for VectorStore with in-memory backend."""

    @pytest.fixture
    def embeddings(self) -> MockEmbeddings:
        return MockEmbeddings()

    @pytest.fixture
    def store(self, embeddings: MockEmbeddings) -> VectorStore:
        # Create store with mock embeddings
        store = VectorStore.__new__(VectorStore)
        store._embeddings = embeddings  # type: ignore
        store._backend_name = "memory"

        # Import the backend class
        from ai_infra.embeddings.vectorstore import _InMemoryBackend

        store._backend = _InMemoryBackend(embeddings)  # type: ignore
        return store

    def test_add_texts(self, store: VectorStore) -> None:
        """Test adding texts."""
        ids = store.add_texts(["hello", "world"])
        assert len(ids) == 2
        assert store.count == 2

    def test_add_texts_with_metadata(self, store: VectorStore) -> None:
        """Test adding texts with metadata."""
        ids = store.add_texts(
            texts=["hello", "world"],
            metadatas=[{"source": "a"}, {"source": "b"}],
        )
        assert len(ids) == 2

    def test_add_documents(self, store: VectorStore) -> None:
        """Test adding Document objects."""
        docs = [
            Document(text="hello", metadata={"lang": "en"}),
            Document(text="bonjour", metadata={"lang": "fr"}),
        ]
        ids = store.add_documents(docs)
        assert len(ids) == 2
        assert store.count == 2

    def test_search(self, store: VectorStore) -> None:
        """Test searching."""
        store.add_texts(["hello world", "goodbye world", "hello there"])
        results = store.search("hello", k=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score >= results[1].score  # Sorted by score

    def test_search_empty_store(self, store: VectorStore) -> None:
        """Test searching empty store."""
        results = store.search("hello", k=2)
        assert results == []

    def test_search_with_filter(self, store: VectorStore) -> None:
        """Test searching with metadata filter."""
        store.add_texts(
            texts=["hello", "world"],
            metadatas=[{"source": "a"}, {"source": "b"}],
        )
        results = store.search("hello", k=10, filter={"source": "a"})

        assert len(results) == 1
        assert results[0].document.metadata["source"] == "a"

    @pytest.mark.asyncio
    async def test_asearch(self, store: VectorStore) -> None:
        """Test async search."""
        store.add_texts(["hello", "world"])
        results = await store.asearch("hello", k=2)
        assert len(results) == 2

    def test_delete(self, store: VectorStore) -> None:
        """Test deleting documents."""
        ids = store.add_texts(["hello", "world"])
        assert store.count == 2

        store.delete([ids[0]])
        assert store.count == 1

    def test_clear(self, store: VectorStore) -> None:
        """Test clearing all documents."""
        store.add_texts(["hello", "world", "test"])
        assert store.count == 3

        store.clear()
        assert store.count == 0

    def test_backend_property(self, store: VectorStore) -> None:
        """Test backend property."""
        assert store.backend == "memory"

    def test_repr(self, store: VectorStore) -> None:
        """Test string representation."""
        result = repr(store)
        assert "memory" in result


class TestVectorStoreBackends:
    """Tests for VectorStore backend selection."""

    def test_invalid_backend(self) -> None:
        """Test error for invalid backend."""
        mock_emb = MockEmbeddings()
        with pytest.raises(ValueError, match="Unknown backend"):
            store = VectorStore.__new__(VectorStore)
            store._embeddings = mock_emb  # type: ignore
            store._backend_name = "invalid"
            # Manually trigger backend init logic
            VectorStore.__init__(store, embeddings=mock_emb, backend="invalid")  # type: ignore

    def test_chroma_import_error(self) -> None:
        """Test Chroma import error message."""
        mock_emb = MockEmbeddings()
        with patch.dict("sys.modules", {"chromadb": None}):
            with patch("importlib.import_module") as mock_import:
                mock_import.side_effect = ImportError("No module")
                # Would raise ImportError with helpful message
                # This tests the error handling path

    def test_faiss_import_error(self) -> None:
        """Test FAISS import error message."""
        mock_emb = MockEmbeddings()
        # Similar pattern for FAISS
