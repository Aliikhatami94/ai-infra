"""Unit tests for Retriever storage backends.

Tests each backend independently with mock data where needed.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Memory Backend Tests
# =============================================================================


class TestMemoryBackend:
    """Comprehensive tests for the in-memory backend."""

    @pytest.fixture
    def backend(self) -> Any:
        """Create a memory backend."""
        from ai_infra.retriever.backends.memory import MemoryBackend

        return MemoryBackend()

    def test_init(self, backend: Any) -> None:
        """Test backend initialization."""
        assert backend.count() == 0

    def test_add_single(self, backend: Any) -> None:
        """Test adding a single document."""
        ids = backend.add(
            embeddings=[[1.0, 0.0, 0.0]],
            texts=["Hello"],
            metadatas=[{"key": "value"}],
        )
        assert len(ids) == 1
        assert backend.count() == 1

    def test_add_batch(self, backend: Any) -> None:
        """Test adding multiple documents at once."""
        embeddings = [[float(i), 0.0, 0.0] for i in range(100)]
        texts = [f"Document {i}" for i in range(100)]
        metadatas = [{"index": i} for i in range(100)]

        ids = backend.add(embeddings, texts, metadatas)
        assert len(ids) == 100
        assert backend.count() == 100

    def test_add_auto_ids(self, backend: Any) -> None:
        """Test auto-generated IDs are unique."""
        ids = backend.add(
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            texts=["A", "B"],
            metadatas=[{}, {}],
        )
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_add_custom_ids(self, backend: Any) -> None:
        """Test using custom IDs."""
        ids = backend.add(
            embeddings=[[1.0, 0.0]],
            texts=["Test"],
            metadatas=[{}],
            ids=["my-custom-id"],
        )
        assert ids == ["my-custom-id"]

    def test_add_empty(self, backend: Any) -> None:
        """Test adding empty list."""
        ids = backend.add([], [], [])
        assert ids == []
        assert backend.count() == 0

    def test_search_basic(self, backend: Any) -> None:
        """Test basic similarity search."""
        backend.add(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            texts=["X axis", "Y axis", "Z axis"],
            metadatas=[{}, {}, {}],
        )

        # Search for X axis
        results = backend.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0]["text"] == "X axis"
        assert results[0]["score"] > 0.99  # Should be almost 1.0

    def test_search_ranking(self, backend: Any) -> None:
        """Test results are ranked by similarity."""
        backend.add(
            embeddings=[[1.0, 0.0], [0.8, 0.2], [0.5, 0.5]],
            texts=["Most similar", "Similar", "Less similar"],
            metadatas=[{}, {}, {}],
        )

        results = backend.search([1.0, 0.0], k=3)
        assert len(results) == 3
        # First should be most similar
        assert results[0]["text"] == "Most similar"
        # Scores should be descending
        assert results[0]["score"] >= results[1]["score"]
        assert results[1]["score"] >= results[2]["score"]

    def test_search_k_limit(self, backend: Any) -> None:
        """Test k limits results."""
        # Add many documents
        for i in range(50):
            backend.add(
                embeddings=[[float(i) / 50, 1.0 - float(i) / 50]],
                texts=[f"Doc {i}"],
                metadatas=[{}],
            )

        results = backend.search([0.5, 0.5], k=5)
        assert len(results) == 5

    def test_search_empty_store(self, backend: Any) -> None:
        """Test search on empty store."""
        results = backend.search([1.0, 0.0], k=10)
        assert results == []

    def test_search_with_filter(self, backend: Any) -> None:
        """Test search with metadata filter."""
        backend.add(
            embeddings=[[1.0, 0.0], [1.0, 0.1], [1.0, 0.2]],
            texts=["Category A", "Category B", "Category A again"],
            metadatas=[{"cat": "A"}, {"cat": "B"}, {"cat": "A"}],
        )

        # Filter by category A
        results = backend.search([1.0, 0.0], k=10, filter={"cat": "A"})
        assert len(results) == 2
        for r in results:
            assert r["metadata"]["cat"] == "A"

    def test_search_filter_no_match(self, backend: Any) -> None:
        """Test filter with no matching documents."""
        backend.add(
            embeddings=[[1.0, 0.0]],
            texts=["Only one"],
            metadatas=[{"cat": "A"}],
        )

        results = backend.search([1.0, 0.0], k=10, filter={"cat": "Z"})
        assert results == []

    def test_delete_single(self, backend: Any) -> None:
        """Test deleting a single document."""
        ids = backend.add(
            embeddings=[[1.0, 0.0]],
            texts=["To delete"],
            metadatas=[{}],
        )

        deleted = backend.delete([ids[0]])
        assert deleted == 1
        assert backend.count() == 0

    def test_delete_multiple(self, backend: Any) -> None:
        """Test deleting multiple documents."""
        ids = backend.add(
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            texts=["A", "B", "C"],
            metadatas=[{}, {}, {}],
        )

        deleted = backend.delete([ids[0], ids[2]])
        assert deleted == 2
        assert backend.count() == 1

    def test_delete_nonexistent(self, backend: Any) -> None:
        """Test deleting nonexistent ID."""
        backend.add(
            embeddings=[[1.0, 0.0]],
            texts=["Existing"],
            metadatas=[{}],
        )

        deleted = backend.delete(["nonexistent-id"])
        assert deleted == 0
        assert backend.count() == 1

    def test_delete_empty_list(self, backend: Any) -> None:
        """Test deleting empty list."""
        backend.add(
            embeddings=[[1.0, 0.0]],
            texts=["Test"],
            metadatas=[{}],
        )

        deleted = backend.delete([])
        assert deleted == 0
        assert backend.count() == 1

    def test_clear(self, backend: Any) -> None:
        """Test clearing all documents."""
        backend.add(
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            texts=["A", "B"],
            metadatas=[{}, {}],
        )
        assert backend.count() == 2

        backend.clear()
        assert backend.count() == 0

    def test_count(self, backend: Any) -> None:
        """Test count tracking."""
        assert backend.count() == 0

        backend.add([[1.0]], ["A"], [{}])
        assert backend.count() == 1

        backend.add([[2.0]], ["B"], [{}])
        assert backend.count() == 2

        backend.clear()
        assert backend.count() == 0


# =============================================================================
# SQLite Backend Tests
# =============================================================================


class TestSQLiteBackend:
    """Tests for the SQLite storage backend."""

    @pytest.fixture
    def backend(self) -> Any:
        """Create a SQLite backend with temp file."""
        from ai_infra.retriever.backends.sqlite import SQLiteBackend

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        backend = SQLiteBackend(path=db_path)
        yield backend

        # Cleanup
        backend._conn.close()
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_init(self, backend: Any) -> None:
        """Test backend initialization creates table."""
        assert backend.count() == 0

    def test_persistence(self) -> None:
        """Test data persists across connections."""
        from ai_infra.retriever.backends.sqlite import SQLiteBackend

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # First connection - add data
            backend1 = SQLiteBackend(path=db_path)
            backend1.add([[1.0, 0.0]], ["Persistent"], [{"key": "value"}])
            backend1._conn.close()

            # Second connection - verify data exists
            backend2 = SQLiteBackend(path=db_path)
            assert backend2.count() == 1
            results = backend2.search([1.0, 0.0], k=1)
            assert results[0]["text"] == "Persistent"
            backend2._conn.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_add_and_search(self, backend: Any) -> None:
        """Test adding and searching documents."""
        backend.add(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            texts=["First", "Second"],
            metadatas=[{"idx": 1}, {"idx": 2}],
        )

        results = backend.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0]["text"] == "First"

    def test_delete(self, backend: Any) -> None:
        """Test deleting documents."""
        ids = backend.add(
            embeddings=[[1.0, 0.0]],
            texts=["To delete"],
            metadatas=[{}],
        )

        deleted = backend.delete(ids)
        assert deleted == 1
        assert backend.count() == 0

    def test_clear(self, backend: Any) -> None:
        """Test clearing all documents."""
        backend.add(
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            texts=["A", "B"],
            metadatas=[{}, {}],
        )

        backend.clear()
        assert backend.count() == 0

    def test_filter(self, backend: Any) -> None:
        """Test metadata filtering."""
        backend.add(
            embeddings=[[1.0, 0.0], [1.0, 0.1]],
            texts=["Type A", "Type B"],
            metadatas=[{"type": "A"}, {"type": "B"}],
        )

        results = backend.search([1.0, 0.0], k=10, filter={"type": "A"})
        assert len(results) == 1
        assert results[0]["text"] == "Type A"


# =============================================================================
# Chroma Backend Tests (with mocks)
# =============================================================================


class TestChromaBackendWithMocks:
    """Tests for Chroma backend using mocks."""

    @pytest.fixture
    def mock_chromadb(self) -> MagicMock:
        """Create mock chromadb module."""
        mock = MagicMock()
        mock_collection = MagicMock()
        mock_collection.add = MagicMock()
        mock_collection.query = MagicMock(
            return_value={
                "ids": [["id1", "id2"]],
                "documents": [["Doc 1", "Doc 2"]],
                "distances": [[0.1, 0.3]],
                "metadatas": [[{"key": "a"}, {"key": "b"}]],
            }
        )
        mock_collection.count = MagicMock(return_value=2)
        mock_collection.delete = MagicMock()

        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        mock.Client.return_value = mock_client
        mock.PersistentClient.return_value = mock_client

        # Add Settings mock
        mock.config.Settings = MagicMock(return_value=MagicMock())

        return mock

    def test_init_in_memory(self, mock_chromadb: MagicMock) -> None:
        """Test in-memory initialization."""
        with patch.dict(
            "sys.modules", {"chromadb": mock_chromadb, "chromadb.config": mock_chromadb.config}
        ):
            # Force reimport
            import importlib

            import ai_infra.retriever.backends.chroma as chroma_module

            importlib.reload(chroma_module)

            # Now test
            _ = chroma_module.ChromaBackend()
            mock_chromadb.Client.assert_called_once()

    def test_init_persistent(self, mock_chromadb: MagicMock) -> None:
        """Test persistent initialization."""
        with patch.dict(
            "sys.modules", {"chromadb": mock_chromadb, "chromadb.config": mock_chromadb.config}
        ):
            import importlib

            import ai_infra.retriever.backends.chroma as chroma_module

            importlib.reload(chroma_module)

            _ = chroma_module.ChromaBackend(persist_directory="./test_db")
            mock_chromadb.PersistentClient.assert_called()


# =============================================================================
# Pinecone Backend Tests (with mocks)
# =============================================================================


class TestPineconeBackendWithMocks:
    """Tests for Pinecone backend using mocks."""

    @pytest.fixture
    def mock_pinecone(self) -> MagicMock:
        """Create mock pinecone module."""
        mock = MagicMock()

        # Mock index
        mock_index = MagicMock()
        mock_index.upsert = MagicMock()
        mock_index.query = MagicMock(
            return_value=MagicMock(
                matches=[
                    MagicMock(
                        id="id1",
                        score=0.95,
                        metadata={"text": "Result 1", "key": "a"},
                    ),
                    MagicMock(
                        id="id2",
                        score=0.85,
                        metadata={"text": "Result 2", "key": "b"},
                    ),
                ]
            )
        )
        mock_index.delete = MagicMock()
        mock_index.describe_index_stats = MagicMock(return_value=MagicMock(total_vector_count=10))

        # Mock Pinecone class
        mock_pc = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock.Pinecone.return_value = mock_pc

        return mock

    def test_add_with_namespace(self, mock_pinecone: MagicMock) -> None:
        """Test adding vectors with namespace."""
        with patch.dict("sys.modules", {"pinecone": mock_pinecone}):
            import importlib

            import ai_infra.retriever.backends.pinecone as pinecone_module

            importlib.reload(pinecone_module)

            backend = pinecone_module.PineconeBackend(
                api_key="test-key",
                index_name="test-index",
                namespace="test-ns",
            )

            ids = backend.add(
                embeddings=[[0.1, 0.2, 0.3]],
                texts=["Hello"],
                metadatas=[{"key": "value"}],
            )
            assert len(ids) == 1


# =============================================================================
# Qdrant Backend Tests (with mocks)
# =============================================================================


class TestQdrantBackendWithMocks:
    """Tests for Qdrant backend using mocks."""

    @pytest.fixture
    def mock_qdrant(self) -> MagicMock:
        """Create mock qdrant-client module."""
        mock = MagicMock()

        # Mock client
        mock_client = MagicMock()
        mock_client.upsert = MagicMock()
        mock_client.search = MagicMock(
            return_value=[
                MagicMock(
                    id="id1",
                    score=0.95,
                    payload={"text": "Result 1", "metadata": {"key": "a"}},
                ),
            ]
        )
        mock_client.delete = MagicMock()
        mock_client.count = MagicMock(return_value=MagicMock(count=5))
        mock_client.collection_exists = MagicMock(return_value=True)
        mock_client.create_collection = MagicMock()

        mock.QdrantClient.return_value = mock_client

        # Add models
        mock_models = MagicMock()
        mock_models.Distance = MagicMock()
        mock_models.Distance.COSINE = "Cosine"
        mock_models.VectorParams = MagicMock()
        mock_models.PointStruct = MagicMock()
        mock_models.Filter = MagicMock()
        mock_models.FieldCondition = MagicMock()
        mock_models.MatchValue = MagicMock()

        return mock, mock_models

    def test_init(self, mock_qdrant: tuple[MagicMock, MagicMock]) -> None:
        """Test Qdrant backend initialization."""
        mock, mock_models = mock_qdrant
        with patch.dict(
            "sys.modules",
            {"qdrant_client": mock, "qdrant_client.models": mock_models},
        ):
            import importlib

            import ai_infra.retriever.backends.qdrant as qdrant_module

            importlib.reload(qdrant_module)

            _ = qdrant_module.QdrantBackend(
                url="http://localhost:6333",
                collection_name="test",
                embedding_dimension=384,  # correct param name
            )
            mock.QdrantClient.assert_called()


# =============================================================================
# FAISS Backend Tests (with mocks)
# =============================================================================


class TestFAISSBackendWithMocks:
    """Tests for FAISS backend using mocks."""

    @pytest.fixture
    def mock_faiss(self) -> MagicMock:
        """Create mock faiss module."""
        mock = MagicMock()

        # Mock index
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_index.add = MagicMock()
        mock_index.search = MagicMock(
            return_value=(
                [[0.1, 0.3]],  # distances
                [[0, 1]],  # indices
            )
        )

        mock.IndexFlatIP.return_value = mock_index
        mock.IndexFlatL2.return_value = mock_index

        return mock

    def test_init_flat_ip(self, mock_faiss: MagicMock) -> None:
        """Test initialization with IndexFlatIP."""
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            import importlib

            import ai_infra.retriever.backends.faiss as faiss_module

            importlib.reload(faiss_module)

            _ = faiss_module.FAISSBackend(dimension=384)
            mock_faiss.IndexFlatIP.assert_called_with(384)


# =============================================================================
# Backend Factory Tests
# =============================================================================


class TestBackendFactory:
    """Tests for the backend factory function."""

    def test_get_memory_backend(self) -> None:
        """Test creating memory backend."""
        from ai_infra.retriever.backends import get_backend

        backend = get_backend("memory")
        from ai_infra.retriever.backends.memory import MemoryBackend

        assert isinstance(backend, MemoryBackend)

    def test_get_sqlite_backend(self) -> None:
        """Test creating SQLite backend."""
        from ai_infra.retriever.backends import get_backend

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            backend = get_backend("sqlite", path=db_path)
            from ai_infra.retriever.backends.sqlite import SQLiteBackend

            assert isinstance(backend, SQLiteBackend)
        finally:
            backend._conn.close()
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_get_invalid_backend(self) -> None:
        """Test invalid backend name raises error."""
        from ai_infra.retriever.backends import get_backend

        with pytest.raises((ValueError, KeyError)):
            get_backend("invalid_backend_xyz")

    def test_backend_kwargs_passed(self) -> None:
        """Test kwargs are passed to backend constructor."""
        from ai_infra.retriever.backends import get_backend

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            backend = get_backend("sqlite", path=db_path, table_name="custom_table")
            assert backend._table_name == "custom_table"
        finally:
            backend._conn.close()
            if os.path.exists(db_path):
                os.unlink(db_path)


# =============================================================================
# Base Backend Interface Tests
# =============================================================================


class TestBaseBackendInterface:
    """Tests for the base backend interface."""

    def test_abstract_methods(self) -> None:
        """Test BaseBackend has required abstract methods."""
        from ai_infra.retriever.backends.base import BaseBackend

        # Check required methods exist
        assert hasattr(BaseBackend, "add")
        assert hasattr(BaseBackend, "search")
        assert hasattr(BaseBackend, "delete")
        assert hasattr(BaseBackend, "clear")
        assert hasattr(BaseBackend, "count")

    def test_cannot_instantiate_directly(self) -> None:
        """Test BaseBackend cannot be instantiated directly."""
        from ai_infra.retriever.backends.base import BaseBackend

        with pytest.raises(TypeError):
            BaseBackend()  # type: ignore


# =============================================================================
# Backend Consistency Tests
# =============================================================================


class TestBackendConsistency:
    """Tests that all backends have consistent behavior."""

    @pytest.fixture(params=["memory", "sqlite"])
    def backend(self, request: Any) -> Any:
        """Create backends for testing."""
        from ai_infra.retriever.backends import get_backend

        backend_name = request.param

        if backend_name == "sqlite":
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name
            backend = get_backend("sqlite", path=db_path)
            yield backend
            backend._conn.close()
            if os.path.exists(db_path):
                os.unlink(db_path)
        else:
            yield get_backend(backend_name)

    def test_add_returns_ids(self, backend: Any) -> None:
        """Test add returns list of IDs."""
        ids = backend.add(
            embeddings=[[1.0, 0.0]],
            texts=["Test"],
            metadatas=[{}],
        )
        assert isinstance(ids, list)
        assert len(ids) == 1
        assert isinstance(ids[0], str)

    def test_search_returns_list_of_dicts(self, backend: Any) -> None:
        """Test search returns list of dicts with required keys."""
        backend.add(
            embeddings=[[1.0, 0.0]],
            texts=["Test"],
            metadatas=[{"key": "value"}],
        )

        results = backend.search([1.0, 0.0], k=1)
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "id" in r
            assert "text" in r
            assert "score" in r
            assert "metadata" in r

    def test_count_returns_int(self, backend: Any) -> None:
        """Test count returns integer."""
        count = backend.count()
        assert isinstance(count, int)
        assert count >= 0

    def test_delete_returns_count(self, backend: Any) -> None:
        """Test delete returns deleted count."""
        ids = backend.add(
            embeddings=[[1.0, 0.0]],
            texts=["Test"],
            metadatas=[{}],
        )

        deleted = backend.delete(ids)
        assert isinstance(deleted, int)
        assert deleted >= 0

    def test_clear_empties_store(self, backend: Any) -> None:
        """Test clear empties the store."""
        backend.add(
            embeddings=[[1.0, 0.0]],
            texts=["Test"],
            metadatas=[{}],
        )

        backend.clear()
        assert backend.count() == 0
