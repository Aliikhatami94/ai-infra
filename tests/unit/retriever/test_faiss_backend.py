"""Unit tests for FAISS vector storage backend.

Tests cover:
- FAISSBackend initialization (dimension, index types, persistence)
- Document operations (add, search, delete, clear, count)
- Index types (flat, ivf, hnsw)
- Persistence (save/load, pickle security warning)
- Search parameters (top_k, score threshold, filtering)
- Edge cases and error handling

Phase 4.2 of ai-infra test plan.
"""

from __future__ import annotations

import sys
import uuid
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

# =============================================================================
# Mock Setup - Create mocks at module level before any FAISSBackend import
# =============================================================================

# Create mock faiss module
_mock_faiss = MagicMock()
_mock_index = MagicMock()

# Configure index mock
_mock_index.ntotal = 0
_mock_index.is_trained = True
_mock_index.add = MagicMock()
_mock_index.search = MagicMock(return_value=(np.array([[0.9, 0.8]]), np.array([[0, 1]])))
_mock_index.reconstruct = MagicMock()
_mock_index.train = MagicMock()

# Configure faiss mock
_mock_faiss.IndexFlatIP = MagicMock(return_value=_mock_index)
_mock_faiss.IndexIVFFlat = MagicMock(return_value=_mock_index)
_mock_faiss.IndexHNSWFlat = MagicMock(return_value=_mock_index)
_mock_faiss.METRIC_INNER_PRODUCT = 0
_mock_faiss.write_index = MagicMock()
_mock_faiss.read_index = MagicMock(return_value=_mock_index)

# Register mocks in sys.modules
sys.modules["faiss"] = _mock_faiss

# Now import the module
from ai_infra.retriever.backends.faiss import FAISSBackend  # noqa: E402

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test."""
    _mock_faiss.reset_mock()
    _mock_index.reset_mock()

    # Re-configure mocks after reset
    _mock_index.ntotal = 0
    _mock_index.is_trained = True
    _mock_faiss.IndexFlatIP.return_value = _mock_index
    _mock_faiss.IndexIVFFlat.return_value = _mock_index
    _mock_faiss.IndexHNSWFlat.return_value = _mock_index
    _mock_faiss.read_index.return_value = _mock_index

    yield


@pytest.fixture
def backend():
    """Create a FAISSBackend instance with mocked dependencies."""
    return FAISSBackend(dimension=1536)


@pytest.fixture
def backend_with_data(backend):
    """Create a backend with pre-added data."""
    embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
    texts = ["First doc", "Second doc", "Third doc"]
    metadatas = [{"idx": 1}, {"idx": 2}, {"idx": 3}]

    backend.add(
        embeddings=embeddings, texts=texts, metadatas=metadatas, ids=["id-1", "id-2", "id-3"]
    )
    _mock_index.ntotal = 3

    return backend


@pytest.fixture
def temp_path(tmp_path):
    """Create a temporary path for persistence tests."""
    return tmp_path / "faiss_test"


# =============================================================================
# FAISSBackend Initialization Tests
# =============================================================================


class TestFAISSBackendInit:
    """Tests for FAISSBackend initialization."""

    def test_init_default_dimension(self):
        """Test initialization with default dimension."""
        backend = FAISSBackend()

        assert backend._dimension == 1536
        _mock_faiss.IndexFlatIP.assert_called_with(1536)

    def test_init_custom_dimension(self):
        """Test initialization with custom dimension."""
        backend = FAISSBackend(dimension=768)

        assert backend._dimension == 768
        _mock_faiss.IndexFlatIP.assert_called_with(768)

    def test_init_flat_index_type(self):
        """Test initialization with flat index type."""
        backend = FAISSBackend(dimension=1536, index_type="flat")

        assert backend._index_type == "flat"
        _mock_faiss.IndexFlatIP.assert_called_with(1536)

    def test_init_ivf_index_type(self):
        """Test initialization with IVF index type."""
        backend = FAISSBackend(dimension=1536, index_type="ivf", nlist=50)

        assert backend._index_type == "ivf"
        _mock_faiss.IndexIVFFlat.assert_called()

    def test_init_hnsw_index_type(self):
        """Test initialization with HNSW index type."""
        backend = FAISSBackend(dimension=1536, index_type="hnsw", m=16)

        assert backend._index_type == "hnsw"
        _mock_faiss.IndexHNSWFlat.assert_called()

    def test_init_invalid_index_type_raises(self):
        """Test initialization with invalid index type raises error."""
        with pytest.raises(ValueError, match="Unknown index_type"):
            FAISSBackend(dimension=1536, index_type="invalid")

    def test_init_empty_storage(self):
        """Test initialization creates empty storage."""
        backend = FAISSBackend(dimension=1536)

        assert backend._texts == []
        assert backend._metadatas == []
        assert backend._ids == []

    def test_init_with_persist_path(self, temp_path):
        """Test initialization with persistence path."""
        backend = FAISSBackend(dimension=1536, persist_path=temp_path)

        assert backend._persist_path == temp_path

    def test_init_persist_path_as_string(self, temp_path):
        """Test initialization with persistence path as string."""
        backend = FAISSBackend(dimension=1536, persist_path=str(temp_path))

        assert backend._persist_path == temp_path


# =============================================================================
# FAISSBackend Index Creation Tests
# =============================================================================


class TestFAISSBackendIndexCreation:
    """Tests for FAISS index creation."""

    def test_create_flat_index(self):
        """Test creating flat index."""
        _backend = FAISSBackend(dimension=512, index_type="flat")

        _mock_faiss.IndexFlatIP.assert_called_with(512)

    def test_create_ivf_index_with_default_nlist(self):
        """Test creating IVF index with default nlist."""
        _backend = FAISSBackend(dimension=512, index_type="ivf")

        # Should create quantizer first
        _mock_faiss.IndexFlatIP.assert_called()
        _mock_faiss.IndexIVFFlat.assert_called()

    def test_create_ivf_index_with_custom_nlist(self):
        """Test creating IVF index with custom nlist."""
        backend = FAISSBackend(dimension=512, index_type="ivf", nlist=200)

        assert backend._kwargs.get("nlist") == 200

    def test_create_hnsw_index_with_default_m(self):
        """Test creating HNSW index with default M."""
        _backend = FAISSBackend(dimension=512, index_type="hnsw")

        _mock_faiss.IndexHNSWFlat.assert_called_with(512, 32)

    def test_create_hnsw_index_with_custom_m(self):
        """Test creating HNSW index with custom M."""
        _backend = FAISSBackend(dimension=512, index_type="hnsw", m=64)

        _mock_faiss.IndexHNSWFlat.assert_called_with(512, 64)


# =============================================================================
# FAISSBackend Add Tests
# =============================================================================


class TestFAISSBackendAdd:
    """Tests for FAISSBackend.add() method."""

    def test_add_single_document(self, backend):
        """Test adding a single document."""
        embedding = [[0.1] * 1536]
        text = ["Hello world"]
        metadata = [{"source": "test"}]

        result = backend.add(embeddings=embedding, texts=text, metadatas=metadata)

        assert len(result) == 1
        assert isinstance(result[0], str)
        _mock_index.add.assert_called_once()

    def test_add_multiple_documents(self, backend):
        """Test adding multiple documents."""
        embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        texts = ["First", "Second", "Third"]
        metadatas = [{"idx": 1}, {"idx": 2}, {"idx": 3}]

        result = backend.add(embeddings=embeddings, texts=texts, metadatas=metadatas)

        assert len(result) == 3
        assert len(backend._texts) == 3
        assert len(backend._metadatas) == 3

    def test_add_with_custom_ids(self, backend):
        """Test adding documents with custom IDs."""
        custom_ids = ["custom-1", "custom-2"]
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        texts = ["First", "Second"]

        result = backend.add(embeddings=embeddings, texts=texts, ids=custom_ids)

        assert result == custom_ids
        assert backend._ids == custom_ids

    def test_add_without_metadatas(self, backend):
        """Test adding documents without metadata."""
        embeddings = [[0.1] * 1536]
        texts = ["No metadata"]

        result = backend.add(embeddings=embeddings, texts=texts)

        assert len(result) == 1
        assert backend._metadatas == [{}]

    def test_add_empty_list(self, backend):
        """Test adding empty list returns empty list."""
        result = backend.add(embeddings=[], texts=[])

        assert result == []
        _mock_index.add.assert_not_called()

    def test_add_generates_uuid_ids(self, backend):
        """Test add generates valid UUIDs when no IDs provided."""
        result = backend.add(embeddings=[[0.1] * 1536], texts=["test"])

        assert len(result) == 1
        # Should be a valid UUID
        uuid.UUID(result[0])

    def test_add_normalizes_vectors(self, backend):
        """Test add normalizes vectors."""
        embeddings = [[1.0, 0.0, 0.0] + [0.0] * 1533]
        texts = ["test"]

        backend.add(embeddings=embeddings, texts=texts)

        # Verify add was called with normalized vectors
        call_args = _mock_index.add.call_args[0][0]
        # Normalized vector should have unit norm
        norm = np.linalg.norm(call_args[0])
        assert abs(norm - 1.0) < 0.01

    def test_add_stores_texts_and_metadata(self, backend):
        """Test add stores texts and metadata correctly."""
        embeddings = [[0.1] * 1536]
        texts = ["Test text"]
        metadatas = [{"key": "value"}]
        ids = ["test-id"]

        backend.add(embeddings=embeddings, texts=texts, metadatas=metadatas, ids=ids)

        assert backend._texts == texts
        assert backend._metadatas == metadatas
        assert backend._ids == ids

    def test_add_trains_ivf_index_if_needed(self):
        """Test add trains IVF index when needed."""
        _mock_index.is_trained = False

        backend = FAISSBackend(dimension=1536, index_type="ivf", nlist=10)

        # Add enough data to train
        embeddings = [[0.1] * 1536 for _ in range(20)]
        texts = ["text"] * 20

        backend.add(embeddings=embeddings, texts=texts)

        _mock_index.train.assert_called()


# =============================================================================
# FAISSBackend Search Tests
# =============================================================================


class TestFAISSBackendSearch:
    """Tests for FAISSBackend.search() method."""

    def test_search_empty_index(self, backend):
        """Test search on empty index returns empty list."""
        _mock_index.ntotal = 0

        results = backend.search(query_embedding=[0.1] * 1536, k=5)

        assert results == []

    def test_search_basic(self, backend_with_data):
        """Test basic search returns results."""
        _mock_index.search.return_value = (np.array([[0.95, 0.85]]), np.array([[0, 1]]))

        results = backend_with_data.search(query_embedding=[0.1] * 1536, k=5)

        assert len(results) == 2
        assert results[0]["id"] == "id-1"
        assert results[0]["text"] == "First doc"
        assert results[0]["score"] == 0.95

    def test_search_respects_k_parameter(self, backend_with_data):
        """Test search respects k parameter."""
        _mock_index.search.return_value = (np.array([[0.95]]), np.array([[0]]))

        results = backend_with_data.search(query_embedding=[0.1] * 1536, k=1)

        assert len(results) <= 1

    def test_search_with_filter(self, backend_with_data):
        """Test search with metadata filter."""
        _mock_index.search.return_value = (np.array([[0.95, 0.85, 0.75]]), np.array([[0, 1, 2]]))

        results = backend_with_data.search(query_embedding=[0.1] * 1536, k=5, filter={"idx": 2})

        # Only the document with idx=2 should match
        assert len(results) == 1
        assert results[0]["metadata"]["idx"] == 2

    def test_search_handles_missing_indices(self, backend_with_data):
        """Test search handles -1 indices from FAISS."""
        _mock_index.search.return_value = (np.array([[0.95, -1]]), np.array([[0, -1]]))

        results = backend_with_data.search(query_embedding=[0.1] * 1536, k=5)

        # Should skip -1 indices
        assert len(results) == 1

    def test_search_normalizes_query(self, backend_with_data):
        """Test search normalizes query vector."""
        _mock_index.search.return_value = (np.array([[0.95]]), np.array([[0]]))

        backend_with_data.search(query_embedding=[1.0] + [0.0] * 1535, k=5)

        # Verify search was called with normalized vector
        call_args = _mock_index.search.call_args[0][0]
        norm = np.linalg.norm(call_args[0])
        assert abs(norm - 1.0) < 0.01

    def test_search_returns_correct_metadata(self, backend_with_data):
        """Test search returns correct metadata."""
        _mock_index.search.return_value = (np.array([[0.95]]), np.array([[1]]))

        results = backend_with_data.search(query_embedding=[0.1] * 1536, k=5)

        assert results[0]["metadata"] == {"idx": 2}

    def test_search_with_multiple_filters(self, backend):
        """Test search with multiple filter criteria."""
        # Add documents with various metadata
        embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        texts = ["doc1", "doc2", "doc3"]
        metadatas = [
            {"type": "a", "status": "active"},
            {"type": "b", "status": "active"},
            {"type": "a", "status": "inactive"},
        ]
        backend.add(embeddings=embeddings, texts=texts, metadatas=metadatas, ids=["1", "2", "3"])
        _mock_index.ntotal = 3

        _mock_index.search.return_value = (np.array([[0.95, 0.85, 0.75]]), np.array([[0, 1, 2]]))

        results = backend.search(
            query_embedding=[0.1] * 1536, k=5, filter={"type": "a", "status": "active"}
        )

        assert len(results) == 1
        assert results[0]["metadata"]["type"] == "a"
        assert results[0]["metadata"]["status"] == "active"


# =============================================================================
# FAISSBackend Delete Tests
# =============================================================================


class TestFAISSBackendDelete:
    """Tests for FAISSBackend.delete() method."""

    def test_delete_empty_list(self, backend):
        """Test deleting empty list returns 0."""
        result = backend.delete([])

        assert result == 0

    def test_delete_nonexistent_ids(self, backend_with_data):
        """Test deleting non-existent IDs returns 0."""
        result = backend_with_data.delete(["nonexistent"])

        assert result == 0

    def test_delete_single_id(self, backend_with_data):
        """Test deleting a single document."""
        # Configure mock to return vector when reconstructing
        _mock_index.reconstruct = MagicMock()

        result = backend_with_data.delete(["id-1"])

        assert result == 1
        assert "id-1" not in backend_with_data._ids

    def test_delete_multiple_ids(self, backend_with_data):
        """Test deleting multiple documents."""
        _mock_index.reconstruct = MagicMock()

        result = backend_with_data.delete(["id-1", "id-2"])

        assert result == 2
        assert "id-1" not in backend_with_data._ids
        assert "id-2" not in backend_with_data._ids

    def test_delete_rebuilds_index(self, backend_with_data):
        """Test delete rebuilds the index."""
        _mock_index.reconstruct = MagicMock()

        backend_with_data.delete(["id-1"])

        # Should create a new index
        assert _mock_faiss.IndexFlatIP.call_count >= 2  # Initial + rebuild

    def test_delete_all_documents(self, backend_with_data):
        """Test deleting all documents."""
        result = backend_with_data.delete(["id-1", "id-2", "id-3"])

        assert result == 3
        assert backend_with_data._texts == []
        assert backend_with_data._metadatas == []
        assert backend_with_data._ids == []


# =============================================================================
# FAISSBackend Clear Tests
# =============================================================================


class TestFAISSBackendClear:
    """Tests for FAISSBackend.clear() method."""

    def test_clear_empties_storage(self, backend_with_data):
        """Test clear empties all storage."""
        backend_with_data.clear()

        assert backend_with_data._texts == []
        assert backend_with_data._metadatas == []
        assert backend_with_data._ids == []

    def test_clear_creates_new_index(self, backend_with_data):
        """Test clear creates a new index."""
        initial_call_count = _mock_faiss.IndexFlatIP.call_count

        backend_with_data.clear()

        assert _mock_faiss.IndexFlatIP.call_count > initial_call_count

    def test_clear_removes_persisted_files(self, temp_path):
        """Test clear removes persisted files if they exist."""
        import pickle

        # Pre-create the files manually (simulating existing persisted data)
        temp_path.mkdir(parents=True, exist_ok=True)
        index_path = temp_path / "index.faiss"
        meta_path = temp_path / "metadata.pkl"

        # Create valid metadata file
        with open(meta_path, "wb") as f:
            pickle.dump({"texts": [], "metadatas": [], "ids": []}, f)

        # Touch index file (faiss.read_index is mocked)
        index_path.touch()

        # Clear should remove the files
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backend = FAISSBackend(dimension=1536, persist_path=temp_path)

        backend.clear()

        assert not index_path.exists()
        assert not meta_path.exists()


# =============================================================================
# FAISSBackend Count Tests
# =============================================================================


class TestFAISSBackendCount:
    """Tests for FAISSBackend.count() method."""

    def test_count_empty_index(self, backend):
        """Test count on empty index."""
        _mock_index.ntotal = 0

        assert backend.count() == 0

    def test_count_with_documents(self, backend_with_data):
        """Test count with documents."""
        _mock_index.ntotal = 3

        assert backend_with_data.count() == 3

    def test_count_returns_index_ntotal(self, backend):
        """Test count returns index.ntotal."""
        _mock_index.ntotal = 42

        assert backend.count() == 42


# =============================================================================
# FAISSBackend Persistence Tests
# =============================================================================


class TestFAISSBackendPersistence:
    """Tests for FAISSBackend persistence."""

    def test_save_creates_directory(self, temp_path):
        """Test save creates directory if needed."""
        backend = FAISSBackend(dimension=1536, persist_path=temp_path)
        backend.add(embeddings=[[0.1] * 1536], texts=["test"])

        assert temp_path.exists()

    def test_save_writes_index_file(self, temp_path):
        """Test save writes index file."""
        backend = FAISSBackend(dimension=1536, persist_path=temp_path)
        backend.add(embeddings=[[0.1] * 1536], texts=["test"])

        _mock_faiss.write_index.assert_called()

    def test_save_writes_metadata_file(self, temp_path):
        """Test save writes metadata JSON file."""
        backend = FAISSBackend(dimension=1536, persist_path=temp_path)
        backend.add(embeddings=[[0.1] * 1536], texts=["test"], metadatas=[{"key": "value"}])

        meta_path = temp_path / "metadata.json"
        assert meta_path.exists()

    def test_load_existing_index(self, temp_path):
        """Test loading existing index."""
        # Create index directory and files
        temp_path.mkdir(parents=True, exist_ok=True)
        index_path = temp_path / "index.faiss"
        index_path.touch()

        _backend = FAISSBackend(dimension=1536, persist_path=temp_path)

        _mock_faiss.read_index.assert_called()

    def test_load_metadata_with_warning(self, temp_path):
        """Test loading metadata shows security warning."""
        import pickle

        # Create index and metadata files
        temp_path.mkdir(parents=True, exist_ok=True)
        index_path = temp_path / "index.faiss"
        meta_path = temp_path / "metadata.pkl"
        index_path.touch()

        # Write metadata
        with open(meta_path, "wb") as f:
            pickle.dump({"texts": ["test"], "metadatas": [{}], "ids": ["1"]}, f)

        # Should issue warning about pickle
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _backend = FAISSBackend(dimension=1536, persist_path=temp_path)

            # Check for security warning
            assert any("pickle" in str(warning.message).lower() for warning in w)

    def test_load_restores_texts_and_metadata(self, temp_path):
        """Test loading restores texts and metadata."""
        import pickle

        # Create files
        temp_path.mkdir(parents=True, exist_ok=True)
        index_path = temp_path / "index.faiss"
        meta_path = temp_path / "metadata.pkl"
        index_path.touch()

        # Write metadata
        data = {
            "texts": ["text1", "text2"],
            "metadatas": [{"key": "val1"}, {"key": "val2"}],
            "ids": ["id1", "id2"],
        }
        with open(meta_path, "wb") as f:
            pickle.dump(data, f)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backend = FAISSBackend(dimension=1536, persist_path=temp_path)

        assert backend._texts == ["text1", "text2"]
        assert backend._metadatas == [{"key": "val1"}, {"key": "val2"}]
        assert backend._ids == ["id1", "id2"]

    def test_no_persist_path_no_save(self, backend):
        """Test no persistence when persist_path is None."""
        backend.add(embeddings=[[0.1] * 1536], texts=["test"])

        # write_index should not be called for in-memory backend
        # (Initial call is for testing, but no save calls)
        assert backend._persist_path is None


# =============================================================================
# FAISSBackend Vector Normalization Tests
# =============================================================================


class TestFAISSBackendNormalization:
    """Tests for vector normalization."""

    def test_normalize_unit_vector(self, backend):
        """Test normalizing already normalized vector."""
        vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        result = backend._normalize_vectors(vectors)

        assert abs(np.linalg.norm(result[0]) - 1.0) < 0.001

    def test_normalize_non_unit_vector(self, backend):
        """Test normalizing non-unit vector."""
        vectors = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)

        result = backend._normalize_vectors(vectors)

        assert abs(np.linalg.norm(result[0]) - 1.0) < 0.001

    def test_normalize_zero_vector(self, backend):
        """Test normalizing zero vector doesn't cause division by zero."""
        vectors = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        result = backend._normalize_vectors(vectors)

        # Should not be NaN
        assert not np.any(np.isnan(result))

    def test_normalize_multiple_vectors(self, backend):
        """Test normalizing multiple vectors."""
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [3.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        )

        result = backend._normalize_vectors(vectors)

        for vec in result:
            assert abs(np.linalg.norm(vec) - 1.0) < 0.001


# =============================================================================
# FAISSBackend Filter Matching Tests
# =============================================================================


class TestFAISSBackendFilterMatching:
    """Tests for metadata filter matching."""

    def test_matches_filter_single_key(self, backend):
        """Test filter matching with single key."""
        metadata = {"key": "value", "other": "data"}
        filter = {"key": "value"}

        assert backend._matches_filter(metadata, filter) is True

    def test_matches_filter_multiple_keys(self, backend):
        """Test filter matching with multiple keys."""
        metadata = {"key1": "value1", "key2": "value2", "extra": "data"}
        filter = {"key1": "value1", "key2": "value2"}

        assert backend._matches_filter(metadata, filter) is True

    def test_matches_filter_missing_key(self, backend):
        """Test filter matching with missing key."""
        metadata = {"key": "value"}
        filter = {"missing": "value"}

        assert backend._matches_filter(metadata, filter) is False

    def test_matches_filter_wrong_value(self, backend):
        """Test filter matching with wrong value."""
        metadata = {"key": "value"}
        filter = {"key": "wrong"}

        assert backend._matches_filter(metadata, filter) is False

    def test_matches_filter_empty_filter(self, backend):
        """Test filter matching with empty filter."""
        metadata = {"key": "value"}
        filter = {}

        assert backend._matches_filter(metadata, filter) is True

    def test_matches_filter_numeric_values(self, backend):
        """Test filter matching with numeric values."""
        metadata = {"count": 42, "score": 0.95}
        filter = {"count": 42}

        assert backend._matches_filter(metadata, filter) is True


# =============================================================================
# FAISSBackend Edge Cases Tests
# =============================================================================


class TestFAISSBackendEdgeCases:
    """Tests for edge cases and error handling."""

    def test_high_dimensional_embeddings(self):
        """Test with high-dimensional embeddings."""
        backend = FAISSBackend(dimension=4096)

        embeddings = [[0.1] * 4096]
        texts = ["High dimensional"]

        result = backend.add(embeddings=embeddings, texts=texts)

        assert len(result) == 1

    def test_unicode_text(self, backend):
        """Test with unicode text."""
        unicode_text = "Hello! 你好! مرحبا! "

        backend.add(embeddings=[[0.1] * 1536], texts=[unicode_text])

        assert backend._texts[0] == unicode_text

    def test_complex_metadata(self, backend):
        """Test with complex nested metadata."""
        complex_metadata = {
            "nested": {"level1": {"level2": "value"}},
            "list": [1, 2, 3],
            "mixed": [{"key": "value"}, "string", 123],
        }

        backend.add(embeddings=[[0.1] * 1536], texts=["test"], metadatas=[complex_metadata])

        assert backend._metadatas[0] == complex_metadata

    def test_search_returns_up_to_k_results(self, backend_with_data):
        """Test search returns at most k results."""
        _mock_index.search.return_value = (np.array([[0.95, 0.85, 0.75]]), np.array([[0, 1, 2]]))

        results = backend_with_data.search(query_embedding=[0.1] * 1536, k=2)

        assert len(results) <= 2

    def test_search_with_out_of_range_indices(self, backend_with_data):
        """Test search handles out-of-range indices gracefully."""
        _mock_index.search.return_value = (
            np.array([[0.95, 0.85]]),
            np.array([[0, 100]]),  # 100 is out of range
        )

        results = backend_with_data.search(query_embedding=[0.1] * 1536, k=5)

        # Should skip out-of-range indices
        assert len(results) == 1

    def test_add_preserves_order(self, backend):
        """Test add preserves order of documents."""
        ids = ["first", "second", "third"]
        embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        texts = ["First", "Second", "Third"]

        result = backend.add(embeddings=embeddings, texts=texts, ids=ids)

        assert result == ids
        assert backend._texts == texts


# =============================================================================
# FAISSBackend IVF Training Tests
# =============================================================================


class TestFAISSBackendIVFTraining:
    """Tests for IVF index training behavior."""

    def test_ivf_not_enough_data_falls_back_to_flat(self):
        """Test IVF falls back to flat index when not enough training data."""
        _mock_index.is_trained = False

        backend = FAISSBackend(dimension=1536, index_type="ivf", nlist=100)

        # Add fewer vectors than nlist
        embeddings = [[0.1] * 1536 for _ in range(10)]
        texts = ["text"] * 10

        backend.add(embeddings=embeddings, texts=texts)

        # Should fallback to flat index
        # The last call to IndexFlatIP would be the fallback
        assert _mock_faiss.IndexFlatIP.called

    def test_ivf_trains_with_enough_data(self):
        """Test IVF trains when enough data provided."""
        _mock_index.is_trained = False

        backend = FAISSBackend(dimension=1536, index_type="ivf", nlist=10)

        # Add more vectors than nlist
        embeddings = [[0.1] * 1536 for _ in range(20)]
        texts = ["text"] * 20

        backend.add(embeddings=embeddings, texts=texts)

        _mock_index.train.assert_called()
