"""Unit tests for Qdrant vector storage backend.

Tests cover:
- QdrantBackend initialization (local, remote, API key)
- Document operations (add, search, delete, clear, count)
- Search with filtering (metadata, payload filters)
- Collection management (creation, recreation)
- Edge cases and error handling

Phase 4.3 of ai-infra test plan.
"""

from __future__ import annotations

import os
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Mock Setup - Create mocks at module level before any QdrantBackend import
# =============================================================================

# Create mock qdrant_client module
_mock_qdrant_client = MagicMock()
_mock_client_instance = MagicMock()

# Create mock models
_mock_models = MagicMock()
_mock_distance = MagicMock()
_mock_distance.COSINE = "Cosine"
_mock_models.Distance = _mock_distance
_mock_models.VectorParams = MagicMock()
_mock_models.PointStruct = MagicMock(
    side_effect=lambda id, vector, payload: {"id": id, "vector": vector, "payload": payload}
)
_mock_models.PointIdsList = MagicMock()
_mock_models.FieldCondition = MagicMock()
_mock_models.Filter = MagicMock()
_mock_models.MatchValue = MagicMock()

# Configure client mock
_mock_collection = MagicMock()
_mock_collection.name = "ai_infra_retriever"
_mock_collections_response = MagicMock()
_mock_collections_response.collections = []

_mock_client_instance.get_collections.return_value = _mock_collections_response
_mock_client_instance.create_collection = MagicMock()
_mock_client_instance.delete_collection = MagicMock()
_mock_client_instance.upsert = MagicMock()
_mock_client_instance.search = MagicMock(return_value=[])
_mock_client_instance.delete = MagicMock()

_mock_collection_info = MagicMock()
_mock_collection_info.points_count = 0
_mock_client_instance.get_collection.return_value = _mock_collection_info

_mock_qdrant_client.QdrantClient = MagicMock(return_value=_mock_client_instance)

# Register mocks in sys.modules
sys.modules["qdrant_client"] = _mock_qdrant_client
sys.modules["qdrant_client.models"] = _mock_models

# Now import the module
from ai_infra.retriever.backends.qdrant import QdrantBackend  # noqa: E402

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test."""
    _mock_qdrant_client.reset_mock()
    _mock_client_instance.reset_mock()
    _mock_models.reset_mock()

    # Re-configure mocks after reset
    _mock_collections_response.collections = []
    _mock_client_instance.get_collections.return_value = _mock_collections_response
    _mock_client_instance.search.return_value = []
    _mock_collection_info.points_count = 0
    _mock_client_instance.get_collection.return_value = _mock_collection_info
    _mock_qdrant_client.QdrantClient.return_value = _mock_client_instance

    # Re-configure models
    _mock_models.Distance = _mock_distance
    _mock_models.VectorParams = MagicMock()
    _mock_models.PointStruct = MagicMock(
        side_effect=lambda id, vector, payload: {"id": id, "vector": vector, "payload": payload}
    )
    _mock_models.PointIdsList = MagicMock()
    _mock_models.FieldCondition = MagicMock()
    _mock_models.Filter = MagicMock()
    _mock_models.MatchValue = MagicMock()

    yield


@pytest.fixture
def backend():
    """Create a QdrantBackend instance with mocked dependencies."""
    return QdrantBackend(url="http://localhost:6333")


@pytest.fixture
def backend_with_data(backend):
    """Create a backend with simulated data."""
    # Simulate existing data
    _mock_collection_info.points_count = 3
    return backend


# =============================================================================
# QdrantBackend Initialization Tests
# =============================================================================


class TestQdrantBackendInit:
    """Tests for QdrantBackend initialization."""

    def test_init_with_url(self):
        """Test initialization with URL."""
        _backend = QdrantBackend(url="http://localhost:6333")

        _mock_qdrant_client.QdrantClient.assert_called_with(
            url="http://localhost:6333", api_key=None
        )

    def test_init_with_host_port(self):
        """Test initialization with host and port."""
        _backend = QdrantBackend(host="127.0.0.1", port=6334)

        _mock_qdrant_client.QdrantClient.assert_called_with(
            host="127.0.0.1", port=6334, api_key=None
        )

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        _backend = QdrantBackend(url="https://cloud.qdrant.io", api_key="test-key")

        _mock_qdrant_client.QdrantClient.assert_called_with(
            url="https://cloud.qdrant.io", api_key="test-key"
        )

    def test_init_api_key_from_env(self):
        """Test initialization reads API key from environment."""
        with patch.dict(os.environ, {"QDRANT_API_KEY": "env-api-key"}):
            _backend = QdrantBackend(url="https://cloud.qdrant.io")

            _mock_qdrant_client.QdrantClient.assert_called_with(
                url="https://cloud.qdrant.io", api_key="env-api-key"
            )

    def test_init_custom_collection_name(self):
        """Test initialization with custom collection name."""
        backend = QdrantBackend(url="http://localhost:6333", collection_name="my_collection")

        assert backend._collection_name == "my_collection"

    def test_init_custom_embedding_dimension(self):
        """Test initialization with custom embedding dimension."""
        backend = QdrantBackend(url="http://localhost:6333", embedding_dimension=768)

        assert backend._embedding_dimension == 768

    def test_init_default_collection_name(self):
        """Test initialization uses default collection name."""
        backend = QdrantBackend(url="http://localhost:6333")

        assert backend._collection_name == "ai_infra_retriever"

    def test_init_default_embedding_dimension(self):
        """Test initialization uses default embedding dimension."""
        backend = QdrantBackend(url="http://localhost:6333")

        assert backend._embedding_dimension == 1536


# =============================================================================
# QdrantBackend Collection Management Tests
# =============================================================================


class TestQdrantBackendCollectionManagement:
    """Tests for collection management."""

    def test_creates_collection_if_not_exists(self):
        """Test collection is created if it doesn't exist."""
        _mock_collections_response.collections = []

        _backend = QdrantBackend(url="http://localhost:6333", collection_name="new_collection")

        _mock_client_instance.create_collection.assert_called_once()

    def test_skips_creation_if_collection_exists(self):
        """Test collection creation is skipped if collection exists."""
        _mock_existing_collection = MagicMock()
        _mock_existing_collection.name = "existing_collection"
        _mock_collections_response.collections = [_mock_existing_collection]

        _backend = QdrantBackend(url="http://localhost:6333", collection_name="existing_collection")

        _mock_client_instance.create_collection.assert_not_called()

    def test_creates_collection_with_correct_params(self):
        """Test collection is created with correct vector params."""
        _mock_collections_response.collections = []

        _backend = QdrantBackend(
            url="http://localhost:6333",
            collection_name="test_collection",
            embedding_dimension=768,
        )

        _mock_client_instance.create_collection.assert_called_once()
        call_kwargs = _mock_client_instance.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test_collection"


# =============================================================================
# QdrantBackend Add Tests
# =============================================================================


class TestQdrantBackendAdd:
    """Tests for QdrantBackend.add() method."""

    def test_add_single_document(self, backend):
        """Test adding a single document."""
        embedding = [[0.1] * 1536]
        text = ["Hello world"]
        metadata = [{"source": "test"}]

        result = backend.add(embeddings=embedding, texts=text, metadatas=metadata)

        assert len(result) == 1
        _mock_client_instance.upsert.assert_called_once()

    def test_add_multiple_documents(self, backend):
        """Test adding multiple documents."""
        embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        texts = ["First", "Second", "Third"]
        metadatas = [{"idx": 1}, {"idx": 2}, {"idx": 3}]

        result = backend.add(embeddings=embeddings, texts=texts, metadatas=metadatas)

        assert len(result) == 3
        _mock_client_instance.upsert.assert_called_once()

    def test_add_with_custom_ids(self, backend):
        """Test adding documents with custom IDs."""
        custom_ids = ["custom-1", "custom-2"]
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        texts = ["First", "Second"]

        result = backend.add(embeddings=embeddings, texts=texts, ids=custom_ids)

        assert result == custom_ids

    def test_add_without_metadatas(self, backend):
        """Test adding documents without metadata."""
        embeddings = [[0.1] * 1536]
        texts = ["No metadata"]

        result = backend.add(embeddings=embeddings, texts=texts)

        assert len(result) == 1

    def test_add_empty_list(self, backend):
        """Test adding empty list returns empty list."""
        result = backend.add(embeddings=[], texts=[])

        assert result == []
        _mock_client_instance.upsert.assert_not_called()

    def test_add_generates_uuid_ids(self, backend):
        """Test add generates valid UUIDs when no IDs provided."""
        result = backend.add(embeddings=[[0.1] * 1536], texts=["test"])

        assert len(result) == 1
        # Should be a valid UUID
        uuid.UUID(result[0])

    def test_add_stores_text_in_payload(self, backend):
        """Test add stores text in payload."""
        embeddings = [[0.1] * 1536]
        texts = ["Test text"]

        backend.add(embeddings=embeddings, texts=texts)

        # Verify upsert was called
        _mock_client_instance.upsert.assert_called_once()
        call_kwargs = _mock_client_instance.upsert.call_args[1]
        assert call_kwargs["collection_name"] == backend._collection_name

    def test_add_includes_metadata_in_payload(self, backend):
        """Test add includes metadata in payload."""
        embeddings = [[0.1] * 1536]
        texts = ["Test"]
        metadatas = [{"key": "value", "source": "test"}]

        backend.add(embeddings=embeddings, texts=texts, metadatas=metadatas)

        _mock_client_instance.upsert.assert_called_once()


# =============================================================================
# QdrantBackend Search Tests
# =============================================================================


class TestQdrantBackendSearch:
    """Tests for QdrantBackend.search() method."""

    def test_search_empty_results(self, backend):
        """Test search returns empty list when no results."""
        _mock_client_instance.search.return_value = []

        results = backend.search(query_embedding=[0.1] * 1536, k=5)

        assert results == []

    def test_search_returns_results(self, backend):
        """Test search returns formatted results."""
        mock_hit = MagicMock()
        mock_hit.id = "doc-1"
        mock_hit.score = 0.95
        mock_hit.payload = {"text": "Hello world", "source": "test"}

        _mock_client_instance.search.return_value = [mock_hit]

        results = backend.search(query_embedding=[0.1] * 1536, k=5)

        assert len(results) == 1
        assert results[0]["id"] == "doc-1"
        assert results[0]["text"] == "Hello world"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"] == {"source": "test"}

    def test_search_respects_k_parameter(self, backend):
        """Test search respects k parameter."""
        backend.search(query_embedding=[0.1] * 1536, k=10)

        call_kwargs = _mock_client_instance.search.call_args[1]
        assert call_kwargs["limit"] == 10

    def test_search_with_filter(self, backend):
        """Test search with metadata filter."""
        backend.search(query_embedding=[0.1] * 1536, k=5, filter={"source": "test"})

        call_kwargs = _mock_client_instance.search.call_args[1]
        assert call_kwargs["query_filter"] is not None

    def test_search_without_filter(self, backend):
        """Test search without filter passes None."""
        backend.search(query_embedding=[0.1] * 1536, k=5)

        call_kwargs = _mock_client_instance.search.call_args[1]
        assert call_kwargs["query_filter"] is None

    def test_search_multiple_results(self, backend):
        """Test search with multiple results."""
        mock_hits = []
        for i in range(3):
            hit = MagicMock()
            hit.id = f"doc-{i}"
            hit.score = 0.9 - (i * 0.1)
            hit.payload = {"text": f"Document {i}"}
            mock_hits.append(hit)

        _mock_client_instance.search.return_value = mock_hits

        results = backend.search(query_embedding=[0.1] * 1536, k=5)

        assert len(results) == 3
        assert results[0]["score"] > results[1]["score"] > results[2]["score"]

    def test_search_with_empty_payload(self, backend):
        """Test search handles empty payload."""
        mock_hit = MagicMock()
        mock_hit.id = "doc-1"
        mock_hit.score = 0.95
        mock_hit.payload = None

        _mock_client_instance.search.return_value = [mock_hit]

        results = backend.search(query_embedding=[0.1] * 1536, k=5)

        assert results[0]["text"] == ""
        assert results[0]["metadata"] == {}

    def test_search_converts_id_to_string(self, backend):
        """Test search converts ID to string."""
        mock_hit = MagicMock()
        mock_hit.id = 12345  # Numeric ID
        mock_hit.score = 0.95
        mock_hit.payload = {"text": "Test"}

        _mock_client_instance.search.return_value = [mock_hit]

        results = backend.search(query_embedding=[0.1] * 1536, k=5)

        assert results[0]["id"] == "12345"
        assert isinstance(results[0]["id"], str)


# =============================================================================
# QdrantBackend Filter Tests
# =============================================================================


class TestQdrantBackendFiltering:
    """Tests for Qdrant filtering functionality."""

    def test_filter_single_condition(self, backend):
        """Test filter with single condition."""
        backend.search(query_embedding=[0.1] * 1536, k=5, filter={"source": "wikipedia"})

        # Verify FieldCondition was called
        _mock_models.FieldCondition.assert_called()

    def test_filter_multiple_conditions(self, backend):
        """Test filter with multiple conditions."""
        backend.search(
            query_embedding=[0.1] * 1536, k=5, filter={"source": "wikipedia", "language": "en"}
        )

        # Should create multiple FieldConditions
        assert _mock_models.FieldCondition.call_count == 2

    def test_filter_creates_must_filter(self, backend):
        """Test filter creates 'must' filter (AND logic)."""
        backend.search(query_embedding=[0.1] * 1536, k=5, filter={"key": "value"})

        # Verify Filter was called with must conditions
        _mock_models.Filter.assert_called()


# =============================================================================
# QdrantBackend Delete Tests
# =============================================================================


class TestQdrantBackendDelete:
    """Tests for QdrantBackend.delete() method."""

    def test_delete_empty_list(self, backend):
        """Test deleting empty list returns 0."""
        result = backend.delete([])

        assert result == 0
        _mock_client_instance.delete.assert_not_called()

    def test_delete_single_id(self, backend):
        """Test deleting a single document."""
        result = backend.delete(["doc-1"])

        assert result == 1
        _mock_client_instance.delete.assert_called_once()

    def test_delete_multiple_ids(self, backend):
        """Test deleting multiple documents."""
        result = backend.delete(["doc-1", "doc-2", "doc-3"])

        assert result == 3
        _mock_client_instance.delete.assert_called_once()

    def test_delete_uses_points_selector(self, backend):
        """Test delete uses PointIdsList selector."""
        ids_to_delete = ["id-1", "id-2"]

        backend.delete(ids_to_delete)

        _mock_models.PointIdsList.assert_called_with(points=ids_to_delete)

    def test_delete_uses_correct_collection(self, backend):
        """Test delete uses correct collection name."""
        backend.delete(["doc-1"])

        call_kwargs = _mock_client_instance.delete.call_args[1]
        assert call_kwargs["collection_name"] == backend._collection_name


# =============================================================================
# QdrantBackend Clear Tests
# =============================================================================


class TestQdrantBackendClear:
    """Tests for QdrantBackend.clear() method."""

    def test_clear_deletes_collection(self, backend):
        """Test clear deletes the collection."""
        backend.clear()

        _mock_client_instance.delete_collection.assert_called_once_with(backend._collection_name)

    def test_clear_recreates_collection(self, backend):
        """Test clear recreates the collection."""
        backend.clear()

        _mock_client_instance.create_collection.assert_called()

    def test_clear_preserves_dimension(self, backend):
        """Test clear preserves embedding dimension."""
        original_dim = backend._embedding_dimension

        backend.clear()

        # Collection should be recreated with same dimension
        _mock_client_instance.create_collection.assert_called()
        # The dimension should still be the same
        assert backend._embedding_dimension == original_dim


# =============================================================================
# QdrantBackend Count Tests
# =============================================================================


class TestQdrantBackendCount:
    """Tests for QdrantBackend.count() method."""

    def test_count_empty_collection(self, backend):
        """Test count on empty collection."""
        _mock_collection_info.points_count = 0

        assert backend.count() == 0

    def test_count_with_documents(self, backend):
        """Test count with documents."""
        _mock_collection_info.points_count = 42

        assert backend.count() == 42

    def test_count_returns_integer(self, backend):
        """Test count returns integer."""
        _mock_collection_info.points_count = 10

        result = backend.count()

        assert isinstance(result, int)

    def test_count_handles_none_points_count(self, backend):
        """Test count handles None points_count."""
        _mock_collection_info.points_count = None

        result = backend.count()

        assert result == 0


# =============================================================================
# QdrantBackend Edge Cases Tests
# =============================================================================


class TestQdrantBackendEdgeCases:
    """Tests for edge cases and error handling."""

    def test_high_dimensional_embeddings(self):
        """Test with high-dimensional embeddings."""
        backend = QdrantBackend(url="http://localhost:6333", embedding_dimension=4096)

        embeddings = [[0.1] * 4096]
        texts = ["High dimensional"]

        result = backend.add(embeddings=embeddings, texts=texts)

        assert len(result) == 1

    def test_unicode_text(self, backend):
        """Test with unicode text."""
        unicode_text = "Hello! 你好! مرحبا! 🎉"

        backend.add(embeddings=[[0.1] * 1536], texts=[unicode_text])

        _mock_client_instance.upsert.assert_called_once()

    def test_complex_metadata(self, backend):
        """Test with complex nested metadata."""
        complex_metadata = {
            "nested": {"level1": {"level2": "value"}},
            "list": [1, 2, 3],
            "mixed": [{"key": "value"}, "string", 123],
        }

        backend.add(embeddings=[[0.1] * 1536], texts=["test"], metadatas=[complex_metadata])

        _mock_client_instance.upsert.assert_called_once()

    def test_large_batch_add(self, backend):
        """Test adding large batch of documents."""
        count = 100
        embeddings = [[0.1] * 1536 for _ in range(count)]
        texts = [f"Document {i}" for i in range(count)]

        result = backend.add(embeddings=embeddings, texts=texts)

        assert len(result) == count

    def test_search_score_is_float(self, backend):
        """Test search score is converted to float."""
        mock_hit = MagicMock()
        mock_hit.id = "doc-1"
        mock_hit.score = 0.95  # Could be numpy float
        mock_hit.payload = {"text": "Test"}

        _mock_client_instance.search.return_value = [mock_hit]

        results = backend.search(query_embedding=[0.1] * 1536, k=5)

        assert isinstance(results[0]["score"], float)


# =============================================================================
# QdrantBackend Qdrant Cloud Tests
# =============================================================================


class TestQdrantBackendCloud:
    """Tests for Qdrant Cloud specific functionality."""

    def test_cloud_url_connection(self):
        """Test connection to Qdrant Cloud URL."""
        _backend = QdrantBackend(url="https://your-cluster.qdrant.io", api_key="cloud-api-key")

        _mock_qdrant_client.QdrantClient.assert_called_with(
            url="https://your-cluster.qdrant.io", api_key="cloud-api-key"
        )

    def test_cloud_api_key_required(self):
        """Test cloud connection with API key."""
        with patch.dict(os.environ, {"QDRANT_API_KEY": "env-key"}):
            _backend = QdrantBackend(url="https://cloud.qdrant.io")

            call_args = _mock_qdrant_client.QdrantClient.call_args[1]
            assert call_args["api_key"] == "env-key"


# =============================================================================
# QdrantBackend Local Mode Tests
# =============================================================================


class TestQdrantBackendLocalMode:
    """Tests for local/self-hosted mode."""

    def test_local_default_host_port(self):
        """Test local mode with default host and port."""
        _backend = QdrantBackend()

        _mock_qdrant_client.QdrantClient.assert_called_with(
            host="localhost", port=6333, api_key=None
        )

    def test_local_custom_host(self):
        """Test local mode with custom host."""
        _backend = QdrantBackend(host="192.168.1.100")

        _mock_qdrant_client.QdrantClient.assert_called_with(
            host="192.168.1.100", port=6333, api_key=None
        )

    def test_local_custom_port(self):
        """Test local mode with custom port."""
        _backend = QdrantBackend(port=6334)

        _mock_qdrant_client.QdrantClient.assert_called_with(
            host="localhost", port=6334, api_key=None
        )

    def test_url_takes_precedence_over_host_port(self):
        """Test URL takes precedence over host/port."""
        _backend = QdrantBackend(url="http://custom:9999", host="localhost", port=6333)

        _mock_qdrant_client.QdrantClient.assert_called_with(url="http://custom:9999", api_key=None)


# =============================================================================
# QdrantBackend Result Format Tests
# =============================================================================


class TestQdrantBackendResultFormat:
    """Tests for search result format."""

    def test_result_has_required_keys(self, backend):
        """Test result dict has all required keys."""
        mock_hit = MagicMock()
        mock_hit.id = "doc-1"
        mock_hit.score = 0.95
        mock_hit.payload = {"text": "Hello"}

        _mock_client_instance.search.return_value = [mock_hit]

        results = backend.search(query_embedding=[0.1] * 1536, k=5)

        required_keys = {"id", "text", "score", "metadata"}
        assert required_keys.issubset(results[0].keys())

    def test_text_extracted_from_payload(self, backend):
        """Test text is extracted and removed from metadata."""
        mock_hit = MagicMock()
        mock_hit.id = "doc-1"
        mock_hit.score = 0.95
        mock_hit.payload = {"text": "Extracted text", "other": "data"}

        _mock_client_instance.search.return_value = [mock_hit]

        results = backend.search(query_embedding=[0.1] * 1536, k=5)

        assert results[0]["text"] == "Extracted text"
        assert "text" not in results[0]["metadata"]
        assert results[0]["metadata"]["other"] == "data"

    def test_missing_text_defaults_to_empty(self, backend):
        """Test missing text defaults to empty string."""
        mock_hit = MagicMock()
        mock_hit.id = "doc-1"
        mock_hit.score = 0.95
        mock_hit.payload = {"other": "data"}  # No "text" key

        _mock_client_instance.search.return_value = [mock_hit]

        results = backend.search(query_embedding=[0.1] * 1536, k=5)

        assert results[0]["text"] == ""
