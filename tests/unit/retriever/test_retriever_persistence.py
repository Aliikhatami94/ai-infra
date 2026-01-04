"""Tests for Retriever persistence (save/load).

Tests cover:
- Saving and loading retriever state (v2 JSON + numpy format)
- State integrity after load
- Error handling for invalid paths
- Metadata preservation
- Legacy pickle format migration
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_infra.retriever.retriever import Retriever


class TestRetrieverSave:
    """Tests for Retriever.save() method."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings for testing."""
        mock = MagicMock()
        mock.embed.return_value = [0.1, 0.2, 0.3]
        mock.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock.provider = "huggingface"
        mock.model = "test-model"
        return mock

    @pytest.fixture
    def retriever_with_data(self, mock_embeddings: MagicMock) -> Retriever:
        """Create a retriever with test data."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings
            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings
            r.add_text("Paris is the capital of France")
            return r

    def test_save_creates_directory(self, retriever_with_data: Retriever) -> None:
        """Test that save creates a directory with state.json and embeddings.npy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_retriever"
            result = retriever_with_data.save(path)

            assert result.is_dir()
            assert (result / "state.json").exists()
            assert (result / "embeddings.npy").exists()

    def test_save_creates_valid_json(self, retriever_with_data: Retriever) -> None:
        """Test that save creates a valid JSON state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_retriever"
            result = retriever_with_data.save(path)

            state_path = result / "state.json"
            with open(state_path) as f:
                state = json.load(f)

            assert state["version"] == 2
            assert "created_at" in state
            assert "backend_name" in state
            assert state["backend_name"] == "memory"
            assert "backend_data" in state
            assert "ids" in state["backend_data"]
            assert "texts" in state["backend_data"]

    def test_save_to_directory_without_suffix(self, retriever_with_data: Retriever) -> None:
        """Test that saving to a path without suffix creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = retriever_with_data.save(tmpdir)

            assert result.is_dir()
            assert (result / "state.json").exists()

    def test_save_handles_legacy_pkl_path(self, retriever_with_data: Retriever) -> None:
        """Test that save with .pkl path converts to directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"
            result = retriever_with_data.save(path)

            # Should create directory named 'test' (without .pkl)
            expected_dir = Path(tmpdir) / "test"
            assert result == expected_dir
            assert result.is_dir()
            assert (result / "state.json").exists()

    def test_save_creates_parent_directories(self, retriever_with_data: Retriever) -> None:
        """Test that save creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "retriever"
            result = retriever_with_data.save(path)

            assert result.exists()
            assert result.is_dir()

    def test_save_fails_for_non_memory_backend(self) -> None:
        """Test that save fails for backends that don't support it."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            mock_emb = MagicMock()
            mock_emb.provider = "huggingface"
            mock_emb.model = "test"
            MockEmb.return_value = mock_emb

            with patch("ai_infra.retriever.backends.get_backend") as mock_backend:
                mock_backend.return_value = MagicMock()
                mock_backend.return_value.__class__.__name__ = "PostgresBackend"

                r = Retriever(auto_configure=False, backend="memory")
                r._embeddings = mock_emb
                r._backend = MagicMock()
                r._backend_name = "postgres"

                with tempfile.TemporaryDirectory() as tmpdir:
                    with pytest.raises(ValueError, match="doesn't support save"):
                        r.save(Path(tmpdir) / "test")

    def test_save_includes_embeddings_config(self, retriever_with_data: Retriever) -> None:
        """Test that saved state includes embeddings configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test"
            retriever_with_data.save(path)

            with open(path / "state.json") as f:
                state = json.load(f)

            assert "embeddings_provider" in state
            assert "embeddings_model" in state

    def test_save_includes_chunk_count(self, retriever_with_data: Retriever) -> None:
        """Test that saved state includes chunk count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test"
            retriever_with_data.save(path)

            with open(path / "state.json") as f:
                state = json.load(f)

            assert "chunk_count" in state
            assert state["chunk_count"] >= 1


class TestRetrieverLoad:
    """Tests for Retriever.load() class method."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings for testing."""
        mock = MagicMock()
        mock.embed.return_value = [0.1, 0.2, 0.3]
        mock.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock.provider = "huggingface"
        mock.model = "test-model"
        return mock

    @pytest.fixture
    def saved_retriever_path(self, mock_embeddings: MagicMock) -> Path:
        """Create a saved retriever and return its path."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            tmpdir = tempfile.mkdtemp()
            path = Path(tmpdir) / "test_retriever"

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings
            r.add_text("Test content for loading")
            r.save(path)

            return path

    def test_load_restores_retriever(self, saved_retriever_path: Path) -> None:
        """Test that load restores a retriever from saved state."""
        loaded = Retriever.load(saved_retriever_path)

        assert loaded is not None
        assert loaded.count >= 1

    def test_load_from_directory(self, saved_retriever_path: Path) -> None:
        """Test that load works when given a directory path."""
        loaded = Retriever.load(saved_retriever_path)
        assert loaded is not None

    def test_load_raises_for_missing_file(self) -> None:
        """Test that load raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            Retriever.load("/nonexistent/path/retriever")

    def test_load_preserves_chunk_settings(self, mock_embeddings: MagicMock) -> None:
        """Test that load preserves chunk size and overlap settings."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test"

                r = Retriever(
                    auto_configure=False,
                    backend="memory",
                    chunk_size=1000,
                    chunk_overlap=100,
                )
                r._embeddings = mock_embeddings
                r.add_text("Test content")
                r.save(path)

                loaded = Retriever.load(path)

                assert loaded._chunk_size == 1000
                assert loaded._chunk_overlap == 100


class TestRetrieverLegacyPickleLoad:
    """Tests for loading legacy pickle format retrievers."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings for testing."""
        mock = MagicMock()
        mock.embed.return_value = [0.1, 0.2, 0.3]
        mock.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock.provider = "huggingface"
        mock.model = "test-model"
        return mock

    def _create_legacy_pickle(self, path: Path, mock_embeddings: MagicMock) -> None:
        """Create a legacy v1 pickle file for testing migration."""
        state = {
            "version": 1,
            "backend_name": "memory",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "similarity": "cosine",
            "doc_ids": ["doc-1"],
            "embeddings_provider": mock_embeddings.provider,
            "embeddings_model": mock_embeddings.model,
            "backend_data": {
                "ids": ["chunk-1"],
                "texts": ["Legacy pickle content"],
                "metadatas": [{"doc_id": "doc-1"}],
                "embeddings": [[0.1, 0.2, 0.3]],
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def test_load_legacy_pickle_emits_deprecation_warning(self, mock_embeddings: MagicMock) -> None:
        """Test that loading legacy pickle emits deprecation warning."""
        import warnings

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "retriever.pkl"
            self._create_legacy_pickle(pkl_path, mock_embeddings)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                Retriever.load(pkl_path)

                deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(deprecation_warnings) >= 1

    def test_load_legacy_pickle_restores_data(self, mock_embeddings: MagicMock) -> None:
        """Test that loading legacy pickle restores data correctly."""
        import warnings

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "retriever.pkl"
            self._create_legacy_pickle(pkl_path, mock_embeddings)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                loaded = Retriever.load(pkl_path)

            assert loaded.count == 1
            assert loaded._chunk_size == 500


class TestRetrieverMigrate:
    """Tests for Retriever.migrate() method."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings for testing."""
        mock = MagicMock()
        mock.embed.return_value = [0.1, 0.2, 0.3]
        mock.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock.provider = "huggingface"
        mock.model = "test-model"
        return mock

    def _create_legacy_pickle(self, path: Path, mock_embeddings: MagicMock) -> None:
        """Create a legacy v1 pickle file for testing migration."""
        state = {
            "version": 1,
            "backend_name": "memory",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "similarity": "cosine",
            "doc_ids": ["doc-1"],
            "embeddings_provider": mock_embeddings.provider,
            "embeddings_model": mock_embeddings.model,
            "backend_data": {
                "ids": ["chunk-1"],
                "texts": ["Legacy pickle content"],
                "metadatas": [{"doc_id": "doc-1"}],
                "embeddings": [[0.1, 0.2, 0.3]],
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def test_migrate_creates_v2_format(self, mock_embeddings: MagicMock) -> None:
        """Test that migrate creates new v2 format directory."""
        import warnings

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "retriever.pkl"
            self._create_legacy_pickle(pkl_path, mock_embeddings)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result = Retriever.migrate(pkl_path)

            assert result.is_dir()
            assert (result / "state.json").exists()
            assert (result / "embeddings.npy").exists()

    def test_migrate_removes_pickle_when_requested(self, mock_embeddings: MagicMock) -> None:
        """Test that migrate can remove the legacy pickle file."""
        import warnings

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "retriever.pkl"
            self._create_legacy_pickle(pkl_path, mock_embeddings)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                Retriever.migrate(pkl_path, remove_pickle=True)

            assert not pkl_path.exists()

    def test_migrate_preserves_data(self, mock_embeddings: MagicMock) -> None:
        """Test that migration preserves all data."""
        import warnings

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "retriever.pkl"
            self._create_legacy_pickle(pkl_path, mock_embeddings)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result = Retriever.migrate(pkl_path)

            # Load from new format
            loaded = Retriever.load(result)
            assert loaded.count == 1
            assert loaded._chunk_size == 500


class TestRetrieverAutoSave:
    """Tests for auto-save functionality."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings for testing."""
        mock = MagicMock()
        mock.embed.return_value = [0.1, 0.2, 0.3]
        mock.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock.provider = "huggingface"
        mock.model = "test-model"
        return mock

    def test_auto_save_on_add(self, mock_embeddings: MagicMock) -> None:
        """Test that auto_save triggers save after add operations."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "auto"

                r = Retriever(
                    auto_configure=False,
                    backend="memory",
                    persist_path=path,
                    auto_save=True,
                )
                r._embeddings = mock_embeddings
                r.add_text("Auto-saved content")

                # Directory with state.json should exist after add
                assert path.is_dir()
                assert (path / "state.json").exists()

    def test_auto_save_disabled(self, mock_embeddings: MagicMock) -> None:
        """Test that auto_save=False prevents automatic saving."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "no_auto"

                r = Retriever(
                    auto_configure=False,
                    backend="memory",
                    persist_path=path,
                    auto_save=False,
                )
                r._embeddings = mock_embeddings
                r.add_text("Not auto-saved")

                # Directory should NOT exist after add
                assert not path.exists()


class TestRetrieverPersistenceRoundTrip:
    """Tests for full save/load round-trip scenarios."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings for testing."""
        mock = MagicMock()
        mock.embed.return_value = [0.1, 0.2, 0.3]
        mock.embed_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock.provider = "huggingface"
        mock.model = "test-model"
        return mock

    def test_roundtrip_preserves_data_count(self, mock_embeddings: MagicMock) -> None:
        """Test that save/load preserves the document count."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "roundtrip"

                r = Retriever(auto_configure=False, backend="memory")
                r._embeddings = mock_embeddings
                r.add_text("Document 1")
                r.add_text("Document 2")

                original_count = r.count
                r.save(path)

                loaded = Retriever.load(path)

                assert loaded.count == original_count

    def test_roundtrip_preserves_doc_count(self, mock_embeddings: MagicMock) -> None:
        """Test that save/load preserves document count."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "roundtrip"

                r = Retriever(auto_configure=False, backend="memory")
                r._embeddings = mock_embeddings
                r.add_text("Test document 1")
                r.add_text("More test document 2")

                original_doc_count = len(r._doc_ids)
                r.save(path)

                loaded = Retriever.load(path)

                assert len(loaded._doc_ids) == original_doc_count

    def test_roundtrip_preserves_similarity_setting(self, mock_embeddings: MagicMock) -> None:
        """Test that save/load preserves similarity setting."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "roundtrip"

                r = Retriever(
                    auto_configure=False,
                    backend="memory",
                    similarity="cosine",
                )
                r._embeddings = mock_embeddings
                r.add_text("Test content")

                r.save(path)

                loaded = Retriever.load(path)

                assert loaded._similarity == "cosine"

    def test_roundtrip_preserves_embeddings(self, mock_embeddings: MagicMock) -> None:
        """Test that save/load preserves embeddings correctly."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "roundtrip"

                r = Retriever(auto_configure=False, backend="memory")
                r._embeddings = mock_embeddings
                r.add_text("Test content")

                # Get original embeddings
                original_embeddings = r._backend._embeddings[0].copy()
                r.save(path)

                loaded = Retriever.load(path)
                loaded_embeddings = loaded._backend._embeddings[0]

                np.testing.assert_array_almost_equal(original_embeddings, loaded_embeddings)


class TestRetrieverLoadOnInit:
    """Tests for loading from persist_path on initialization."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings for testing."""
        mock = MagicMock()
        mock.embed.return_value = [0.1, 0.2, 0.3]
        mock.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock.provider = "huggingface"
        mock.model = "test-model"
        return mock

    def test_init_loads_from_existing_persist_path(self, mock_embeddings: MagicMock) -> None:
        """Test that Retriever loads from existing persist_path on init."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "persist"

                # Create and save a retriever
                r1 = Retriever(auto_configure=False, backend="memory")
                r1._embeddings = mock_embeddings
                r1.add_text("Persisted content")
                r1.save(path)
                original_count = r1.count

                # Create new retriever with same persist_path
                r2 = Retriever(
                    auto_configure=False,
                    backend="memory",
                    persist_path=path,
                )

                assert r2.count == original_count

    def test_init_creates_new_if_no_persist_file(self, mock_embeddings: MagicMock) -> None:
        """Test that Retriever creates new store if persist file doesn't exist."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "new"

                r = Retriever(
                    auto_configure=False,
                    backend="memory",
                    persist_path=path,
                )
                r._embeddings = mock_embeddings

                assert r.count == 0
