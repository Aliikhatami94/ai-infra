"""Tests for Retriever persistence (save/load).

Tests cover:
- Saving and loading retriever state
- State integrity after load
- Error handling for invalid paths
- Metadata preservation
- Sidecar JSON file creation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_save_creates_pickle_file(self, retriever_with_data: Retriever) -> None:
        """Test that save creates a pickle file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"
            result = retriever_with_data.save(path)

            assert result.exists()
            assert result.suffix == ".pkl"

    def test_save_creates_json_sidecar(self, retriever_with_data: Retriever) -> None:
        """Test that save creates a JSON metadata sidecar file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"
            result = retriever_with_data.save(path)

            json_path = result.with_suffix(".json")
            assert json_path.exists()

            with open(json_path) as f:
                metadata = json.load(f)

            assert "version" in metadata
            assert "created_at" in metadata
            assert "backend" in metadata
            assert metadata["backend"] == "memory"

    def test_save_to_directory_uses_default_filename(self, retriever_with_data: Retriever) -> None:
        """Test that saving to a directory uses default filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = retriever_with_data.save(tmpdir)

            assert result.name == "retriever.pkl"
            assert result.parent == Path(tmpdir)

    def test_save_creates_parent_directories(self, retriever_with_data: Retriever) -> None:
        """Test that save creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "retriever.pkl"
            result = retriever_with_data.save(path)

            assert result.exists()
            assert result.parent.exists()

    def test_save_fails_for_non_memory_backend(self) -> None:
        """Test that save fails for backends that don't support it."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            mock_emb = MagicMock()
            mock_emb.provider = "huggingface"
            mock_emb.model = "test"
            MockEmb.return_value = mock_emb

            # Mock a non-memory backend
            with patch("ai_infra.retriever.backends.get_backend") as mock_backend:
                mock_backend.return_value = MagicMock()
                mock_backend.return_value.__class__.__name__ = "PostgresBackend"

                r = Retriever(auto_configure=False, backend="memory")
                r._embeddings = mock_emb
                # Override with a non-memory backend
                r._backend = MagicMock()
                r._backend_name = "postgres"

                with tempfile.TemporaryDirectory() as tmpdir:
                    with pytest.raises(ValueError, match="doesn't support save"):
                        r.save(Path(tmpdir) / "test.pkl")

    def test_save_includes_embeddings_config(self, retriever_with_data: Retriever) -> None:
        """Test that saved state includes embeddings configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"
            retriever_with_data.save(path)

            json_path = path.with_suffix(".json")
            with open(json_path) as f:
                metadata = json.load(f)

            assert "embeddings_provider" in metadata
            assert "embeddings_model" in metadata

    def test_save_includes_chunk_count(self, retriever_with_data: Retriever) -> None:
        """Test that saved metadata includes chunk count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"
            retriever_with_data.save(path)

            json_path = path.with_suffix(".json")
            with open(json_path) as f:
                metadata = json.load(f)

            assert "chunk_count" in metadata
            assert metadata["chunk_count"] >= 1


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
            path = Path(tmpdir) / "test.pkl"

            r = Retriever(auto_configure=False, backend="memory")
            r._embeddings = mock_embeddings
            r.add_text("Test content for loading")
            r.save(path)

            return path

    def test_load_restores_retriever(self, saved_retriever_path: Path) -> None:
        """Test that load restores a retriever from saved state."""
        # Expect pickle warning
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            loaded = Retriever.load(saved_retriever_path)

        assert loaded is not None
        assert loaded.count >= 1

    def test_load_from_directory(self, saved_retriever_path: Path) -> None:
        """Test that load works when given a directory path."""
        # Rename to use default filename
        directory = saved_retriever_path.parent
        default_path = directory / "retriever.pkl"
        saved_retriever_path.rename(default_path)
        saved_retriever_path.with_suffix(".json").rename(default_path.with_suffix(".json"))

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            loaded = Retriever.load(directory)

        assert loaded is not None

    def test_load_raises_for_missing_file(self) -> None:
        """Test that load raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            Retriever.load("/nonexistent/path/retriever.pkl")

    def test_load_preserves_chunk_settings(self, mock_embeddings: MagicMock) -> None:
        """Test that load preserves chunk size and overlap settings."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test.pkl"

                r = Retriever(
                    auto_configure=False,
                    backend="memory",
                    chunk_size=1000,
                    chunk_overlap=100,
                )
                r._embeddings = mock_embeddings
                r.add_text("Test content")
                r.save(path)

                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    loaded = Retriever.load(path)

                assert loaded._chunk_size == 1000
                assert loaded._chunk_overlap == 100

    def test_load_emits_pickle_warning(self, saved_retriever_path: Path) -> None:
        """Test that load emits a security warning for pickle files."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Retriever.load(saved_retriever_path)

            pickle_warnings = [x for x in w if "pickle" in str(x.message).lower()]
            assert len(pickle_warnings) >= 1


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
                path = Path(tmpdir) / "auto.pkl"

                r = Retriever(
                    auto_configure=False,
                    backend="memory",
                    persist_path=path,
                    auto_save=True,
                )
                r._embeddings = mock_embeddings
                r.add_text("Auto-saved content")

                # File should exist after add
                assert path.exists()

    def test_auto_save_disabled(self, mock_embeddings: MagicMock) -> None:
        """Test that auto_save=False prevents automatic saving."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "no_auto.pkl"

                r = Retriever(
                    auto_configure=False,
                    backend="memory",
                    persist_path=path,
                    auto_save=False,
                )
                r._embeddings = mock_embeddings
                r.add_text("Not auto-saved")

                # File should NOT exist after add
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
                path = Path(tmpdir) / "roundtrip.pkl"

                r = Retriever(auto_configure=False, backend="memory")
                r._embeddings = mock_embeddings
                r.add_text("Document 1")
                r.add_text("Document 2")

                original_count = r.count
                r.save(path)

                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    loaded = Retriever.load(path)

                assert loaded.count == original_count

    def test_roundtrip_preserves_doc_count(self, mock_embeddings: MagicMock) -> None:
        """Test that save/load preserves document count."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "roundtrip.pkl"

                r = Retriever(auto_configure=False, backend="memory")
                r._embeddings = mock_embeddings
                r.add_text("Test document 1")
                r.add_text("More test document 2")

                original_doc_count = len(r._doc_ids)
                r.save(path)

                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    loaded = Retriever.load(path)

                assert len(loaded._doc_ids) == original_doc_count

    def test_roundtrip_preserves_similarity_setting(self, mock_embeddings: MagicMock) -> None:
        """Test that save/load preserves similarity setting."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "roundtrip.pkl"

                r = Retriever(
                    auto_configure=False,
                    backend="memory",
                    similarity="cosine",
                )
                r._embeddings = mock_embeddings
                r.add_text("Test content")

                r.save(path)

                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    loaded = Retriever.load(path)

                assert loaded._similarity == "cosine"


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
                path = Path(tmpdir) / "persist.pkl"

                # Create and save a retriever
                r1 = Retriever(auto_configure=False, backend="memory")
                r1._embeddings = mock_embeddings
                r1.add_text("Persisted content")
                r1.save(path)
                original_count = r1.count

                # Create new retriever with same persist_path
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
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
                path = Path(tmpdir) / "new.pkl"

                r = Retriever(
                    auto_configure=False,
                    backend="memory",
                    persist_path=path,
                )
                r._embeddings = mock_embeddings

                assert r.count == 0
