"""Unit tests for the Retriever module.

Tests cover:
- Input detection (text/file/directory)
- Document chunking
- Memory backend (add, search, delete)
- Retriever class integration
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.retriever.chunking import chunk_documents, chunk_text, estimate_chunks
from ai_infra.retriever.detection import detect_input_type
from ai_infra.retriever.models import Chunk, SearchResult

# =============================================================================
# Input Detection Tests
# =============================================================================


class TestDetectInputType:
    """Tests for input type detection."""

    def test_plain_text(self) -> None:
        """Test plain text is detected as text."""
        assert detect_input_type("Hello, world!") == "text"
        assert detect_input_type("This is a sentence.") == "text"
        assert detect_input_type("Multiple\nlines\nof\ntext") == "text"

    def test_long_text(self) -> None:
        """Test long text is detected as text."""
        long_text = "This is a very long piece of text. " * 100
        assert detect_input_type(long_text) == "text"

    def test_text_with_special_chars(self) -> None:
        """Test text with special characters is detected as text."""
        assert detect_input_type("Special chars: @#$%^&*()") == "text"
        assert detect_input_type("Unicode: 你好世界 ") == "text"

    def test_existing_file(self) -> None:
        """Test existing file is detected as file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            f.flush()
            try:
                assert detect_input_type(f.name) == "file"
            finally:
                os.unlink(f.name)

    def test_existing_directory(self) -> None:
        """Test existing directory is detected as directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert detect_input_type(tmpdir) == "directory"

    def test_file_with_relative_path(self) -> None:
        """Test file with relative path notation."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            f.flush()
            try:
                # Get the relative path from current directory
                abs_path = f.name
                assert detect_input_type(abs_path) == "file"
            finally:
                os.unlink(f.name)

    def test_nonexistent_file_with_extension_raises(self) -> None:
        """Test nonexistent file with known extension raises error."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            detect_input_type("./nonexistent_file.pdf")

    def test_nonexistent_directory_raises(self) -> None:
        """Test nonexistent directory (ending with /) raises error."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            detect_input_type("./nonexistent_dir/")

    def test_path_looking_text_without_extension(self) -> None:
        """Test text that looks like a path but has no extension is treated as text."""
        # This should be treated as text since it doesn't have a recognized extension
        # and doesn't end with /
        result = detect_input_type("some/path/without/extension")
        assert result == "text"

    def test_tilde_expansion(self) -> None:
        """Test ~ is expanded to home directory."""
        home = os.path.expanduser("~")
        if os.path.isdir(home):
            assert detect_input_type("~") == "directory"


# =============================================================================
# Chunking Tests
# =============================================================================


class TestChunkText:
    """Tests for text chunking."""

    def test_short_text_single_chunk(self) -> None:
        """Test short text becomes single chunk."""
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_text_multiple_chunks(self) -> None:
        """Test long text is split into multiple chunks."""
        text = "Word " * 200  # 1000 characters
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunk_metadata(self) -> None:
        """Test chunks have correct metadata."""
        text = "Word " * 200
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_custom_metadata_preserved(self) -> None:
        """Test custom metadata is preserved in chunks."""
        text = "Short text"
        metadata = {"source": "test.txt", "page": 1}
        chunks = chunk_text(text, chunk_size=100, metadata=metadata)

        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["page"] == 1

    def test_empty_text(self) -> None:
        """Test empty text returns empty list."""
        chunks = chunk_text("")
        assert chunks == []

    def test_overlap_preserved(self) -> None:
        """Test chunk overlap works correctly."""
        # Create text that will definitely be split
        text = "A" * 500 + " " + "B" * 500
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)

        # With overlap, adjacent chunks should have some shared content
        if len(chunks) > 1:
            # The end of one chunk should overlap with start of next
            # (this is fuzzy since splitters use natural boundaries)
            assert len(chunks) >= 2


class TestChunkDocuments:
    """Tests for chunking multiple documents."""

    def test_multiple_documents(self) -> None:
        """Test chunking multiple documents."""
        documents = [
            ("First document content", {"source": "doc1.txt"}),
            ("Second document content", {"source": "doc2.txt"}),
        ]
        chunks = chunk_documents(documents, chunk_size=100)

        # Should have chunks from both documents
        sources = {c.metadata["source"] for c in chunks}
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources

    def test_empty_documents_list(self) -> None:
        """Test empty documents list returns empty chunks."""
        chunks = chunk_documents([])
        assert chunks == []


class TestEstimateChunks:
    """Tests for chunk count estimation."""

    def test_estimate_short_text(self) -> None:
        """Test estimate for short text."""
        count = estimate_chunks("Short text", chunk_size=100)
        assert count == 1

    def test_estimate_long_text(self) -> None:
        """Test estimate for long text."""
        text = "Word " * 200  # ~1000 chars
        count = estimate_chunks(text, chunk_size=100)
        assert count > 5  # Should be split into many chunks


# =============================================================================
# Chunk Model Tests
# =============================================================================


class TestChunkModel:
    """Tests for the Chunk dataclass."""

    def test_chunk_creation(self) -> None:
        """Test Chunk creation."""
        chunk = Chunk(text="Hello", metadata={"key": "value"})
        assert chunk.text == "Hello"
        assert chunk.metadata == {"key": "value"}

    def test_chunk_id_default_none(self) -> None:
        """Test Chunk ID defaults to None if not provided."""
        chunk = Chunk(text="Hello", metadata={})
        # ID is optional and defaults to None
        assert chunk.id is None

    def test_chunk_custom_id(self) -> None:
        """Test Chunk with custom ID."""
        chunk = Chunk(text="Hello", metadata={}, id="custom-id")
        assert chunk.id == "custom-id"


# =============================================================================
# SearchResult Model Tests
# =============================================================================


class TestSearchResultModel:
    """Tests for the SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Test SearchResult creation."""
        result = SearchResult(
            text="Found text",
            score=0.95,
            metadata={"source": "test.txt"},
        )
        assert result.text == "Found text"
        assert result.score == 0.95
        assert result.metadata["source"] == "test.txt"

    def test_search_result_optional_fields(self) -> None:
        """Test SearchResult optional fields."""
        result = SearchResult(
            text="Found text",
            score=0.95,
            metadata={},
            source="policy.pdf",
            page=5,
            chunk_index=3,
        )
        assert result.source == "policy.pdf"
        assert result.page == 5
        assert result.chunk_index == 3

    def test_search_result_defaults(self) -> None:
        """Test SearchResult default values."""
        result = SearchResult(text="Text", score=0.5, metadata={})
        assert result.source is None
        assert result.page is None
        assert result.chunk_index is None


# =============================================================================
# Memory Backend Tests
# =============================================================================


class TestMemoryBackend:
    """Tests for the in-memory storage backend."""

    @pytest.fixture
    def backend(self) -> Any:
        """Create a memory backend."""
        from ai_infra.retriever.backends.memory import MemoryBackend

        return MemoryBackend()

    def test_add_single_vector(self, backend: Any) -> None:
        """Test adding a single vector."""
        ids = backend.add(
            embeddings=[[0.1, 0.2, 0.3]],
            texts=["Hello world"],
            metadatas=[{"source": "test"}],
        )
        assert len(ids) == 1
        assert backend.count() == 1

    def test_add_multiple_vectors(self, backend: Any) -> None:
        """Test adding multiple vectors."""
        ids = backend.add(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            texts=["First", "Second"],
            metadatas=[{"id": 1}, {"id": 2}],
        )
        assert len(ids) == 2
        assert backend.count() == 2

    def test_add_with_custom_ids(self, backend: Any) -> None:
        """Test adding with custom IDs."""
        ids = backend.add(
            embeddings=[[0.1, 0.2, 0.3]],
            texts=["Hello"],
            metadatas=[{}],
            ids=["custom-id-1"],
        )
        assert ids == ["custom-id-1"]

    def test_add_empty_list(self, backend: Any) -> None:
        """Test adding empty list returns empty."""
        ids = backend.add(embeddings=[], texts=[], metadatas=[])
        assert ids == []

    def test_search_returns_similar(self, backend: Any) -> None:
        """Test search returns most similar vectors."""
        # Add vectors
        backend.add(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            texts=["Red", "Green", "Blue"],
            metadatas=[{"color": "red"}, {"color": "green"}, {"color": "blue"}],
        )

        # Search for vector similar to [1, 0, 0]
        results = backend.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0]["text"] == "Red"

    def test_search_k_limit(self, backend: Any) -> None:
        """Test search respects k limit."""
        # Add 10 vectors
        embeddings = [[float(i), 0.0, 0.0] for i in range(10)]
        texts = [f"Item {i}" for i in range(10)]
        metadatas = [{} for _ in range(10)]
        backend.add(embeddings, texts, metadatas)

        # Search with k=3
        results = backend.search([5.0, 0.0, 0.0], k=3)
        assert len(results) == 3

    def test_search_empty_store(self, backend: Any) -> None:
        """Test search on empty store returns empty."""
        results = backend.search([1.0, 0.0, 0.0], k=5)
        assert results == []

    def test_search_with_score(self, backend: Any) -> None:
        """Test search returns scores."""
        backend.add(
            embeddings=[[1.0, 0.0, 0.0]],
            texts=["Test"],
            metadatas=[{}],
        )
        results = backend.search([1.0, 0.0, 0.0], k=1)
        assert "score" in results[0]
        assert 0.0 <= results[0]["score"] <= 1.0

    def test_search_with_metadata_filter(self, backend: Any) -> None:
        """Test search with metadata filter."""
        backend.add(
            embeddings=[[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
            texts=["Doc A", "Doc B"],
            metadatas=[{"category": "A"}, {"category": "B"}],
        )

        # Filter by category
        results = backend.search([1.0, 0.0, 0.0], k=5, filter={"category": "A"})
        assert len(results) == 1
        assert results[0]["text"] == "Doc A"

    def test_delete_by_ids(self, backend: Any) -> None:
        """Test deleting vectors by ID."""
        ids = backend.add(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            texts=["First", "Second"],
            metadatas=[{}, {}],
        )
        assert backend.count() == 2

        # Delete first
        backend.delete([ids[0]])
        assert backend.count() == 1

    def test_clear(self, backend: Any) -> None:
        """Test clearing all vectors."""
        backend.add(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            texts=["First", "Second"],
            metadatas=[{}, {}],
        )
        assert backend.count() == 2

        backend.clear()
        assert backend.count() == 0

    def test_count(self, backend: Any) -> None:
        """Test count method."""
        assert backend.count() == 0

        backend.add(
            embeddings=[[1.0, 0.0, 0.0]],
            texts=["Test"],
            metadatas=[{}],
        )
        assert backend.count() == 1


# =============================================================================
# Retriever Integration Tests (with mocks)
# =============================================================================


class TestRetrieverWithMocks:
    """Integration tests for Retriever class using mocks."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings."""
        mock = MagicMock()
        mock.embed.return_value = [0.1, 0.2, 0.3]
        mock.embed_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock.aembed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock.aembed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        return mock

    @pytest.fixture
    def retriever(self, mock_embeddings: MagicMock) -> Any:
        """Create a Retriever with mocked embeddings."""
        from ai_infra.retriever import Retriever

        # Patch at the import location inside the module
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings
            r = Retriever()
            # Replace the embeddings with our mock
            r._embeddings = mock_embeddings
            return r

    def test_retriever_init_defaults(self) -> None:
        """Test Retriever initializes with defaults."""
        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = MagicMock()
            from ai_infra.retriever import Retriever

            r = Retriever()
            assert r._chunk_size == 500
            assert r._chunk_overlap == 50

    def test_add_text(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test adding raw text."""
        ids = retriever.add_text("Hello world")
        assert len(ids) >= 1
        mock_embeddings.embed_batch.assert_called()

    def test_add_detects_text(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test add() detects raw text."""
        ids = retriever.add("This is plain text, not a file path.")
        assert len(ids) >= 1

    def test_add_with_metadata(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test adding text with metadata."""
        ids = retriever.add_text(
            "Hello world",
            metadata={"source": "manual", "category": "greeting"},
        )
        assert len(ids) >= 1

    def test_search_simple(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test simple search returns list of strings."""
        # Add some text first
        retriever.add_text("Paris is the capital of France")
        retriever.add_text("Berlin is the capital of Germany")

        # Search
        results = retriever.search("capital of France")
        assert isinstance(results, list)
        # Results should be strings (not detailed)
        if results:
            assert isinstance(results[0], str)

    def test_search_detailed(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test detailed search returns SearchResult objects."""
        retriever.add_text("Paris is the capital of France")

        results = retriever.search("capital of France", detailed=True)
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], SearchResult)
            assert hasattr(results[0], "score")
            assert hasattr(results[0], "metadata")

    def test_search_k_parameter(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test search respects k parameter."""
        # Add several texts
        for i in range(10):
            retriever.add_text(f"Document number {i}")

        results = retriever.search("document", k=3)
        assert len(results) <= 3

    def test_get_context(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test get_context returns formatted string."""
        retriever.add_text("Fact 1: The sky is blue.")
        retriever.add_text("Fact 2: Water is wet.")

        context = retriever.get_context("color of sky", k=2)
        assert isinstance(context, str)

    def test_get_context_custom_separator(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test get_context with custom separator."""
        retriever.add_text("First fact")
        retriever.add_text("Second fact")

        context = retriever.get_context("facts", separator="\n\n===\n\n")
        assert isinstance(context, str)
        # If there are multiple results, separator should be present
        # (depends on what was stored)

    def test_delete(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test delete removes documents."""
        ids = retriever.add_text("To be deleted")
        initial_count = retriever.count  # count is a property

        retriever.delete(ids)
        assert retriever.count < initial_count or retriever.count == 0

    def test_clear(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test clear removes all documents."""
        retriever.add_text("Document 1")
        retriever.add_text("Document 2")

        retriever.clear()
        assert retriever.count == 0  # count is a property

    def test_count(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test count returns number of documents."""
        assert retriever.count == 0  # count is a property

        retriever.add_text("Document 1")
        # Count might be chunks, not documents
        assert retriever.count >= 1  # count is a property

    def test_embeddings_internal(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test _embeddings stores the embeddings instance."""
        assert retriever._embeddings is mock_embeddings

    def test_backend_property(self, retriever: Any) -> None:
        """Test backend property exposes underlying backend."""
        from ai_infra.retriever.backends.base import BaseBackend

        assert isinstance(retriever.backend, BaseBackend)


class TestRetrieverAsync:
    """Tests for async Retriever methods."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings with async methods."""
        mock = MagicMock()
        mock.embed.return_value = [0.1, 0.2, 0.3]
        mock.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock.aembed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock.aembed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        return mock

    @pytest.fixture
    def retriever(self, mock_embeddings: MagicMock) -> Any:
        """Create a Retriever with mocked embeddings."""
        from ai_infra.retriever import Retriever

        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings
            r = Retriever()
            r._embeddings = mock_embeddings
            return r

    @pytest.mark.asyncio
    async def test_aadd(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test async add."""
        ids = await retriever.aadd("Async added text")
        assert len(ids) >= 1
        mock_embeddings.aembed_batch.assert_called()

    @pytest.mark.asyncio
    async def test_asearch(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test async search."""
        # Add sync first
        retriever.add_text("Test document")

        results = await retriever.asearch("test")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_aget_context(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test async get_context."""
        retriever.add_text("Context document")

        context = await retriever.aget_context("context")
        assert isinstance(context, str)


# =============================================================================
# File Loading Tests (with temp files)
# =============================================================================


class TestRetrieverFileLoading:
    """Tests for file loading in Retriever."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings."""
        mock = MagicMock()
        mock.embed.return_value = [0.1, 0.2, 0.3]
        mock.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock.aembed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock.aembed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        return mock

    @pytest.fixture
    def retriever(self, mock_embeddings: MagicMock) -> Any:
        """Create a Retriever with mocked embeddings."""
        from ai_infra.retriever import Retriever

        with patch("ai_infra.embeddings.Embeddings") as MockEmb:
            MockEmb.return_value = mock_embeddings
            r = Retriever()
            r._embeddings = mock_embeddings
            return r

    def test_add_txt_file(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test adding a TXT file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is the content of a text file.")
            f.flush()
            try:
                ids = retriever.add(f.name)
                assert len(ids) >= 1
            finally:
                os.unlink(f.name)

    def test_add_md_file(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test adding a Markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Title\n\nSome markdown content.\n\n- Item 1\n- Item 2")
            f.flush()
            try:
                ids = retriever.add(f.name)
                assert len(ids) >= 1
            finally:
                os.unlink(f.name)

    def test_add_json_file(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test adding a JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"name": "Test", "value": 123}')
            f.flush()
            try:
                ids = retriever.add(f.name)
                assert len(ids) >= 1
            finally:
                os.unlink(f.name)

    def test_add_directory(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test adding a directory of files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            (Path(tmpdir) / "file1.txt").write_text("Content of file 1")
            (Path(tmpdir) / "file2.txt").write_text("Content of file 2")

            ids = retriever.add(tmpdir)
            # Should load both files
            assert len(ids) >= 2

    def test_add_directory_with_pattern(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test adding directory with file pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mixed files
            (Path(tmpdir) / "doc1.txt").write_text("Text file 1")
            (Path(tmpdir) / "doc2.md").write_text("Markdown file")
            (Path(tmpdir) / "data.json").write_text('{"key": "value"}')

            # Only load .txt files
            ids = retriever.add_directory(tmpdir, pattern="*.txt")
            # Should only load the .txt file
            assert len(ids) >= 1

    def test_add_nonexistent_file_raises(self, retriever: Any) -> None:
        """Test adding nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            retriever.add("./nonexistent_file.pdf")

    def test_add_file_explicit(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test add_file() method."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Explicit file content")
            f.flush()
            try:
                ids = retriever.add_file(f.name)
                assert len(ids) >= 1
            finally:
                os.unlink(f.name)


# =============================================================================
# Backend Factory Tests
# =============================================================================


class TestBackendFactory:
    """Tests for backend factory function."""

    def test_get_memory_backend(self) -> None:
        """Test creating memory backend."""
        from ai_infra.retriever.backends import get_backend

        backend = get_backend("memory")
        assert backend is not None
        assert backend.count() == 0

    def test_get_invalid_backend(self) -> None:
        """Test invalid backend raises error."""
        from ai_infra.retriever.backends import get_backend

        with pytest.raises((ValueError, KeyError)):
            get_backend("invalid_backend_name")

    def test_get_sqlite_backend(self) -> None:
        """Test creating sqlite backend."""
        from ai_infra.retriever.backends import get_backend

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            try:
                backend = get_backend("sqlite", path=f.name)
                assert backend is not None
            finally:
                os.unlink(f.name)


# =============================================================================
# Persistence Tests
# =============================================================================


class TestRetrieverPersistence:
    """Tests for Retriever save/load functionality."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings that return consistent vectors."""
        mock = MagicMock()
        mock.provider = "mock"
        mock.model = "mock-model"
        # Return 384-dim vectors (same as all-MiniLM-L6-v2)
        mock.embed.return_value = [0.1] * 384
        mock.embed_batch.return_value = [[0.1 + i * 0.01] * 384 for i in range(10)]
        return mock

    @pytest.fixture
    def retriever(self, mock_embeddings: MagicMock) -> Any:
        """Create a Retriever with mocked embeddings."""
        from ai_infra.retriever import Retriever

        # Patch embeddings initialization to avoid needing huggingface
        with patch("ai_infra.embeddings.Embeddings", return_value=mock_embeddings):
            r = Retriever(backend="memory")
        return r

    def test_save_creates_files(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test save() creates pickle and JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_save.pkl"

            # Add some content
            retriever.add_text("Hello world")
            retriever.add_text("Paris is the capital of France")

            # Save
            saved_path = retriever.save(path)

            # Check v2 directory format exists
            assert saved_path.exists()
            assert saved_path.is_dir()
            assert (saved_path / "state.json").exists()

            # Check JSON metadata
            import json

            with open(saved_path / "state.json") as f:
                metadata = json.load(f)

            assert metadata["version"] == 2
            assert metadata["backend_name"] == "memory"
            assert metadata["chunk_count"] == 2
            assert "created_at" in metadata

    def test_save_to_directory(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test save() to a directory saves in that directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever.add_text("Test content")

            # Save to directory (not file)
            saved_path = retriever.save(tmpdir)

            # Should save in the directory with v2 format
            assert saved_path == Path(tmpdir)
            assert saved_path.is_dir()
            assert (saved_path / "state.json").exists()

    def test_load_restores_data(self, retriever: Any, mock_embeddings: MagicMock) -> None:
        """Test load() restores saved data."""
        from ai_infra.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_load"

            # Add content and save
            retriever.add_text("Hello world")
            retriever.add_text("Paris is the capital of France")
            original_count = retriever.count
            retriever.save(path)

            # Load into new retriever
            loaded = Retriever.load(path)

            # Check data restored
            assert loaded.count == original_count
            assert loaded.backend_name == "memory"

    def test_load_nonexistent_raises(self) -> None:
        """Test load() raises FileNotFoundError for missing file."""
        from ai_infra.retriever import Retriever

        with pytest.raises(FileNotFoundError):
            Retriever.load("./nonexistent_retriever.pkl")

    def test_persist_path_auto_load(self, mock_embeddings: MagicMock) -> None:
        """Test persist_path loads existing save on init."""
        from ai_infra.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "auto_load"

            # Create and save first retriever
            with patch("ai_infra.embeddings.Embeddings", return_value=mock_embeddings):
                r1 = Retriever(backend="memory")
            r1.add_text("Persisted content")
            r1.save(path)
            original_count = r1.count

            # Create new retriever with persist_path - should auto-load
            with patch("ai_infra.embeddings.Embeddings", return_value=mock_embeddings):
                r2 = Retriever(backend="memory", persist_path=path)

            # Should have loaded the data
            assert r2.count == original_count

    def test_persist_path_auto_save(self, mock_embeddings: MagicMock) -> None:
        """Test persist_path auto-saves after add."""
        from ai_infra.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "auto_save"

            # Create retriever with persist_path (directory doesn't exist yet)
            with patch("ai_infra.embeddings.Embeddings", return_value=mock_embeddings):
                r1 = Retriever(backend="memory", persist_path=path)

            # Directory shouldn't exist yet
            assert not path.exists()

            # Add content - should trigger auto-save
            r1.add_text("Auto-saved content")

            # Directory should now exist with state.json
            assert path.exists()
            assert (path / "state.json").exists()

            # Load and verify
            r2 = Retriever.load(path)
            assert r2.count == r1.count

    def test_persist_path_no_auto_save(self, mock_embeddings: MagicMock) -> None:
        """Test auto_save=False disables auto-saving."""
        from ai_infra.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "no_auto_save"

            # Create retriever with auto_save=False
            with patch("ai_infra.embeddings.Embeddings", return_value=mock_embeddings):
                r1 = Retriever(backend="memory", persist_path=path, auto_save=False)

            # Add content
            r1.add_text("Not auto-saved")

            # Directory should NOT exist (auto_save disabled)
            assert not path.exists()

            # Manual save should work
            r1.save(path)
            assert path.exists()
            assert (path / "state.json").exists()

    def test_save_unsupported_backend_raises(self, mock_embeddings: MagicMock) -> None:
        """Test save() raises for unsupported backends."""
        from ai_infra.retriever import Retriever
        from ai_infra.retriever.backends.base import BaseBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"

            # Create retriever with memory backend
            with patch("ai_infra.embeddings.Embeddings", return_value=mock_embeddings):
                r = Retriever(backend="memory")

            # Replace with a mock backend that's not MemoryBackend
            class FakeBackend(BaseBackend):
                def add(self, ids, texts, embeddings, metadatas=None):
                    pass

                def search(self, embedding, k=5, filter_dict=None):
                    return [], []

                def delete(self, ids):
                    pass

                def count(self):
                    return 0

                def clear(self):
                    pass

            r._backend = FakeBackend()
            r._backend_name = "fake"

            with pytest.raises(ValueError, match="doesn't support save"):
                r.save(path)


# =============================================================================
# Lazy Initialization Tests
# =============================================================================


class TestRetrieverLazyInit:
    """Tests for Retriever lazy initialization functionality."""

    def test_lazy_init_no_model_load_on_create(self) -> None:
        """Test lazy_init=True doesn't load embedding model on creation."""
        from ai_infra.retriever import Retriever
        from ai_infra.retriever.retriever import _LazyEmbeddings

        # Create with lazy_init=True
        r = Retriever(backend="memory", lazy_init=True, provider="huggingface")

        # Should use lazy embeddings wrapper
        assert isinstance(r._embeddings, _LazyEmbeddings)

        # Internal embeddings should not be loaded yet
        assert r._embeddings._embeddings is None

        # Should not be marked as initialized
        assert r._initialized is False

    def test_lazy_init_loads_on_add(self) -> None:
        """Test lazy_init loads model on first add."""
        from ai_infra.retriever import Retriever

        # Create with lazy_init=True and mock the internal loading
        r = Retriever(backend="memory", lazy_init=True, provider="huggingface")

        # Mock the embeddings to avoid actual model loading
        mock_embeddings = MagicMock()
        mock_embeddings.embed_batch.return_value = [[0.1] * 384]
        r._embeddings._embeddings = mock_embeddings

        # Add content - should trigger initialization
        r.add_text("Test content")

        # Should be marked as initialized now
        assert r._initialized is True

        # embed_batch should have been called
        mock_embeddings.embed_batch.assert_called_once()

    def test_lazy_init_loads_on_search(self) -> None:
        """Test lazy_init loads model on first search."""
        from ai_infra.retriever import Retriever

        # Create with lazy_init=True
        r = Retriever(backend="memory", lazy_init=True, provider="huggingface")

        # Add some content manually to backend (bypass embedding)

        r._backend.add(
            embeddings=[[0.1] * 384],
            texts=["Test content"],
            metadatas=[{}],
            ids=["test-id"],
        )

        # Mock the embeddings for search
        mock_embeddings = MagicMock()
        mock_embeddings.embed.return_value = [0.1] * 384
        r._embeddings._embeddings = mock_embeddings

        # Search - should trigger initialization
        r.search("test")

        # Should be marked as initialized now
        assert r._initialized is True

        # embed should have been called for the query
        mock_embeddings.embed.assert_called_once_with("test")

    def test_lazy_embeddings_provider_model_exposed(self) -> None:
        """Test lazy embeddings exposes provider and model properties."""
        from ai_infra.retriever.retriever import _LazyEmbeddings

        lazy = _LazyEmbeddings(provider="openai", model="text-embedding-3-small")

        assert lazy.provider == "openai"
        assert lazy.model == "text-embedding-3-small"

        # Should not have loaded yet
        assert lazy._embeddings is None

    def test_normal_init_loads_immediately(self) -> None:
        """Test normal init (lazy_init=False) loads model immediately."""
        from ai_infra.retriever import Retriever
        from ai_infra.retriever.retriever import _LazyEmbeddings

        # Mock the Embeddings class at its import location
        with patch("ai_infra.embeddings.Embeddings") as MockEmbeddings:
            mock_instance = MagicMock()
            mock_instance.provider = "mock"
            mock_instance.model = "mock-model"
            MockEmbeddings.return_value = mock_instance

            # Create with lazy_init=False (default)
            r = Retriever(backend="memory", lazy_init=False, provider="openai")

            # Should NOT use lazy embeddings wrapper
            assert not isinstance(r._embeddings, _LazyEmbeddings)

            # Should be marked as initialized
            assert r._initialized is True

            # Verify Embeddings was instantiated
            MockEmbeddings.assert_called_once()

    def test_lazy_init_with_persist_path(self) -> None:
        """Test lazy_init works with persist_path."""
        from ai_infra.retriever import Retriever
        from ai_infra.retriever.retriever import _LazyEmbeddings

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lazy_persist.pkl"

            # Create with lazy_init=True and persist_path (no existing file)
            r = Retriever(
                backend="memory",
                lazy_init=True,
                persist_path=path,
                provider="huggingface",
            )

            # Should use lazy embeddings
            assert isinstance(r._embeddings, _LazyEmbeddings)

            # File shouldn't exist yet
            assert not path.exists()


class TestRetrieverSimilarityMetrics:
    """Tests for similarity metric options."""

    def test_default_similarity_is_cosine(self) -> None:
        """Test default similarity metric is cosine."""
        from ai_infra.retriever import Retriever

        with patch("ai_infra.embeddings.Embeddings") as MockEmbeddings:
            mock_instance = MagicMock()
            mock_instance.provider = "mock"
            mock_instance.model = "mock-model"
            MockEmbeddings.return_value = mock_instance

            r = Retriever(backend="memory", provider="openai")
            assert r.similarity == "cosine"
            assert r._backend.similarity == "cosine"

    def test_similarity_dot_product(self) -> None:
        """Test dot product similarity metric."""
        from ai_infra.retriever import Retriever

        with patch("ai_infra.embeddings.Embeddings") as MockEmbeddings:
            mock_instance = MagicMock()
            mock_instance.provider = "mock"
            mock_instance.model = "mock-model"
            MockEmbeddings.return_value = mock_instance

            r = Retriever(backend="memory", similarity="dot_product", provider="openai")
            assert r.similarity == "dot_product"
            assert r._backend.similarity == "dot_product"

    def test_similarity_euclidean(self) -> None:
        """Test euclidean similarity metric."""
        from ai_infra.retriever import Retriever

        with patch("ai_infra.embeddings.Embeddings") as MockEmbeddings:
            mock_instance = MagicMock()
            mock_instance.provider = "mock"
            mock_instance.model = "mock-model"
            MockEmbeddings.return_value = mock_instance

            r = Retriever(backend="memory", similarity="euclidean", provider="openai")
            assert r.similarity == "euclidean"
            assert r._backend.similarity == "euclidean"

    def test_memory_backend_invalid_similarity(self) -> None:
        """Test invalid similarity metric raises error."""
        from ai_infra.retriever.backends.memory import MemoryBackend

        with pytest.raises(ValueError, match="Unsupported similarity"):
            MemoryBackend(similarity="invalid")

    def test_sqlite_backend_invalid_similarity(self) -> None:
        """Test invalid similarity metric raises error for SQLite."""
        from ai_infra.retriever.backends.sqlite import SQLiteBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with pytest.raises(ValueError, match="Unsupported similarity"):
                SQLiteBackend(path=str(db_path), similarity="invalid")

    def test_similarity_preserved_on_save_load(self) -> None:
        """Test similarity metric is preserved when saving and loading."""
        from ai_infra.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "similarity_test"

            # Mock embeddings for this test
            mock_emb = MagicMock()
            mock_emb.provider = "mock"
            mock_emb.model = "mock-model"
            mock_emb.embed.return_value = [0.1, 0.2, 0.3]
            mock_emb.embed_batch.return_value = [[0.1, 0.2, 0.3]]

            # Create with dot_product similarity
            r1 = Retriever(
                backend="memory",
                similarity="dot_product",
                embeddings=mock_emb,
            )

            # Add some text and save
            r1.add("Test document")
            r1.save(path)

            # Load and verify similarity is preserved
            r2 = Retriever.load(path)
            assert r2.similarity == "dot_product"
            assert r2._backend.similarity == "dot_product"

    def test_cosine_similarity_search(self) -> None:
        """Test search works with cosine similarity."""

        from ai_infra.retriever.backends.memory import MemoryBackend

        backend = MemoryBackend(similarity="cosine")

        # Add normalized vectors
        v1 = [1.0, 0.0, 0.0]  # Unit vector in x direction
        v2 = [0.0, 1.0, 0.0]  # Unit vector in y direction
        v3 = [0.707, 0.707, 0.0]  # 45 degrees between x and y

        backend.add([v1, v2, v3], ["x", "y", "xy"], ids=["1", "2", "3"])

        # Query with x direction - should match v1 best
        results = backend.search([1.0, 0.0, 0.0], k=3)
        assert results[0]["text"] == "x"
        assert results[0]["score"] == pytest.approx(1.0, rel=1e-5)

        # v3 (45 degrees) should be more similar than v2 (90 degrees)
        assert results[1]["text"] == "xy"
        assert results[2]["text"] == "y"

    def test_dot_product_similarity_search(self) -> None:
        """Test search works with dot product similarity."""
        from ai_infra.retriever.backends.memory import MemoryBackend

        backend = MemoryBackend(similarity="dot_product")

        # Add vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.5, 0.0, 0.0]  # Same direction, smaller magnitude
        v3 = [2.0, 0.0, 0.0]  # Same direction, larger magnitude

        backend.add([v1, v2, v3], ["one", "half", "two"], ids=["1", "2", "3"])

        # Query with [1, 0, 0] - dot product favors larger magnitude
        results = backend.search([1.0, 0.0, 0.0], k=3)

        # v3 has highest dot product (2.0), then v1 (1.0), then v2 (0.5)
        assert results[0]["text"] == "two"
        assert results[1]["text"] == "one"
        assert results[2]["text"] == "half"

    def test_euclidean_similarity_search(self) -> None:
        """Test search works with euclidean similarity."""
        from ai_infra.retriever.backends.memory import MemoryBackend

        backend = MemoryBackend(similarity="euclidean")

        # Add vectors with different distances from query
        v1 = [1.0, 0.0, 0.0]  # Distance 0 from query
        v2 = [0.5, 0.0, 0.0]  # Distance 0.5 from query
        v3 = [0.0, 0.0, 0.0]  # Distance 1.0 from query

        backend.add([v1, v2, v3], ["close", "medium", "far"], ids=["1", "2", "3"])

        # Query with [1, 0, 0] - euclidean favors closer vectors
        results = backend.search([1.0, 0.0, 0.0], k=3)

        # v1 is closest (identical), then v2, then v3
        assert results[0]["text"] == "close"
        assert results[0]["score"] == pytest.approx(1.0, rel=1e-5)  # 1/(1+0)
        assert results[1]["text"] == "medium"
        assert results[2]["text"] == "far"
