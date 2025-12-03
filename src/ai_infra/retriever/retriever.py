"""Main Retriever class for semantic search and RAG.

The Retriever provides a dead-simple API for semantic search:
- Add text, files, or directories with `add()`
- Search with `search()` or `get_context()`
- Zero configuration required, sensible defaults

Example:
    >>> from ai_infra import Retriever
    >>>
    >>> # Dead simple - just works
    >>> r = Retriever()
    >>> r.add("Paris is the capital of France")
    >>> r.add("Berlin is the capital of Germany")
    >>> r.search("What is the capital of France?")
    ['Paris is the capital of France']
    >>>
    >>> # From files
    >>> r.add("./docs/")  # Loads all supported files
    >>> r.add("./report.pdf")  # Or a single file
    >>>
    >>> # Get context for LLM
    >>> context = r.get_context("revenue growth", k=5)
    >>> prompt = f"Based on this context:\\n{context}\\n\\nAnswer: ..."
"""

from __future__ import annotations

import asyncio
import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from ai_infra.retriever.backends import BaseBackend, get_backend
from ai_infra.retriever.chunking import chunk_documents, chunk_text
from ai_infra.retriever.detection import detect_input_type
from ai_infra.retriever.loaders import load_directory, load_file
from ai_infra.retriever.models import Chunk, SearchResult

if TYPE_CHECKING:
    from ai_infra.embeddings import Embeddings


class Retriever:
    """Semantic search made simple.

    The Retriever automatically handles:
    - Embedding generation (via any provider)
    - Text chunking for long documents
    - File loading (PDF, DOCX, TXT, CSV, JSON, HTML)
    - Directory scanning
    - Vector storage and search

    Progressive complexity:
    - Zero-config: `Retriever()` uses memory storage and auto-detected embeddings
    - Simple: `Retriever(backend="postgres", connection_string="...")`
    - Advanced: Pass your own `Embeddings` instance for full control

    Example - Dead simple:
        >>> r = Retriever()
        >>> r.add("Your text here")
        >>> r.add("./documents/")  # Add all files from a directory
        >>> results = r.search("query")  # Returns list of strings

    Example - Production with PostgreSQL:
        >>> r = Retriever(
        ...     backend="postgres",
        ...     connection_string="postgresql://user:pass@localhost/db",
        ... )
        >>> r.add("./knowledge_base/")
        >>> results = r.search("query", detailed=True)  # Returns SearchResult objects

    Example - LLM context generation:
        >>> r = Retriever()
        >>> r.add("./docs/")
        >>> context = r.get_context("user question", k=5)
        >>> prompt = f"Context:\\n{context}\\n\\nQuestion: {question}"
    """

    def __init__(
        self,
        # Embedding configuration
        provider: str | None = None,
        model: str | None = None,
        embeddings: "Embeddings | None" = None,
        # Backend configuration
        backend: str = "memory",
        # Similarity metric
        similarity: str = "cosine",
        # Chunking configuration
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        # Persistence configuration
        persist_path: str | Path | None = None,
        auto_save: bool = True,
        # Lazy initialization
        lazy_init: bool = False,
        # Backend-specific options
        **backend_config: Any,
    ) -> None:
        """Initialize the Retriever.

        Args:
            provider: Embedding provider (openai, google, voyage, cohere).
                     Auto-detected from environment if not specified.
            model: Embedding model name. Uses provider default if not specified.
            embeddings: Pre-configured Embeddings instance. If provided,
                       `provider` and `model` are ignored.
            backend: Storage backend name. Options:
                - "memory": In-memory (default, no persistence)
                - "postgres": PostgreSQL with pgvector (production)
                - "sqlite": SQLite file (lightweight persistence)
                - "chroma": ChromaDB (good for prototyping)
                - "faiss": FAISS (high-performance local)
                - "pinecone": Pinecone (managed cloud)
                - "qdrant": Qdrant (cloud or self-hosted)
            similarity: Similarity metric for search. Options:
                - "cosine": Cosine similarity (default). Best general choice.
                - "euclidean": Euclidean distance-based similarity.
                - "dot_product": Dot product. Best for normalized embeddings.
            chunk_size: Maximum characters per chunk (default 500).
            chunk_overlap: Overlapping characters between chunks (default 50).
            persist_path: Path to save/load retriever state. If provided and the
                         file exists, the retriever loads from it. Works with
                         memory backend to add persistence.
            auto_save: If True (default) and persist_path is set, automatically
                      saves after each add operation.
            lazy_init: If True, defer loading the embedding model until first
                      use (add or search). Makes server startup faster.
            **backend_config: Backend-specific options:
                - postgres: connection_string, host, port, user, password, database
                - sqlite: path
                - chroma: persist_directory, collection_name
                - faiss: persist_path, index_type
                - pinecone: api_key, environment, index_name, namespace
                - qdrant: url, api_key, collection_name

        Example:
            >>> # Auto-detect everything
            >>> r = Retriever()

            >>> # Custom embedding provider
            >>> r = Retriever(provider="openai", model="text-embedding-3-large")

            >>> # Production with PostgreSQL
            >>> r = Retriever(
            ...     backend="postgres",
            ...     connection_string="postgresql://user:pass@localhost/db",
            ... )

            >>> # Persistent local storage with FAISS
            >>> r = Retriever(
            ...     backend="faiss",
            ...     persist_path="./vector_store",
            ... )

            >>> # Memory backend with disk persistence (survives restarts)
            >>> r = Retriever(
            ...     backend="memory",
            ...     persist_path="./cache/embeddings.pkl",
            ... )

            >>> # Lazy initialization (fast startup)
            >>> r = Retriever(
            ...     provider="huggingface",
            ...     lazy_init=True,  # Model loads on first add/search
            ... )

            >>> # Custom similarity metric
            >>> r = Retriever(
            ...     similarity="dot_product",  # Best for normalized embeddings
            ... )
        """
        # Store persistence config
        self._persist_path = Path(persist_path) if persist_path else None
        self._auto_save = auto_save
        self._lazy_init = lazy_init
        self._initialized = False
        self._similarity = similarity

        # Store config for lazy init
        self._init_provider = provider
        self._init_model = model
        self._init_embeddings = embeddings
        self._init_backend_config = backend_config

        # Check if we should load from existing save
        if self._persist_path and self._persist_path.exists():
            try:
                # Load from saved state
                loaded = Retriever.load(self._persist_path)

                # Copy state from loaded retriever
                self._embeddings = loaded._embeddings
                self._chunk_size = loaded._chunk_size
                self._chunk_overlap = loaded._chunk_overlap
                self._backend_name = loaded._backend_name
                self._backend = loaded._backend
                self._doc_ids = loaded._doc_ids
                self._similarity = (
                    loaded._similarity if hasattr(loaded, "_similarity") else "cosine"
                )
                self._initialized = True
                return
            except Exception as e:
                # Failed to load (corrupt file, incompatible format, etc.)
                # Fall through to fresh initialization
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to load from {self._persist_path}, starting fresh: {e}"
                )

        # Store chunking config
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Initialize backend (always - it's lightweight)
        self._backend_name = backend
        self._backend = get_backend(backend, similarity=similarity, **backend_config)

        # Track added documents for deduplication
        self._doc_ids: set[str] = set()

        # Initialize embeddings (unless lazy)
        if lazy_init:
            # Use lazy wrapper - will load model on first use
            self._embeddings = _LazyEmbeddings(provider=provider, model=model)
            self._initialized = False
        else:
            # Immediate initialization
            if embeddings is not None:
                self._embeddings = embeddings
            else:
                from ai_infra.embeddings import Embeddings as EmbeddingsClass

                self._embeddings = EmbeddingsClass(provider=provider, model=model)
            self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure embeddings are initialized (for lazy init).

        Called before any operation that needs the embedding model.
        Thread-safe and idempotent.
        """
        if self._initialized:
            return

        # If using lazy embeddings, they'll initialize on first use
        # Just mark as initialized so we don't check again
        self._initialized = True

    @property
    def backend(self) -> BaseBackend:
        """Get the storage backend."""
        return self._backend

    @property
    def backend_name(self) -> str:
        """Get the backend name."""
        return self._backend_name

    @property
    def similarity(self) -> str:
        """Get the similarity metric used for search."""
        return self._similarity

    @property
    def count(self) -> int:
        """Get the number of chunks in the store."""
        return self._backend.count()

    # =========================================================================
    # Add methods
    # =========================================================================

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """Add content to the retriever with smart type detection.

        Automatically detects whether `content` is:
        - Raw text: Chunks and embeds directly
        - File path: Loads the file, chunks, and embeds
        - Directory path: Loads all files, chunks, and embeds

        Args:
            content: Text, file path, or directory path.
            metadata: Optional metadata to attach to all chunks.
            chunk: Whether to chunk long text (default True).

        Returns:
            List of IDs for the added chunks.

        Raises:
            FileNotFoundError: If content looks like a path but doesn't exist.

        Example:
            >>> r = Retriever()
            >>> r.add("Some text to search later")
            >>> r.add("./document.pdf")
            >>> r.add("./documents/")  # All files in directory
        """
        input_type = detect_input_type(content)

        if input_type == "text":
            return self.add_text(content, metadata=metadata, chunk=chunk)
        elif input_type == "file":
            return self.add_file(content, metadata=metadata, chunk=chunk)
        else:  # directory
            return self.add_directory(content, metadata=metadata, chunk=chunk)

    def add_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """Add raw text to the retriever.

        Args:
            text: The text to add.
            metadata: Optional metadata for all chunks.
            chunk: Whether to chunk long text (default True).

        Returns:
            List of IDs for the added chunks.

        Example:
            >>> r.add_text("Paris is the capital of France")
            >>> r.add_text(long_document, metadata={"source": "wikipedia"})
        """
        if not text.strip():
            return []

        if chunk:
            chunks = chunk_text(
                text,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                metadata=metadata,
            )
        else:
            # Single chunk
            chunks = [Chunk(text=text, metadata=metadata or {})]

        return self._add_chunks(chunks)

    def add_file(
        self,
        path: str,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """Add a file to the retriever.

        Supports: PDF, DOCX, TXT, MD, CSV, JSON, HTML

        Args:
            path: Path to the file.
            metadata: Optional metadata for all chunks.
            chunk: Whether to chunk long content (default True).

        Returns:
            List of IDs for the added chunks.

        Example:
            >>> r.add_file("./report.pdf")
            >>> r.add_file("./notes.md", metadata={"category": "notes"})
        """
        documents = load_file(path)
        return self._add_documents(documents, metadata=metadata, chunk=chunk)

    def add_directory(
        self,
        path: str,
        pattern: str = "*",
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """Add all files from a directory.

        Args:
            path: Path to the directory.
            pattern: Glob pattern for file matching (e.g., "*.pdf").
            recursive: Whether to search subdirectories (default True).
            metadata: Optional metadata for all chunks.
            chunk: Whether to chunk long content (default True).

        Returns:
            List of IDs for the added chunks.

        Example:
            >>> r.add_directory("./docs/")  # All files
            >>> r.add_directory("./docs/", pattern="*.md")  # Only markdown
        """
        documents = load_directory(path, pattern=pattern, recursive=recursive)
        return self._add_documents(documents, metadata=metadata, chunk=chunk)

    def _add_documents(
        self,
        documents: list[tuple[str, dict[str, Any]]],
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """Add loaded documents to the store."""
        if not documents:
            return []

        # Merge metadata
        if metadata:
            documents = [(text, {**doc_meta, **metadata}) for text, doc_meta in documents]

        if chunk:
            chunks = chunk_documents(
                documents,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        else:
            chunks = [Chunk(text=text, metadata=doc_meta) for text, doc_meta in documents]

        return self._add_chunks(chunks)

    def _add_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Add chunks to the backend after embedding."""
        if not chunks:
            return []

        # Ensure embeddings are initialized (for lazy init)
        self._ensure_initialized()

        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Generate embeddings
        embeddings = self._embeddings.embed_batch(texts)

        # Add to backend
        self._backend.add(
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        # Track IDs
        self._doc_ids.update(ids)

        # Auto-save if persist_path is configured
        if self._persist_path and self._auto_save:
            self.save(self._persist_path)

        return ids

    # =========================================================================
    # Search methods
    # =========================================================================

    @overload
    def search(
        self,
        query: str,
        k: int = ...,
        filter: dict[str, Any] | None = ...,
        detailed: Literal[False] = ...,
        min_score: float | None = ...,
    ) -> list[str]:
        pass

    @overload
    def search(
        self,
        query: str,
        k: int = ...,
        filter: dict[str, Any] | None = ...,
        detailed: Literal[True] = ...,
        min_score: float | None = ...,
    ) -> list[SearchResult]:
        pass

    def search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        detailed: bool = False,
        min_score: float | None = None,
    ) -> list[str] | list[SearchResult]:
        """Search for similar content.

        Args:
            query: The search query.
            k: Number of results to return (default 5).
            filter: Optional metadata filter (backend-dependent).
            detailed: If True, return SearchResult objects with scores
                     and metadata. If False (default), return plain strings.
            min_score: Optional minimum similarity score threshold (0-1).
                      Results below this score are filtered out.

        Returns:
            List of matching texts (or SearchResult objects if detailed=True).

        Example:
            >>> # Simple - just get the text
            >>> results = r.search("capital of France")
            >>> print(results[0])  # "Paris is the capital of France"

            >>> # Detailed - get scores and metadata
            >>> results = r.search("capital of France", detailed=True)
            >>> for result in results:
            ...     print(f"{result.score:.2f}: {result.text}")

            >>> # With minimum score threshold
            >>> results = r.search("query", min_score=0.7, detailed=True)
        """
        if self._backend.count() == 0:
            return [] if not detailed else []

        # Ensure embeddings are initialized (for lazy init)
        self._ensure_initialized()

        # Embed query
        query_embedding = self._embeddings.embed(query)

        # Search backend
        results = self._backend.search(
            query_embedding=query_embedding,
            k=k,
            filter=filter,
        )

        # Filter by min_score if specified
        if min_score is not None:
            results = [r for r in results if r.get("score", 0) >= min_score]

        if detailed:
            return [
                SearchResult(
                    text=r["text"],
                    score=r["score"],
                    metadata=r.get("metadata", {}),
                    source=r.get("metadata", {}).get("source"),
                    page=r.get("metadata", {}).get("page"),
                    chunk_index=r.get("metadata", {}).get("chunk_index"),
                )
                for r in results
            ]
        else:
            return [r["text"] for r in results]

    def get_context(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Get context string for LLM prompts.

        Convenience method that searches and formats results as a single
        string suitable for including in an LLM prompt.

        Args:
            query: The search query.
            k: Number of results to include (default 5).
            filter: Optional metadata filter.
            separator: String to join results (default newline with divider).

        Returns:
            Formatted context string.

        Example:
            >>> context = r.get_context("revenue growth", k=3)
            >>> prompt = f'''Based on this context:
            ... {context}
            ...
            ... Answer the question: What was the revenue growth?'''
        """
        results = self.search(query, k=k, filter=filter, detailed=False)
        return separator.join(results)

    # =========================================================================
    # Async methods
    # =========================================================================

    async def aadd(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """Async version of add().

        Args:
            content: Text, file path, or directory path.
            metadata: Optional metadata.
            chunk: Whether to chunk long text.

        Returns:
            List of IDs for the added chunks.
        """
        input_type = detect_input_type(content)

        if input_type == "text":
            return await self.aadd_text(content, metadata=metadata, chunk=chunk)
        elif input_type == "file":
            # File loading is CPU-bound, run in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.add_file(content, metadata=metadata, chunk=chunk)
            )
        else:  # directory
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.add_directory(content, metadata=metadata, chunk=chunk)
            )

    async def aadd_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """Async version of add_text()."""
        if not text.strip():
            return []

        if chunk:
            chunks = chunk_text(
                text,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                metadata=metadata,
            )
        else:
            chunks = [Chunk(text=text, metadata=metadata or {})]

        return await self._aadd_chunks(chunks)

    async def _aadd_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Async add chunks to the backend."""
        if not chunks:
            return []

        # Ensure embeddings are initialized (for lazy init)
        self._ensure_initialized()

        ids = [str(uuid.uuid4()) for _ in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Async embedding
        embeddings = await self._embeddings.aembed_batch(texts)

        # Add to backend (sync for now, most backends don't have async)
        self._backend.add(
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        self._doc_ids.update(ids)
        return ids

    async def asearch(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        detailed: bool = False,
    ) -> list[str] | list[SearchResult]:
        """Async version of search().

        Args:
            query: The search query.
            k: Number of results.
            filter: Optional metadata filter.
            detailed: Return SearchResult objects if True.

        Returns:
            List of matching texts or SearchResult objects.
        """
        if self._backend.count() == 0:
            return []

        # Ensure embeddings are initialized (for lazy init)
        self._ensure_initialized()

        # Async embed
        query_embedding = await self._embeddings.aembed(query)

        # Search (sync for now)
        results = self._backend.search(
            query_embedding=query_embedding,
            k=k,
            filter=filter,
        )

        if detailed:
            return [
                SearchResult(
                    text=r["text"],
                    score=r["score"],
                    metadata=r.get("metadata", {}),
                    source=r.get("metadata", {}).get("source"),
                    page=r.get("metadata", {}).get("page"),
                    chunk_index=r.get("metadata", {}).get("chunk_index"),
                )
                for r in results
            ]
        else:
            return [r["text"] for r in results]

    async def aget_context(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Async version of get_context()."""
        results = await self.asearch(query, k=k, filter=filter, detailed=False)
        # results is list[str] here since detailed=False
        return separator.join(results)  # type: ignore[arg-type]

    # =========================================================================
    # Management methods
    # =========================================================================

    def delete(self, ids: list[str]) -> int:
        """Delete chunks by ID.

        Args:
            ids: List of chunk IDs to delete.

        Returns:
            Number of chunks deleted.

        Example:
            >>> ids = r.add("Some temporary content")
            >>> r.delete(ids)
        """
        deleted = self._backend.delete(ids)
        self._doc_ids -= set(ids)
        return deleted

    def clear(self) -> None:
        """Clear all content from the retriever.

        Example:
            >>> r.clear()
            >>> print(r.count)  # 0
        """
        self._backend.clear()
        self._doc_ids.clear()

    def __repr__(self) -> str:
        return (
            f"Retriever("
            f"backend={self._backend_name!r}, "
            f"provider={self._embeddings.provider!r}, "
            f"count={self.count}"
            f")"
        )

    def __len__(self) -> int:
        return self.count

    # =========================================================================
    # Persistence methods
    # =========================================================================

    def save(self, path: str | Path) -> Path:
        """Save the retriever state to disk for later loading.

        Serializes the backend data, embeddings config, and metadata to a pickle file.
        Also creates a JSON sidecar file with human-readable metadata.

        Only works with in-memory-like backends (memory, faiss). For database
        backends (postgres, sqlite, chroma), data is already persisted.

        Args:
            path: Path to save the retriever state. Can be a file path or directory.
                  If directory, creates 'retriever.pkl' inside it.

        Returns:
            Path to the saved pickle file.

        Raises:
            ValueError: If the backend doesn't support serialization.

        Example:
            >>> r = Retriever()
            >>> r.add("Some text to search")
            >>> r.save("./cache/my_retriever.pkl")

            >>> # Later...
            >>> r2 = Retriever.load("./cache/my_retriever.pkl")
            >>> r2.search("text")
        """
        path = Path(path)

        # If path is a directory, add default filename
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            path = path / "retriever.pkl"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Get backend state (only memory backend supports this for now)
        if not hasattr(self._backend, "_ids"):
            raise ValueError(
                f"Backend '{self._backend_name}' doesn't support save(). "
                f"Use a persistent backend like 'sqlite' or 'postgres' instead, "
                f"or use 'memory' backend with save/load."
            )

        # Serialize the state
        state = {
            "version": 1,
            "backend_name": self._backend_name,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "similarity": self._similarity,
            "doc_ids": list(self._doc_ids),
            "embeddings_provider": self._embeddings.provider,
            "embeddings_model": self._embeddings.model,
            # Backend data (memory backend specific)
            "backend_data": {
                "ids": self._backend._ids,
                "texts": self._backend._texts,
                "metadatas": self._backend._metadatas,
                "embeddings": [e.tolist() for e in self._backend._embeddings],
            },
        }

        # Save pickle
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save JSON sidecar with human-readable metadata
        metadata_path = path.with_suffix(".json")
        metadata = {
            "version": 1,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "backend": self._backend_name,
            "embeddings_provider": self._embeddings.provider,
            "embeddings_model": self._embeddings.model,
            "similarity": self._similarity,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "doc_count": len(self._doc_ids),
            "chunk_count": self.count,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return path

    @classmethod
    def load(cls, path: str | Path) -> "Retriever":
        """Load a retriever from a previously saved state.

        Args:
            path: Path to the saved retriever pickle file, or directory containing it.

        Returns:
            A fully initialized Retriever with the loaded data.

        Raises:
            FileNotFoundError: If the save file doesn't exist.
            ValueError: If the save file is corrupted or incompatible.

        Example:
            >>> # Save a retriever
            >>> r = Retriever()
            >>> r.add("Hello world")
            >>> r.save("./cache/retriever.pkl")

            >>> # Load it later (even after restart)
            >>> r2 = Retriever.load("./cache/retriever.pkl")
            >>> r2.search("hello")
            ['Hello world']
        """
        path = Path(path)

        # If path is a directory, look for default filename
        if path.is_dir():
            path = path / "retriever.pkl"

        if not path.exists():
            raise FileNotFoundError(f"No saved retriever found at: {path}")

        # Load the state
        with open(path, "rb") as f:
            state = pickle.load(f)

        # Validate version
        version = state.get("version", 0)
        if version != 1:
            raise ValueError(f"Unsupported save file version: {version}")

        # Create retriever instance without calling __init__
        # This avoids loading the embedding model (we have embeddings already)
        retriever = object.__new__(cls)

        # Restore config
        retriever._persist_path = None
        retriever._auto_save = False
        retriever._chunk_size = state["chunk_size"]
        retriever._chunk_overlap = state["chunk_overlap"]
        retriever._backend_name = state["backend_name"]
        retriever._similarity = state.get("similarity", "cosine")  # Default for old saves
        retriever._doc_ids = set(state["doc_ids"])

        # Create a lazy embeddings placeholder that stores provider/model info
        # Actual embedding model only loads if user calls add() after load
        retriever._embeddings = _LazyEmbeddings(
            provider=state["embeddings_provider"],
            model=state["embeddings_model"],
        )

        # Initialize backend with loaded data (include similarity)
        retriever._backend = get_backend(state["backend_name"], similarity=retriever._similarity)

        # Restore backend data
        backend_data = state["backend_data"]
        import numpy as np

        retriever._backend._ids = backend_data["ids"]
        retriever._backend._texts = backend_data["texts"]
        retriever._backend._metadatas = backend_data["metadatas"]
        retriever._backend._embeddings = [
            np.array(e, dtype=np.float32) for e in backend_data["embeddings"]
        ]

        return retriever


class _LazyEmbeddings:
    """Lazy embeddings wrapper that only loads the model when needed.

    Used by Retriever.load() to avoid loading the embedding model
    until the user actually calls add() on the loaded retriever.
    """

    def __init__(self, provider: str, model: str) -> None:
        self._provider = provider
        self._model = model
        self._embeddings = None

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def _ensure_loaded(self) -> None:
        if self._embeddings is None:
            from ai_infra.embeddings import Embeddings as EmbeddingsClass

            self._embeddings = EmbeddingsClass(provider=self._provider, model=self._model)

    def embed(self, text: str) -> list[float]:
        self._ensure_loaded()
        return self._embeddings.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        return self._embeddings.embed_batch(texts)

    async def aembed(self, text: str) -> list[float]:
        self._ensure_loaded()
        return await self._embeddings.aembed(text)
