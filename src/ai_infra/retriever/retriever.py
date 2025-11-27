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
import uuid
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
        # Chunking configuration
        chunk_size: int = 500,
        chunk_overlap: int = 50,
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
            chunk_size: Maximum characters per chunk (default 500).
            chunk_overlap: Overlapping characters between chunks (default 50).
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
        """
        # Initialize embeddings
        if embeddings is not None:
            self._embeddings = embeddings
        else:
            from ai_infra.embeddings import Embeddings as EmbeddingsClass

            self._embeddings = EmbeddingsClass(provider=provider, model=model)

        # Store chunking config
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Initialize backend
        self._backend_name = backend
        self._backend = get_backend(backend, **backend_config)

        # Track added documents for deduplication
        self._doc_ids: set[str] = set()

    @property
    def backend(self) -> BaseBackend:
        """Get the storage backend."""
        return self._backend

    @property
    def backend_name(self) -> str:
        """Get the backend name."""
        return self._backend_name

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
    ) -> list[str]:
        pass

    @overload
    def search(
        self,
        query: str,
        k: int = ...,
        filter: dict[str, Any] | None = ...,
        detailed: Literal[True] = ...,
    ) -> list[SearchResult]:
        pass

    def search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        detailed: bool = False,
    ) -> list[str] | list[SearchResult]:
        """Search for similar content.

        Args:
            query: The search query.
            k: Number of results to return (default 5).
            filter: Optional metadata filter (backend-dependent).
            detailed: If True, return SearchResult objects with scores
                     and metadata. If False (default), return plain strings.

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
        """
        if self._backend.count() == 0:
            return [] if not detailed else []

        # Embed query
        query_embedding = self._embeddings.embed(query)

        # Search backend
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
