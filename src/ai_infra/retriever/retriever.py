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

Zero-config with environment variables:
    >>> # Set DATABASE_URL for automatic postgres backend
    >>> # No API keys? Uses free local HuggingFace embeddings
    >>> r = Retriever()  # Auto-configures from environment!
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from ai_infra.retriever.backends import BaseBackend, get_backend
from ai_infra.retriever.chunking import chunk_documents, chunk_text
from ai_infra.retriever.detection import detect_input_type
from ai_infra.retriever.loaders import load_directory, load_file
from ai_infra.retriever.models import Chunk, SearchResult

if TYPE_CHECKING:
    from ai_infra.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Known embedding dimensions for common models
# Used to auto-configure backend without requiring explicit dimension
KNOWN_EMBEDDING_DIMENSIONS: dict[str, int] = {
    # HuggingFace / Sentence Transformers (free, local)
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "intfloat/e5-small-v2": 384,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
    "thenlper/gte-small": 384,
    "thenlper/gte-base": 768,
    "thenlper/gte-large": 1024,
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Voyage AI
    "voyage-2": 1024,
    "voyage-02": 1024,
    "voyage-large-2": 1536,
    "voyage-code-2": 1536,
    "voyage-lite-02-instruct": 1024,
    # Cohere
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    # Google
    "models/embedding-001": 768,
    "models/text-embedding-004": 768,
}

# Default models for providers (when auto-configured)
_DEFAULT_PROVIDER_MODELS: dict[str, str] = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
    "voyage": "voyage-2",
    "cohere": "embed-english-v3.0",
    "google_genai": "models/text-embedding-004",
}


def _get_embedding_dimension(provider: str | None, model: str | None) -> int:
    """Get embedding dimension for a provider/model combination.

    Args:
        provider: Embedding provider name (or None for auto-detected)
        model: Model name (or None for provider default)

    Returns:
        Embedding dimension

    Raises:
        ValueError: If dimension cannot be determined
    """
    # Determine effective model
    if model:
        effective_model = model
    elif provider and provider in _DEFAULT_PROVIDER_MODELS:
        effective_model = _DEFAULT_PROVIDER_MODELS[provider]
    else:
        # Default to HuggingFace's default
        effective_model = _DEFAULT_PROVIDER_MODELS["huggingface"]

    # Look up dimension
    if effective_model in KNOWN_EMBEDDING_DIMENSIONS:
        return KNOWN_EMBEDDING_DIMENSIONS[effective_model]

    # Check without provider prefix (e.g., "all-MiniLM-L6-v2")
    for known_model, dim in KNOWN_EMBEDDING_DIMENSIONS.items():
        if known_model.endswith(f"/{effective_model}") or effective_model.endswith(
            known_model.split("/")[-1]
        ):
            return dim

    # Unknown model - raise helpful error
    raise ValueError(
        f"Unknown embedding dimension for model '{effective_model}'. "
        f"Please specify embedding_dimension explicitly, or use a known model: "
        f"{', '.join(sorted(KNOWN_EMBEDDING_DIMENSIONS.keys())[:5])}..."
    )


def _require_svc_infra() -> None:
    """Check that svc-infra is installed, raise helpful error if not."""
    try:
        import svc_infra.loaders  # noqa: F401
    except ImportError:
        raise ImportError(
            "svc-infra is required for remote content loading. "
            "Install with: pip install 'ai-infra[loaders]' or pip install svc-infra"
        ) from None


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    Handles Python 3.10+ event loop deprecation properly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in an async context - can't use run_until_complete
        raise RuntimeError(
            "Cannot call sync method from within an async context. "
            "Use the async version instead (e.g., add_from_github instead of add_from_github_sync)"
        )

    # Create new event loop and run
    return asyncio.run(coro)


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
        embeddings: Embeddings | None = None,
        # Backend configuration
        backend: str | None = None,
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
        # Auto-configuration
        auto_configure: bool = True,
        # Backend-specific options
        **backend_config: Any,
    ) -> None:
        """Initialize the Retriever.

        Args:
            provider: Embedding provider (openai, google, voyage, cohere, huggingface).
                     If not specified and auto_configure=True, auto-detects from
                     environment (falls back to free huggingface if no API keys).
            model: Embedding model name. Uses provider default if not specified.
            embeddings: Pre-configured Embeddings instance. If provided,
                       `provider` and `model` are ignored.
            backend: Storage backend name. Options:
                - None: Auto-detect from DATABASE_URL (default if auto_configure=True)
                - "memory": In-memory (no persistence)
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
            auto_configure: If True (default), auto-detect configuration from
                           environment variables:
                           - DATABASE_URL → backend="postgres" with auto dimension
                           - OPENAI_API_KEY → provider="openai"
                           - VOYAGE_API_KEY → provider="voyage"
                           - COHERE_API_KEY → provider="cohere"
                           - GOOGLE_API_KEY → provider="google_genai"
                           - No API keys → provider="huggingface" (free local)
            **backend_config: Backend-specific options:
                - postgres: connection_string, embedding_dimension, table_name
                - sqlite: path
                - chroma: persist_directory, collection_name
                - faiss: persist_path, index_type
                - pinecone: api_key, environment, index_name, namespace
                - qdrant: url, api_key, collection_name

        Example:
            >>> # Zero-config (auto-detects from environment)
            >>> r = Retriever()

            >>> # Explicit configuration (overrides auto-detect)
            >>> r = Retriever(provider="openai", model="text-embedding-3-large")

            >>> # Production with PostgreSQL (auto-detects DATABASE_URL)
            >>> # Just set DATABASE_URL env var!
            >>> r = Retriever()

            >>> # Or explicit postgres config
            >>> r = Retriever(
            ...     backend="postgres",
            ...     connection_string="postgresql://user:pass@localhost/db",
            ... )

            >>> # Disable auto-configuration
            >>> r = Retriever(auto_configure=False, backend="memory")

            >>> # Lazy initialization (fast startup)
            >>> r = Retriever(lazy_init=True)
        """
        # Instance attributes (explicit for mypy; assigned below)
        self._embeddings: Embeddings | _LazyEmbeddings
        self._chunk_size: int
        self._chunk_overlap: int
        self._backend_name: str
        self._backend: BaseBackend
        self._doc_ids: set[str]

        self._persist_path: Path | None
        self._auto_save: bool
        self._lazy_init: bool
        self._initialized: bool
        self._similarity: str

        self._init_provider: str | None
        self._init_model: str | None
        self._init_embeddings: Embeddings | None
        self._init_backend_config: dict[str, Any]

        # =====================================================================
        # Auto-configuration from environment
        # =====================================================================
        if auto_configure:
            # Auto-detect backend from DATABASE_URL (also check DATABASE_URL_PRIVATE for Railway)
            if backend is None:
                database_url = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URL_PRIVATE")
                if database_url:
                    backend = "postgres"
                    # Set connection_string if not already provided
                    if "connection_string" not in backend_config:
                        backend_config["connection_string"] = database_url
                    logger.debug("Auto-configured postgres backend from DATABASE_URL")
                else:
                    backend = "memory"

            # Auto-detect embedding provider from API keys (if not specified)
            if provider is None and embeddings is None:
                if os.getenv("OPENAI_API_KEY"):
                    provider = "openai"
                    logger.debug("Auto-configured openai provider from OPENAI_API_KEY")
                elif os.getenv("VOYAGE_API_KEY"):
                    provider = "voyage"
                    logger.debug("Auto-configured voyage provider from VOYAGE_API_KEY")
                elif os.getenv("COHERE_API_KEY"):
                    provider = "cohere"
                    logger.debug("Auto-configured cohere provider from COHERE_API_KEY")
                elif os.getenv("GOOGLE_API_KEY"):
                    provider = "google_genai"
                    logger.debug("Auto-configured google_genai provider from GOOGLE_API_KEY")
                else:
                    # Default to free local embeddings (no API key needed!)
                    provider = "huggingface"
                    model = model or "sentence-transformers/all-MiniLM-L6-v2"
                    logger.debug("No API keys found, using free huggingface embeddings")

            # Auto-detect embedding dimension for postgres backend
            if backend == "postgres" and "embedding_dimension" not in backend_config:
                try:
                    dim = _get_embedding_dimension(provider, model)
                    backend_config["embedding_dimension"] = dim
                    logger.debug(f"Auto-configured embedding_dimension={dim} for model")
                except ValueError:
                    # Unknown dimension - let postgres backend use its default
                    pass
        else:
            # No auto-configure - use defaults
            if backend is None:
                backend = "memory"

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
                logger.warning(f"Failed to load from {self._persist_path}, starting fresh: {e}")

        # Store chunking config
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Initialize backend (always - it's lightweight)
        self._backend_name = backend
        self._backend = get_backend(backend, similarity=similarity, **backend_config)

        # Track added documents for deduplication
        self._doc_ids = set()

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

    # =========================================================================
    # Remote content loading (delegates to svc-infra loaders)
    # =========================================================================

    async def add_from_github(
        self,
        repo: str,
        path: str = "",
        branch: str = "main",
        pattern: str = "*.md",
        token: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
        **loader_kwargs: Any,
    ) -> list[str]:
        """Load and embed files from a GitHub repository.

        Delegates to svc_infra.loaders.GitHubLoader for fetching, then embeds.

        Args:
            repo: Repository in "owner/repo" format (e.g., "nfraxlab/svc-infra")
            path: Path within repo (e.g., "docs", "examples/src")
            branch: Branch name (default: "main")
            pattern: Glob pattern for files (e.g., "*.md", "*.py", "*")
            token: GitHub token for private repos or higher rate limits.
                   Falls back to GITHUB_TOKEN env var.
            metadata: Additional metadata to attach to all chunks.
            chunk: Whether to chunk long content (default True)
            **loader_kwargs: Additional args passed to GitHubLoader
                (e.g., skip_patterns, recursive)

        Returns:
            List of chunk IDs added.

        Raises:
            ImportError: If svc-infra is not installed.

        Example:
            >>> retriever = Retriever()
            >>>
            >>> # Load documentation
            >>> await retriever.add_from_github(
            ...     "nfraxlab/svc-infra",
            ...     path="docs",
            ...     pattern="*.md",
            ...     metadata={"package": "svc-infra", "type": "docs"}
            ... )
            >>>
            >>> # Load Python examples
            >>> await retriever.add_from_github(
            ...     "nfraxlab/ai-infra",
            ...     path="examples",
            ...     pattern="*.py",
            ...     metadata={"type": "examples"}
            ... )
        """
        _require_svc_infra()
        from svc_infra.loaders import GitHubLoader

        loader = GitHubLoader(
            repo=repo,
            path=path,
            branch=branch,
            pattern=pattern,
            token=token,
            extra_metadata=metadata,
            **loader_kwargs,
        )

        contents = await loader.load()

        chunk_ids: list[str] = []
        for content in contents:
            # LoadedContent.metadata already has source, repo, path, branch
            ids = self.add_text(
                content.content,
                metadata=content.metadata,
                chunk=chunk,
            )
            chunk_ids.extend(ids)

        logger.debug(f"Added {len(chunk_ids)} chunks from github://{repo}/{path}")
        return chunk_ids

    async def add_from_url(
        self,
        url: str | list[str],
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
        **loader_kwargs: Any,
    ) -> list[str]:
        """Load and embed content from URL(s).

        Delegates to svc_infra.loaders.URLLoader for fetching, then embeds.
        Supports: HTML pages (auto text extraction), raw text, JSON, markdown.

        Args:
            url: Single URL or list of URLs to fetch
            metadata: Additional metadata to attach to all chunks
            chunk: Whether to chunk long content (default True)
            **loader_kwargs: Additional args passed to URLLoader
                (e.g., headers, extract_text, timeout)

        Returns:
            List of chunk IDs added.

        Raises:
            ImportError: If svc-infra is not installed.

        Example:
            >>> retriever = Retriever()
            >>>
            >>> # Load single URL
            >>> await retriever.add_from_url(
            ...     "https://example.com/docs/guide.md",
            ...     metadata={"category": "guides"}
            ... )
            >>>
            >>> # Load multiple URLs at once
            >>> await retriever.add_from_url([
            ...     "https://example.com/page1",
            ...     "https://example.com/page2",
            ... ])
        """
        _require_svc_infra()
        from svc_infra.loaders import URLLoader

        loader = URLLoader(
            urls=url,
            extra_metadata=metadata,
            **loader_kwargs,
        )

        contents = await loader.load()

        chunk_ids: list[str] = []
        for content in contents:
            ids = self.add_text(
                content.content,
                metadata=content.metadata,
                chunk=chunk,
            )
            chunk_ids.extend(ids)

        url_count = 1 if isinstance(url, str) else len(url)
        logger.debug(f"Added {len(chunk_ids)} chunks from {url_count} URL(s)")
        return chunk_ids

    async def add_from_loader(
        self,
        loader: Any,  # BaseLoader from svc-infra
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """Load and embed content from any svc-infra loader.

        Generic method for using any svc_infra.loaders.BaseLoader subclass.

        Args:
            loader: Any loader from svc_infra.loaders (GitHubLoader, URLLoader, etc.)
            metadata: Additional metadata to merge with loader metadata
            chunk: Whether to chunk long content (default True)

        Returns:
            List of chunk IDs added.

        Example:
            >>> from svc_infra.loaders import GitHubLoader
            >>>
            >>> # Custom loader configuration
            >>> loader = GitHubLoader(
            ...     "nfraxlab/svc-infra",
            ...     path="docs",
            ...     skip_patterns=["__pycache__", "*.pyc", "drafts/*"],
            ... )
            >>> await retriever.add_from_loader(loader, metadata={"team": "backend"})
        """
        contents = await loader.load()

        chunk_ids: list[str] = []
        for content in contents:
            merged_metadata = {**content.metadata, **(metadata or {})}
            ids = self.add_text(content.content, metadata=merged_metadata, chunk=chunk)
            chunk_ids.extend(ids)

        logger.debug(f"Added {len(chunk_ids)} chunks from custom loader")
        return chunk_ids

    def add_from_github_sync(
        self,
        repo: str,
        path: str = "",
        branch: str = "main",
        pattern: str = "*.md",
        token: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
        **loader_kwargs: Any,
    ) -> list[str]:
        """Synchronous wrapper for add_from_github().

        See add_from_github() for full documentation.

        Note:
            This creates a new event loop. Do not call from within an async context.
        """
        return cast(
            "list[str]",
            _run_sync(
                self.add_from_github(
                    repo=repo,
                    path=path,
                    branch=branch,
                    pattern=pattern,
                    token=token,
                    metadata=metadata,
                    chunk=chunk,
                    **loader_kwargs,
                )
            ),
        )

    def add_from_url_sync(
        self,
        url: str | list[str],
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
        **loader_kwargs: Any,
    ) -> list[str]:
        """Synchronous wrapper for add_from_url().

        See add_from_url() for full documentation.

        Note:
            This creates a new event loop. Do not call from within an async context.
        """
        return cast(
            "list[str]",
            _run_sync(
                self.add_from_url(
                    url=url,
                    metadata=metadata,
                    chunk=chunk,
                    **loader_kwargs,
                )
            ),
        )

    def add_from_loader_sync(
        self,
        loader: Any,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """Synchronous wrapper for add_from_loader().

        See add_from_loader() for full documentation.

        Note:
            This creates a new event loop. Do not call from within an async context.
        """
        return cast(
            "list[str]",
            _run_sync(self.add_from_loader(loader=loader, metadata=metadata, chunk=chunk)),
        )

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
                None,
                lambda: self.add_directory(content, metadata=metadata, chunk=chunk),
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
        min_score: float | None = None,
    ) -> list[str] | list[SearchResult]:
        """Async version of search().

        Args:
            query: The search query.
            k: Number of results.
            filter: Optional metadata filter.
            detailed: Return SearchResult objects if True.
            min_score: Optional minimum similarity score threshold (0-1).
                      Results below this score are filtered out.

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

    async def aget_context(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Async version of get_context()."""
        results = cast("list[str]", await self.asearch(query, k=k, filter=filter, detailed=False))
        return separator.join(results)

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
        deleted = cast("int", self._backend.delete(ids))
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
        from ai_infra.retriever.backends.memory import MemoryBackend

        if not isinstance(self._backend, MemoryBackend):
            raise ValueError(
                f"Backend '{self._backend_name}' doesn't support save(). "
                f"Use a persistent backend like 'sqlite' or 'postgres' instead, "
                f"or use 'memory' backend with save/load."
            )

        backend = self._backend

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
                "ids": backend._ids,
                "texts": backend._texts,
                "metadatas": backend._metadatas,
                "embeddings": [e.tolist() for e in backend._embeddings],
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
    def load(cls, path: str | Path) -> Retriever:
        """Load a retriever from a previously saved state.

        Args:
            path: Path to the saved retriever pickle file, or directory containing it.

        Returns:
            A fully initialized Retriever with the loaded data.

        Raises:
            FileNotFoundError: If the save file doesn't exist.
            ValueError: If the save file is corrupted or incompatible.

        Security Warning:
            This method uses pickle to load data, which can execute arbitrary code.
            Only load retriever files from trusted sources. A future version will
            migrate to a safer JSON-based format.

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
        import logging
        import warnings

        logger = logging.getLogger("ai_infra.retriever")

        path = Path(path)

        # If path is a directory, look for default filename
        if path.is_dir():
            path = path / "retriever.pkl"

        if not path.exists():
            raise FileNotFoundError(f"No saved retriever found at: {path}")

        # Security warning for pickle files
        if path.suffix == ".pkl":
            warnings.warn(
                "Loading a pickle file can execute arbitrary code. "
                "Only load retriever files from trusted sources. "
                "A future version will migrate to a safer JSON format.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                f"Loading pickle file from {path}. Ensure this file is from a trusted source."
            )

        # Load the state
        with open(path, "rb") as f:
            state = pickle.load(f)

        # Validate version
        version = state.get("version", 0)
        if version != 1:
            raise ValueError(f"Unsupported save file version: {version}")

        backend_name = state.get("backend_name")
        if backend_name != "memory":
            raise ValueError(
                f"Unsupported backend for load(): {backend_name!r}. "
                "Only 'memory' backend save/load is supported."
            )

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
        retriever._backend = get_backend("memory", similarity=retriever._similarity)

        # Restore backend data
        backend_data = state["backend_data"]
        import numpy as np

        from ai_infra.retriever.backends.memory import MemoryBackend

        if not isinstance(retriever._backend, MemoryBackend):
            raise RuntimeError("Expected memory backend during load()")

        backend = retriever._backend
        backend._ids = backend_data["ids"]
        backend._texts = backend_data["texts"]
        backend._metadatas = backend_data["metadatas"]
        backend._embeddings = [np.array(e, dtype=np.float32) for e in backend_data["embeddings"]]

        return retriever


class _LazyEmbeddings:
    """Lazy embeddings wrapper that only loads the model when needed.

    Used by Retriever.load() to avoid loading the embedding model
    until the user actually calls add() on the loaded retriever.
    """

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self._provider = provider
        self._model = model
        self._embeddings: Embeddings | None = None

    @property
    def provider(self) -> str | None:
        return self._provider

    @property
    def model(self) -> str | None:
        return self._model

    def _ensure_loaded(self) -> None:
        if self._embeddings is None:
            from ai_infra.embeddings import Embeddings as EmbeddingsClass

            self._embeddings = EmbeddingsClass(provider=self._provider, model=self._model)

    def embed(self, text: str) -> list[float]:
        self._ensure_loaded()
        assert self._embeddings is not None
        return self._embeddings.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        assert self._embeddings is not None
        return self._embeddings.embed_batch(texts)

    async def aembed(self, text: str) -> list[float]:
        self._ensure_loaded()
        assert self._embeddings is not None
        return await self._embeddings.aembed(text)

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        assert self._embeddings is not None
        return await self._embeddings.aembed_batch(texts)
