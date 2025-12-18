"""Storage backends for the Retriever module.

Each backend provides vector storage and similarity search capabilities.
"""

from ai_infra.retriever.backends.base import BaseBackend, SimilarityMetric
from ai_infra.retriever.backends.memory import MemoryBackend

__all__ = [
    "BaseBackend",
    "MemoryBackend",
    "SimilarityMetric",
    "get_backend",
]


def get_backend(name: str, **config) -> BaseBackend:
    """Get a backend instance by name.

    Args:
        name: Backend name ("memory", "postgres", "sqlite", "chroma", etc.)
        **config: Backend-specific configuration.

    Returns:
        A configured backend instance.

    Raises:
        ValueError: If the backend name is not recognized.

    Example:
        >>> backend = get_backend("memory")
        >>> backend = get_backend("postgres", connection_string="postgresql://...")
        >>> backend = get_backend("chroma", persist_directory="./chroma_db")
    """
    backends = {
        "memory": _get_memory_backend,
        "postgres": _get_postgres_backend,
        "postgresql": _get_postgres_backend,
        "sqlite": _get_sqlite_backend,
        "chroma": _get_chroma_backend,
        "pinecone": _get_pinecone_backend,
        "qdrant": _get_qdrant_backend,
        "faiss": _get_faiss_backend,
    }

    factory = backends.get(name.lower())
    if factory is None:
        available = ", ".join(sorted(backends.keys()))
        raise ValueError(f"Unknown backend: {name!r}\nAvailable backends: {available}")

    return factory(**config)


def _get_memory_backend(**config) -> BaseBackend:
    """Create a memory backend."""
    return MemoryBackend(**config)


def _get_postgres_backend(**config) -> BaseBackend:
    """Create a PostgreSQL backend."""
    try:
        from ai_infra.retriever.backends.postgres import PostgresBackend
    except ImportError as e:
        raise ImportError(
            "PostgreSQL backend requires pgvector and asyncpg. "
            "Install with: pip install pgvector asyncpg psycopg2-binary"
        ) from e
    return PostgresBackend(**config)


def _get_sqlite_backend(**config) -> BaseBackend:
    """Create a SQLite backend."""
    try:
        from ai_infra.retriever.backends.sqlite import SQLiteBackend
    except ImportError as e:
        raise ImportError(
            "SQLite backend requires sqlite-vss. Install with: pip install sqlite-vss"
        ) from e
    return SQLiteBackend(**config)


def _get_chroma_backend(**config) -> BaseBackend:
    """Create a Chroma backend."""
    try:
        from ai_infra.retriever.backends.chroma import ChromaBackend
    except ImportError as e:
        raise ImportError(
            "Chroma backend requires chromadb. Install with: pip install chromadb"
        ) from e
    return ChromaBackend(**config)


def _get_pinecone_backend(**config) -> BaseBackend:
    """Create a Pinecone backend."""
    try:
        from ai_infra.retriever.backends.pinecone import PineconeBackend
    except ImportError as e:
        raise ImportError(
            "Pinecone backend requires pinecone-client. Install with: pip install pinecone-client"
        ) from e
    return PineconeBackend(**config)


def _get_qdrant_backend(**config) -> BaseBackend:
    """Create a Qdrant backend."""
    try:
        from ai_infra.retriever.backends.qdrant import QdrantBackend
    except ImportError as e:
        raise ImportError(
            "Qdrant backend requires qdrant-client. Install with: pip install qdrant-client"
        ) from e
    return QdrantBackend(**config)


def _get_faiss_backend(**config) -> BaseBackend:
    """Create a FAISS backend."""
    try:
        from ai_infra.retriever.backends.faiss import FAISSBackend
    except ImportError as e:
        raise ImportError(
            "FAISS backend requires faiss-cpu. Install with: pip install faiss-cpu"
        ) from e
    return FAISSBackend(**config)
