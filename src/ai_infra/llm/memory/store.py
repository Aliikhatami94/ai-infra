"""Long-term memory store with semantic search.

This module provides MemoryStore, a simple API for storing and retrieving
memories across sessions with optional semantic search.

Example:
    ```python
    from ai_infra.memory import MemoryStore

    # In-memory (dev/testing)
    store = MemoryStore()

    # SQLite (single-instance production)
    store = MemoryStore.sqlite("./memories.db")

    # PostgreSQL (multi-instance production)
    store = MemoryStore.postgres(os.environ["DATABASE_URL"])

    # Store a memory
    store.put(
        namespace=("user_123", "preferences"),
        key="language",
        value={"preference": "Python and TypeScript"},
    )

    # Get by key
    item = store.get(("user_123", "preferences"), "language")

    # Semantic search
    results = store.search(
        ("user_123", "preferences"),
        query="programming languages",
        limit=5,
    )
    ```
"""

from __future__ import annotations

import builtins
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Type alias for namespace
Namespace = str | tuple[str, ...] | list[str]


@dataclass
class MemoryItem:
    """A single memory item."""

    namespace: tuple[str, ...]
    """Hierarchical namespace for the memory."""

    key: str
    """Unique key within the namespace."""

    value: dict[str, Any]
    """The memory content as a dict."""

    created_at: float = field(default_factory=time.time)
    """Unix timestamp when created."""

    updated_at: float = field(default_factory=time.time)
    """Unix timestamp when last updated."""

    expires_at: float | None = None
    """Unix timestamp when this memory expires (optional)."""

    score: float | None = None
    """Similarity score (populated during search)."""


class MemoryStore:
    """Long-term memory store with semantic search.

    Provides a simple interface for storing and retrieving memories
    with optional semantic search capabilities. Supports multiple
    storage backends:

    - In-memory (dev/testing)
    - SQLite (single-instance)
    - PostgreSQL (production)

    Memories are organized by namespace (tuple of strings) and key.
    Namespaces enable multi-user and multi-tenant storage.

    Example:
        ```python
        from ai_infra.memory import MemoryStore

        # Simple in-memory store
        store = MemoryStore()

        # With semantic search (requires embedding provider)
        store = MemoryStore(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )

        # Store user preferences
        store.put(
            namespace=("user_123", "preferences"),
            key="language",
            value={"preference": "User prefers Python"},
        )

        # Search memories
        results = store.search(
            ("user_123", "preferences"),
            query="what programming language",
            limit=5,
        )
        ```
    """

    def __init__(
        self,
        *,
        backend: str = "memory",
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        connection_string: str | None = None,
        path: str | None = None,
    ):
        """Initialize a MemoryStore.

        Args:
            backend: Storage backend ("memory", "sqlite", "postgres")
            embedding_provider: Provider for semantic search (e.g., "openai", "huggingface")
            embedding_model: Model for embeddings (e.g., "text-embedding-3-small")
            connection_string: PostgreSQL connection string (for postgres backend)
            path: File path (for sqlite backend)
        """
        self._backend_type = backend
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._connection_string = connection_string
        self._path = path

        # Initialize backend
        self._backend = self._create_backend()

        # Initialize embeddings if provider specified
        self._embeddings: Embeddings | None = None
        if embedding_provider:
            self._init_embeddings()

    @classmethod
    def sqlite(
        cls,
        path: str,
        *,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
    ) -> MemoryStore:
        """Create a SQLite-backed memory store.

        Args:
            path: Path to SQLite database file
            embedding_provider: Provider for semantic search
            embedding_model: Model for embeddings

        Returns:
            MemoryStore with SQLite backend

        Example:
            ```python
            store = MemoryStore.sqlite(
                "./memories.db",
                embedding_provider="huggingface",
            )
            ```
        """
        return cls(
            backend="sqlite",
            path=path,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )

    @classmethod
    def postgres(
        cls,
        connection_string: str,
        *,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
    ) -> MemoryStore:
        """Create a PostgreSQL-backed memory store.

        Args:
            connection_string: PostgreSQL connection string
            embedding_provider: Provider for semantic search
            embedding_model: Model for embeddings

        Returns:
            MemoryStore with PostgreSQL backend

        Example:
            ```python
            store = MemoryStore.postgres(
                os.environ["DATABASE_URL"],
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
            )
            ```
        """
        return cls(
            backend="postgres",
            connection_string=connection_string,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )

    def _create_backend(self) -> _MemoryBackend:
        """Create the storage backend."""
        if self._backend_type == "memory":
            return _InMemoryBackend()
        elif self._backend_type == "sqlite":
            if not self._path:
                raise ValueError("path required for sqlite backend")
            return _SQLiteBackend(self._path)
        elif self._backend_type == "postgres":
            if not self._connection_string:
                raise ValueError("connection_string required for postgres backend")
            return _PostgresBackend(self._connection_string)
        else:
            raise ValueError(f"Unknown backend: {self._backend_type}")

    def _init_embeddings(self):
        """Initialize embeddings for semantic search."""
        from ai_infra import Embeddings

        self._embeddings = Embeddings(  # type: ignore[assignment]
            provider=self._embedding_provider,
            model=self._embedding_model,
        )

    def _normalize_namespace(self, namespace: Namespace) -> tuple[str, ...]:
        """Convert namespace to tuple format."""
        if isinstance(namespace, str):
            return (namespace,)
        return tuple(namespace)

    # =========================================================================
    # Core Operations
    # =========================================================================

    def put(
        self,
        namespace: Namespace,
        key: str,
        value: dict[str, Any],
        *,
        ttl: int | None = None,
    ) -> MemoryItem:
        """Store a memory.

        Args:
            namespace: Hierarchical namespace (e.g., ("user_123", "preferences"))
            key: Unique key within the namespace
            value: Memory content as a dict
            ttl: Time-to-live in seconds (optional)

        Returns:
            The stored MemoryItem

        Example:
            ```python
            store.put(
                ("user_123", "preferences"),
                "language",
                {"preference": "Python", "reason": "Type hints"},
                ttl=3600,  # Expires in 1 hour
            )
            ```
        """
        ns = self._normalize_namespace(namespace)
        now = time.time()
        expires_at = now + ttl if ttl else None

        item = MemoryItem(
            namespace=ns,
            key=key,
            value=value,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
        )

        # Generate embedding if semantic search enabled
        embedding = None
        if self._embeddings:
            # Embed the value content
            text = self._value_to_text(value)
            embedding = self._embeddings.embed(text)  # type: ignore[attr-defined]

        self._backend.put(item, embedding)
        return item

    def get(
        self,
        namespace: Namespace,
        key: str,
    ) -> MemoryItem | None:
        """Get a memory by key.

        Args:
            namespace: Hierarchical namespace
            key: Memory key

        Returns:
            MemoryItem if found, None otherwise

        Example:
            ```python
            item = store.get(("user_123", "preferences"), "language")
            if item:
                print(item.value["preference"])
            ```
        """
        ns = self._normalize_namespace(namespace)
        item = self._backend.get(ns, key)

        # Check expiration
        if item and item.expires_at and time.time() > item.expires_at:
            self._backend.delete(ns, key)
            return None

        return item

    def delete(
        self,
        namespace: Namespace,
        key: str,
    ) -> bool:
        """Delete a memory.

        Args:
            namespace: Hierarchical namespace
            key: Memory key

        Returns:
            True if deleted, False if not found

        Example:
            ```python
            deleted = store.delete(("user_123", "preferences"), "language")
            ```
        """
        ns = self._normalize_namespace(namespace)
        return self._backend.delete(ns, key)

    def list(
        self,
        namespace: Namespace,
        *,
        limit: int | None = None,
    ) -> builtins.list[MemoryItem]:
        """List all memories in a namespace.

        Args:
            namespace: Hierarchical namespace
            limit: Maximum number of items to return

        Returns:
            List of MemoryItems

        Example:
            ```python
            items = store.list(("user_123", "preferences"))
            for item in items:
                print(f"{item.key}: {item.value}")
            ```
        """
        ns = self._normalize_namespace(namespace)
        items = self._backend.list(ns, limit=limit)

        # Filter expired items
        now = time.time()
        return [item for item in items if not item.expires_at or item.expires_at > now]

    def search(
        self,
        namespace: Namespace,
        query: str,
        *,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> builtins.list[MemoryItem]:
        """Search memories by semantic similarity.

        Requires embedding_provider to be configured.

        Args:
            namespace: Hierarchical namespace to search
            query: Search query
            limit: Maximum results to return
            filter: Filter on value fields (exact match)

        Returns:
            List of MemoryItems sorted by relevance

        Example:
            ```python
            results = store.search(
                ("user_123", "preferences"),
                query="programming languages",
                limit=5,
            )
            for item in results:
                print(f"{item.key} (score: {item.score}): {item.value}")
            ```
        """
        if not self._embeddings:
            raise ValueError(
                "Semantic search requires embedding_provider. "
                "Initialize with: MemoryStore(embedding_provider='openai')"
            )

        ns = self._normalize_namespace(namespace)

        # Generate query embedding
        query_embedding = self._embeddings.embed(query)  # type: ignore[attr-defined]

        # Search backend
        items = self._backend.search(
            ns,
            query_embedding,
            limit=limit,
            filter=filter,
        )

        # Filter expired items
        now = time.time()
        return [item for item in items if not item.expires_at or item.expires_at > now]

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def put_user_memory(
        self,
        user_id: str,
        key: str,
        value: dict[str, Any],
        *,
        category: str = "memories",
        ttl: int | None = None,
    ) -> MemoryItem:
        """Store a user-scoped memory.

        Convenience method that uses (user_id, category) as namespace.

        Args:
            user_id: User identifier
            key: Memory key
            value: Memory content
            category: Memory category (default: "memories")
            ttl: Time-to-live in seconds

        Returns:
            The stored MemoryItem

        Example:
            ```python
            store.put_user_memory(
                "user_123",
                "favorite_color",
                {"color": "blue"},
            )
            ```
        """
        return self.put((user_id, category), key, value, ttl=ttl)

    def get_user_memory(
        self,
        user_id: str,
        key: str,
        *,
        category: str = "memories",
    ) -> MemoryItem | None:
        """Get a user-scoped memory.

        Args:
            user_id: User identifier
            key: Memory key
            category: Memory category (default: "memories")

        Returns:
            MemoryItem if found, None otherwise
        """
        return self.get((user_id, category), key)

    def search_user_memories(
        self,
        user_id: str,
        query: str,
        *,
        category: str = "memories",
        limit: int = 10,
    ) -> builtins.list[MemoryItem]:
        """Search user's memories.

        Args:
            user_id: User identifier
            query: Search query
            category: Memory category (default: "memories")
            limit: Maximum results

        Returns:
            List of matching MemoryItems
        """
        return self.search((user_id, category), query, limit=limit)

    def put_app_memory(
        self,
        key: str,
        value: dict[str, Any],
        *,
        ttl: int | None = None,
    ) -> MemoryItem:
        """Store an app-level memory.

        Uses ("app", "global") namespace.

        Args:
            key: Memory key
            value: Memory content
            ttl: Time-to-live in seconds

        Returns:
            The stored MemoryItem
        """
        return self.put(("app", "global"), key, value, ttl=ttl)

    def get_app_memory(self, key: str) -> MemoryItem | None:
        """Get an app-level memory.

        Args:
            key: Memory key

        Returns:
            MemoryItem if found, None otherwise
        """
        return self.get(("app", "global"), key)

    def search_app_memories(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> builtins.list[MemoryItem]:
        """Search app-level memories.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching MemoryItems
        """
        return self.search(("app", "global"), query, limit=limit)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _value_to_text(self, value: dict[str, Any]) -> str:
        """Convert a value dict to text for embedding."""
        # Simple approach: join all string values
        parts = []
        for k, v in value.items():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                parts.extend(str(item) for item in v if item)
            else:
                parts.append(str(v))
        return " ".join(parts)


# =============================================================================
# Backend Protocol and Implementations
# =============================================================================


class _MemoryBackend:
    """Base class for memory storage backends."""

    def put(
        self,
        item: MemoryItem,
        embedding: builtins.list[float] | None = None,
    ) -> None:
        raise NotImplementedError

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> MemoryItem | None:
        raise NotImplementedError

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> bool:
        raise NotImplementedError

    def list(
        self,
        namespace: tuple[str, ...],
        *,
        limit: int | None = None,
    ) -> builtins.list[MemoryItem]:
        raise NotImplementedError

    def search(
        self,
        namespace: tuple[str, ...],
        query_embedding: builtins.list[float],
        *,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> builtins.list[MemoryItem]:
        raise NotImplementedError


class _InMemoryBackend(_MemoryBackend):
    """In-memory storage backend."""

    def __init__(self):
        # {namespace_str: {key: (item, embedding)}}
        self._data: dict[str, dict[str, tuple[MemoryItem, list[float] | None]]] = {}

    def _ns_key(self, namespace: tuple[str, ...]) -> str:
        return "/".join(namespace)

    def put(
        self,
        item: MemoryItem,
        embedding: builtins.list[float] | None = None,
    ) -> None:
        ns_key = self._ns_key(item.namespace)
        if ns_key not in self._data:
            self._data[ns_key] = {}
        self._data[ns_key][item.key] = (item, embedding)

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> MemoryItem | None:
        ns_key = self._ns_key(namespace)
        if ns_key not in self._data:
            return None
        entry = self._data[ns_key].get(key)
        return entry[0] if entry else None

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> bool:
        ns_key = self._ns_key(namespace)
        if ns_key not in self._data:
            return False
        if key not in self._data[ns_key]:
            return False
        del self._data[ns_key][key]
        return True

    def list(
        self,
        namespace: tuple[str, ...],
        *,
        limit: int | None = None,
    ) -> builtins.list[MemoryItem]:
        ns_key = self._ns_key(namespace)
        if ns_key not in self._data:
            return []
        items = [entry[0] for entry in self._data[ns_key].values()]
        if limit:
            items = items[:limit]
        return items

    def search(
        self,
        namespace: tuple[str, ...],
        query_embedding: builtins.list[float],
        *,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> builtins.list[MemoryItem]:
        ns_key = self._ns_key(namespace)
        if ns_key not in self._data:
            return []

        # Calculate similarities
        results: list[tuple[float, MemoryItem]] = []

        for key, (item, embedding) in self._data[ns_key].items():
            # Apply filter
            if filter:
                if not self._matches_filter(item.value, filter):
                    continue

            # Calculate similarity
            if embedding:
                score = self._cosine_similarity(query_embedding, embedding)
                item_copy = MemoryItem(
                    namespace=item.namespace,
                    key=item.key,
                    value=item.value,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                    expires_at=item.expires_at,
                    score=score,
                )
                results.append((score, item_copy))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)

        return [item for _, item in results[:limit]]

    def _matches_filter(
        self,
        value: dict[str, Any],
        filter: dict[str, Any],
    ) -> bool:
        """Check if value matches filter criteria."""
        for k, v in filter.items():
            if k not in value or value[k] != v:
                return False
        return True

    def _cosine_similarity(
        self,
        a: builtins.list[float],
        b: builtins.list[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


class _SQLiteBackend(_MemoryBackend):
    """SQLite storage backend."""

    def __init__(self, path: str):
        import sqlite3

        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._setup()

    def _setup(self):
        """Create tables if they don't exist."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                embedding BLOB,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL,
                PRIMARY KEY (namespace, key)
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_namespace ON memories(namespace)
        """
        )
        self._conn.commit()

    def put(
        self,
        item: MemoryItem,
        embedding: builtins.list[float] | None = None,
    ) -> None:
        cursor = self._conn.cursor()
        ns_str = "/".join(item.namespace)
        value_json = json.dumps(item.value)
        embedding_blob = json.dumps(embedding).encode() if embedding else None

        cursor.execute(
            """
            INSERT OR REPLACE INTO memories
            (namespace, key, value, embedding, created_at, updated_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                ns_str,
                item.key,
                value_json,
                embedding_blob,
                item.created_at,
                item.updated_at,
                item.expires_at,
            ),
        )
        self._conn.commit()

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> MemoryItem | None:
        cursor = self._conn.cursor()
        ns_str = "/".join(namespace)

        cursor.execute(
            """
            SELECT value, created_at, updated_at, expires_at
            FROM memories
            WHERE namespace = ? AND key = ?
        """,
            (ns_str, key),
        )

        row = cursor.fetchone()
        if not row:
            return None

        return MemoryItem(
            namespace=namespace,
            key=key,
            value=json.loads(row[0]),
            created_at=row[1],
            updated_at=row[2],
            expires_at=row[3],
        )

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> bool:
        cursor = self._conn.cursor()
        ns_str = "/".join(namespace)

        cursor.execute(
            """
            DELETE FROM memories
            WHERE namespace = ? AND key = ?
        """,
            (ns_str, key),
        )
        self._conn.commit()

        return cursor.rowcount > 0

    def list(
        self,
        namespace: tuple[str, ...],
        *,
        limit: int | None = None,
    ) -> builtins.list[MemoryItem]:
        cursor = self._conn.cursor()
        ns_str = "/".join(namespace)

        query = """
            SELECT key, value, created_at, updated_at, expires_at
            FROM memories
            WHERE namespace = ?
        """
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, (ns_str,))

        items = []
        for row in cursor.fetchall():
            items.append(
                MemoryItem(
                    namespace=namespace,
                    key=row[0],
                    value=json.loads(row[1]),
                    created_at=row[2],
                    updated_at=row[3],
                    expires_at=row[4],
                )
            )

        return items

    def search(
        self,
        namespace: tuple[str, ...],
        query_embedding: builtins.list[float],
        *,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> builtins.list[MemoryItem]:
        """Search by similarity (in-memory calculation for SQLite)."""
        cursor = self._conn.cursor()
        ns_str = "/".join(namespace)

        cursor.execute(
            """
            SELECT key, value, embedding, created_at, updated_at, expires_at
            FROM memories
            WHERE namespace = ? AND embedding IS NOT NULL
        """,
            (ns_str,),
        )

        results: list[tuple[float, MemoryItem]] = []

        for row in cursor.fetchall():
            value = json.loads(row[1])

            # Apply filter
            if filter:
                matches = True
                for k, v in filter.items():
                    if k not in value or value[k] != v:
                        matches = False
                        break
                if not matches:
                    continue

            embedding = json.loads(row[2].decode()) if row[2] else None
            if embedding:
                score = self._cosine_similarity(query_embedding, embedding)
                item = MemoryItem(
                    namespace=namespace,
                    key=row[0],
                    value=value,
                    created_at=row[3],
                    updated_at=row[4],
                    expires_at=row[5],
                    score=score,
                )
                results.append((score, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:limit]]

    def _cosine_similarity(
        self,
        a: builtins.list[float],
        b: builtins.list[float],
    ) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


class _PostgresBackend(_MemoryBackend):
    """PostgreSQL storage backend with pgvector support."""

    def __init__(self, connection_string: str):
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "PostgreSQL backend requires 'psycopg2'. Install with: pip install psycopg2-binary"
            ) from e

        self._connection_string = connection_string
        self._conn = psycopg2.connect(connection_string)
        self._has_pgvector = self._check_pgvector()
        self._setup()

    def _check_pgvector(self) -> bool:
        """Check if pgvector extension is available."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            return cursor.fetchone() is not None
        except Exception:
            return False

    def _setup(self):
        """Create tables if they don't exist."""
        cursor = self._conn.cursor()

        # Create table
        if self._has_pgvector:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    embedding vector(1536),
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    expires_at DOUBLE PRECISION,
                    PRIMARY KEY (namespace, key)
                )
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_embedding
                ON memories USING ivfflat (embedding vector_cosine_ops)
            """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    embedding JSONB,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    expires_at DOUBLE PRECISION,
                    PRIMARY KEY (namespace, key)
                )
            """
            )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)
        """
        )
        self._conn.commit()

    def put(
        self,
        item: MemoryItem,
        embedding: builtins.list[float] | None = None,
    ) -> None:
        cursor = self._conn.cursor()
        ns_str = "/".join(item.namespace)

        if self._has_pgvector and embedding:
            cursor.execute(
                """
                INSERT INTO memories
                (namespace, key, value, embedding, created_at, updated_at, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (namespace, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    embedding = EXCLUDED.embedding,
                    updated_at = EXCLUDED.updated_at,
                    expires_at = EXCLUDED.expires_at
            """,
                (
                    ns_str,
                    item.key,
                    json.dumps(item.value),
                    embedding,
                    item.created_at,
                    item.updated_at,
                    item.expires_at,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO memories
                (namespace, key, value, embedding, created_at, updated_at, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (namespace, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    embedding = EXCLUDED.embedding,
                    updated_at = EXCLUDED.updated_at,
                    expires_at = EXCLUDED.expires_at
            """,
                (
                    ns_str,
                    item.key,
                    json.dumps(item.value),
                    json.dumps(embedding) if embedding else None,
                    item.created_at,
                    item.updated_at,
                    item.expires_at,
                ),
            )
        self._conn.commit()

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> MemoryItem | None:
        cursor = self._conn.cursor()
        ns_str = "/".join(namespace)

        cursor.execute(
            """
            SELECT value, created_at, updated_at, expires_at
            FROM memories
            WHERE namespace = %s AND key = %s
        """,
            (ns_str, key),
        )

        row = cursor.fetchone()
        if not row:
            return None

        return MemoryItem(
            namespace=namespace,
            key=key,
            value=row[0] if isinstance(row[0], dict) else json.loads(row[0]),
            created_at=row[1],
            updated_at=row[2],
            expires_at=row[3],
        )

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> bool:
        cursor = self._conn.cursor()
        ns_str = "/".join(namespace)

        cursor.execute(
            """
            DELETE FROM memories
            WHERE namespace = %s AND key = %s
        """,
            (ns_str, key),
        )
        self._conn.commit()

        return bool(cursor.rowcount > 0)

    def list(
        self,
        namespace: tuple[str, ...],
        *,
        limit: int | None = None,
    ) -> builtins.list[MemoryItem]:
        cursor = self._conn.cursor()
        ns_str = "/".join(namespace)

        query = """
            SELECT key, value, created_at, updated_at, expires_at
            FROM memories
            WHERE namespace = %s
        """
        params: list[str | int] = [ns_str]
        if limit:
            query += " LIMIT %s"
            params.append(limit)

        cursor.execute(query, params)

        items = []
        for row in cursor.fetchall():
            items.append(
                MemoryItem(
                    namespace=namespace,
                    key=row[0],
                    value=row[1] if isinstance(row[1], dict) else json.loads(row[1]),
                    created_at=row[2],
                    updated_at=row[3],
                    expires_at=row[4],
                )
            )

        return items

    def search(
        self,
        namespace: tuple[str, ...],
        query_embedding: builtins.list[float],
        *,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> builtins.list[MemoryItem]:
        cursor = self._conn.cursor()
        ns_str = "/".join(namespace)

        if self._has_pgvector:
            # Use pgvector for efficient similarity search
            query = """
                SELECT key, value, created_at, updated_at, expires_at,
                       1 - (embedding <=> %s::vector) as score
                FROM memories
                WHERE namespace = %s AND embedding IS NOT NULL
            """
            params: list = [query_embedding, ns_str]

            if filter:
                for k, v in filter.items():
                    query += " AND value->>%s = %s"
                    params.extend([k, json.dumps(v) if not isinstance(v, str) else v])

            query += " ORDER BY embedding <=> %s::vector LIMIT %s"
            params.extend([query_embedding, limit])

            cursor.execute(query, params)

            items = []
            for row in cursor.fetchall():
                items.append(
                    MemoryItem(
                        namespace=namespace,
                        key=row[0],
                        value=row[1] if isinstance(row[1], dict) else json.loads(row[1]),
                        created_at=row[2],
                        updated_at=row[3],
                        expires_at=row[4],
                        score=row[5],
                    )
                )
            return items
        else:
            # Fallback to in-memory similarity calculation
            cursor.execute(
                """
                SELECT key, value, embedding, created_at, updated_at, expires_at
                FROM memories
                WHERE namespace = %s AND embedding IS NOT NULL
            """,
                (ns_str,),
            )

            results: list[tuple[float, MemoryItem]] = []

            for row in cursor.fetchall():
                value = row[1] if isinstance(row[1], dict) else json.loads(row[1])

                if filter:
                    matches = True
                    for k, v in filter.items():
                        if k not in value or value[k] != v:
                            matches = False
                            break
                    if not matches:
                        continue

                embedding = row[2] if isinstance(row[2], list) else json.loads(row[2])
                if embedding:
                    score = self._cosine_similarity(query_embedding, embedding)
                    item = MemoryItem(
                        namespace=namespace,
                        key=row[0],
                        value=value,
                        created_at=row[3],
                        updated_at=row[4],
                        expires_at=row[5],
                        score=score,
                    )
                    results.append((score, item))

            results.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in results[:limit]]

    def _cosine_similarity(
        self,
        a: builtins.list[float],
        b: builtins.list[float],
    ) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)
