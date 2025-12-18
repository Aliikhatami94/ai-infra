"""SQLite storage backend using sqlite-vss or numpy fallback.

Local file-based persistence. Good for desktop apps and CLI tools.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from typing import TYPE_CHECKING, Any

from ai_infra.retriever.backends.base import BaseBackend, SimilarityMetric

if TYPE_CHECKING:
    import numpy as np


class SQLiteBackend(BaseBackend):
    """SQLite vector storage with configurable similarity search.

    Uses numpy for similarity calculations. Embeddings are stored
    as JSON arrays in SQLite.

    Supports multiple similarity metrics: cosine (default), euclidean, dot_product.

    Note: For very large datasets, consider using the postgres or chroma
    backends which have optimized vector indexes.

    Example:
        >>> backend = SQLiteBackend(path="./vectors.db")
        >>> backend.add([[0.1, 0.2, 0.3]], ["Hello"], [{"source": "test"}])
        >>> results = backend.search([0.1, 0.2, 0.3], k=5)

        >>> # With dot product similarity
        >>> backend = SQLiteBackend(path="./vectors.db", similarity="dot_product")
    """

    # All three metrics are supported
    supported_metrics: tuple[SimilarityMetric, ...] = (
        "cosine",
        "euclidean",
        "dot_product",
    )

    def __init__(
        self,
        path: str = "./retriever.db",
        table_name: str = "embeddings",
        similarity: SimilarityMetric = "cosine",
    ) -> None:
        """Initialize the SQLite backend.

        Args:
            path: Path to the SQLite database file.
            table_name: Table name for storing embeddings.
            similarity: Similarity metric to use ("cosine", "euclidean", "dot_product").
        """
        if similarity not in self.supported_metrics:
            raise ValueError(
                f"Unsupported similarity metric: {similarity!r}. "
                f"Supported: {', '.join(self.supported_metrics)}"
            )

        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "numpy is required for the SQLite backend. Install with: pip install numpy"
            ) from e

        self._np = np
        self.similarity = similarity
        self._path = os.path.expanduser(path)
        self._table_name = table_name

        # Create directory if needed
        dir_path = os.path.dirname(os.path.abspath(self._path))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Connect to database
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row

        # Create table
        self._create_table()

    def _create_table(self) -> None:
        """Create the embeddings table if it doesn't exist."""
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                embedding TEXT NOT NULL
            )
        """
        )
        self._conn.commit()

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add vectors to the store.

        Args:
            embeddings: List of embedding vectors.
            texts: List of text content.
            metadatas: Optional list of metadata dicts.
            ids: Optional list of IDs. Generated if not provided.

        Returns:
            List of IDs for the added documents.
        """
        if not embeddings:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]

        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in embeddings]

        result_ids = []
        for emb, text, meta, doc_id in zip(embeddings, texts, metadatas, ids):
            self._conn.execute(
                f"""
                INSERT OR REPLACE INTO {self._table_name}
                (id, text, metadata, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (
                    doc_id,
                    text,
                    json.dumps(meta),
                    json.dumps(emb),
                ),
            )
            result_ids.append(doc_id)

        self._conn.commit()
        return result_ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors using the configured similarity metric.

        Args:
            query_embedding: The query vector.
            k: Maximum number of results.
            filter: Optional metadata filters.

        Returns:
            List of result dicts with id, text, score, metadata.
        """
        query_vec = self._np.array(query_embedding, dtype=self._np.float32)

        # Fetch all rows (SQLite doesn't have native vector search)
        cursor = self._conn.execute(f"SELECT id, text, metadata, embedding FROM {self._table_name}")

        scores: list[tuple[str, str, dict[str, Any], float]] = []
        for row in cursor:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            # Apply metadata filter
            if filter:
                if not self._matches_filter(metadata, filter):
                    continue

            embedding = self._np.array(json.loads(row["embedding"]), dtype=self._np.float32)
            score = self._compute_similarity(query_vec, embedding)
            scores.append((row["id"], row["text"], metadata, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[3], reverse=True)

        # Take top k
        top_k = scores[:k]

        # Convert to result dicts
        results: list[dict[str, Any]] = []
        for doc_id, text, metadata, score in top_k:
            results.append(
                {
                    "id": doc_id,
                    "text": text,
                    "score": score,
                    "metadata": metadata,
                }
            )

        return results

    def delete(self, ids: list[str]) -> int:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        if not ids:
            return 0

        placeholders = ",".join("?" * len(ids))
        cursor = self._conn.execute(
            f"DELETE FROM {self._table_name} WHERE id IN ({placeholders})",
            ids,
        )
        self._conn.commit()
        return cursor.rowcount

    def clear(self) -> None:
        """Delete all documents."""
        self._conn.execute(f"DELETE FROM {self._table_name}")
        self._conn.commit()

    def count(self) -> int:
        """Get the number of stored documents.

        Returns:
            Total number of documents.
        """
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {self._table_name}")
        return int(cursor.fetchone()[0])

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()

    def _compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between two vectors using the configured metric."""
        if self.similarity == "cosine":
            return self._cosine_similarity(a, b)
        elif self.similarity == "euclidean":
            return self._euclidean_similarity(a, b)
        elif self.similarity == "dot_product":
            return self._dot_product_similarity(a, b)
        else:
            return self._cosine_similarity(a, b)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = self._np.linalg.norm(a)
        norm_b = self._np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(self._np.dot(a, b) / (norm_a * norm_b))

    def _euclidean_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute euclidean distance-based similarity between two vectors."""
        distance = float(self._np.linalg.norm(a - b))
        return 1.0 / (1.0 + distance)

    def _dot_product_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute dot product similarity between two vectors."""
        return float(self._np.dot(a, b))

    @staticmethod
    def _matches_filter(
        metadata: dict[str, Any],
        filter: dict[str, Any],
    ) -> bool:
        """Check if metadata matches the filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
