"""In-memory storage backend using numpy.

Simple and fast, but no persistence. Good for testing and small datasets.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from ai_infra.retriever.backends.base import BaseBackend, SimilarityMetric

if TYPE_CHECKING:
    import numpy as np


class MemoryBackend(BaseBackend):
    """In-memory vector storage using numpy arrays.

    Supports multiple similarity metrics for vector search. Data is not persisted
    and will be lost when the backend is closed or the process ends.

    Args:
        similarity: Similarity metric to use ("cosine", "euclidean", "dot_product").
                   Default is "cosine".

    Example:
        >>> backend = MemoryBackend()
        >>> backend.add([[0.1, 0.2, 0.3]], ["Hello"], [{"source": "test"}])
        >>> results = backend.search([0.1, 0.2, 0.3], k=5)

        >>> # With dot product similarity
        >>> backend = MemoryBackend(similarity="dot_product")
    """

    # All three metrics are supported
    supported_metrics: tuple[SimilarityMetric, ...] = (
        "cosine",
        "euclidean",
        "dot_product",
    )

    def __init__(self, similarity: SimilarityMetric = "cosine") -> None:
        """Initialize the memory backend.

        Args:
            similarity: Similarity metric to use. Options:
                - "cosine": Cosine similarity (default). Range [-1, 1] but
                  typically [0, 1] for positive embeddings.
                - "euclidean": Euclidean distance (converted to similarity).
                  Uses 1 / (1 + distance) for [0, 1] range.
                - "dot_product": Dot product similarity. Best for normalized
                  embeddings where it equals cosine similarity.
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
                "numpy is required for the memory backend. Install with: pip install numpy"
            ) from e

        self._np = np
        self.similarity = similarity
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._embeddings: list[np.ndarray] = []

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

        # Add to storage
        for i, (emb, text, meta, doc_id) in enumerate(zip(embeddings, texts, metadatas, ids)):
            self._ids.append(doc_id)
            self._texts.append(text)
            self._metadatas.append(meta)
            self._embeddings.append(self._np.array(emb, dtype=self._np.float32))

        return ids

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
        if not self._embeddings:
            return []

        query_vec = self._np.array(query_embedding, dtype=self._np.float32)

        # Compute similarity with all stored embeddings
        scores: list[tuple[int, float]] = []
        for i, embedding in enumerate(self._embeddings):
            # Apply metadata filter if provided
            if filter:
                if not self._matches_filter(self._metadatas[i], filter):
                    continue

            score = self._compute_similarity(query_vec, embedding)
            scores.append((i, float(score)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Take top k
        top_k = scores[:k]

        # Convert to result dicts
        results: list[dict[str, Any]] = []
        for idx, score in top_k:
            results.append(
                {
                    "id": self._ids[idx],
                    "text": self._texts[idx],
                    "score": score,
                    "metadata": self._metadatas[idx],
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

        ids_to_delete = set(ids)
        deleted = 0

        # Find indices to delete (in reverse order to avoid index shifting)
        indices_to_delete = []
        for i, doc_id in enumerate(self._ids):
            if doc_id in ids_to_delete:
                indices_to_delete.append(i)
                deleted += 1

        # Delete in reverse order
        for i in sorted(indices_to_delete, reverse=True):
            del self._ids[i]
            del self._texts[i]
            del self._metadatas[i]
            del self._embeddings[i]

        return deleted

    def clear(self) -> None:
        """Delete all documents."""
        self._ids.clear()
        self._texts.clear()
        self._metadatas.clear()
        self._embeddings.clear()

    def count(self) -> int:
        """Get the number of stored documents.

        Returns:
            Total number of documents.
        """
        return len(self._ids)

    def _compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between two vectors using the configured metric.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Similarity score. Higher is always more similar.
        """
        if self.similarity == "cosine":
            return self._cosine_similarity(a, b)
        elif self.similarity == "euclidean":
            return self._euclidean_similarity(a, b)
        elif self.similarity == "dot_product":
            return self._dot_product_similarity(a, b)
        else:
            # Fallback to cosine (shouldn't happen due to validation)
            return self._cosine_similarity(a, b)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity score (range [-1, 1], typically [0, 1] for positive embeddings).
        """
        norm_a = self._np.linalg.norm(a)
        norm_b = self._np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(self._np.dot(a, b) / (norm_a * norm_b))

    def _euclidean_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute euclidean distance-based similarity between two vectors.

        Uses 1 / (1 + distance) to convert distance to similarity score.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Euclidean similarity score (range [0, 1], 1 = identical vectors).
        """
        distance = float(self._np.linalg.norm(a - b))
        return 1.0 / (1.0 + distance)

    def _dot_product_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute dot product similarity between two vectors.

        Best used with normalized embeddings, where it equals cosine similarity.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Dot product score (unbounded, higher = more similar).
        """
        return float(self._np.dot(a, b))

    @staticmethod
    def _matches_filter(
        metadata: dict[str, Any],
        filter: dict[str, Any],
    ) -> bool:
        """Check if metadata matches the filter.

        Args:
            metadata: Document metadata.
            filter: Filter to match.

        Returns:
            True if all filter keys match.
        """
        for key, value in filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
