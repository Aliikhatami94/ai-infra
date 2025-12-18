"""Abstract base class for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

# Similarity metric type alias
SimilarityMetric = Literal["cosine", "euclidean", "dot_product"]


class BaseBackend(ABC):
    """Abstract base class for vector storage backends.

    All backends must implement these methods to provide
    vector storage and similarity search capabilities.

    Backends work with raw embeddings, texts, and metadata dicts
    rather than higher-level objects for maximum flexibility.

    Attributes:
        similarity: The similarity metric used for search. Default is "cosine".
    """

    # Default similarity metric
    similarity: SimilarityMetric = "cosine"

    # Metrics supported by this backend (override in subclasses)
    supported_metrics: tuple[SimilarityMetric, ...] = (
        "cosine",
        "euclidean",
        "dot_product",
    )

    @abstractmethod
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
            texts: List of text content (one per embedding).
            metadatas: Optional list of metadata dicts (one per embedding).
            ids: Optional list of IDs. If not provided, IDs are generated.

        Returns:
            List of IDs for the added documents.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_embedding: The query vector to search for.
            k: Maximum number of results to return.
            filter: Optional metadata filters (backend-dependent).

        Returns:
            List of result dicts with keys:
                - "id": Document ID
                - "text": Text content
                - "score": Similarity score (0-1, higher is better)
                - "metadata": Metadata dict
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Delete all documents from the store."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the store.

        Returns:
            Total number of stored documents.
        """
        ...

    def close(self) -> None:
        """Close any open connections.

        Override this method if the backend needs cleanup.
        """
        pass

    def __enter__(self) -> BaseBackend:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
