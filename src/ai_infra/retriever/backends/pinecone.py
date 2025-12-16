"""Pinecone storage backend.

Managed cloud vector database for production workloads.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from ai_infra.retriever.backends.base import BaseBackend


class PineconeBackend(BaseBackend):
    """Pinecone vector storage backend.

    Uses Pinecone's managed cloud service for vector storage and search.
    Requires a Pinecone account and API key.

    Example:
        >>> backend = PineconeBackend(
        ...     api_key="your-api-key",
        ...     index_name="my-index",
        ... )
        >>> backend.add([[0.1, 0.2, 0.3]], ["Hello"], [{"source": "test"}])
        >>> results = backend.search([0.1, 0.2, 0.3], k=5)
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str = "ai-infra-retriever",
        namespace: str = "",
        environment: str | None = None,
    ) -> None:
        """Initialize the Pinecone backend.

        Args:
            api_key: Pinecone API key. If not provided, uses PINECONE_API_KEY env var.
            index_name: Name of the Pinecone index.
            namespace: Namespace within the index (for multi-tenancy).
            environment: Pinecone environment (deprecated in newer versions).
        """
        try:
            from pinecone import Pinecone
        except ImportError as e:
            raise ImportError(
                "Pinecone backend requires pinecone-client. "
                "Install with: pip install pinecone-client"
            ) from e

        self._api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Pinecone API key is required. "
                "Set PINECONE_API_KEY environment variable or pass api_key parameter."
            )

        self._index_name = index_name
        self._namespace = namespace

        # Initialize Pinecone client
        self._client = Pinecone(api_key=self._api_key)

        # Get or create index
        self._index = self._client.Index(index_name)

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

        vectors = []
        for emb, text, meta, doc_id in zip(embeddings, texts, metadatas, ids):
            # Pinecone metadata must be flat (no nested dicts)
            metadata = {
                "text": text,
                **self._flatten_metadata(meta),
            }

            vectors.append(
                {
                    "id": doc_id,
                    "values": emb,
                    "metadata": metadata,
                }
            )

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=self._namespace)

        return ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_embedding: The query vector.
            k: Maximum number of results.
            filter: Optional metadata filters.

        Returns:
            List of result dicts with id, text, score, metadata.
        """
        # Build filter
        pinecone_filter = None
        if filter:
            pinecone_filter = {key: {"$eq": value} for key, value in filter.items()}

        # Query index
        results = self._index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            namespace=self._namespace,
            filter=pinecone_filter,
        )

        # Convert to result dicts
        search_results: list[dict[str, Any]] = []
        for match in results.get("matches", []):
            score = match.get("score", 0)
            metadata = dict(match.get("metadata", {}))
            text = metadata.pop("text", "")

            search_results.append(
                {
                    "id": match.get("id", ""),
                    "text": text,
                    "score": float(score),
                    "metadata": metadata,
                }
            )

        return search_results

    def delete(self, ids: list[str]) -> int:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        if not ids:
            return 0

        self._index.delete(ids=ids, namespace=self._namespace)
        return len(ids)

    def clear(self) -> None:
        """Delete all documents in the namespace."""
        # Delete all vectors in namespace
        self._index.delete(delete_all=True, namespace=self._namespace)

    def count(self) -> int:
        """Get the number of stored documents.

        Returns:
            Total number of documents in the namespace.
        """
        stats = self._index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get(self._namespace, {})
        return int(namespace_stats.get("vector_count", 0) or 0)

    @staticmethod
    def _flatten_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Flatten metadata for Pinecone (no nested dicts).

        Pinecone only supports string, number, boolean, and list of strings.
        """
        flat: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                # Skip nested dicts
                continue
            elif isinstance(value, (str, int, float, bool)):
                flat[key] = value
            elif isinstance(value, list):
                # Only keep if it's a list of strings
                if all(isinstance(v, str) for v in value):
                    flat[key] = value
            elif value is not None:
                flat[key] = str(value)
        return flat
