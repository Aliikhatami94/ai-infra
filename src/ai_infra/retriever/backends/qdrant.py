"""Qdrant storage backend.

Fast vector database, supports both cloud and self-hosted deployments.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from ai_infra.retriever.backends.base import BaseBackend


class QdrantBackend(BaseBackend):
    """Qdrant vector storage backend.

    Uses Qdrant for vector storage and search. Supports both
    Qdrant Cloud and self-hosted instances.

    Example:
        >>> # Local/self-hosted
        >>> backend = QdrantBackend(url="http://localhost:6333")

        >>> # Qdrant Cloud
        >>> backend = QdrantBackend(
        ...     url="https://your-cluster.qdrant.io",
        ...     api_key="your-api-key",
        ... )

        >>> backend.add([[0.1, 0.2, 0.3]], ["Hello"], [{"source": "test"}])
        >>> results = backend.search([0.1, 0.2, 0.3], k=5)
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "ai_infra_retriever",
        embedding_dimension: int = 1536,
        host: str = "localhost",
        port: int = 6333,
    ) -> None:
        """Initialize the Qdrant backend.

        Args:
            url: Qdrant server URL. If not provided, uses host:port.
            api_key: API key for Qdrant Cloud. Uses QDRANT_API_KEY env var if not provided.
            collection_name: Name of the collection.
            embedding_dimension: Dimension of embedding vectors.
            host: Qdrant host (used if url is not provided).
            port: Qdrant port (used if url is not provided).
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:
            raise ImportError(
                "Qdrant backend requires qdrant-client. Install with: pip install qdrant-client"
            ) from e

        self._collection_name = collection_name
        self._embedding_dimension = embedding_dimension

        # Get API key from env if not provided
        api_key = api_key or os.getenv("QDRANT_API_KEY")

        # Initialize client
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(host=host, port=port, api_key=api_key)

        # Create collection if it doesn't exist
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )

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

        try:
            from qdrant_client.models import PointStruct
        except ImportError as e:
            raise ImportError(
                "Qdrant backend requires qdrant-client. Install with: pip install qdrant-client"
            ) from e

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]

        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in embeddings]

        points = []
        for emb, text, meta, doc_id in zip(embeddings, texts, metadatas, ids):
            # Store text in payload
            payload = {
                "text": text,
                **meta,
            }

            points.append(
                PointStruct(
                    id=doc_id,
                    vector=emb,
                    payload=payload,
                )
            )

        # Upsert points
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

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
        qdrant_filter = None
        if filter:
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
            except ImportError:
                qdrant_filter = None
            else:
                conditions = [
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in filter.items()
                ]
                qdrant_filter = Filter(must=conditions)

        # Search
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=qdrant_filter,
        )

        # Convert to result dicts
        search_results: list[dict[str, Any]] = []
        for hit in results:
            payload = dict(hit.payload) if hit.payload else {}
            text = payload.pop("text", "")

            search_results.append(
                {
                    "id": str(hit.id),
                    "text": text,
                    "score": float(hit.score),
                    "metadata": payload,
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

        try:
            from qdrant_client.models import PointIdsList
        except ImportError as e:
            raise ImportError(
                "Qdrant backend requires qdrant-client. Install with: pip install qdrant-client"
            ) from e

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=PointIdsList(points=ids),
        )

        return len(ids)

    def clear(self) -> None:
        """Delete all documents."""
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:
            raise ImportError(
                "Qdrant backend requires qdrant-client. Install with: pip install qdrant-client"
            ) from e

        # Recreate collection
        self._client.delete_collection(self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._embedding_dimension,
                distance=Distance.COSINE,
            ),
        )

    def count(self) -> int:
        """Get the number of stored chunks.

        Returns:
            Total number of chunks.
        """
        info = self._client.get_collection(self._collection_name)
        points_count = getattr(info, "points_count", 0) or 0
        return int(points_count)
