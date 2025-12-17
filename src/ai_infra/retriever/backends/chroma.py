"""Chroma storage backend.

Easy-to-use vector database, good for prototyping and small-medium datasets.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from ai_infra.retriever.backends.base import BaseBackend

if TYPE_CHECKING:
    import chromadb


class ChromaBackend(BaseBackend):
    """Chroma vector storage backend.

    Uses ChromaDB for vector storage and similarity search.
    Supports both in-memory and persistent modes.

    Example:
        >>> # In-memory (for testing)
        >>> backend = ChromaBackend()

        >>> # Persistent
        >>> backend = ChromaBackend(persist_directory="./chroma_db")

        >>> backend.add([[0.1, 0.2, 0.3]], ["Hello"], [{"source": "test"}])
        >>> results = backend.search([0.1, 0.2, 0.3], k=5)
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = "ai_infra_retriever",
    ) -> None:
        """Initialize the Chroma backend.

        Args:
            persist_directory: Directory for persistent storage.
                If None, uses in-memory storage.
            collection_name: Name of the Chroma collection.
        """
        try:
            import chromadb as cdb
            from chromadb.config import Settings
        except ImportError as e:
            raise ImportError(
                "Chroma backend requires chromadb. Install with: pip install chromadb"
            ) from e

        self._collection_name = collection_name

        # Create client
        if persist_directory:
            self._client: chromadb.ClientAPI = cdb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = cdb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
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

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]

        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in embeddings]

        # Sanitize metadata for Chroma
        sanitized_metadatas = [self._sanitize_metadata(m) for m in metadatas]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=sanitized_metadatas,
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
        # Build where filter
        where = None
        if filter:
            where = {
                key: value
                for key, value in filter.items()
                if isinstance(value, (str, int, float, bool))
            }

        # Query collection
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to result dicts
        search_results: list[dict[str, Any]] = []

        if not results["ids"] or not results["ids"][0]:
            return search_results

        result_ids = results["ids"][0]
        documents = results["documents"][0] if results["documents"] else []
        metadatas_list = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i, doc_id in enumerate(result_ids):
            # Chroma returns distance, convert to similarity score
            # For cosine distance: similarity = 1 - distance
            distance = distances[i] if i < len(distances) else 0
            score = 1 - distance

            text = documents[i] if i < len(documents) else ""
            metadata = metadatas_list[i] if i < len(metadatas_list) else {}

            search_results.append(
                {
                    "id": doc_id,
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

        # Get current count of matching IDs
        existing = self._collection.get(ids=ids)
        count = len(existing["ids"]) if existing["ids"] else 0

        if count > 0:
            self._collection.delete(ids=ids)

        return count

    def clear(self) -> None:
        """Delete all documents."""
        # Recreate the collection to clear it
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Get the number of stored documents.

        Returns:
            Total number of documents.
        """
        return int(self._collection.count())

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize metadata for Chroma storage.

        Chroma only supports str, int, float, bool values.
        """
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                # Skip None values
                continue
            else:
                # Convert to string
                sanitized[key] = str(value)
        return sanitized
