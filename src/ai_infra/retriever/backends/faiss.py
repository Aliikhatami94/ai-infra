"""FAISS backend for high-performance local vector search.

FAISS (Facebook AI Similarity Search) provides high-performance
similarity search with various index types.

Requires: faiss-cpu (or faiss-gpu)
Install: pip install ai-infra[faiss]
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .base import BaseBackend

if TYPE_CHECKING:
    import faiss
    import numpy as np


class FAISSBackend(BaseBackend):
    """FAISS backend for high-performance local vector search.

    FAISS provides efficient similarity search for large-scale datasets.
    Supports various index types for different performance/accuracy tradeoffs.

    Example:
        >>> backend = FAISSBackend(dimension=1536)
        >>> backend.add(embeddings, texts, metadatas)
        >>> results = backend.search(query_embedding, k=5)

    For persistence:
        >>> backend = FAISSBackend(dimension=1536, persist_path="./faiss_index")
        >>> # Data is automatically saved/loaded
    """

    def __init__(
        self,
        dimension: int = 1536,
        index_type: str = "flat",
        persist_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FAISS backend.

        Args:
            dimension: Dimension of embedding vectors. Default 1536 for OpenAI.
            index_type: Type of FAISS index. Options:
                - "flat": Exact search (default, recommended for < 1M vectors)
                - "ivf": Inverted file index (faster for large datasets)
                - "hnsw": Hierarchical NSW (good balance of speed/accuracy)
            persist_path: Path to save/load index. If None, in-memory only.
            **kwargs: Additional arguments (e.g., nlist for IVF).
        """
        try:
            import faiss
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "FAISS is not installed. Install with: pip install faiss-cpu\n"
                "Or install ai-infra with FAISS support: pip install ai-infra[faiss]"
            ) from e

        self._faiss = faiss
        self._np = np
        self._dimension = dimension
        self._index_type = index_type
        self._persist_path = Path(persist_path) if persist_path else None
        self._kwargs = kwargs

        # Storage for texts and metadata (FAISS only stores vectors)
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._ids: list[str] = []

        # Create or load index
        self._index: faiss.Index = self._create_or_load_index()

    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index based on index_type."""
        if self._index_type == "flat":
            # Exact search - best accuracy, slower for large datasets
            return self._faiss.IndexFlatIP(self._dimension)

        elif self._index_type == "ivf":
            # Inverted file index - faster for large datasets
            nlist = self._kwargs.get("nlist", 100)
            quantizer = self._faiss.IndexFlatIP(self._dimension)
            index = self._faiss.IndexIVFFlat(
                quantizer, self._dimension, nlist, self._faiss.METRIC_INNER_PRODUCT
            )
            return index

        elif self._index_type == "hnsw":
            # Hierarchical NSW - good balance
            m = self._kwargs.get("m", 32)
            return self._faiss.IndexHNSWFlat(self._dimension, m)

        else:
            raise ValueError(
                f"Unknown index_type: {self._index_type}. Options: 'flat', 'ivf', 'hnsw'"
            )

    def _create_or_load_index(self) -> faiss.Index:
        """Create new index or load from disk."""
        if self._persist_path and self._persist_path.exists():
            return self._load_index()
        return self._create_index()

    def _save_index(self) -> None:
        """Save index and metadata to disk."""
        if not self._persist_path:
            return

        self._persist_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = self._persist_path / "index.faiss"
        self._faiss.write_index(self._index, str(index_path))

        # Save texts and metadata
        meta_path = self._persist_path / "metadata.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(
                {"texts": self._texts, "metadatas": self._metadatas, "ids": self._ids},
                f,
            )

    def _load_index(self) -> faiss.Index:
        """Load index and metadata from disk.

        Security Warning:
            This method uses pickle to load metadata, which can execute arbitrary code.
            Only load from trusted sources.
        """
        import logging
        import warnings

        if not self._persist_path:
            return self._create_index()

        index_path = self._persist_path / "index.faiss"
        meta_path = self._persist_path / "metadata.pkl"

        if not index_path.exists():
            return self._create_index()

        # Load FAISS index
        index = self._faiss.read_index(str(index_path))

        # Load texts and metadata
        if meta_path.exists():
            # Security warning for pickle files
            warnings.warn(
                "Loading a pickle metadata file can execute arbitrary code. "
                "Only load from trusted sources.",
                UserWarning,
                stacklevel=2,
            )
            logging.getLogger("ai_infra.retriever.faiss").warning(
                f"Loading pickle metadata from {meta_path}. Ensure this is from a trusted source."
            )
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self._texts = data.get("texts", [])
                self._metadatas = data.get("metadatas", [])
                self._ids = data.get("ids", [])

        return index

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity via inner product."""
        norms = self._np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = self._np.where(norms == 0, 1, norms)
        return cast("np.ndarray", vectors / norms)

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add vectors to the FAISS index.

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

        # Convert to numpy and normalize
        vectors = self._np.array(embeddings, dtype=self._np.float32)
        vectors = self._normalize_vectors(vectors)

        # Train IVF index if needed
        if self._index_type == "ivf" and not self._index.is_trained:
            # IVF needs training data
            if len(vectors) >= self._kwargs.get("nlist", 100):
                self._index.train(vectors)
            else:
                # Not enough data to train, recreate as flat index
                self._index = self._faiss.IndexFlatIP(self._dimension)

        # Add to FAISS index
        self._index.add(vectors)

        # Store texts and metadata
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._ids.extend(ids)

        # Persist if configured
        self._save_index()

        return ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector.
            k: Number of results to return.
            filter: Optional metadata filter (applied post-search).

        Returns:
            List of results with text, score, and metadata.
        """
        if self._index.ntotal == 0:
            return []

        # Convert and normalize query
        query = self._np.array([query_embedding], dtype=self._np.float32)
        query = self._normalize_vectors(query)

        # Search - get more results if filtering
        search_k = k * 10 if filter else k
        search_k = min(search_k, self._index.ntotal)

        distances, indices = self._index.search(query, search_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            idx_int = int(idx)
            if idx_int >= len(self._texts):
                continue

            metadata = self._metadatas[idx_int]

            # Apply filter if provided
            if filter:
                if not self._matches_filter(metadata, filter):
                    continue

            results.append(
                {
                    "id": self._ids[idx_int],
                    "text": self._texts[idx_int],
                    "score": float(score),
                    "metadata": metadata,
                }
            )

            if len(results) >= k:
                break

        return results

    def _matches_filter(self, metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID.

        Note: FAISS doesn't support efficient deletion. This rebuilds the index.

        Args:
            ids: List of IDs to delete.

        Returns:
            Number of documents deleted.
        """
        if not ids:
            return 0

        ids_to_delete = set(ids)
        deleted_count = 0

        # Find indices to keep
        keep_indices = []
        for i, doc_id in enumerate(self._ids):
            if doc_id in ids_to_delete:
                deleted_count += 1
            else:
                keep_indices.append(i)

        if deleted_count == 0:
            return 0

        # Rebuild index with remaining vectors
        if keep_indices:
            # Extract remaining vectors
            remaining_vectors = []
            remaining_texts = []
            remaining_metadatas = []
            remaining_ids = []

            # Reconstruct vectors from index
            for i in keep_indices:
                # Get vector from index
                vector = self._np.zeros((1, self._dimension), dtype=self._np.float32)
                self._index.reconstruct(i, vector[0])

                remaining_vectors.append(vector[0])
                remaining_texts.append(self._texts[i])
                remaining_metadatas.append(self._metadatas[i])
                remaining_ids.append(self._ids[i])

            # Create new index
            self._index = self._create_index()
            self._texts = []
            self._metadatas = []
            self._ids = []

            # Re-add remaining vectors
            if remaining_vectors:
                vectors = self._np.array(remaining_vectors, dtype=self._np.float32)

                # Train if needed
                if self._index_type == "ivf" and not self._index.is_trained:
                    if len(vectors) >= self._kwargs.get("nlist", 100):
                        self._index.train(vectors)
                    else:
                        self._index = self._faiss.IndexFlatIP(self._dimension)

                self._index.add(vectors)
                self._texts = remaining_texts
                self._metadatas = remaining_metadatas
                self._ids = remaining_ids
        else:
            # All documents deleted
            self._index = self._create_index()
            self._texts = []
            self._metadatas = []
            self._ids = []

        # Persist changes
        self._save_index()

        return deleted_count

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self._index = self._create_index()
        self._texts = []
        self._metadatas = []
        self._ids = []

        # Clear persisted data
        if self._persist_path:
            index_path = self._persist_path / "index.faiss"
            meta_path = self._persist_path / "metadata.pkl"
            if index_path.exists():
                index_path.unlink()
            if meta_path.exists():
                meta_path.unlink()

    def count(self) -> int:
        """Return number of vectors in the index."""
        return cast("int", self._index.ntotal)
