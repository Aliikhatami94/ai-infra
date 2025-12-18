"""Simple VectorStore abstraction for ai-infra.

Provides a unified interface for vector databases without exposing
any underlying implementation details. Supports in-memory, Chroma,
and FAISS backends.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal, cast

from ai_infra.embeddings.embeddings import Embeddings


@dataclass
class Document:
    """A document with text and optional metadata.

    Attributes:
        text: The document text content.
        metadata: Optional key-value metadata.
        id: Unique document identifier (auto-generated if not provided).
    """

    text: str
    metadata: dict[str, Any] | None = None
    id: str | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """A search result with document and similarity score.

    Attributes:
        document: The matched document.
        score: Similarity score (higher is more similar).
    """

    document: Document
    score: float


class VectorStore:
    """Simple vector store for semantic search.

    Store documents and search by meaning using embeddings.
    Supports multiple backends: in-memory, Chroma, FAISS.

    Features:
        - Simple API: `add()`, `search()`, `delete()`
        - Multiple backends: memory, chroma, faiss
        - Auto-embedding: Just add text, embeddings handled automatically
        - Metadata filtering: Filter search results by metadata

    Example:
        ```python
        from ai_infra import Embeddings, VectorStore

        # Create with embeddings
        embeddings = Embeddings()
        store = VectorStore(embeddings=embeddings)

        # Add documents
        store.add_texts(["Python is great", "JavaScript is popular"])

        # Search
        results = store.search("programming languages", k=2)
        for result in results:
            print(f"{result.score:.2f}: {result.document.text}")
        ```

    Example - With metadata:
        ```python
        store.add_texts(
            texts=["Doc 1", "Doc 2"],
            metadatas=[{"source": "web"}, {"source": "book"}]
        )

        # Filter by metadata
        results = store.search("query", filter={"source": "web"})
        ```

    Example - Persistent storage with Chroma:
        ```python
        store = VectorStore(
            embeddings=embeddings,
            backend="chroma",
            persist_directory="./my_db"
        )
        ```
    """

    def __init__(
        self,
        embeddings: Embeddings,
        backend: Literal["memory", "chroma", "faiss"] = "memory",
        collection_name: str = "default",
        persist_directory: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize vector store.

        Args:
            embeddings: Embeddings instance for generating vectors.
            backend: Storage backend - "memory", "chroma", or "faiss".
            collection_name: Name for the collection (chroma only).
            persist_directory: Directory for persistent storage.
            **kwargs: Additional backend-specific options.

        Example:
            ```python
            # In-memory (default, fast, no persistence)
            store = VectorStore(embeddings=embeddings)

            # Chroma (persistent, full-featured)
            store = VectorStore(
                embeddings=embeddings,
                backend="chroma",
                persist_directory="./db"
            )

            # FAISS (fast similarity search)
            store = VectorStore(embeddings=embeddings, backend="faiss")
            ```
        """
        self._embeddings = embeddings
        self._backend_name = backend
        self._collection_name = collection_name
        self._persist_directory = persist_directory

        # Initialize backend with union type annotation
        self._backend: _VectorBackend
        if backend == "memory":
            self._backend = _InMemoryBackend(embeddings)
        elif backend == "chroma":
            self._backend = _ChromaBackend(
                embeddings=embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory,
                **kwargs,
            )
        elif backend == "faiss":
            self._backend = _FAISSBackend(
                embeddings=embeddings,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Available: memory, chroma, faiss")

    @property
    def backend(self) -> str:
        """Get the backend name."""
        return self._backend_name

    @property
    def embeddings(self) -> Embeddings:
        """Get the embeddings instance."""
        return self._embeddings

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add texts to the store.

        Args:
            texts: List of text strings to add.
            metadatas: Optional metadata for each text.
            ids: Optional IDs (auto-generated if not provided).

        Returns:
            List of document IDs.

        Example:
            ```python
            ids = store.add_texts(
                texts=["Hello world", "Goodbye world"],
                metadatas=[{"source": "greeting"}, {"source": "farewell"}]
            )
            ```
        """
        # Create documents
        docs = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else None
            doc_id = ids[i] if ids else None
            docs.append(Document(text=text, metadata=metadata, id=doc_id))

        return self._backend.add_documents(docs)

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add Document objects to the store.

        Args:
            documents: List of Document objects.

        Returns:
            List of document IDs.

        Example:
            ```python
            docs = [
                Document(text="Hello", metadata={"lang": "en"}),
                Document(text="Bonjour", metadata={"lang": "fr"}),
            ]
            ids = store.add_documents(docs)
            ```
        """
        return self._backend.add_documents(documents)

    def search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query: Search query text.
            k: Number of results to return (default 4).
            filter: Optional metadata filter.

        Returns:
            List of SearchResult with document and score.

        Example:
            ```python
            results = store.search("programming", k=3)
            for r in results:
                print(f"{r.score:.2f}: {r.document.text}")
            ```
        """
        return self._backend.search(query, k=k, filter=filter)

    async def asearch(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Async search for similar documents.

        Args:
            query: Search query text.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of SearchResult with document and score.
        """
        return await self._backend.asearch(query, k=k, filter=filter)

    def delete(self, ids: list[str]) -> bool:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.

        Returns:
            True if deletion was successful.

        Example:
            ```python
            store.delete(["doc-id-1", "doc-id-2"])
            ```
        """
        return self._backend.delete(ids)

    def clear(self) -> bool:
        """Delete all documents from the store.

        Returns:
            True if successful.
        """
        return self._backend.clear()

    @property
    def count(self) -> int:
        """Get the number of documents in the store."""
        return self._backend.count()

    def __repr__(self) -> str:
        return f"VectorStore(backend={self._backend_name!r}, embeddings={self._embeddings!r})"


# =============================================================================
# Backend Implementations
# =============================================================================


class _VectorBackend:
    """Base class for vector store backends."""

    def add_documents(self, documents: list[Document]) -> list[str]:
        raise NotImplementedError

    def search(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        raise NotImplementedError

    async def asearch(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        raise NotImplementedError

    def delete(self, ids: list[str]) -> bool:
        raise NotImplementedError

    def clear(self) -> bool:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class _InMemoryBackend(_VectorBackend):
    """Simple in-memory vector store."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._documents: dict[str, Document] = {}
        self._vectors: dict[str, list[float]] = {}

    def add_documents(self, documents: list[Document]) -> list[str]:
        texts = [doc.text for doc in documents]
        vectors = self._embeddings.embed_batch(texts)

        ids = []
        for doc, vector in zip(documents, vectors):
            doc_id = doc.id or str(uuid.uuid4())
            self._documents[doc_id] = doc
            self._vectors[doc_id] = vector
            ids.append(doc_id)

        return ids

    def search(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        if not self._documents:
            return []

        query_vector = self._embeddings.embed(query)

        # Calculate similarities
        scores: list[tuple[str, float]] = []
        for doc_id, doc_vector in self._vectors.items():
            doc = self._documents[doc_id]

            # Apply filter
            if filter:
                if not self._matches_filter(doc.metadata or {}, filter):
                    continue

            score = self._cosine_similarity(query_vector, doc_vector)
            scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        for doc_id, score in scores[:k]:
            results.append(SearchResult(document=self._documents[doc_id], score=score))

        return results

    async def asearch(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        if not self._documents:
            return []

        query_vector = await self._embeddings.aembed(query)

        scores: list[tuple[str, float]] = []
        for doc_id, doc_vector in self._vectors.items():
            doc = self._documents[doc_id]

            if filter:
                if not self._matches_filter(doc.metadata or {}, filter):
                    continue

            score = self._cosine_similarity(query_vector, doc_vector)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scores[:k]:
            results.append(SearchResult(document=self._documents[doc_id], score=score))

        return results

    def delete(self, ids: list[str]) -> bool:
        for doc_id in ids:
            self._documents.pop(doc_id, None)
            self._vectors.pop(doc_id, None)
        return True

    def clear(self) -> bool:
        self._documents.clear()
        self._vectors.clear()
        return True

    def count(self) -> int:
        return len(self._documents)

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        import math

        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class _ChromaBackend(_VectorBackend):
    """Chroma vector store backend."""

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = "default",
        persist_directory: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._embeddings = embeddings

        try:
            import chromadb
        except ImportError as e:
            raise ImportError("Chroma backend requires: pip install chromadb") from e

        # Create client
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        # Create embedding function wrapper
        self._embedding_function = _ChromaEmbeddingFunction(embeddings)

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]

        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        return ids

    def search(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        where = filter if filter else None

        results = self._collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            ids = results["ids"][0] if results["ids"] else [None] * len(docs)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

            for text, doc_id, metadata, distance in zip(docs, ids, metadatas, distances):
                # Convert distance to similarity (Chroma uses L2 distance)
                # Smaller distance = higher similarity
                score = 1.0 / (1.0 + distance)
                doc = Document(text=text, metadata=metadata, id=doc_id)
                search_results.append(SearchResult(document=doc, score=score))

        return search_results

    async def asearch(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        # Chroma doesn't have native async, run in executor
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search(query, k, filter))

    def delete(self, ids: list[str]) -> bool:
        self._collection.delete(ids=ids)
        return True

    def clear(self) -> bool:
        # Delete all by getting all IDs
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
        return True

    def count(self) -> int:
        return cast("int", self._collection.count())


class _ChromaEmbeddingFunction:
    """Wrapper to make Embeddings compatible with Chroma."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embeddings.embed_batch(input)


class _FAISSBackend(_VectorBackend):
    """FAISS vector store backend."""

    def __init__(
        self,
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> None:
        self._embeddings = embeddings
        self._documents: dict[str, Document] = {}
        self._id_to_index: dict[str, int] = {}
        self._index_to_id: dict[int, str] = {}
        self._index: Any = None
        self._dimension: int | None = None

    def _ensure_index(self, dimension: int) -> None:
        """Initialize FAISS index if needed."""
        if self._index is None:
            try:
                import faiss
            except ImportError as e:
                raise ImportError("FAISS backend requires: pip install faiss-cpu") from e

            self._dimension = dimension
            self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine)

    def add_documents(self, documents: list[Document]) -> list[str]:
        import numpy as np

        texts = [doc.text for doc in documents]
        vectors = self._embeddings.embed_batch(texts)

        if not vectors:
            return []

        # Initialize index with first vector's dimension
        self._ensure_index(len(vectors[0]))

        # Normalize vectors for cosine similarity
        vectors_array = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
        vectors_array = vectors_array / norms

        # Add to index
        start_idx = self._index.ntotal
        self._index.add(vectors_array)

        ids = []
        for i, doc in enumerate(documents):
            doc_id = doc.id or str(uuid.uuid4())
            idx = start_idx + i
            self._documents[doc_id] = doc
            self._id_to_index[doc_id] = idx
            self._index_to_id[idx] = doc_id
            ids.append(doc_id)

        return ids

    def search(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        import numpy as np

        if self._index is None or self._index.ntotal == 0:
            return []

        query_vector = self._embeddings.embed(query)
        query_array = np.array([query_vector], dtype=np.float32)

        # Normalize for cosine similarity
        norm = np.linalg.norm(query_array)
        query_array = query_array / norm

        # Search
        search_k = min(k * 2, self._index.ntotal)  # Over-fetch for filtering
        scores, indices = self._index.search(query_array, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty results
                continue

            doc_id = self._index_to_id.get(int(idx))
            if doc_id is None:
                continue

            doc = self._documents.get(doc_id)
            if doc is None:
                continue

            # Apply filter
            if filter:
                if not self._matches_filter(doc.metadata or {}, filter):
                    continue

            results.append(SearchResult(document=doc, score=float(score)))

            if len(results) >= k:
                break

        return results

    async def asearch(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search(query, k, filter))

    def delete(self, ids: list[str]) -> bool:
        # FAISS doesn't support deletion well, we just remove from our mapping
        # The vectors remain in the index but won't be returned
        for doc_id in ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                idx = self._id_to_index.pop(doc_id, None)
                if idx is not None:
                    self._index_to_id.pop(idx, None)
        return True

    def clear(self) -> bool:
        self._documents.clear()
        self._id_to_index.clear()
        self._index_to_id.clear()
        self._index = None
        self._dimension = None
        return True

    def count(self) -> int:
        return len(self._documents)

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
        for key, value in filter.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
