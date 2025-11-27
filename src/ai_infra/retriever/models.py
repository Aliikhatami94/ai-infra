"""Data models for the Retriever module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """A search result from the Retriever.

    Attributes:
        text: The chunk text content.
        score: Similarity score (0-1, higher is more similar).
        metadata: User-provided metadata for this chunk.
        source: Source file path (if loaded from file).
        page: Page number (if from PDF).
        chunk_index: Index of this chunk within the source.
    """

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    page: int | None = None
    chunk_index: int | None = None

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        source_info = f", source={self.source!r}" if self.source else ""
        return f"SearchResult(score={self.score:.3f}, text={text_preview!r}{source_info})"


@dataclass
class Chunk:
    """A chunk of text with metadata, used internally.

    Attributes:
        text: The chunk text content.
        metadata: Metadata about this chunk (source, page, index, etc.).
        embedding: The embedding vector (populated after embedding).
        id: Unique identifier for this chunk.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    id: str | None = None

    @property
    def source(self) -> str | None:
        """Get the source file path from metadata."""
        return self.metadata.get("source")

    @property
    def page(self) -> int | None:
        """Get the page number from metadata."""
        return self.metadata.get("page")

    @property
    def chunk_index(self) -> int | None:
        """Get the chunk index from metadata."""
        return self.metadata.get("chunk_index")

    def to_search_result(self, score: float) -> SearchResult:
        """Convert to a SearchResult with the given score."""
        return SearchResult(
            text=self.text,
            score=score,
            metadata=self.metadata,
            source=self.source,
            page=self.page,
            chunk_index=self.chunk_index,
        )
