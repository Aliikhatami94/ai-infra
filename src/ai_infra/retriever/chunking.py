"""Document chunking utilities for the Retriever module.

Uses LangChain's text splitters internally but exposes a simple API.
"""

from __future__ import annotations

from typing import Any

from ai_infra.retriever.models import Chunk


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split text into chunks with overlap.

    Uses LangChain's RecursiveCharacterTextSplitter internally for
    intelligent splitting at natural boundaries (paragraphs, sentences, etc.).

    Args:
        text: The text to split into chunks.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        metadata: Optional metadata to attach to all chunks.

    Returns:
        List of Chunk objects with text and metadata.

    Example:
        >>> chunks = chunk_text("Long document...", chunk_size=500)
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.chunk_index}: {len(chunk.text)} chars")
    """
    # Import LangChain splitter lazily
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as e:
        raise ImportError(
            "langchain-text-splitters is required for chunking. "
            "Install it with: pip install langchain-text-splitters"
        ) from e

    # Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split text
    texts = splitter.split_text(text)

    # Create chunks with metadata
    base_metadata = metadata or {}
    chunks = []
    for i, chunk_text in enumerate(texts):
        chunk_metadata = {
            **base_metadata,
            "chunk_index": i,
            "total_chunks": len(texts),
        }
        chunks.append(
            Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
            )
        )

    return chunks


def chunk_documents(
    documents: list[tuple[str, dict[str, Any]]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split multiple documents into chunks.

    Args:
        documents: List of (text, metadata) tuples.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of Chunk objects from all documents.
    """
    all_chunks = []
    for text, metadata in documents:
        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=metadata,
        )
        all_chunks.extend(chunks)
    return all_chunks


def estimate_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
    """Estimate the number of chunks for a given text.

    This is a rough estimate without actually performing the split.

    Args:
        text: The text to estimate chunks for.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        Estimated number of chunks.
    """
    if not text:
        return 0
    text_len = len(text)
    if text_len <= chunk_size:
        return 1
    # Account for overlap
    effective_chunk_size = chunk_size - chunk_overlap
    return max(1, (text_len - chunk_overlap) // effective_chunk_size + 1)
