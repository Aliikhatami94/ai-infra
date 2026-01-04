"""Benchmark fixtures and configuration for ai-infra."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Sample messages for LLM benchmarks."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for embedding benchmarks."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming industries.",
        "Python is a popular programming language.",
        "Natural language processing enables text understanding.",
        "Vector embeddings capture semantic meaning.",
    ] * 20  # 100 texts for batch testing


@pytest.fixture
def sample_documents() -> list[str]:
    """Sample documents for retriever benchmarks."""
    return [
        f"Document {i}: This is a sample document about topic {i % 10}. "
        f"It contains relevant information for testing retrieval accuracy."
        for i in range(100)
    ]


@pytest.fixture
def sample_query() -> str:
    """Sample query for search benchmarks."""
    return "What is the topic of the document?"
