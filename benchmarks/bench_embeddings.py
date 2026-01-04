"""Benchmarks for embeddings operations.

Run with:
    make benchmark
    pytest benchmarks/bench_embeddings.py --benchmark-only
"""

from __future__ import annotations

import pytest


class TestEmbeddingsInit:
    """Benchmark Embeddings initialization."""

    def test_embeddings_import(self, benchmark):
        """Benchmark embeddings module import time."""

        def import_embeddings():
            import importlib

            import ai_infra.embeddings

            importlib.reload(ai_infra.embeddings)

        benchmark(import_embeddings)

    @pytest.mark.skip(reason="Requires API key - run manually")
    def test_embeddings_init(self, benchmark):
        """Benchmark Embeddings class instantiation."""
        from ai_infra import Embeddings

        def create_embeddings():
            return Embeddings()

        result = benchmark(create_embeddings)
        assert result is not None


class TestVectorStore:
    """Benchmark VectorStore operations."""

    def test_vectorstore_init(self, benchmark, sample_texts):
        """Benchmark VectorStore initialization."""
        from ai_infra.embeddings import VectorStore

        # Mock embeddings function for benchmarking
        def mock_embed(text: str) -> list[float]:
            # Simple hash-based mock embedding
            h = hash(text)
            return [(h >> i) & 0xFF for i in range(0, 64, 8)]

        def mock_embed_batch(texts: list[str]) -> list[list[float]]:
            return [mock_embed(t) for t in texts]

        class MockEmbeddings:
            def embed(self, text: str) -> list[float]:
                return mock_embed(text)

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return mock_embed_batch(texts)

        def create_store():
            return VectorStore(embeddings=MockEmbeddings())

        result = benchmark(create_store)
        assert result is not None


class TestRetrieverOperations:
    """Benchmark Retriever operations."""

    def test_retriever_import(self, benchmark):
        """Benchmark retriever module import time."""

        def import_retriever():
            import importlib

            import ai_infra.retriever

            importlib.reload(ai_infra.retriever)

        benchmark(import_retriever)
