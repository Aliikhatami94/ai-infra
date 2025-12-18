"""Integration tests for Embeddings providers.

These tests require API keys and make real API calls.
They are skipped by default unless explicitly enabled via environment variables.

Run with: pytest tests/integration/test_embeddings.py -v
"""

from __future__ import annotations

import os

import pytest

from ai_infra.embeddings import Embeddings

# Skip markers for different providers
SKIP_NO_OPENAI = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

SKIP_NO_VOYAGE = pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY not set",
)

SKIP_NO_COHERE = pytest.mark.skipif(
    not os.environ.get("COHERE_API_KEY"),
    reason="COHERE_API_KEY not set",
)

SKIP_NO_GOOGLE = pytest.mark.skipif(
    not (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
    ),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not set",
)


# =============================================================================
# OpenAI Embeddings Tests
# =============================================================================


@SKIP_NO_OPENAI
@pytest.mark.integration
class TestOpenAIEmbeddings:
    """Integration tests for OpenAI embeddings."""

    def test_embed_single_text(self):
        """Test embedding a single text."""
        embeddings = Embeddings(provider="openai")
        vector = embeddings.embed("Hello, world!")

        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    def test_embed_batch(self):
        """Test embedding multiple texts."""
        embeddings = Embeddings(provider="openai")
        texts = ["Hello", "World", "Test"]
        vectors = embeddings.embed_batch(texts)

        assert isinstance(vectors, list)
        assert len(vectors) == 3
        assert all(isinstance(v, list) for v in vectors)
        assert all(len(v) > 0 for v in vectors)

    @pytest.mark.asyncio
    async def test_async_embed(self):
        """Test async embedding."""
        embeddings = Embeddings(provider="openai")
        vector = await embeddings.aembed("Hello, async world!")

        assert isinstance(vector, list)
        assert len(vector) > 0

    @pytest.mark.asyncio
    async def test_async_embed_batch(self):
        """Test async batch embedding."""
        embeddings = Embeddings(provider="openai")
        texts = ["Async", "Batch", "Test"]
        vectors = await embeddings.aembed_batch(texts)

        assert isinstance(vectors, list)
        assert len(vectors) == 3

    def test_cosine_similarity(self):
        """Test that similar texts have higher similarity."""
        embeddings = Embeddings(provider="openai")

        v1 = embeddings.embed("The cat sat on the mat")
        v2 = embeddings.embed("A feline rested on the rug")
        v3 = embeddings.embed("Quantum physics explains particles")

        # Calculate cosine similarity
        from ai_infra.embeddings.embeddings import cosine_similarity

        sim_similar = cosine_similarity(v1, v2)
        sim_different = cosine_similarity(v1, v3)

        # Similar sentences should have higher similarity
        assert sim_similar > sim_different

    def test_embedding_dimensions(self):
        """Test embedding dimensions match expected."""
        embeddings = Embeddings(provider="openai", model="text-embedding-3-small")
        vector = embeddings.embed("Test text")

        # text-embedding-3-small has 1536 dimensions by default
        assert len(vector) == 1536

    def test_custom_dimensions(self):
        """Test custom embedding dimensions (OpenAI-specific feature)."""
        embeddings = Embeddings(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=512,  # Reduced dimensions
        )
        vector = embeddings.embed("Test text")

        assert len(vector) == 512


# =============================================================================
# Google Embeddings Tests
# =============================================================================


@SKIP_NO_GOOGLE
@pytest.mark.integration
class TestGoogleEmbeddings:
    """Integration tests for Google embeddings."""

    def test_embed_single_text(self):
        """Test embedding a single text."""
        embeddings = Embeddings(provider="google")
        vector = embeddings.embed("Hello, world!")

        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    def test_embed_batch(self):
        """Test embedding multiple texts."""
        embeddings = Embeddings(provider="google")
        texts = ["Hello", "World", "Test"]
        vectors = embeddings.embed_batch(texts)

        assert isinstance(vectors, list)
        assert len(vectors) == 3

    @pytest.mark.asyncio
    async def test_async_embed(self):
        """Test async embedding."""
        embeddings = Embeddings(provider="google")
        vector = await embeddings.aembed("Hello, async world!")

        assert isinstance(vector, list)
        assert len(vector) > 0


# =============================================================================
# Voyage Embeddings Tests
# =============================================================================


@SKIP_NO_VOYAGE
@pytest.mark.integration
class TestVoyageEmbeddings:
    """Integration tests for Voyage AI embeddings."""

    def test_embed_single_text(self):
        """Test embedding a single text."""
        embeddings = Embeddings(provider="voyage")
        vector = embeddings.embed("Hello, world!")

        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    def test_embed_batch(self):
        """Test embedding multiple texts."""
        embeddings = Embeddings(provider="voyage")
        texts = ["Hello", "World", "Test"]
        vectors = embeddings.embed_batch(texts)

        assert isinstance(vectors, list)
        assert len(vectors) == 3


# =============================================================================
# Cohere Embeddings Tests
# =============================================================================


@SKIP_NO_COHERE
@pytest.mark.integration
class TestCohereEmbeddings:
    """Integration tests for Cohere embeddings."""

    def test_embed_single_text(self):
        """Test embedding a single text."""
        embeddings = Embeddings(provider="cohere")
        vector = embeddings.embed("Hello, world!")

        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    def test_embed_batch(self):
        """Test embedding multiple texts."""
        embeddings = Embeddings(provider="cohere")
        texts = ["Hello", "World", "Test"]
        vectors = embeddings.embed_batch(texts)

        assert isinstance(vectors, list)
        assert len(vectors) == 3


# =============================================================================
# Provider Discovery Tests
# =============================================================================


@pytest.mark.integration
class TestEmbeddingsProviderDiscovery:
    """Tests for embeddings provider discovery."""

    def test_list_providers(self):
        """Test listing all available providers."""
        providers = Embeddings.list_providers()

        assert isinstance(providers, list)
        assert "openai" in providers
        assert "google" in providers
        assert "voyage" in providers
        assert "cohere" in providers

    def test_list_configured_providers(self):
        """Test listing configured providers."""
        providers = Embeddings.list_configured_providers()

        assert isinstance(providers, list)
        # At least one provider should be configured (huggingface is always available)
        assert len(providers) > 0


# =============================================================================
# VectorStore Integration Tests
# =============================================================================


@SKIP_NO_OPENAI
@pytest.mark.integration
class TestVectorStoreIntegration:
    """Integration tests for VectorStore with real embeddings."""

    def test_add_and_search(self):
        """Test adding documents and searching."""
        from ai_infra.embeddings import VectorStore

        embeddings = Embeddings(provider="openai")
        store = VectorStore(embeddings=embeddings)

        # Add documents
        docs = [
            "Python is a programming language",
            "Machine learning uses algorithms",
            "The weather is sunny today",
        ]
        store.add_texts(docs)

        # Search for programming-related content
        results = store.search("coding in Python", k=2)

        assert len(results) > 0
        # First result should be about Python
        assert (
            "python" in results[0][0].lower() or "programming" in results[0][0].lower()
        )

    def test_search_with_metadata(self):
        """Test search returns metadata."""
        from ai_infra.embeddings import VectorStore

        embeddings = Embeddings(provider="openai")
        store = VectorStore(embeddings=embeddings)

        # Add documents with metadata
        docs = ["Document about AI", "Document about databases"]
        metadatas = [{"topic": "ai"}, {"topic": "databases"}]
        store.add_texts(docs, metadatas=metadatas)

        results = store.search("artificial intelligence", k=1)

        assert len(results) > 0
        # Result should include the document text
        assert "AI" in results[0][0] or "ai" in results[0][0].lower()
