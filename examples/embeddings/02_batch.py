#!/usr/bin/env python
"""Batch Embeddings Example.

This example demonstrates:
- Batch embedding for multiple texts
- Async batch embedding for concurrent processing
- Performance comparison between single and batch
- Different providers with batch operations
- VectorStore integration for similarity search

Batch embedding is more efficient than embedding texts one at a time,
especially when using API-based providers.
"""

import asyncio
import time

from ai_infra import Embeddings, VectorStore
from ai_infra.embeddings.vectorstore import Document

# =============================================================================
# Example 1: Basic Batch Embedding
# =============================================================================


def basic_batch():
    """Embed multiple texts in a single batch."""
    print("=" * 60)
    print("1. Basic Batch Embedding")
    print("=" * 60)

    embeddings = Embeddings()

    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing understands human language",
        "Computer vision enables machines to interpret images",
        "Reinforcement learning learns through trial and error",
    ]

    print(f"\nProvider: {embeddings.provider}")
    print(f"Texts to embed: {len(texts)}")

    # Batch embedding - single API call for all texts
    start = time.time()
    vectors = embeddings.embed_batch(texts)
    elapsed = time.time() - start

    print(f"\n✓ Generated {len(vectors)} embeddings in {elapsed:.2f}s")
    print(f"  Dimensions per vector: {len(vectors[0])}")

    # Show preview
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        print(f"  {i + 1}. '{text[:40]}...' → [{vector[0]:.4f}, ...]")


# =============================================================================
# Example 2: Single vs Batch Performance
# =============================================================================


def performance_comparison():
    """Compare single embedding vs batch embedding performance."""
    print("\n" + "=" * 60)
    print("2. Single vs Batch Performance")
    print("=" * 60)

    embeddings = Embeddings()

    # Generate sample texts
    texts = [f"Sample document number {i} with some content" for i in range(20)]

    print(f"\nEmbedding {len(texts)} texts...")

    # Single embedding (one at a time)
    start = time.time()
    single_vectors = [embeddings.embed(text) for text in texts]
    single_time = time.time() - start

    # Batch embedding (all at once)
    start = time.time()
    batch_vectors = embeddings.embed_batch(texts)
    batch_time = time.time() - start

    print(f"\n  Single embeddings: {single_time:.2f}s")
    print(f"  Batch embeddings:  {batch_time:.2f}s")

    if single_time > 0:
        speedup = single_time / batch_time
        print(f"  Speedup: {speedup:.1f}x faster with batch")

    # Verify results are the same
    match = all(
        abs(s - b) < 0.0001 for sv, bv in zip(single_vectors, batch_vectors) for s, b in zip(sv, bv)
    )
    print(f"\n  Results match: {match}")


# =============================================================================
# Example 3: Async Batch Embedding
# =============================================================================


async def async_batch():
    """Use async batch embedding for non-blocking operations."""
    print("\n" + "=" * 60)
    print("3. Async Batch Embedding")
    print("=" * 60)

    embeddings = Embeddings()

    texts = [
        "Async programming improves concurrency",
        "Coroutines enable non-blocking I/O",
        "Event loops manage async execution",
        "Futures represent pending results",
        "Tasks wrap coroutines for execution",
    ]

    print(f"\nAsync embedding {len(texts)} texts...")

    # Async batch embedding
    start = time.time()
    vectors = await embeddings.aembed_batch(texts)
    elapsed = time.time() - start

    print(f"\n✓ Completed in {elapsed:.2f}s")
    print(f"  Generated {len(vectors)} vectors of {len(vectors[0])} dimensions")


# =============================================================================
# Example 4: Large Batch Handling
# =============================================================================


def large_batch():
    """Handle large batches of text."""
    print("\n" + "=" * 60)
    print("4. Large Batch Handling")
    print("=" * 60)

    embeddings = Embeddings()

    # Generate a large batch
    num_texts = 100
    texts = [
        f"Document {i}: This is sample content for testing embeddings" for i in range(num_texts)
    ]

    print(f"\nEmbedding {num_texts} texts...")

    start = time.time()
    vectors = embeddings.embed_batch(texts)
    elapsed = time.time() - start

    print(f"\n✓ Embedded {len(vectors)} texts in {elapsed:.2f}s")
    print(f"  Rate: {len(vectors) / elapsed:.1f} texts/second")
    print(f"  Dimensions: {len(vectors[0])}")


# =============================================================================
# Example 5: VectorStore Integration
# =============================================================================


def vectorstore_integration():
    """Use embeddings with VectorStore for similarity search."""
    print("\n" + "=" * 60)
    print("5. VectorStore Integration")
    print("=" * 60)

    # Create embeddings and vector store
    embeddings = Embeddings()
    store = VectorStore(embeddings=embeddings)

    # Sample documents about programming languages
    documents = [
        "Python is great for data science and machine learning",
        "JavaScript powers interactive web applications",
        "Rust provides memory safety without garbage collection",
        "Go is designed for concurrent programming",
        "TypeScript adds static typing to JavaScript",
        "Java is widely used in enterprise applications",
        "C++ offers high performance and low-level control",
        "Ruby focuses on developer happiness and productivity",
    ]

    print(f"\nAdding {len(documents)} documents to store...")

    # Add documents (automatically batches embeddings)
    store.add_texts(documents)

    print("✓ Documents added")

    # Search for similar documents
    queries = [
        "best language for AI",
        "web development",
        "fast compiled language",
    ]

    print("\nSimilarity search:")
    for query in queries:
        print(f"\n  Query: '{query}'")
        results = store.search(query, k=2)
        for result in results:
            print(f"    {result.score:.3f}: {result.document.text[:50]}...")


# =============================================================================
# Example 6: Batch with Metadata
# =============================================================================


def batch_with_metadata():
    """Batch embed documents with metadata."""
    print("\n" + "=" * 60)
    print("6. Batch Embedding with Metadata")
    print("=" * 60)

    embeddings = Embeddings()
    store = VectorStore(embeddings=embeddings)

    # Documents with metadata
    docs = [
        Document(
            text="Python tutorial for beginners",
            metadata={"category": "tutorial", "lang": "python"},
        ),
        Document(
            text="Advanced JavaScript patterns",
            metadata={"category": "guide", "lang": "javascript"},
        ),
        Document(
            text="Python data analysis cookbook",
            metadata={"category": "cookbook", "lang": "python"},
        ),
        Document(
            text="JavaScript testing best practices",
            metadata={"category": "guide", "lang": "javascript"},
        ),
        Document(
            text="Python web framework comparison",
            metadata={"category": "comparison", "lang": "python"},
        ),
    ]

    print(f"\nAdding {len(docs)} documents with metadata...")

    # Add documents
    store.add_documents(docs)

    print("✓ Documents added")

    # Search with metadata filter
    print("\nSearch 'programming' filtered by lang='python':")
    results = store.search("programming", k=3, filter={"lang": "python"})

    for result in results:
        meta = result.document.metadata
        print(f"  {result.score:.3f}: {result.document.text}")
        print(f"          metadata: {meta}")


# =============================================================================
# Example 7: Multiple Providers Batch
# =============================================================================


def multi_provider_batch():
    """Compare batch embedding across providers."""
    print("\n" + "=" * 60)
    print("7. Multi-Provider Batch Comparison")
    print("=" * 60)

    import os

    texts = [
        "Artificial intelligence is changing the world",
        "Machine learning enables predictive analytics",
        "Neural networks learn from data",
    ]

    # Collect available providers
    providers = [("huggingface", Embeddings(provider="huggingface"))]

    if os.getenv("OPENAI_API_KEY"):
        providers.append(("openai", Embeddings(provider="openai")))
    if os.getenv("VOYAGE_API_KEY"):
        providers.append(("voyage", Embeddings(provider="voyage")))

    print(f"\nBatch embedding {len(texts)} texts with {len(providers)} providers:\n")

    for name, emb in providers:
        start = time.time()
        vectors = emb.embed_batch(texts)
        elapsed = time.time() - start

        print(f"  {name}:")
        print(f"    Model: {emb.model}")
        print(f"    Dimensions: {len(vectors[0])}")
        print(f"    Time: {elapsed:.3f}s")


# =============================================================================
# Example 8: Chunked Batch Processing
# =============================================================================


def chunked_batch():
    """Process very large datasets in chunks."""
    print("\n" + "=" * 60)
    print("8. Chunked Batch Processing")
    print("=" * 60)

    embeddings = Embeddings()

    # Simulate a large dataset
    num_docs = 500
    all_texts = [f"Document {i} with content about topic {i % 10}" for i in range(num_docs)]

    chunk_size = 100
    all_vectors = []

    print(f"\nProcessing {num_docs} documents in chunks of {chunk_size}...")

    start = time.time()
    for i in range(0, len(all_texts), chunk_size):
        chunk = all_texts[i : i + chunk_size]
        vectors = embeddings.embed_batch(chunk)
        all_vectors.extend(vectors)
        print(f"  Processed {min(i + chunk_size, num_docs)}/{num_docs}")

    elapsed = time.time() - start

    print(f"\n✓ Completed in {elapsed:.2f}s")
    print(f"  Total vectors: {len(all_vectors)}")
    print(f"  Rate: {len(all_vectors) / elapsed:.1f} docs/second")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Batch Embeddings Examples")
    print("=" * 60)
    print("\nBatch embedding is more efficient than one-at-a-time.")
    print("Especially important for API-based providers.\n")

    basic_batch()
    performance_comparison()
    await async_batch()
    large_batch()
    vectorstore_integration()
    batch_with_metadata()
    multi_provider_batch()
    chunked_batch()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. embed_batch() is faster than multiple embed() calls")
    print("  2. Use aembed_batch() for async operations")
    print("  3. VectorStore handles batching automatically")
    print("  4. Process large datasets in chunks")


if __name__ == "__main__":
    asyncio.run(main())
