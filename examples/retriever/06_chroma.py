#!/usr/bin/env python
"""Chroma Backend for Retriever Example.

This example demonstrates:
- Using Chroma for vector storage
- In-memory and persistent modes
- Collection management
- Metadata filtering
- Local-first development

Chroma is an open-source embedding database that runs locally
or in a server. Great for development and small-to-medium deployments.

Requirements:
  pip install 'ai-infra[chroma]'
"""

import tempfile
from pathlib import Path

from ai_infra import Retriever

# =============================================================================
# Example 1: In-Memory Chroma
# =============================================================================


def in_memory_chroma():
    """Use Chroma in ephemeral in-memory mode."""
    print("=" * 60)
    print("1. In-Memory Chroma")
    print("=" * 60)

    # Default Chroma is in-memory (ephemeral)
    retriever = Retriever(backend="chroma")

    print(f"\n  Backend: {retriever.backend}")
    print("  Mode: In-memory (ephemeral)")

    # Add documents
    retriever.add(
        [
            "Chroma is an open-source embedding database",
            "Vector search enables semantic retrieval",
            "Embeddings capture meaning in dense vectors",
        ]
    )

    print(f"\n  Added {len(retriever)} documents")

    # Search
    results = retriever.search("semantic search", k=2)
    print("\n  Search 'semantic search':")
    for r in results:
        print(f"    {r.score:.3f}: {r.document.text[:50]}...")


# =============================================================================
# Example 2: Persistent Chroma
# =============================================================================


def persistent_chroma():
    """Use Chroma with persistent storage."""
    print("\n" + "=" * 60)
    print("2. Persistent Chroma")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir) / "chroma_data"

        # First session
        print("\n[Session 1] Creating persistent collection...")
        retriever1 = Retriever(
            backend="chroma",
            path=str(persist_dir),
        )

        retriever1.add(
            [
                "First document in persistent storage",
                "Second document saved to disk",
            ]
        )
        print(f"  Added {len(retriever1)} documents")

        # Close first session
        del retriever1

        # Second session
        print("\n[Session 2] Reopening persistent collection...")
        retriever2 = Retriever(
            backend="chroma",
            path=str(persist_dir),
        )

        print(f"  Found {len(retriever2)} existing documents")
        print("\n✓ Data persisted between sessions!")


# =============================================================================
# Example 3: Named Collections
# =============================================================================


def named_collections():
    """Work with multiple named collections."""
    print("\n" + "=" * 60)
    print("3. Named Collections")
    print("=" * 60)

    # Documentation collection
    docs_retriever = Retriever(backend="chroma", collection="documentation")
    docs_retriever.add(
        [
            "API documentation for developers",
            "Installation guide and requirements",
        ]
    )

    # FAQ collection
    faq_retriever = Retriever(backend="chroma", collection="faq")
    faq_retriever.add(
        [
            "How do I reset my password?",
            "What payment methods are accepted?",
        ]
    )

    print("\n  Collections created:")
    print(f"    documentation: {len(docs_retriever)} docs")
    print(f"    faq: {len(faq_retriever)} docs")

    # Search specific collection
    results = docs_retriever.search("API", k=1)
    if results:
        print(f"\n  Documentation search: {results[0].document.text}")

    results = faq_retriever.search("password", k=1)
    if results:
        print(f"  FAQ search: {results[0].document.text}")


# =============================================================================
# Example 4: Metadata Filtering
# =============================================================================


def metadata_filtering():
    """Filter search results with metadata."""
    print("\n" + "=" * 60)
    print("4. Metadata Filtering")
    print("=" * 60)

    retriever = Retriever(backend="chroma")

    # Add documents with metadata
    docs = [
        ("Python tutorial for beginners", {"lang": "python", "level": "beginner"}),
        ("Advanced Python decorators", {"lang": "python", "level": "advanced"}),
        ("JavaScript async patterns", {"lang": "javascript", "level": "intermediate"}),
        ("Python data structures", {"lang": "python", "level": "intermediate"}),
    ]

    for text, metadata in docs:
        retriever.add_text(text, metadata=metadata)

    print(f"\n  Added {len(docs)} documents with metadata")

    # Filter by language
    print("\n  Search 'programming' filtered by lang='python':")
    results = retriever.search("programming", k=3, filter={"lang": "python"})
    for r in results:
        print(f"    {r.score:.3f}: {r.document.text} ({r.document.metadata})")

    # Filter by level
    print("\n  Search 'tutorial' filtered by level='beginner':")
    results = retriever.search("tutorial", k=2, filter={"level": "beginner"})
    for r in results:
        print(f"    {r.score:.3f}: {r.document.text}")


# =============================================================================
# Example 5: Chroma Client Modes
# =============================================================================


def client_modes():
    """Different Chroma client configurations."""
    print("\n" + "=" * 60)
    print("5. Chroma Client Modes")
    print("=" * 60)

    modes = {
        "Ephemeral (in-memory)": {
            "config": 'Retriever(backend="chroma")',
            "use_case": "Testing, development, temporary data",
        },
        "Persistent (local)": {
            "config": 'Retriever(backend="chroma", path="./chroma_data")',
            "use_case": "Local development, single-user apps",
        },
        "Client-Server": {
            "config": 'Retriever(backend="chroma", host="localhost", port=8000)',
            "use_case": "Multi-process access, team development",
        },
    }

    print("\nChroma can run in different modes:")
    for mode, config in modes.items():
        print(f"\n  {mode}:")
        print(f"    Config: {config['config']}")
        print(f"    Use case: {config['use_case']}")


# =============================================================================
# Example 6: Document Updates
# =============================================================================


def document_updates():
    """Update and delete documents."""
    print("\n" + "=" * 60)
    print("6. Document Updates")
    print("=" * 60)

    retriever = Retriever(backend="chroma")

    # Add initial documents with IDs
    retriever.add_text(
        "Original product description v1",
        metadata={"doc_id": "product_1", "version": "1"},
    )

    print("\n  Initial document added")

    results = retriever.search("product", k=1)
    print(f"  Current: {results[0].document.text}")

    # Note: For updates, typically you'd delete and re-add
    # or use document IDs for replacement
    print("\n  (For updates, delete old and add new document)")


# =============================================================================
# Example 7: Batch Operations
# =============================================================================


def batch_operations():
    """Efficient batch operations."""
    print("\n" + "=" * 60)
    print("7. Batch Operations")
    print("=" * 60)

    import time

    retriever = Retriever(backend="chroma")

    # Generate documents
    num_docs = 200
    docs = [f"Document {i} about topic {i % 20}" for i in range(num_docs)]

    # Batch add
    print(f"\n  Adding {num_docs} documents in batch...")
    start = time.time()
    retriever.add(docs)
    elapsed = time.time() - start

    print(f"  ✓ Added in {elapsed:.2f}s ({num_docs / elapsed:.0f} docs/sec)")

    # Batch search
    queries = [f"topic {i}" for i in range(10)]

    print(f"\n  Running {len(queries)} searches...")
    start = time.time()
    for query in queries:
        retriever.search(query, k=5)
    elapsed = time.time() - start

    print(f"  ✓ Searched in {elapsed:.2f}s ({len(queries) / elapsed:.0f} queries/sec)")


# =============================================================================
# Example 8: Integration with LLM
# =============================================================================


def llm_integration():
    """Use Chroma retriever with LLM context."""
    print("\n" + "=" * 60)
    print("8. LLM Integration")
    print("=" * 60)

    retriever = Retriever(backend="chroma")

    # Build knowledge base
    retriever.add(
        [
            "The Widget Pro costs $99 and includes a 2-year warranty",
            "Free shipping is available on all orders over $50",
            "Returns are accepted within 30 days of purchase",
            "Customer support is available 24/7 via chat or phone",
        ]
    )

    query = "What is the return policy?"

    # Get context for LLM
    context = retriever.get_context(query, k=2)

    # Build prompt for LLM (shown as example)
    _prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

    print(f"\n  Query: {query}")
    print("\n  Context retrieved:")
    print(f"  {context[:200]}...")
    print("\n  Ready to send to LLM!")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Chroma Backend Examples")
    print("=" * 60)
    print("\nChroma is an open-source embedding database:")
    print("  - Runs locally or as server")
    print("  - In-memory or persistent modes")
    print("  - Great for development")
    print("  - Easy metadata filtering\n")

    in_memory_chroma()
    persistent_chroma()
    named_collections()
    metadata_filtering()
    client_modes()
    document_updates()
    batch_operations()
    llm_integration()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Default mode is in-memory (ephemeral)")
    print("  2. Use path parameter for persistence")
    print("  3. Named collections separate different data")
    print("  4. Powerful metadata filtering")
    print("  5. Easy local development, easy to scale")


if __name__ == "__main__":
    main()
