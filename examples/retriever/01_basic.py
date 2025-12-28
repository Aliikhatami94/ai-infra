#!/usr/bin/env python
"""Basic Retriever Example.

This example demonstrates:
- Zero-config retriever setup
- Adding texts with add() and add_text()
- Basic similarity search with search()
- Getting context for LLM with get_context()
- Understanding search results

The Retriever is your all-in-one RAG solution - it handles
embeddings, vector storage, and similarity search automatically.
"""

from ai_infra import Retriever

# =============================================================================
# Example 1: Zero-Config Setup
# =============================================================================


def zero_config():
    """Create a retriever with automatic configuration."""
    print("=" * 60)
    print("1. Zero-Config Retriever Setup")
    print("=" * 60)

    # Simplest possible setup - uses environment defaults
    retriever = Retriever()

    print(f"\n  Embedding provider: {retriever.embeddings.provider}")
    print(f"  Embedding model: {retriever.embeddings.model}")
    print(f"  Backend: {retriever.backend}")

    # Add some content
    retriever.add("Python is great for data science")
    retriever.add("JavaScript powers the web")

    print("\n✓ Retriever ready with 2 documents")


# =============================================================================
# Example 2: Adding Content
# =============================================================================


def adding_content():
    """Demonstrate different ways to add content."""
    print("\n" + "=" * 60)
    print("2. Adding Content")
    print("=" * 60)

    retriever = Retriever()

    # Method 1: add() - simple text
    print("\nMethod 1: add() for simple text")
    retriever.add("Machine learning is a subset of AI")
    print("  ✓ Added single text")

    # Method 2: add() with multiple texts
    print("\nMethod 2: add() with multiple texts")
    retriever.add(
        [
            "Deep learning uses neural networks",
            "Natural language processing handles text",
            "Computer vision processes images",
        ]
    )
    print("  ✓ Added 3 texts as list")

    # Method 3: add_text() with metadata
    print("\nMethod 3: add_text() with metadata")
    retriever.add_text(
        "Reinforcement learning learns from rewards",
        metadata={"category": "ml", "level": "advanced"},
    )
    print("  ✓ Added text with metadata")

    # Method 4: add_text() with source
    print("\nMethod 4: add_text() with source")
    retriever.add_text(
        "Transfer learning reuses trained models",
        source="ml_textbook.pdf",
    )
    print("  ✓ Added text with source reference")

    print(f"\nTotal documents: {len(retriever)}")


# =============================================================================
# Example 3: Basic Search
# =============================================================================


def basic_search():
    """Perform basic similarity search."""
    print("\n" + "=" * 60)
    print("3. Basic Similarity Search")
    print("=" * 60)

    retriever = Retriever()

    # Add knowledge base
    docs = [
        "Python is excellent for data analysis and machine learning",
        "JavaScript is the language of web browsers",
        "Rust provides memory safety without garbage collection",
        "Go was designed for concurrent programming at Google",
        "TypeScript adds static typing to JavaScript",
        "Java is popular in enterprise applications",
        "C++ offers high performance for systems programming",
    ]
    retriever.add(docs)

    print(f"\nKnowledge base: {len(docs)} documents")

    # Search for similar documents
    query = "best language for AI development"
    print(f"\nQuery: '{query}'")

    results = retriever.search(query, k=3)

    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. Score: {result.score:.3f}")
        print(f"     Text: {result.document.text}")


# =============================================================================
# Example 4: Understanding Search Results
# =============================================================================


def search_results():
    """Explore search result properties."""
    print("\n" + "=" * 60)
    print("4. Understanding Search Results")
    print("=" * 60)

    retriever = Retriever()

    # Add documents with metadata
    retriever.add_text(
        "Python has simple syntax and is beginner-friendly",
        metadata={"language": "python", "difficulty": "easy"},
        source="programming_guide.md",
    )
    retriever.add_text(
        "Machine learning requires understanding of statistics",
        metadata={"topic": "ml", "difficulty": "hard"},
        source="ml_intro.pdf",
    )

    results = retriever.search("easy programming language", k=1)

    if results:
        result = results[0]

        print("\nSearchResult properties:")
        print(f"  score:    {result.score:.4f} (0-1, higher is better)")
        print(f"  text:     {result.document.text[:50]}...")
        print(f"  metadata: {result.document.metadata}")
        print(f"  source:   {result.document.source}")


# =============================================================================
# Example 5: Get Context for LLM
# =============================================================================


def get_context():
    """Use get_context() for LLM prompts."""
    print("\n" + "=" * 60)
    print("5. Get Context for LLM")
    print("=" * 60)

    retriever = Retriever()

    # Add knowledge about a product
    retriever.add(
        [
            "The Widget Pro costs $99 and includes free shipping",
            "Widget Pro has a 2-year warranty on all parts",
            "Widget Pro is available in blue, red, and black colors",
            "Widget Pro weighs 1.5 pounds and measures 10x5x3 inches",
            "Contact support@widget.com for Widget Pro issues",
        ]
    )

    # Get formatted context for LLM
    query = "What colors does Widget Pro come in?"
    context = retriever.get_context(query, k=2)

    print(f"\nQuery: '{query}'")
    print("\nContext for LLM:")
    print("-" * 40)
    print(context)
    print("-" * 40)

    # Example of using in a prompt
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

    print("\nExample prompt for LLM:")
    print(prompt)


# =============================================================================
# Example 6: Filtering by Metadata
# =============================================================================


def metadata_filtering():
    """Filter search results by metadata."""
    print("\n" + "=" * 60)
    print("6. Metadata Filtering")
    print("=" * 60)

    retriever = Retriever()

    # Add documents with categories
    docs = [
        {
            "text": "Python basics for beginners",
            "metadata": {"category": "tutorial", "lang": "python"},
        },
        {
            "text": "Advanced Python decorators",
            "metadata": {"category": "advanced", "lang": "python"},
        },
        {
            "text": "JavaScript async/await guide",
            "metadata": {"category": "tutorial", "lang": "js"},
        },
        {
            "text": "Python web frameworks overview",
            "metadata": {"category": "guide", "lang": "python"},
        },
        {"text": "JavaScript DOM manipulation", "metadata": {"category": "tutorial", "lang": "js"}},
    ]

    for doc in docs:
        retriever.add_text(doc["text"], metadata=doc["metadata"])

    print(f"\nAdded {len(docs)} documents with metadata")

    # Search with metadata filter
    print("\nSearch 'programming' filtered by lang='python':")
    results = retriever.search("programming", k=3, filter={"lang": "python"})

    for result in results:
        print(f"  {result.score:.3f}: {result.document.text}")


# =============================================================================
# Example 7: Controlling Result Count
# =============================================================================


def result_count():
    """Control how many results to return."""
    print("\n" + "=" * 60)
    print("7. Controlling Result Count")
    print("=" * 60)

    retriever = Retriever()

    # Add a larger set of documents
    topics = ["AI", "ML", "DL", "NLP", "CV", "RL", "robotics", "automation"]
    for topic in topics:
        retriever.add(f"{topic} is an important area of technology")

    query = "artificial intelligence"

    # Different k values
    for k in [1, 3, 5]:
        results = retriever.search(query, k=k)
        print(f"\nk={k}: Found {len(results)} results")
        for r in results:
            print(f"  {r.score:.3f}: {r.document.text[:40]}...")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Basic Retriever Examples")
    print("=" * 60)
    print("\nThe Retriever is your all-in-one RAG solution.")
    print("It handles embeddings, storage, and search automatically.\n")

    zero_config()
    adding_content()
    basic_search()
    search_results()
    get_context()
    metadata_filtering()
    result_count()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Retriever() works out of the box (zero config)")
    print("  2. add() and add_text() for adding content")
    print("  3. search() returns ranked results with scores")
    print("  4. get_context() formats results for LLM prompts")
    print("  5. Use metadata for filtering and organization")


if __name__ == "__main__":
    main()
