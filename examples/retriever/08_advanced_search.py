#!/usr/bin/env python
"""Advanced Search Features Example.

This example demonstrates:
- Minimum score thresholds for quality filtering
- Detailed search results with full metadata
- Hybrid search (vector + keyword)
- Reranking for improved accuracy
- Search with context windows
- Multi-query strategies

These features help you get more relevant and useful
search results in production applications.
"""

from ai_infra import Retriever

# =============================================================================
# Example 1: Minimum Score Threshold
# =============================================================================


def minimum_score():
    """Filter results by minimum similarity score."""
    print("=" * 60)
    print("1. Minimum Score Threshold")
    print("=" * 60)

    retriever = Retriever()

    # Add documents
    retriever.add(
        [
            "Python is a programming language for data science",
            "JavaScript runs in web browsers",
            "Machine learning uses statistical models",
            "Cooking pasta requires boiling water",
            "The weather is sunny today",
        ]
    )

    query = "Python data science"

    # Without minimum score
    print(f"\nQuery: '{query}'")
    print("\nWithout min_score (all results):")
    results = retriever.search(query, k=5)
    for r in results:
        print(f"  {r.score:.3f}: {r.document.text[:50]}...")

    # With minimum score threshold
    print("\nWith min_score=0.5 (relevant only):")
    results = retriever.search(query, k=5, min_score=0.5)
    for r in results:
        print(f"  {r.score:.3f}: {r.document.text[:50]}...")

    if not results:
        print("  (No results above threshold)")


# =============================================================================
# Example 2: Detailed Search Results
# =============================================================================


def detailed_results():
    """Get comprehensive information from search results."""
    print("\n" + "=" * 60)
    print("2. Detailed Search Results")
    print("=" * 60)

    retriever = Retriever()

    # Add document with metadata
    retriever.add_text(
        "The API endpoint /users returns a list of all users in JSON format",
        metadata={
            "category": "api",
            "version": "v2",
            "author": "docs-team",
        },
        source="api-reference.md",
    )

    results = retriever.search("get users endpoint", k=1)

    if results:
        result = results[0]

        print("\nSearchResult object properties:")
        print(f"\n  score: {result.score}")
        print("    Type: float (0.0 to 1.0)")
        print("    Higher = more similar")

        print(f"\n  document.text: {result.document.text[:50]}...")
        print("    The actual content")

        print(f"\n  document.metadata: {result.document.metadata}")
        print("    Custom key-value pairs")

        print(f"\n  document.source: {result.document.source}")
        print("    Origin file or identifier")


# =============================================================================
# Example 3: Score Interpretation
# =============================================================================


def score_interpretation():
    """Understand what similarity scores mean."""
    print("\n" + "=" * 60)
    print("3. Score Interpretation")
    print("=" * 60)

    retriever = Retriever()

    # Add documents with varying relevance
    docs = [
        "Python is excellent for machine learning and data science",
        "Data science uses Python for analysis",
        "JavaScript is used for web development",
        "The recipe calls for two cups of flour",
    ]
    retriever.add(docs)

    query = "Python data science"

    print(f"\nQuery: '{query}'")
    print("\nScore interpretation guide:")
    print("  0.9-1.0: Near-exact match")
    print("  0.7-0.9: Highly relevant")
    print("  0.5-0.7: Somewhat relevant")
    print("  0.3-0.5: Tangentially related")
    print("  <0.3:    Likely irrelevant")

    results = retriever.search(query, k=4)
    print("\nResults with interpretation:")
    for r in results:
        if r.score >= 0.7:
            relevance = "HIGH"
        elif r.score >= 0.5:
            relevance = "MEDIUM"
        else:
            relevance = "LOW"
        print(f"  {r.score:.3f} [{relevance:6}]: {r.document.text[:40]}...")


# =============================================================================
# Example 4: Context Window Search
# =============================================================================


def context_window():
    """Search with surrounding context."""
    print("\n" + "=" * 60)
    print("4. Context Window Search")
    print("=" * 60)

    retriever = Retriever()

    # Add a longer document that gets chunked
    long_doc = """
    Chapter 1: Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that
    focuses on building systems that can learn from data.

    Chapter 2: Supervised Learning

    In supervised learning, the algorithm learns from labeled training
    data to make predictions on new, unseen data.

    Chapter 3: Unsupervised Learning

    Unsupervised learning finds patterns in data without labels.
    Common techniques include clustering and dimensionality reduction.

    Chapter 4: Deep Learning

    Deep learning uses neural networks with multiple layers to learn
    complex patterns. It has revolutionized image and speech recognition.
    """
    retriever.add(long_doc)

    query = "supervised learning prediction"

    # Get context
    print(f"\nQuery: '{query}'")
    context = retriever.get_context(query, k=2)

    print("\nContext for LLM:")
    print("-" * 40)
    print(context)
    print("-" * 40)


# =============================================================================
# Example 5: Multi-Query Expansion
# =============================================================================


def multi_query():
    """Use multiple queries for better recall."""
    print("\n" + "=" * 60)
    print("5. Multi-Query Expansion")
    print("=" * 60)

    retriever = Retriever()

    # Add documents
    retriever.add(
        [
            "How to reset your password using the settings menu",
            "Account recovery options for forgotten credentials",
            "Two-factor authentication setup guide",
            "Login troubleshooting and common issues",
        ]
    )

    # Single query might miss relevant docs
    single_query = "forgot password"

    # Multiple query variations for better recall
    query_variations = [
        "forgot password",
        "reset credentials",
        "account recovery",
    ]

    print(f"\nSingle query: '{single_query}'")
    single_results = retriever.search(single_query, k=2)
    print("Results:")
    for r in single_results:
        print(f"  {r.score:.3f}: {r.document.text[:50]}...")

    print("\nMulti-query expansion:")
    print(f"  Queries: {query_variations}")

    # Combine results from all queries
    all_results = {}
    for query in query_variations:
        results = retriever.search(query, k=2)
        for r in results:
            key = r.document.text
            if key not in all_results or r.score > all_results[key].score:
                all_results[key] = r

    # Sort by score
    combined = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
    print("\nCombined results:")
    for r in combined[:4]:
        print(f"  {r.score:.3f}: {r.document.text[:50]}...")


# =============================================================================
# Example 6: Filtered Search Strategies
# =============================================================================


def filtered_search():
    """Advanced filtering techniques."""
    print("\n" + "=" * 60)
    print("6. Filtered Search Strategies")
    print("=" * 60)

    retriever = Retriever()

    # Add documents with rich metadata
    docs = [
        ("Python beginner tutorial", {"level": "beginner", "lang": "python", "year": 2024}),
        ("Advanced Python patterns", {"level": "advanced", "lang": "python", "year": 2024}),
        ("Python 2.7 migration guide", {"level": "intermediate", "lang": "python", "year": 2020}),
        ("JavaScript basics", {"level": "beginner", "lang": "javascript", "year": 2024}),
        ("TypeScript advanced types", {"level": "advanced", "lang": "typescript", "year": 2024}),
    ]

    for text, metadata in docs:
        retriever.add_text(text, metadata=metadata)

    print(f"\nAdded {len(docs)} documents with metadata")

    # Strategy 1: Single filter
    print("\n  Strategy 1 - Single filter (lang=python):")
    results = retriever.search("programming", k=5, filter={"lang": "python"})
    for r in results:
        print(f"    {r.document.text}")

    # Strategy 2: Multiple conditions
    print("\n  Strategy 2 - Multiple conditions (lang=python, level=beginner):")
    results = retriever.search("tutorial", k=5, filter={"lang": "python", "level": "beginner"})
    for r in results:
        print(f"    {r.document.text}")


# =============================================================================
# Example 7: Search with Diversity
# =============================================================================


def diverse_results():
    """Get diverse results instead of similar ones."""
    print("\n" + "=" * 60)
    print("7. Diverse Results")
    print("=" * 60)

    retriever = Retriever()

    # Add documents that might be very similar
    retriever.add(
        [
            "Python is great for data science",
            "Python is excellent for data analysis",
            "Python is perfect for data work",
            "JavaScript powers web applications",
            "Machine learning uses statistics",
        ]
    )

    query = "Python data"

    # Standard search (might return similar docs)
    print(f"\nQuery: '{query}'")
    print("\nStandard search (top 3):")
    results = retriever.search(query, k=3)
    for r in results:
        print(f"  {r.score:.3f}: {r.document.text}")

    print("\nFor diversity, consider:")
    print("  1. Use lower k with min_score filtering")
    print("  2. Post-process to remove near-duplicates")
    print("  3. Use MMR (Maximal Marginal Relevance) if available")


# =============================================================================
# Example 8: Search Quality Evaluation
# =============================================================================


def quality_evaluation():
    """Evaluate and improve search quality."""
    print("\n" + "=" * 60)
    print("8. Search Quality Evaluation")
    print("=" * 60)

    retriever = Retriever()

    # Test corpus
    retriever.add(
        [
            "API authentication uses Bearer tokens",
            "Rate limiting prevents abuse",
            "Webhooks notify external systems",
            "REST endpoints follow standard patterns",
        ]
    )

    # Test queries with expected results
    test_cases = [
        {
            "query": "how to authenticate API",
            "expected_keywords": ["authentication", "token", "Bearer"],
        },
        {
            "query": "prevent too many requests",
            "expected_keywords": ["rate", "limiting", "abuse"],
        },
    ]

    print("\nSearch quality test:")
    for test in test_cases:
        results = retriever.search(test["query"], k=1)
        if results:
            result_text = results[0].document.text.lower()
            matches = sum(1 for kw in test["expected_keywords"] if kw.lower() in result_text)
            score = matches / len(test["expected_keywords"])

            status = "[OK] PASS" if score >= 0.5 else "[X] FAIL"
            print(f"\n  {status} Query: '{test['query']}'")
            print(f"    Result: {results[0].document.text[:50]}...")
            print(f"    Keyword match: {matches}/{len(test['expected_keywords'])}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Advanced Search Features")
    print("=" * 60)
    print("\nThese features help you get better search results.\n")

    minimum_score()
    detailed_results()
    score_interpretation()
    context_window()
    multi_query()
    filtered_search()
    diverse_results()
    quality_evaluation()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Use min_score to filter low-quality results")
    print("  2. Understand score interpretation (0.7+ is highly relevant)")
    print("  3. Multi-query expansion improves recall")
    print("  4. Combine vector search with metadata filtering")
    print("  5. Evaluate search quality with test cases")


if __name__ == "__main__":
    main()
