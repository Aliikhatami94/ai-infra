#!/usr/bin/env python
"""Async Retriever Operations Example.

This example demonstrates:
- Async search with asearch()
- Concurrent operations
- Async embeddings integration
- Performance comparison (sync vs async)
- Batch async operations
- Error handling in async context

Use async operations for web servers, APIs, and
applications that need non-blocking I/O.
"""

import asyncio
import time

from ai_infra import Retriever

# =============================================================================
# Example 1: Basic Async Search
# =============================================================================


async def basic_async_search():
    """Use async search method."""
    print("=" * 60)
    print("1. Basic Async Search")
    print("=" * 60)

    retriever = Retriever()

    # Add documents (sync is fine for setup)
    retriever.add(
        [
            "Python async/await enables concurrent programming",
            "Asyncio is Python's async I/O framework",
            "Coroutines are the building blocks of async code",
            "Event loops manage async task execution",
        ]
    )

    # Async search
    query = "async programming"
    print(f"\nAsync search: '{query}'")

    results = await retriever.asearch(query, k=2)

    print(f"\nResults ({len(results)} found):")
    for r in results:
        print(f"  {r.score:.3f}: {r.document.text[:50]}...")


# =============================================================================
# Example 2: Concurrent Searches
# =============================================================================


async def concurrent_searches():
    """Run multiple searches concurrently."""
    print("\n" + "=" * 60)
    print("2. Concurrent Searches")
    print("=" * 60)

    retriever = Retriever()

    # Build knowledge base
    topics = [
        "Machine learning uses statistical models to learn patterns",
        "Deep learning employs neural networks with many layers",
        "Natural language processing handles text understanding",
        "Computer vision enables image recognition",
        "Reinforcement learning learns through trial and error",
        "Transfer learning reuses pretrained models",
    ]
    retriever.add(topics)

    # Multiple queries to run concurrently
    queries = [
        "neural networks",
        "text processing",
        "image recognition",
        "learning from rewards",
    ]

    print(f"\nRunning {len(queries)} searches concurrently...")

    # Concurrent execution
    start = time.time()
    tasks = [retriever.asearch(q, k=1) for q in queries]
    all_results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    print(f"Completed in {elapsed:.3f}s")

    for query, results in zip(queries, all_results):
        if results:
            print(f"\n  '{query}':")
            print(f"    → {results[0].document.text[:50]}...")


# =============================================================================
# Example 3: Sync vs Async Performance
# =============================================================================


async def performance_comparison():
    """Compare sync and async performance."""
    print("\n" + "=" * 60)
    print("3. Sync vs Async Performance")
    print("=" * 60)

    retriever = Retriever()

    # Add documents
    for i in range(100):
        retriever.add(f"Document {i} about topic {i % 10}")

    queries = [f"topic {i}" for i in range(10)]

    # Sync (sequential)
    print(f"\n  Running {len(queries)} searches...")

    start = time.time()
    for query in queries:
        retriever.search(query, k=3)
    sync_time = time.time() - start

    # Async (concurrent)
    start = time.time()
    tasks = [retriever.asearch(query, k=3) for query in queries]
    await asyncio.gather(*tasks)
    async_time = time.time() - start

    print(f"\n  Sync (sequential):  {sync_time:.3f}s")
    print(f"  Async (concurrent): {async_time:.3f}s")

    if sync_time > async_time:
        speedup = sync_time / async_time
        print(f"\n  Async is {speedup:.1f}x faster!")


# =============================================================================
# Example 4: Async Context Manager
# =============================================================================


async def async_context():
    """Use retriever in async context."""
    print("\n" + "=" * 60)
    print("4. Async Context Patterns")
    print("=" * 60)

    retriever = Retriever()
    retriever.add(
        [
            "Context managers manage resources",
            "Async context for non-blocking I/O",
        ]
    )

    # Pattern 1: Simple async function
    async def search_one(query: str) -> str:
        results = await retriever.asearch(query, k=1)
        return results[0].document.text if results else ""

    result = await search_one("resource management")
    print("\n  Pattern 1 (simple):")
    print(f"    {result[:50]}...")

    # Pattern 2: Gather multiple
    async def search_many(queries: list[str]) -> list[str]:
        tasks = [search_one(q) for q in queries]
        return await asyncio.gather(*tasks)

    results = await search_many(["context", "async"])
    print("\n  Pattern 2 (gather):")
    for r in results:
        print(f"    {r[:50]}...")


# =============================================================================
# Example 5: Async with Timeout
# =============================================================================


async def async_with_timeout():
    """Handle timeouts in async operations."""
    print("\n" + "=" * 60)
    print("5. Async with Timeout")
    print("=" * 60)

    retriever = Retriever()
    retriever.add(["Test document for timeout example"])

    async def search_with_timeout(query: str, timeout: float):
        try:
            result = await asyncio.wait_for(
                retriever.asearch(query, k=1),
                timeout=timeout,
            )
            return result
        except TimeoutError:
            print(f"    ⚠ Search timed out after {timeout}s")
            return []

    print("\n  Testing with 5s timeout (should succeed):")
    results = await search_with_timeout("test", timeout=5.0)
    print(f"    ✓ Got {len(results)} results")

    print("\n  Timeout pattern protects against slow operations")


# =============================================================================
# Example 6: Async Error Handling
# =============================================================================


async def async_error_handling():
    """Handle errors in async context."""
    print("\n" + "=" * 60)
    print("6. Async Error Handling")
    print("=" * 60)

    retriever = Retriever()
    retriever.add(["Error handling in async code"])

    async def safe_search(query: str):
        try:
            results = await retriever.asearch(query, k=1)
            return {"status": "success", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # Successful search
    result = await safe_search("error handling")
    print(f"\n  Status: {result['status']}")
    if result["status"] == "success":
        print(f"  Results: {len(result['results'])} found")


# =============================================================================
# Example 7: Async Stream Processing
# =============================================================================


async def stream_processing():
    """Process search results as a stream."""
    print("\n" + "=" * 60)
    print("7. Async Stream Processing")
    print("=" * 60)

    retriever = Retriever()

    # Add numbered documents
    for i in range(20):
        retriever.add(f"Stream document {i}: Content about item {i}")

    # Process queries as they complete (not in order)
    queries = [f"item {i}" for i in range(5)]

    print(f"\n  Processing {len(queries)} queries as stream...")

    async def process_query(query: str, idx: int):
        await asyncio.sleep(0.1 * (5 - idx))  # Simulate varying latency
        results = await retriever.asearch(query, k=1)
        return idx, query, results

    tasks = [process_query(q, i) for i, q in enumerate(queries)]

    # Process as they complete
    for coro in asyncio.as_completed(tasks):
        idx, query, results = await coro
        print(f"    Query {idx} ('{query}'): {len(results)} results")


# =============================================================================
# Example 8: Async Web Server Pattern
# =============================================================================


async def web_server_pattern():
    """Pattern for using retriever in async web server."""
    print("\n" + "=" * 60)
    print("8. Web Server Pattern")
    print("=" * 60)

    # Create retriever once (at startup)
    retriever = Retriever()
    retriever.add(
        [
            "API documentation",
            "User guide",
            "FAQ answers",
        ]
    )

    # Handler pattern (called per request)
    async def handle_search_request(query: str) -> dict:
        """Simulates a web request handler."""
        results = await retriever.asearch(query, k=3)
        return {
            "query": query,
            "results": [
                {
                    "text": r.document.text,
                    "score": r.score,
                }
                for r in results
            ],
        }

    # Simulate concurrent requests
    print("\n  Simulating concurrent web requests...")

    requests = ["API", "guide", "FAQ"]
    handlers = [handle_search_request(r) for r in requests]
    responses = await asyncio.gather(*handlers)

    for response in responses:
        print(f"\n  Request: '{response['query']}'")
        print(f"    Results: {len(response['results'])} items")

    print("\n  Pattern: FastAPI/Starlette example:")
    print("""
    from fastapi import FastAPI
    app = FastAPI()
    retriever = Retriever()  # Created once

    @app.get("/search")
    async def search(q: str):
        results = await retriever.asearch(q, k=5)
        return {"results": [...]}
""")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Async Retriever Operations")
    print("=" * 60)
    print("\nAsync operations enable non-blocking I/O.")
    print("Essential for web servers and concurrent applications.\n")

    await basic_async_search()
    await concurrent_searches()
    await performance_comparison()
    await async_context()
    await async_with_timeout()
    await async_error_handling()
    await stream_processing()
    await web_server_pattern()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Use asearch() for async operations")
    print("  2. asyncio.gather() for concurrent searches")
    print("  3. asyncio.wait_for() for timeouts")
    print("  4. Async is essential for web servers")
    print("  5. Create retriever once, reuse across requests")


if __name__ == "__main__":
    asyncio.run(main())
