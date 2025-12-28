#!/usr/bin/env python
"""SQLite Backend for Retriever Example.

This example demonstrates:
- Using SQLite for persistent vector storage
- Database file configuration
- Persisting and reloading retrievers
- Managing multiple collections
- SQLite-specific features

SQLite is perfect for local development, single-user apps,
and scenarios where you need persistence without a server.
"""

import os
import tempfile
from pathlib import Path

from ai_infra import Retriever

# =============================================================================
# Example 1: Basic SQLite Setup
# =============================================================================


def basic_sqlite():
    """Create a retriever with SQLite backend."""
    print("=" * 60)
    print("1. Basic SQLite Setup")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "vectors.db"

        # Create retriever with SQLite backend
        retriever = Retriever(backend="sqlite", path=str(db_path))

        print(f"\n  Backend: {retriever.backend}")
        print(f"  Database: {db_path}")

        # Add some content
        retriever.add(
            [
                "Python is great for AI development",
                "SQLite provides lightweight persistence",
                "Vector databases store embeddings",
            ]
        )

        print(f"\n✓ Added {len(retriever)} documents")
        print(f"  Database size: {db_path.stat().st_size:,} bytes")


# =============================================================================
# Example 2: Persistent Storage
# =============================================================================


def persistent_storage():
    """Demonstrate persistence across sessions."""
    print("\n" + "=" * 60)
    print("2. Persistent Storage")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "persistent.db"

        # First session: create and populate
        print("\n[Session 1] Creating database...")
        retriever1 = Retriever(backend="sqlite", path=str(db_path))
        retriever1.add(
            [
                "Machine learning trains models on data",
                "Deep learning uses neural networks",
                "Reinforcement learning uses rewards",
            ]
        )
        print(f"  Added {len(retriever1)} documents")

        # Close first session (just let it go out of scope)
        del retriever1

        # Second session: reload and search
        print("\n[Session 2] Reopening database...")
        retriever2 = Retriever(backend="sqlite", path=str(db_path))
        print(f"  Found {len(retriever2)} existing documents")

        results = retriever2.search("neural network", k=1)
        if results:
            print(f"\n  Search result: {results[0].document.text}")

        print("\n✓ Data persisted across sessions!")


# =============================================================================
# Example 3: Named Collections
# =============================================================================


def named_collections():
    """Use different collections in the same database."""
    print("\n" + "=" * 60)
    print("3. Named Collections")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "multi_collection.db"

        # Collection for documentation
        docs_retriever = Retriever(
            backend="sqlite",
            path=str(db_path),
            collection="documentation",
        )
        docs_retriever.add(
            [
                "API endpoint documentation",
                "Installation guide",
            ]
        )

        # Collection for code
        code_retriever = Retriever(
            backend="sqlite",
            path=str(db_path),
            collection="code",
        )
        code_retriever.add(
            [
                "def process_data(x): return x * 2",
                "class DataProcessor: pass",
            ]
        )

        print(f"\n  'documentation' collection: {len(docs_retriever)} docs")
        print(f"  'code' collection: {len(code_retriever)} docs")

        # Search in specific collections
        doc_results = docs_retriever.search("API", k=1)
        code_results = code_retriever.search("function", k=1)

        print(
            "\n  Documentation search 'API':",
            doc_results[0].document.text[:40] if doc_results else "No results",
        )
        print(
            "  Code search 'function':",
            code_results[0].document.text[:40] if code_results else "No results",
        )


# =============================================================================
# Example 4: Incremental Updates
# =============================================================================


def incremental_updates():
    """Add documents incrementally over time."""
    print("\n" + "=" * 60)
    print("4. Incremental Updates")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "incremental.db"

        retriever = Retriever(backend="sqlite", path=str(db_path))

        # Day 1: Initial documents
        retriever.add(["Initial document from day 1"])
        print(f"\n  Day 1: {len(retriever)} documents")

        # Day 2: Add more
        retriever.add(["New document from day 2", "Another day 2 doc"])
        print(f"  Day 2: {len(retriever)} documents")

        # Day 3: Add more
        retriever.add(["Final document from day 3"])
        print(f"  Day 3: {len(retriever)} documents")

        print("\n✓ Documents accumulated over time")


# =============================================================================
# Example 5: Database in Common Locations
# =============================================================================


def common_locations():
    """Show common database locations."""
    print("\n" + "=" * 60)
    print("5. Common Database Locations")
    print("=" * 60)

    locations = {
        "Current directory": "./vectors.db",
        "User data directory": "~/.local/share/myapp/vectors.db",
        "App config": "~/.config/myapp/vectors.db",
        "Temporary": "/tmp/myapp_vectors.db",
    }

    print("\nCommon locations for SQLite databases:")
    for name, path in locations.items():
        expanded = Path(path).expanduser()
        print(f"\n  {name}:")
        print(f"    {path}")
        print(f"    → {expanded}")

    print("\nExample usage:")
    print('  retriever = Retriever(backend="sqlite", path="./data/vectors.db")')


# =============================================================================
# Example 6: Environment Variable Configuration
# =============================================================================


def env_configuration():
    """Configure SQLite via environment variables."""
    print("\n" + "=" * 60)
    print("6. Environment Variable Configuration")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "env_config.db"

        # Save original env
        original_backend = os.environ.get("RETRIEVER_BACKEND")
        original_path = os.environ.get("RETRIEVER_SQLITE_PATH")

        try:
            # Set environment variables
            os.environ["RETRIEVER_BACKEND"] = "sqlite"
            os.environ["RETRIEVER_SQLITE_PATH"] = str(db_path)

            # Create retriever (picks up env vars)
            retriever = Retriever()

            print(f"\n  RETRIEVER_BACKEND: {os.environ.get('RETRIEVER_BACKEND')}")
            print(f"  RETRIEVER_SQLITE_PATH: {os.environ.get('RETRIEVER_SQLITE_PATH')}")
            print(f"\n  Actual backend: {retriever.backend}")

        finally:
            # Restore original env
            if original_backend:
                os.environ["RETRIEVER_BACKEND"] = original_backend
            else:
                os.environ.pop("RETRIEVER_BACKEND", None)

            if original_path:
                os.environ["RETRIEVER_SQLITE_PATH"] = original_path
            else:
                os.environ.pop("RETRIEVER_SQLITE_PATH", None)


# =============================================================================
# Example 7: Full-Text Search Hybrid
# =============================================================================


def fulltext_hybrid():
    """Combine vector search with SQLite FTS."""
    print("\n" + "=" * 60)
    print("7. Full-Text Search Hybrid")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "hybrid.db"

        retriever = Retriever(backend="sqlite", path=str(db_path))

        # Add technical documentation
        docs = [
            "The API endpoint /users accepts GET and POST requests",
            "Authentication requires Bearer tokens in the header",
            "Rate limiting is set to 100 requests per minute",
            "Error responses use standard HTTP status codes",
            "Pagination uses cursor-based navigation",
        ]
        retriever.add(docs)

        # Vector similarity search
        print("\nVector search for 'how to authenticate':")
        results = retriever.search("how to authenticate", k=2)
        for r in results:
            print(f"  {r.score:.3f}: {r.document.text[:50]}...")

        print("\n(For exact keyword matching, combine with SQLite FTS)")


# =============================================================================
# Example 8: Large Dataset Handling
# =============================================================================


def large_dataset():
    """Handle larger datasets with SQLite."""
    print("\n" + "=" * 60)
    print("8. Large Dataset Handling")
    print("=" * 60)

    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "large.db"

        retriever = Retriever(backend="sqlite", path=str(db_path))

        # Generate sample documents
        num_docs = 500
        docs = [f"Document {i}: Content about topic {i % 50}" for i in range(num_docs)]

        print(f"\nAdding {num_docs} documents...")
        start = time.time()
        retriever.add(docs)
        add_time = time.time() - start

        print(f"  ✓ Added in {add_time:.2f}s")
        print(f"  Database size: {db_path.stat().st_size / 1024:.1f} KB")

        # Search performance
        start = time.time()
        for _ in range(10):
            retriever.search("topic 25", k=5)
        search_time = time.time() - start

        print(f"\n  10 searches in {search_time:.3f}s")
        print(f"  Average: {search_time / 10 * 1000:.1f}ms per search")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SQLite Backend Examples")
    print("=" * 60)
    print("\nSQLite provides persistent storage without a server.")
    print("Perfect for local apps, development, and single-user scenarios.\n")

    basic_sqlite()
    persistent_storage()
    named_collections()
    incremental_updates()
    common_locations()
    env_configuration()
    fulltext_hybrid()
    large_dataset()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. backend='sqlite' enables file-based persistence")
    print("  2. Data survives across sessions")
    print("  3. Use collection parameter for multiple collections")
    print("  4. Configure via code or environment variables")
    print("  5. Great for local development and single-user apps")


if __name__ == "__main__":
    main()
