#!/usr/bin/env python
"""Pinecone Backend for Retriever Example.

This example demonstrates:
- Connecting to Pinecone managed cloud
- Index configuration
- Namespace organization
- Metadata filtering
- Production scaling

Pinecone is a fully managed vector database service
optimized for production ML workloads at scale.

Requirements:
  pip install 'ai-infra[pinecone]'
  PINECONE_API_KEY environment variable
"""

import os

from ai_infra import Retriever

# =============================================================================
# Example 1: Basic Pinecone Setup
# =============================================================================


def basic_pinecone():
    """Connect to Pinecone and add vectors."""
    print("=" * 60)
    print("1. Basic Pinecone Setup")
    print("=" * 60)

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("\n  [!] Set PINECONE_API_KEY to run this example:")
        print("    export PINECONE_API_KEY='your-api-key'")
        return

    # Create retriever with Pinecone
    retriever = Retriever(
        backend="pinecone",
        index="my-index",  # Your Pinecone index name
    )

    print(f"\n  Backend: {retriever.backend}")
    print("  [OK] Connected to Pinecone")

    # Add documents
    retriever.add(
        [
            "Pinecone provides fully managed vector search",
            "Serverless deployment for automatic scaling",
            "Low-latency similarity search at scale",
        ]
    )

    print(f"\n  Added {len(retriever)} documents")


# =============================================================================
# Example 2: Index Configuration
# =============================================================================


def index_configuration():
    """Configure Pinecone index settings."""
    print("\n" + "=" * 60)
    print("2. Index Configuration")
    print("=" * 60)

    print("\nIndex configuration options:")

    configs = {
        "Serverless": {
            "cloud": "aws",
            "region": "us-east-1",
            "description": "Auto-scaling, pay per use",
        },
        "Pod-based": {
            "pod_type": "p1.x1",
            "replicas": 2,
            "description": "Dedicated resources, predictable performance",
        },
    }

    for name, config in configs.items():
        print(f"\n  {name}:")
        for key, value in config.items():
            print(f"    {key}: {value}")

    print("\nExample index creation (via Pinecone console or API):")
    print("""
    import pinecone

    pinecone.init(api_key="xxx")
    pinecone.create_index(
        name="my-index",
        dimension=1536,  # Match your embedding model
        metric="cosine",
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
""")


# =============================================================================
# Example 3: Using Namespaces
# =============================================================================


def using_namespaces():
    """Organize data with namespaces."""
    print("\n" + "=" * 60)
    print("3. Using Namespaces")
    print("=" * 60)

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("\n  [!] Set PINECONE_API_KEY to run this example")
        print("\n  Namespaces allow data segregation:")
        print("    - Multi-tenant applications")
        print("    - Environment separation (dev/staging/prod)")
        print("    - Category organization")
        return

    # Different namespaces for different purposes
    namespaces = {
        "docs": "Documentation content",
        "faq": "Frequently asked questions",
        "products": "Product information",
    }

    print("\n  Namespaces in single index:")
    for ns, description in namespaces.items():
        print(f"    {ns}: {description}")

    # Example usage
    print("\n  Usage:")
    print("""
    # Each namespace is isolated
    docs_retriever = Retriever(
        backend="pinecone",
        index="my-index",
        namespace="docs"
    )

    faq_retriever = Retriever(
        backend="pinecone",
        index="my-index",
        namespace="faq"
    )
""")


# =============================================================================
# Example 4: Metadata Filtering
# =============================================================================


def metadata_filtering():
    """Filter search results with metadata."""
    print("\n" + "=" * 60)
    print("4. Metadata Filtering")
    print("=" * 60)

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("\n  [!] Set PINECONE_API_KEY to run this example")
        print("\n  Pinecone supports rich metadata filtering:")
        return

    # Example with metadata
    print("\n  Adding documents with metadata:")
    print("""
    retriever.add_text(
        "Python ML tutorial",
        metadata={
            "category": "tutorial",
            "language": "python",
            "difficulty": "beginner",
            "year": 2024
        }
    )
""")

    print("\n  Search with filters:")
    print("""
    # Filter by exact match
    results = retriever.search(
        "machine learning",
        filter={"language": "python"}
    )

    # Numeric comparison
    results = retriever.search(
        "tutorial",
        filter={"year": {"$gte": 2023}}
    )

    # Combined filters
    results = retriever.search(
        "guide",
        filter={
            "category": "tutorial",
            "difficulty": {"$in": ["beginner", "intermediate"]}
        }
    )
""")


# =============================================================================
# Example 5: Environment Configuration
# =============================================================================


def environment_config():
    """Configure Pinecone via environment variables."""
    print("\n" + "=" * 60)
    print("5. Environment Configuration")
    print("=" * 60)

    env_vars = {
        "PINECONE_API_KEY": "Your Pinecone API key (required)",
        "PINECONE_ENVIRONMENT": "Pinecone environment (e.g., us-west1-gcp)",
        "PINECONE_INDEX": "Default index name",
        "RETRIEVER_BACKEND": "Set to 'pinecone' for default",
    }

    print("\nEnvironment variables:")
    for var, description in env_vars.items():
        value = os.getenv(var, "(not set)")
        if "KEY" in var and value != "(not set)":
            value = value[:8] + "..."
        print(f"\n  {var}")
        print(f"    {description}")
        print(f"    Current: {value}")


# =============================================================================
# Example 6: Production Best Practices
# =============================================================================


def production_practices():
    """Best practices for production use."""
    print("\n" + "=" * 60)
    print("6. Production Best Practices")
    print("=" * 60)

    practices = [
        {
            "title": "Use Serverless for Variable Workloads",
            "description": "Automatic scaling, cost-effective for bursty traffic",
        },
        {
            "title": "Index Dimension Matching",
            "description": "Ensure index dimension matches embedding model output",
        },
        {
            "title": "Namespace Strategy",
            "description": "Use namespaces for multi-tenancy and environment separation",
        },
        {
            "title": "Metadata for Filtering",
            "description": "Index metadata fields you'll filter on frequently",
        },
        {
            "title": "Batch Upserts",
            "description": "Use batch operations for bulk data ingestion",
        },
        {
            "title": "Monitor Usage",
            "description": "Track vector count, query latency, and error rates",
        },
    ]

    print("\nProduction recommendations:")
    for i, practice in enumerate(practices, 1):
        print(f"\n  {i}. {practice['title']}")
        print(f"     {practice['description']}")


# =============================================================================
# Example 7: Batch Upserts
# =============================================================================


def batch_upserts():
    """Efficiently upsert large amounts of data."""
    print("\n" + "=" * 60)
    print("7. Batch Upserts")
    print("=" * 60)

    print("\nFor large data ingestion, batch your upserts:")
    print("""
    # Generate documents
    documents = [f"Document {i} content" for i in range(10000)]

    # Batch size recommendation: 100-1000 vectors
    batch_size = 100

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        retriever.add(batch)
        print(f"Upserted {min(i + batch_size, len(documents))}/{len(documents)}")
""")

    print("\n  Tips for large ingestion:")
    print("    - Use parallel upserts for speed")
    print("    - Monitor rate limits")
    print("    - Validate dimension before upsert")


# =============================================================================
# Example 8: Cost Optimization
# =============================================================================


def cost_optimization():
    """Optimize Pinecone costs."""
    print("\n" + "=" * 60)
    print("8. Cost Optimization")
    print("=" * 60)

    tips = {
        "Right-size your index": "Use pod type appropriate for workload",
        "Use namespaces wisely": "Same index, multiple logical partitions",
        "Serverless for variable loads": "Pay per query, auto-scaling",
        "Optimize embedding dimensions": "Smaller dimensions = lower costs",
        "Delete unused data": "Remove stale vectors to reduce storage",
        "Monitor and alert": "Set up cost alerts in Pinecone console",
    }

    print("\nCost optimization strategies:")
    for tip, description in tips.items():
        print(f"\n  • {tip}")
        print(f"    {description}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Pinecone Backend Examples")
    print("=" * 60)
    print("\nPinecone is a fully managed vector database:")
    print("  - Zero infrastructure management")
    print("  - Automatic scaling")
    print("  - Low-latency at scale")
    print("  - Rich metadata filtering\n")

    basic_pinecone()
    index_configuration()
    using_namespaces()
    metadata_filtering()
    environment_config()
    production_practices()
    batch_upserts()
    cost_optimization()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Set PINECONE_API_KEY environment variable")
    print("  2. Create index with matching dimensions")
    print("  3. Use namespaces for data organization")
    print("  4. Leverage metadata for filtered search")
    print("  5. Batch operations for large data")


if __name__ == "__main__":
    main()
