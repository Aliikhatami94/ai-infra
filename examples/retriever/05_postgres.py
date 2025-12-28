#!/usr/bin/env python
"""PostgreSQL + pgvector Backend for Retriever Example.

This example demonstrates:
- Connecting to PostgreSQL with pgvector extension
- Production-grade vector storage
- Connection string configuration
- Concurrent access and scaling
- Best practices for production

PostgreSQL with pgvector is ideal for production deployments
where you need ACID transactions, concurrent access, and scaling.

Requirements:
  - PostgreSQL 15+ with pgvector extension
  - pip install 'ai-infra[postgres]'
"""

import os

from ai_infra import Retriever

# =============================================================================
# Example 1: Basic PostgreSQL Setup
# =============================================================================


def basic_postgres():
    """Connect to PostgreSQL with pgvector."""
    print("=" * 60)
    print("1. Basic PostgreSQL Setup")
    print("=" * 60)

    # Check for connection string
    conn_str = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

    if not conn_str:
        print("\n  ⚠ Set DATABASE_URL to run this example:")
        print("    export DATABASE_URL='postgresql://user:pass@localhost:5432/mydb'")
        return

    # Create retriever with PostgreSQL backend
    retriever = Retriever(backend="postgres", connection_string=conn_str)

    print(f"\n  Backend: {retriever.backend}")
    print("  ✓ Connected to PostgreSQL")

    # Add some content
    retriever.add(
        [
            "PostgreSQL is a powerful relational database",
            "pgvector extension enables vector similarity search",
            "Production-ready with ACID transactions",
        ]
    )

    print(f"\n  Added {len(retriever)} documents")


# =============================================================================
# Example 2: Connection String Formats
# =============================================================================


def connection_strings():
    """Show different connection string formats."""
    print("\n" + "=" * 60)
    print("2. Connection String Formats")
    print("=" * 60)

    formats = {
        "Standard": "postgresql://user:password@localhost:5432/dbname",
        "With SSL": "postgresql://user:password@host:5432/dbname?sslmode=require",
        "Unix Socket": "postgresql://user:password@/dbname?host=/var/run/postgresql",
        "Multiple Hosts": "postgresql://user:pass@host1:5432,host2:5432/db?target_session_attrs=any",
    }

    print("\nSupported connection string formats:")
    for name, fmt in formats.items():
        print(f"\n  {name}:")
        print(f"    {fmt}")

    print("\nEnvironment variables:")
    print("  DATABASE_URL - Primary connection string")
    print("  POSTGRES_URL - Alternative name")
    print("  PGHOST, PGUSER, PGPASSWORD, PGDATABASE - Individual components")


# =============================================================================
# Example 3: Table and Collection Configuration
# =============================================================================


def table_configuration():
    """Configure table names and collections."""
    print("\n" + "=" * 60)
    print("3. Table and Collection Configuration")
    print("=" * 60)

    conn_str = os.getenv("DATABASE_URL")
    if not conn_str:
        print("\n  ⚠ Set DATABASE_URL to run this example")
        return

    # Custom table name
    _retriever = Retriever(
        backend="postgres",
        connection_string=conn_str,
        collection="my_documents",  # Creates 'my_documents' table
    )

    print("\n  Using collection (table): my_documents")

    # You can have multiple collections
    collections = ["products", "support_docs", "user_guides"]
    print("\n  Multiple collections in same database:")
    for coll in collections:
        print(f"    - {coll}")


# =============================================================================
# Example 4: Index Configuration for Performance
# =============================================================================


def index_configuration():
    """Configure vector indexes for search performance."""
    print("\n" + "=" * 60)
    print("4. Index Configuration")
    print("=" * 60)

    print("\npgvector supports different index types:")

    indexes = {
        "IVFFlat": {
            "description": "Good for medium datasets (100K-1M vectors)",
            "params": "lists = 100",
            "search": "probes = 10",
        },
        "HNSW": {
            "description": "Best for high-recall requirements",
            "params": "m = 16, ef_construction = 64",
            "search": "ef_search = 40",
        },
    }

    for name, config in indexes.items():
        print(f"\n  {name}:")
        print(f"    {config['description']}")
        print(f"    Build params: {config['params']}")
        print(f"    Search params: {config['search']}")

    print("\nExample SQL for creating HNSW index:")
    print("""
    CREATE INDEX ON my_documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""")


# =============================================================================
# Example 5: Production Configuration
# =============================================================================


def production_config():
    """Production-ready configuration options."""
    print("\n" + "=" * 60)
    print("5. Production Configuration")
    print("=" * 60)

    config = """
    # Production environment variables

    # Connection (use connection pooling in production)
    DATABASE_URL=postgresql://user:pass@pooler.host:5432/db

    # Connection pool settings
    PGPOOL_MIN_SIZE=5
    PGPOOL_MAX_SIZE=20

    # Timeouts
    PGTIMEOUT=30

    # SSL (required for cloud providers)
    PGSSLMODE=require

    # Embeddings
    EMBEDDING_PROVIDER=openai
    OPENAI_API_KEY=sk-xxx

    # Retriever
    RETRIEVER_BACKEND=postgres
    RETRIEVER_COLLECTION=production_docs
    """

    print("\nRecommended production settings:")
    for line in config.strip().split("\n"):
        if line.strip():
            print(f"  {line.strip()}")


# =============================================================================
# Example 6: Concurrent Access
# =============================================================================


def concurrent_access():
    """Handle concurrent reads and writes."""
    print("\n" + "=" * 60)
    print("6. Concurrent Access")
    print("=" * 60)

    print("\nPostgreSQL handles concurrent access automatically:")

    points = [
        "ACID transactions ensure data consistency",
        "Multiple readers don't block each other",
        "Writers use row-level locking",
        "Connection pooling improves throughput",
    ]

    for point in points:
        print(f"  • {point}")

    print("\nExample with connection pool:")
    print("""
    from ai_infra import Retriever
    import asyncio

    async def search_concurrently():
        retriever = Retriever(backend="postgres")

        # Multiple concurrent searches
        queries = ["query 1", "query 2", "query 3"]
        tasks = [retriever.asearch(q, k=5) for q in queries]
        results = await asyncio.gather(*tasks)
        return results
""")


# =============================================================================
# Example 7: Backup and Recovery
# =============================================================================


def backup_recovery():
    """Backup and restore vector data."""
    print("\n" + "=" * 60)
    print("7. Backup and Recovery")
    print("=" * 60)

    print("\nBackup strategies for PostgreSQL vectors:")

    strategies = {
        "pg_dump": {
            "command": "pg_dump -Fc mydb > backup.dump",
            "restore": "pg_restore -d mydb backup.dump",
            "notes": "Full database backup including vectors",
        },
        "Table export": {
            "command": "COPY my_documents TO '/path/backup.csv' CSV",
            "restore": "COPY my_documents FROM '/path/backup.csv' CSV",
            "notes": "Export specific collection only",
        },
        "Continuous archiving": {
            "command": "Configure WAL archiving",
            "restore": "Point-in-time recovery",
            "notes": "Best for production with RPO requirements",
        },
    }

    for name, config in strategies.items():
        print(f"\n  {name}:")
        print(f"    Backup: {config['command']}")
        print(f"    Restore: {config['restore']}")
        print(f"    Notes: {config['notes']}")


# =============================================================================
# Example 8: Monitoring and Maintenance
# =============================================================================


def monitoring():
    """Monitor and maintain PostgreSQL vectors."""
    print("\n" + "=" * 60)
    print("8. Monitoring and Maintenance")
    print("=" * 60)

    queries = {
        "Check table size": """
            SELECT pg_size_pretty(pg_total_relation_size('my_documents'));
        """,
        "Count documents": """
            SELECT COUNT(*) FROM my_documents;
        """,
        "Check index usage": """
            SELECT indexrelname, idx_scan, idx_tup_read
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public';
        """,
        "Vacuum and analyze": """
            VACUUM ANALYZE my_documents;
        """,
    }

    print("\nUseful monitoring queries:")
    for name, sql in queries.items():
        print(f"\n  -- {name}")
        print(f"  {sql.strip()}")


# =============================================================================
# Example 9: Cloud Provider Setup
# =============================================================================


def cloud_providers():
    """Setup with cloud PostgreSQL providers."""
    print("\n" + "=" * 60)
    print("9. Cloud Provider Setup")
    print("=" * 60)

    providers = {
        "Supabase": {
            "enable_pgvector": "Enable from dashboard Extensions",
            "connection": "postgresql://postgres:xxx@db.xxx.supabase.co:5432/postgres",
        },
        "Neon": {
            "enable_pgvector": "CREATE EXTENSION IF NOT EXISTS vector;",
            "connection": "postgresql://user:xxx@xxx.neon.tech/neondb?sslmode=require",
        },
        "AWS RDS": {
            "enable_pgvector": "CREATE EXTENSION IF NOT EXISTS vector;",
            "connection": "postgresql://user:xxx@xxx.rds.amazonaws.com:5432/db",
        },
        "Google Cloud SQL": {
            "enable_pgvector": "CREATE EXTENSION IF NOT EXISTS vector;",
            "connection": "postgresql://user:xxx@/db?host=/cloudsql/project:region:instance",
        },
    }

    for provider, config in providers.items():
        print(f"\n  {provider}:")
        print(f"    Enable pgvector: {config['enable_pgvector']}")
        print(f"    Connection: {config['connection'][:50]}...")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PostgreSQL + pgvector Backend Examples")
    print("=" * 60)
    print("\nPostgreSQL with pgvector is ideal for production:")
    print("  - ACID transactions")
    print("  - Concurrent access")
    print("  - Horizontal scaling")
    print("  - Cloud provider support\n")

    basic_postgres()
    connection_strings()
    table_configuration()
    index_configuration()
    production_config()
    concurrent_access()
    backup_recovery()
    monitoring()
    cloud_providers()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Use DATABASE_URL for connection configuration")
    print("  2. pgvector extension must be enabled")
    print("  3. Choose appropriate index type for your scale")
    print("  4. Use connection pooling in production")
    print("  5. Cloud providers have easy pgvector setup")


if __name__ == "__main__":
    main()
