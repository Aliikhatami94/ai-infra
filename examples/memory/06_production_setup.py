#!/usr/bin/env python
"""Production Memory Setup Example.

This example demonstrates:
- PostgreSQL backend configuration
- Connection pooling and performance
- Multi-process / multi-instance setups
- Monitoring and observability
- Migration and maintenance patterns

Production-ready memory infrastructure for
enterprise AI applications.
"""


# =============================================================================
# Example 1: PostgreSQL Backend Setup
# =============================================================================


def postgres_setup():
    """Set up PostgreSQL for production memory."""
    print("=" * 60)
    print("1. PostgreSQL Backend Setup")
    print("=" * 60)

    print("\n  Environment variables:")
    print("""
    # .env file
    DATABASE_URL=postgresql://user:pass@localhost:5432/ai_memories
    OPENAI_API_KEY=sk-...
""")

    print("\n  Initialize with PostgreSQL:")
    print("""
    from ai_infra.memory import MemoryStore
    from ai_infra.llm.tools.custom import ConversationMemory

    # Long-term memory store
    memory_store = MemoryStore.postgres(
        os.environ["DATABASE_URL"],
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",  # Cheaper, faster
    )

    # Conversation history
    conv_memory = ConversationMemory(
        backend="postgres",
        connection_string=os.environ["DATABASE_URL"],
        embedding_provider="openai",
    )
""")

    print("\n  Required PostgreSQL extensions:")
    print("    - pgvector (for embedding similarity search)")
    print("    - uuid-ossp (optional, for UUID generation)")


# =============================================================================
# Example 2: Database Schema
# =============================================================================


def database_schema():
    """PostgreSQL schema for memory storage."""
    print("\n" + "=" * 60)
    print("2. Database Schema")
    print("=" * 60)

    print("\n  MemoryStore schema (auto-created):")
    print("""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS memories (
        id SERIAL PRIMARY KEY,
        namespace TEXT NOT NULL,
        key TEXT NOT NULL,
        value JSONB NOT NULL,
        embedding vector(1536),  -- OpenAI text-embedding-3-small
        created_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL,
        expires_at DOUBLE PRECISION,
        UNIQUE(namespace, key)
    );

    CREATE INDEX idx_memories_namespace ON memories(namespace);
    CREATE INDEX idx_memories_expires ON memories(expires_at)
        WHERE expires_at IS NOT NULL;

    -- Vector similarity index
    CREATE INDEX idx_memories_embedding ON memories
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
""")

    print("\n  ConversationMemory schema:")
    print("""
    CREATE TABLE IF NOT EXISTS conversation_chunks (
        id SERIAL PRIMARY KEY,
        chunk_id TEXT UNIQUE NOT NULL,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        text TEXT NOT NULL,
        summary TEXT,
        metadata JSONB DEFAULT '{}',
        embedding vector(1536),
        created_at DOUBLE PRECISION NOT NULL
    );

    CREATE INDEX idx_chunks_user ON conversation_chunks(user_id);
    CREATE INDEX idx_chunks_session ON conversation_chunks(user_id, session_id);
    CREATE INDEX idx_chunks_embedding ON conversation_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
""")


# =============================================================================
# Example 3: Connection Pooling
# =============================================================================


def connection_pooling():
    """Configure connection pooling for performance."""
    print("\n" + "=" * 60)
    print("3. Connection Pooling")
    print("=" * 60)

    print("\n  Use connection pooler (recommended):")
    print("""
    # PgBouncer configuration (pgbouncer.ini)
    [databases]
    ai_memories = host=localhost port=5432 dbname=ai_memories

    [pgbouncer]
    pool_mode = transaction
    max_client_conn = 1000
    default_pool_size = 20
    min_pool_size = 5

    # Connect via PgBouncer
    DATABASE_URL=postgresql://user:pass@localhost:6432/ai_memories
""")

    print("\n  SQLAlchemy pool settings:")
    print("""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import QueuePool

    engine = create_engine(
        os.environ["DATABASE_URL"],
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
""")


# =============================================================================
# Example 4: Multi-Instance Deployment
# =============================================================================


def multi_instance():
    """Deploy across multiple instances."""
    print("\n" + "=" * 60)
    print("4. Multi-Instance Deployment")
    print("=" * 60)

    print("\n  Architecture for horizontal scaling:")
    print("""
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  Instance 1 │  │  Instance 2 │  │  Instance 3 │
    │  (memory)   │  │  (memory)   │  │  (memory)   │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
                    ┌───────▼───────┐
                    │   PgBouncer   │
                    │ (connection   │
                    │   pooler)     │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │  PostgreSQL   │
                    │  (primary)    │
                    └───────────────┘
""")

    print("\n  Each instance uses the same database:")
    print("""
    # All instances connect to same PostgreSQL
    memory_store = MemoryStore.postgres(
        os.environ["DATABASE_URL"],  # Same for all
        embedding_provider="openai",
    )

    # Data is automatically shared across instances
""")


# =============================================================================
# Example 5: Embedding Provider Configuration
# =============================================================================


def embedding_providers():
    """Configure different embedding providers."""
    print("\n" + "=" * 60)
    print("5. Embedding Provider Configuration")
    print("=" * 60)

    print("\n  OpenAI (recommended):")
    print("""
    store = MemoryStore.postgres(
        database_url,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",  # 1536 dims, $0.02/1M
    )

    # Or higher quality
    store = MemoryStore.postgres(
        database_url,
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",  # 3072 dims, $0.13/1M
    )
""")

    print("\n  Google (Gemini):")
    print("""
    store = MemoryStore.postgres(
        database_url,
        embedding_provider="google",
        embedding_model="text-embedding-004",
    )
""")

    print("\n  Local embeddings (cost-free):")
    print("""
    # Using sentence-transformers (no API cost)
    store = MemoryStore.postgres(
        database_url,
        embedding_provider="huggingface",
        embedding_model="all-MiniLM-L6-v2",  # 384 dims
    )
""")


# =============================================================================
# Example 6: Caching Layer
# =============================================================================


def caching_layer():
    """Add Redis caching for frequently accessed memories."""
    print("\n" + "=" * 60)
    print("6. Caching Layer")
    print("=" * 60)

    print("\n  Redis cache for hot memories:")
    print("""
    import redis
    import json
    from functools import wraps

    redis_client = redis.Redis.from_url(os.environ["REDIS_URL"])

    def cached_memory(store: MemoryStore, ttl: int = 300):
        '''Decorator for caching memory lookups.'''

        def decorator(func):
            @wraps(func)
            def wrapper(namespace, key):
                # Try cache first
                cache_key = f"memory:{'/'.join(namespace)}:{key}"
                cached = redis_client.get(cache_key)
                if cached:
                    return MemoryItem(**json.loads(cached))

                # Miss - hit database
                item = func(namespace, key)
                if item:
                    redis_client.setex(
                        cache_key,
                        ttl,
                        json.dumps(item.__dict__),
                    )
                return item

            return wrapper
        return decorator

    # Wrap the store's get method
    @cached_memory(memory_store, ttl=300)
    def get_memory(namespace, key):
        return memory_store.get(namespace, key)
""")


# =============================================================================
# Example 7: Monitoring and Metrics
# =============================================================================


def monitoring():
    """Add observability to memory operations."""
    print("\n" + "=" * 60)
    print("7. Monitoring and Metrics")
    print("=" * 60)

    print("\n  Prometheus metrics:")
    print("""
    from prometheus_client import Counter, Histogram

    MEMORY_OPS = Counter(
        'memory_operations_total',
        'Total memory operations',
        ['operation', 'namespace', 'status']
    )

    MEMORY_LATENCY = Histogram(
        'memory_operation_seconds',
        'Memory operation latency',
        ['operation']
    )

    class InstrumentedMemoryStore:
        def __init__(self, store: MemoryStore):
            self._store = store

        def get(self, namespace, key):
            with MEMORY_LATENCY.labels(operation='get').time():
                try:
                    result = self._store.get(namespace, key)
                    status = 'hit' if result else 'miss'
                except Exception:
                    status = 'error'
                    raise
                finally:
                    ns_label = namespace[0] if namespace else 'unknown'
                    MEMORY_OPS.labels(
                        operation='get',
                        namespace=ns_label,
                        status=status,
                    ).inc()
                return result

        # ... wrap other methods ...
""")

    print("\n  Logging:")
    print("""
    import structlog

    logger = structlog.get_logger()

    class LoggedMemoryStore:
        def get(self, namespace, key):
            result = self._store.get(namespace, key)
            logger.info(
                "memory_get",
                namespace=namespace,
                key=key,
                found=result is not None,
            )
            return result
""")


# =============================================================================
# Example 8: Backup and Recovery
# =============================================================================


def backup_recovery():
    """Backup and restore memory data."""
    print("\n" + "=" * 60)
    print("8. Backup and Recovery")
    print("=" * 60)

    print("\n  PostgreSQL backup:")
    print("""
    # Full database backup
    pg_dump -Fc ai_memories > backup_$(date +%Y%m%d).dump

    # Restore
    pg_restore -d ai_memories backup_20250115.dump
""")

    print("\n  Table-level export:")
    print("""
    # Export memories to JSON
    psql ai_memories -c "
        COPY (
            SELECT json_agg(row_to_json(t))
            FROM memories t
        ) TO '/tmp/memories.json'
    "

    # Export conversations
    psql ai_memories -c "
        COPY (
            SELECT json_agg(row_to_json(t))
            FROM conversation_chunks t
        ) TO '/tmp/conversations.json'
    "
""")

    print("\n  Automated backup with cron:")
    print("""
    # /etc/cron.d/memory-backup
    0 2 * * * postgres pg_dump -Fc ai_memories > /backups/$(date +\\%Y\\%m\\%d).dump
    0 3 * * 0 find /backups -mtime +30 -delete  # Keep 30 days
""")


# =============================================================================
# Example 9: Memory Maintenance
# =============================================================================


def maintenance():
    """Routine maintenance tasks."""
    print("\n" + "=" * 60)
    print("9. Memory Maintenance")
    print("=" * 60)

    print("\n  Cleanup expired memories:")
    print("""
    import time

    def cleanup_expired():
        '''Remove expired memories.'''
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()

        cur.execute('''
            DELETE FROM memories
            WHERE expires_at IS NOT NULL
              AND expires_at < %s
        ''', (time.time(),))

        deleted = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()

        return deleted

    # Run as scheduled job
    @scheduler.scheduled_job('cron', hour=3)
    def nightly_cleanup():
        deleted = cleanup_expired()
        logger.info(f"Cleaned up {deleted} expired memories")
""")

    print("\n  Vacuum for performance:")
    print("""
    -- Run periodically for optimal performance
    VACUUM ANALYZE memories;
    VACUUM ANALYZE conversation_chunks;

    -- Reindex if needed
    REINDEX INDEX idx_memories_embedding;
    REINDEX INDEX idx_chunks_embedding;
""")

    print("\n  Monitor table size:")
    print("""
    SELECT
        relname AS table,
        pg_size_pretty(pg_total_relation_size(relid)) AS total_size
    FROM pg_catalog.pg_statio_user_tables
    WHERE relname IN ('memories', 'conversation_chunks');
""")


# =============================================================================
# Example 10: Full Production Configuration
# =============================================================================


def full_production_config():
    """Complete production configuration."""
    print("\n" + "=" * 60)
    print("10. Full Production Configuration")
    print("=" * 60)

    print("\n  Environment variables (.env):")
    print("""
    # Database
    DATABASE_URL=postgresql://user:pass@pgbouncer:6432/ai_memories
    REDIS_URL=redis://redis:6379/0

    # Embeddings
    OPENAI_API_KEY=sk-...
    EMBEDDING_MODEL=text-embedding-3-small

    # Performance
    DB_POOL_SIZE=10
    DB_MAX_OVERFLOW=20
    CACHE_TTL=300

    # Monitoring
    ENABLE_METRICS=true
    LOG_LEVEL=info
""")

    print("\n  Production initialization:")
    print("""
    from ai_infra.memory import MemoryStore
    from ai_infra.llm.tools.custom import ConversationMemory
    import structlog

    logger = structlog.get_logger()

    class ProductionMemory:
        def __init__(self):
            self.database_url = os.environ["DATABASE_URL"]

            # Initialize stores
            self.facts = MemoryStore.postgres(
                self.database_url,
                embedding_provider="openai",
                embedding_model=os.environ.get(
                    "EMBEDDING_MODEL",
                    "text-embedding-3-small"
                ),
            )

            self.conversations = ConversationMemory(
                backend="postgres",
                connection_string=self.database_url,
                embedding_provider="openai",
            )

            # Optional: Redis cache
            if os.environ.get("REDIS_URL"):
                self._setup_cache()

            logger.info("Production memory initialized")

        def health_check(self) -> bool:
            '''Check memory system health.'''
            try:
                # Test database connection
                self.facts.list(("health", "check"), limit=1)
                return True
            except Exception as e:
                logger.error("Memory health check failed", error=str(e))
                return False

    # FastAPI health endpoint
    @app.get("/health")
    async def health():
        memory_ok = memory.health_check()
        return {
            "status": "healthy" if memory_ok else "unhealthy",
            "memory": "ok" if memory_ok else "error",
        }
""")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Production Memory Setup")
    print("=" * 60)
    print("\nEnterprise-ready memory infrastructure.\n")

    postgres_setup()
    database_schema()
    connection_pooling()
    multi_instance()
    embedding_providers()
    caching_layer()
    monitoring()
    backup_recovery()
    maintenance()
    full_production_config()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nProduction checklist:")
    print("  ✓ PostgreSQL with pgvector extension")
    print("  ✓ Connection pooler (PgBouncer)")
    print("  ✓ Embedding provider configured")
    print("  ✓ Monitoring and metrics")
    print("  ✓ Backup strategy")
    print("  ✓ Maintenance jobs scheduled")


if __name__ == "__main__":
    main()
