#!/usr/bin/env python
"""Long-Term Memory Store Example.

This example demonstrates:
- Basic put/get/delete operations
- Namespace organization
- Semantic search across memories
- TTL (time-to-live) expiration
- Storage backends (memory, SQLite, Postgres)

MemoryStore provides persistent key-value storage
with optional semantic search for long-term memory.
"""

import time

from ai_infra.llm.memory import MemoryStore

# =============================================================================
# Example 1: Basic Operations
# =============================================================================


def basic_operations():
    """Put, get, and delete memories."""
    print("=" * 60)
    print("1. Basic Operations")
    print("=" * 60)

    # Create in-memory store (dev/testing)
    store = MemoryStore()

    # Store a memory
    item = store.put(
        namespace=("user_123", "preferences"),
        key="language",
        value={"preference": "Python", "reason": "Type hints and ecosystem"},
    )

    print(f"\n  Stored: {item.key}")
    print(f"    Namespace: {item.namespace}")
    print(f"    Value: {item.value}")

    # Retrieve it
    retrieved = store.get(("user_123", "preferences"), "language")
    print(f"\n  Retrieved: {retrieved.value}")

    # Update (put with same key)
    store.put(
        namespace=("user_123", "preferences"),
        key="language",
        value={"preference": "Python", "reason": "Updated reason: AI/ML support"},
    )
    updated = store.get(("user_123", "preferences"), "language")
    print(f"\n  Updated: {updated.value}")

    # Delete
    deleted = store.delete(("user_123", "preferences"), "language")
    print(f"\n  Deleted: {deleted}")

    # Verify deletion
    gone = store.get(("user_123", "preferences"), "language")
    print(f"  After delete: {gone}")


# =============================================================================
# Example 2: Namespace Organization
# =============================================================================


def namespace_organization():
    """Use namespaces to organize memories."""
    print("\n" + "=" * 60)
    print("2. Namespace Organization")
    print("=" * 60)

    store = MemoryStore()

    # Different namespaces for different data
    # User preferences
    store.put(("user_123", "preferences"), "theme", {"value": "dark"})
    store.put(("user_123", "preferences"), "language", {"value": "Python"})

    # User facts (things we learned about them)
    store.put(("user_123", "facts"), "job", {"value": "Software Engineer"})
    store.put(("user_123", "facts"), "company", {"value": "TechCorp"})

    # Session-specific context
    store.put(
        ("user_123", "session_abc", "context"),
        "project",
        {
            "name": "API refactor",
            "status": "in progress",
        },
    )

    # App-level configuration
    store.put(("app", "config"), "rate_limit", {"requests_per_minute": 100})

    # List memories in a namespace
    prefs = store.list(("user_123", "preferences"))
    print(f"\n  User preferences ({len(prefs)} items):")
    for item in prefs:
        print(f"    {item.key}: {item.value}")

    facts = store.list(("user_123", "facts"))
    print(f"\n  User facts ({len(facts)} items):")
    for item in facts:
        print(f"    {item.key}: {item.value}")

    # Namespaces are isolated
    other_user = store.list(("user_456", "preferences"))
    print(f"\n  Other user preferences: {len(other_user)} items")


# =============================================================================
# Example 3: Convenience Methods
# =============================================================================


def convenience_methods():
    """Use helper methods for common patterns."""
    print("\n" + "=" * 60)
    print("3. Convenience Methods")
    print("=" * 60)

    store = MemoryStore()

    # User-scoped memories
    store.put_user_memory(
        user_id="user_123",
        key="favorite_color",
        value={"color": "blue", "shade": "navy"},
    )

    item = store.get_user_memory("user_123", "favorite_color")
    print(f"\n  User memory: {item.value}")

    # With custom category
    store.put_user_memory(
        user_id="user_123",
        key="auth_token",
        value={"token": "abc123"},
        category="credentials",  # Instead of default "memories"
    )

    # App-level memories
    store.put_app_memory(
        key="system_version",
        value={"version": "2.0.0", "deployed": "2025-01-01"},
    )

    app_item = store.get_app_memory("system_version")
    print(f"  App memory: {app_item.value}")


# =============================================================================
# Example 4: TTL Expiration
# =============================================================================


def ttl_expiration():
    """Use TTL for temporary memories."""
    print("\n" + "=" * 60)
    print("4. TTL Expiration")
    print("=" * 60)

    store = MemoryStore()

    # Store with 1 second TTL (for demo)
    store.put(
        namespace=("user_123", "temp"),
        key="session_token",
        value={"token": "xyz789"},
        ttl=1,  # Expires in 1 second
    )

    # Immediately available
    item = store.get(("user_123", "temp"), "session_token")
    print(f"\n  Before expiry: {item.value if item else None}")

    # Wait for expiration
    print("  Waiting 1.5 seconds...")
    time.sleep(1.5)

    # Now expired
    item = store.get(("user_123", "temp"), "session_token")
    print(f"  After expiry: {item}")

    # Practical TTL values
    print("\n  Common TTL patterns:")
    print("    Session: 3600 (1 hour)")
    print("    Cache: 300 (5 minutes)")
    print("    Token: 86400 (24 hours)")
    print("    Remember: None (permanent)")


# =============================================================================
# Example 5: Semantic Search
# =============================================================================


def semantic_search():
    """Search memories by meaning."""
    print("\n" + "=" * 60)
    print("5. Semantic Search")
    print("=" * 60)

    print("\n  Note: Semantic search requires embedding_provider")
    print("\n  Setup with embeddings:")
    print("""
    store = MemoryStore(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    )

    # Store memories (automatically embedded)
    store.put(("user_123", "facts"), "lang", {
        "value": "User prefers Python for data science"
    })
    store.put(("user_123", "facts"), "editor", {
        "value": "User uses VS Code with Vim keybindings"
    })
    store.put(("user_123", "facts"), "os", {
        "value": "User runs macOS on M2 MacBook Pro"
    })

    # Search by meaning (not exact match)
    results = store.search(
        namespace=("user_123", "facts"),
        query="programming tools preferences",
        limit=3,
    )

    for item in results:
        print(f"{item.key}: {item.value}")
""")


# =============================================================================
# Example 6: Storage Backends
# =============================================================================


def storage_backends():
    """Different storage backends for different needs."""
    print("\n" + "=" * 60)
    print("6. Storage Backends")
    print("=" * 60)

    # In-memory (dev/testing)
    print("\n  In-memory (not persisted):")
    print("    store = MemoryStore()")

    # SQLite (single-instance production)
    print("\n  SQLite (persisted to file):")
    print('    store = MemoryStore.sqlite("./memories.db")')
    print("    # Or with embeddings:")
    print("""    store = MemoryStore.sqlite(
        "./memories.db",
        embedding_provider="openai",
    )""")

    # PostgreSQL (multi-instance production)
    print("\n  PostgreSQL (production):")
    print('    store = MemoryStore.postgres(os.environ["DATABASE_URL"])')
    print("    # Or with embeddings:")
    print("""    store = MemoryStore.postgres(
        os.environ["DATABASE_URL"],
        embedding_provider="openai",
    )""")

    print("\n  Backend comparison:")
    print("    | Backend    | Persistence | Multi-process | Semantic |")
    print("    |------------|-------------|---------------|----------|")
    print("    | In-memory  | [X]          | [X]            | [OK]       |")
    print("    | SQLite     | [OK]          | [!] (limited)  | [OK]       |")
    print("    | PostgreSQL | [OK]          | [OK]            | [OK]       |")


# =============================================================================
# Example 7: Multi-Tenant Patterns
# =============================================================================


def multi_tenant():
    """Organize memories for multi-tenant applications."""
    print("\n" + "=" * 60)
    print("7. Multi-Tenant Patterns")
    print("=" * 60)

    print("\n  Namespace hierarchy for multi-tenancy:")
    print("""
    # Organization -> User -> Category
    store.put(
        namespace=("org_acme", "user_123", "preferences"),
        key="theme",
        value={"theme": "dark"},
    )

    # Get all preferences for a user in an org
    items = store.list(("org_acme", "user_123", "preferences"))

    # Organization-level settings
    store.put(
        namespace=("org_acme", "settings"),
        key="plan",
        value={"plan": "enterprise", "seats": 100},
    )
""")

    print("\n  Alternative: Single namespace, filter by value")
    print("""
    # Store tenant info in value
    store.put(
        namespace=("memories",),
        key="user_123_pref_theme",
        value={
            "org_id": "acme",
            "user_id": "123",
            "key": "theme",
            "value": "dark",
        },
    )

    # Search with filter
    results = store.search(
        ("memories",),
        query="user preferences",
        filter={"org_id": "acme", "user_id": "123"},
    )
""")


# =============================================================================
# Example 8: MemoryItem Details
# =============================================================================


def memory_item_details():
    """Inspect MemoryItem fields."""
    print("\n" + "=" * 60)
    print("8. MemoryItem Details")
    print("=" * 60)

    store = MemoryStore()

    item = store.put(
        namespace=("user_123", "facts"),
        key="favorite_food",
        value={"food": "pizza", "topping": "pepperoni"},
        ttl=3600,
    )

    print("\n  MemoryItem fields:")
    print(f"    namespace: {item.namespace}")
    print(f"    key: {item.key}")
    print(f"    value: {item.value}")
    print(f"    created_at: {item.created_at}")
    print(f"    updated_at: {item.updated_at}")
    print(f"    expires_at: {item.expires_at}")


# =============================================================================
# Example 9: List with Limit
# =============================================================================


def list_with_limit():
    """List memories with pagination."""
    print("\n" + "=" * 60)
    print("9. List with Limit")
    print("=" * 60)

    store = MemoryStore()

    # Store multiple items
    for i in range(10):
        store.put(
            namespace=("user_123", "items"),
            key=f"item_{i}",
            value={"index": i},
        )

    # List all
    all_items = store.list(("user_123", "items"))
    print(f"\n  All items: {len(all_items)}")

    # List with limit
    limited = store.list(("user_123", "items"), limit=3)
    print(f"  With limit=3: {len(limited)}")
    for item in limited:
        print(f"    {item.key}: {item.value}")


# =============================================================================
# Example 10: Agent Integration Pattern
# =============================================================================


def agent_integration():
    """Use MemoryStore with agents."""
    print("\n" + "=" * 60)
    print("10. Agent Integration Pattern")
    print("=" * 60)

    print("\n  Pattern: Agent with long-term memory")
    print("""
    from ai_infra import Agent, LLM
    from ai_infra.memory import MemoryStore
    from langchain_core.tools import tool

    # Initialize store
    store = MemoryStore.sqlite("./memories.db", embedding_provider="openai")

    @tool
    def remember(user_id: str, key: str, value: str) -> str:
        '''Store something to remember about the user.'''
        store.put_user_memory(user_id, key, {"value": value})
        return f"Remembered: {key}"

    @tool
    def recall(user_id: str, query: str) -> str:
        '''Recall memories about the user matching a query.'''
        results = store.search_user_memories(user_id, query, limit=5)
        if not results:
            return "No relevant memories found."
        return "\\n".join(f"- {r.key}: {r.value['value']}" for r in results)

    # Create agent with memory tools
    agent = Agent(
        tools=[remember, recall],
        system_prompt="You have access to long-term memory. Use remember() to store facts and recall() to search memories.",
    )

    # Usage
    agent.run(
        "Remember that I prefer dark mode",
        user_id="user_123",
    )

    agent.run(
        "What do you know about my preferences?",
        user_id="user_123",
    )
""")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Long-Term Memory Store Examples")
    print("=" * 60)
    print("\nPersistent key-value storage with semantic search.\n")

    basic_operations()
    namespace_organization()
    convenience_methods()
    ttl_expiration()
    semantic_search()
    storage_backends()
    multi_tenant()
    memory_item_details()
    list_with_limit()
    agent_integration()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Namespaces organize memories (user/org/category)")
    print("  2. TTL enables temporary/session memories")
    print("  3. Semantic search finds memories by meaning")
    print("  4. SQLite for single-instance, Postgres for production")
    print("  5. Convenience methods simplify user/app memory access")


if __name__ == "__main__":
    main()
