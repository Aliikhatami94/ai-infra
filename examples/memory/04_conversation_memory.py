#!/usr/bin/env python
"""Conversation Memory (RAG over Chat History) Example.

This example demonstrates:
- Indexing past conversations
- Semantic search over conversation history
- Multi-session conversation recall
- Chunking strategies for long conversations
- Integration with agents

ConversationMemory lets agents search and recall
information from past conversations.
"""


# =============================================================================
# Example 1: Basic Conversation Indexing
# =============================================================================


def basic_indexing():
    """Index a conversation for later search."""
    print("=" * 60)
    print("1. Basic Conversation Indexing")
    print("=" * 60)

    print("\n  Index a completed conversation:")
    print("""
    from ai_infra.llm.tools.custom import ConversationMemory

    # Create conversation memory
    memory = ConversationMemory(
        backend="sqlite",
        path="./conversations.db",
        embedding_provider="openai",
    )

    # A past conversation
    messages = [
        HumanMessage(content="How do I debug Python code?"),
        AIMessage(content="You can use pdb, breakpoints, or print statements."),
        HumanMessage(content="What's the best approach for async code?"),
        AIMessage(content="Use asyncio.debug mode and logging for async debugging."),
    ]

    # Index it
    memory.index_conversation(
        user_id="user_123",
        session_id="session_abc",
        messages=messages,
        metadata={
            "topic": "debugging",
            "date": "2025-01-15",
        },
    )

    print("Conversation indexed!")
""")


# =============================================================================
# Example 2: Searching Conversations
# =============================================================================


def searching_conversations():
    """Search past conversations by meaning."""
    print("\n" + "=" * 60)
    print("2. Searching Conversations")
    print("=" * 60)

    print("\n  Search across all indexed conversations:")
    print("""
    # Search for debugging-related conversations
    results = memory.search(
        user_id="user_123",
        query="how to fix async bugs",
        limit=5,
    )

    for result in results:
        print(f"Score: {result.score:.2f}")
        print(f"Session: {result.chunk.session_id}")
        print(f"Context: {result.context[:200]}...")
        print()
""")

    print("\n  Filter by session:")
    print("""
    # Search within a specific session
    results = memory.search(
        user_id="user_123",
        query="debugging tips",
        session_id="session_abc",  # Only this session
        limit=5,
    )
""")


# =============================================================================
# Example 3: Chunking Strategies
# =============================================================================


def chunking_strategies():
    """How conversations are chunked for indexing."""
    print("\n" + "=" * 60)
    print("3. Chunking Strategies")
    print("=" * 60)

    print("\n  Conversations are chunked for better search:")
    print("""
    memory = ConversationMemory(
        backend="sqlite",
        path="./conversations.db",
        embedding_provider="openai",
        chunk_size=10,      # Max messages per chunk
        chunk_overlap=2,    # Overlap between chunks
    )
""")

    print("\n  Example: 20-message conversation with chunk_size=10:")
    print("    Chunk 1: messages 1-10")
    print("    Chunk 2: messages 9-18  (overlap of 2)")
    print("    Chunk 3: messages 17-20 (overlap of 2)")

    print("\n  Benefits of chunking:")
    print("    - Better embedding quality (focused context)")
    print("    - More precise search results")
    print("    - Overlap preserves context at boundaries")


# =============================================================================
# Example 4: Metadata and Filtering
# =============================================================================


def metadata_filtering():
    """Use metadata to organize and filter conversations."""
    print("\n" + "=" * 60)
    print("4. Metadata and Filtering")
    print("=" * 60)

    print("\n  Add metadata when indexing:")
    print("""
    memory.index_conversation(
        user_id="user_123",
        session_id="session_xyz",
        messages=messages,
        metadata={
            "topic": "python-debugging",
            "date": "2025-01-15",
            "channel": "slack",
            "agent": "support-bot",
            "resolved": True,
        },
    )
""")

    print("\n  Metadata is stored with each chunk")
    print("  Can be used for filtering in advanced backends")


# =============================================================================
# Example 5: SearchResult Structure
# =============================================================================


def search_result_structure():
    """Understanding search results."""
    print("\n" + "=" * 60)
    print("5. SearchResult Structure")
    print("=" * 60)

    print("\n  SearchResult fields:")
    print("""
    result = results[0]

    result.score        # Similarity score (0-1)
    result.context      # Formatted conversation text
    result.chunk        # ConversationChunk object

    result.chunk.chunk_id    # Unique chunk ID
    result.chunk.user_id     # User who had conversation
    result.chunk.session_id  # Session identifier
    result.chunk.messages    # The actual messages
    result.chunk.text        # Text representation
    result.chunk.summary     # Optional summary
    result.chunk.metadata    # Custom metadata
    result.chunk.created_at  # When indexed
""")


# =============================================================================
# Example 6: Storage Backends
# =============================================================================


def storage_backends():
    """Different backends for conversation storage."""
    print("\n" + "=" * 60)
    print("6. Storage Backends")
    print("=" * 60)

    print("\n  In-memory (dev/testing):")
    print("""
    memory = ConversationMemory(
        backend="memory",
        embedding_provider="openai",
    )
""")

    print("\n  SQLite (single-server):")
    print("""
    memory = ConversationMemory(
        backend="sqlite",
        path="./conversations.db",
        embedding_provider="openai",
    )
""")

    print("\n  PostgreSQL (production):")
    print("""
    memory = ConversationMemory(
        backend="postgres",
        connection_string=os.environ["DATABASE_URL"],
        embedding_provider="openai",
    )
""")


# =============================================================================
# Example 7: Async Operations
# =============================================================================


def async_operations():
    """Async methods for web applications."""
    print("\n" + "=" * 60)
    print("7. Async Operations")
    print("=" * 60)

    print("\n  Use async methods in web apps:")
    print("""
    # Async indexing
    await memory.aindex_conversation(
        user_id="user_123",
        session_id="session_abc",
        messages=messages,
    )

    # Async search
    results = await memory.asearch(
        user_id="user_123",
        query="debugging async code",
        limit=5,
    )
""")


# =============================================================================
# Example 8: Delete Operations
# =============================================================================


def delete_operations():
    """Clean up old conversations."""
    print("\n" + "=" * 60)
    print("8. Delete Operations")
    print("=" * 60)

    print("\n  Delete a specific conversation:")
    print("""
    # Delete a single conversation
    deleted_count = memory.delete_conversation(
        user_id="user_123",
        session_id="session_abc",
    )
    print(f"Deleted {deleted_count} chunks")
""")

    print("\n  Delete all user conversations:")
    print("""
    # GDPR-compliant user data deletion
    deleted_count = memory.delete_user_conversations(user_id="user_123")
    print(f"Deleted {deleted_count} total chunks")
""")


# =============================================================================
# Example 9: Create Memory Tool for Agents
# =============================================================================


def memory_tool():
    """Create a recall tool for agents."""
    print("\n" + "=" * 60)
    print("9. Create Memory Tool for Agents")
    print("=" * 60)

    print("\n  Create a tool agents can use to recall:")
    print("""
    from ai_infra.llm.tools.custom import ConversationMemory, create_memory_tool

    # Create memory
    memory = ConversationMemory(
        backend="sqlite",
        path="./conversations.db",
        embedding_provider="openai",
    )

    # Create agent tool
    recall_tool = create_memory_tool(
        memory,
        name="recall_conversation",
        description="Search past conversations with this user",
    )

    # Use with agent
    from ai_infra import Agent

    agent = Agent(
        tools=[recall_tool, other_tools...],
        system_prompt="Use recall_conversation to find relevant past discussions.",
    )

    # Agent can now search history
    response = agent.run(
        "What did we discuss about API authentication last week?",
        user_id="user_123",
    )
""")


# =============================================================================
# Example 10: Complete Integration Pattern
# =============================================================================


def complete_integration():
    """Full pattern for conversation memory with agents."""
    print("\n" + "=" * 60)
    print("10. Complete Integration Pattern")
    print("=" * 60)

    print("\n  Pattern: Agent with conversation recall")
    print("""
    from ai_infra import Agent
    from ai_infra.llm.tools.custom import ConversationMemory, create_memory_tool

    class ConversationAgent:
        def __init__(self):
            # Initialize conversation memory
            self.memory = ConversationMemory(
                backend="sqlite",
                path="./conversations.db",
                embedding_provider="openai",
            )

            # Create agent with recall capability
            self.agent = Agent(
                tools=[
                    create_memory_tool(self.memory),
                    # ... other tools
                ],
                system_prompt='''You are a helpful assistant with memory.
                Use recall_conversation to find relevant past discussions
                before answering questions about previous topics.''',
            )

        async def chat(
            self,
            user_id: str,
            session_id: str,
            message: str,
            history: list[dict],
        ) -> str:
            # Run agent
            response = await self.agent.arun(
                message,
                messages=history,
                user_id=user_id,
            )

            return response

        async def end_session(
            self,
            user_id: str,
            session_id: str,
            messages: list,
        ):
            # Index completed conversation for future recall
            await self.memory.aindex_conversation(
                user_id=user_id,
                session_id=session_id,
                messages=messages,
                metadata={
                    "date": datetime.now().isoformat(),
                },
            )

    # Usage
    agent = ConversationAgent()

    # During chat
    response = await agent.chat(
        user_id="user_123",
        session_id="session_abc",
        message="What debugging tips did you give me?",
        history=[...],
    )

    # When session ends
    await agent.end_session(
        user_id="user_123",
        session_id="session_abc",
        messages=full_conversation,
    )
""")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Conversation Memory Examples")
    print("=" * 60)
    print("\nRAG over chat history for long-term recall.\n")

    basic_indexing()
    searching_conversations()
    chunking_strategies()
    metadata_filtering()
    search_result_structure()
    storage_backends()
    async_operations()
    delete_operations()
    memory_tool()
    complete_integration()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Index conversations after they complete")
    print("  2. Chunking improves search quality")
    print("  3. Metadata enables organization and filtering")
    print("  4. create_memory_tool() makes agent integration easy")
    print("  5. SQLite for single-server, Postgres for production")


if __name__ == "__main__":
    main()
