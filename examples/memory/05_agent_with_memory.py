#!/usr/bin/env python
"""Agent with Full Memory Stack Example.

This example demonstrates:
- Agent with short-term context management
- Agent with long-term memory store
- Agent with conversation recall
- Combining all memory features
- Production-ready memory patterns

Build agents that remember across sessions
and conversations.
"""


# =============================================================================
# Example 1: Agent with Context Management
# =============================================================================


def agent_with_context():
    """Agent that manages its own context window."""
    print("=" * 60)
    print("1. Agent with Context Management")
    print("=" * 60)

    print("\n  Use SummarizationMiddleware for auto context management:")
    print("""
    from ai_infra import Agent
    from ai_infra.memory import SummarizationMiddleware

    agent = Agent(
        tools=[...],
        middleware=[
            SummarizationMiddleware(
                trigger_tokens=4000,  # Summarize when context > 4000 tokens
                keep_messages=10,     # Always keep last 10 messages
            )
        ]
    )

    # Long conversation - context is auto-managed
    for message in long_conversation:
        response = agent.run(message)
        # No context overflow errors!
""")


# =============================================================================
# Example 2: Agent with Long-Term Memory
# =============================================================================


def agent_with_memory_store():
    """Agent that stores and recalls user facts."""
    print("\n" + "=" * 60)
    print("2. Agent with Long-Term Memory")
    print("=" * 60)

    print("\n  Agent with remember/recall tools:")
    print("""
    from ai_infra import Agent
    from ai_infra.memory import MemoryStore
    from langchain_core.tools import tool

    # Initialize memory store
    store = MemoryStore.sqlite(
        "./user_memories.db",
        embedding_provider="openai",
    )

    @tool
    def remember(fact: str) -> str:
        '''Store a fact about the current user.'''
        key = f"fact_{int(time.time())}"
        store.put_user_memory(
            user_id=_current_user_id,  # From context
            key=key,
            value={"fact": fact},
        )
        return f"I'll remember: {fact}"

    @tool
    def recall(query: str) -> str:
        '''Recall facts about the current user.'''
        results = store.search_user_memories(
            user_id=_current_user_id,
            query=query,
            limit=5,
        )
        if not results:
            return "I don't recall anything about that."
        return "\\n".join(
            f"- {r.value['fact']}" for r in results
        )

    agent = Agent(
        tools=[remember, recall],
        system_prompt='''You have long-term memory.
        Use remember() to store important facts the user tells you.
        Use recall() when the user asks about past conversations.''',
    )

    # Example conversation
    agent.run("My favorite color is blue")  # Agent uses remember()
    agent.run("I work as a software engineer")  # Agent uses remember()
    agent.run("What do you know about me?")  # Agent uses recall()
""")


# =============================================================================
# Example 3: Agent with Conversation Recall
# =============================================================================


def agent_with_conversation_recall():
    """Agent that can search past conversations."""
    print("\n" + "=" * 60)
    print("3. Agent with Conversation Recall")
    print("=" * 60)

    print("\n  Agent with conversation history search:")
    print("""
    from ai_infra import Agent
    from ai_infra.llm.tools.custom import ConversationMemory, create_memory_tool

    # Initialize conversation memory
    conv_memory = ConversationMemory(
        backend="sqlite",
        path="./conversations.db",
        embedding_provider="openai",
    )

    # Create recall tool
    recall_tool = create_memory_tool(conv_memory)

    agent = Agent(
        tools=[recall_tool],
        system_prompt='''You can search past conversations.
        Use recall_conversation when the user asks about
        previous discussions or topics from earlier chats.''',
    )

    # User asks about past topic
    agent.run("What did we decide about the API design last week?")
    # Agent searches conversation history and summarizes findings
""")


# =============================================================================
# Example 4: Full Memory Stack
# =============================================================================


def full_memory_stack():
    """Agent with all memory features combined."""
    print("\n" + "=" * 60)
    print("4. Full Memory Stack")
    print("=" * 60)

    print("\n  Combine all memory features:")
    print("""
    from ai_infra import Agent
    from ai_infra.memory import (
        MemoryStore,
        SummarizationMiddleware,
    )
    from ai_infra.llm.tools.custom import ConversationMemory, create_memory_tool
    from langchain_core.tools import tool

    class FullyMemorizedAgent:
        def __init__(self, db_path: str = "./agent.db"):
            # 1. Long-term memory (user facts)
            self.memory_store = MemoryStore.sqlite(
                db_path.replace(".db", "_facts.db"),
                embedding_provider="openai",
            )

            # 2. Conversation history (past sessions)
            self.conv_memory = ConversationMemory(
                backend="sqlite",
                path=db_path.replace(".db", "_convos.db"),
                embedding_provider="openai",
            )

            # Create tools
            self.tools = self._create_tools()

            # 3. Create agent with context management
            self.agent = Agent(
                tools=self.tools,
                middleware=[
                    SummarizationMiddleware(
                        trigger_tokens=4000,
                        keep_messages=10,
                    )
                ],
                system_prompt=self._get_system_prompt(),
            )

        def _create_tools(self):
            store = self.memory_store

            @tool
            def remember_fact(fact: str) -> str:
                '''Store an important fact about the user.'''
                key = f"fact_{int(time.time())}"
                store.put_user_memory(
                    user_id=self._current_user,
                    key=key,
                    value={"fact": fact},
                )
                return f"Remembered: {fact}"

            @tool
            def recall_facts(query: str) -> str:
                '''Recall stored facts about the user.'''
                results = store.search_user_memories(
                    user_id=self._current_user,
                    query=query,
                    limit=5,
                )
                if not results:
                    return "No relevant facts found."
                return "\\n".join(f"- {r.value['fact']}" for r in results)

            return [
                remember_fact,
                recall_facts,
                create_memory_tool(self.conv_memory),
            ]

        def _get_system_prompt(self):
            return '''You are an assistant with comprehensive memory:

            1. REMEMBER FACTS: Use remember_fact() to store important
               user information (preferences, projects, goals).

            2. RECALL FACTS: Use recall_facts() when you need to
               remember something about the user.

            3. SEARCH HISTORY: Use recall_conversation() when the
               user asks about past conversations or decisions.

            Be proactive about remembering relevant information.'''

        async def chat(
            self,
            user_id: str,
            session_id: str,
            message: str,
            history: list,
        ) -> str:
            self._current_user = user_id
            return await self.agent.arun(message, messages=history)

        async def end_session(
            self,
            user_id: str,
            session_id: str,
            messages: list,
        ):
            # Index conversation for future recall
            await self.conv_memory.aindex_conversation(
                user_id=user_id,
                session_id=session_id,
                messages=messages,
            )
""")


# =============================================================================
# Example 5: Memory-Enhanced Chat Loop
# =============================================================================


def memory_chat_loop():
    """Interactive chat with memory."""
    print("\n" + "=" * 60)
    print("5. Memory-Enhanced Chat Loop")
    print("=" * 60)

    print("\n  Interactive chat with persistence:")
    print("""
    import asyncio

    async def chat_with_memory():
        agent = FullyMemorizedAgent()
        user_id = "user_123"
        session_id = f"session_{int(time.time())}"
        history = []

        print("Chat with memory! Type 'quit' to exit.")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break

            history.append({"role": "user", "content": user_input})

            response = await agent.chat(
                user_id=user_id,
                session_id=session_id,
                message=user_input,
                history=history,
            )

            print(f"AI: {response}")
            history.append({"role": "assistant", "content": response})

        # Save conversation for future recall
        await agent.end_session(user_id, session_id, history)
        print("Conversation saved to memory!")

    asyncio.run(chat_with_memory())
""")


# =============================================================================
# Example 6: Memory with User Context
# =============================================================================


def memory_with_context():
    """Pass user context to memory operations."""
    print("\n" + "=" * 60)
    print("6. Memory with User Context")
    print("=" * 60)

    print("\n  Thread-safe user context pattern:")
    print("""
    import contextvars
    from ai_infra import Agent
    from ai_infra.memory import MemoryStore
    from langchain_core.tools import tool

    # Thread-safe user context
    current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar('user_id')

    store = MemoryStore.sqlite("./memories.db", embedding_provider="openai")

    @tool
    def remember(fact: str) -> str:
        '''Remember a fact about the current user.'''
        user_id = current_user_id.get()
        store.put_user_memory(user_id, f"fact_{time.time()}", {"fact": fact})
        return f"Remembered: {fact}"

    @tool
    def recall(query: str) -> str:
        '''Recall facts about the current user.'''
        user_id = current_user_id.get()
        results = store.search_user_memories(user_id, query, limit=5)
        return "\\n".join(f"- {r.value['fact']}" for r in results)

    agent = Agent(tools=[remember, recall])

    # Usage with context
    async def handle_request(user_id: str, message: str):
        token = current_user_id.set(user_id)
        try:
            return await agent.arun(message)
        finally:
            current_user_id.reset(token)
""")


# =============================================================================
# Example 7: Selective Memory
# =============================================================================


def selective_memory():
    """Let the agent decide what to remember."""
    print("\n" + "=" * 60)
    print("7. Selective Memory")
    print("=" * 60)

    print("\n  Agent decides what's worth remembering:")
    print("""
    SYSTEM_PROMPT = '''You are an assistant with memory capabilities.

    WHAT TO REMEMBER (use remember_fact):
    - User preferences (colors, languages, tools)
    - Personal facts (job, location, interests)
    - Project details (names, tech stacks, goals)
    - Decisions and conclusions reached
    - Explicit requests to remember something

    WHAT NOT TO REMEMBER:
    - Trivial greetings or small talk
    - Temporary context (what we're working on right now)
    - Sensitive information (passwords, keys, secrets)
    - Information the user wants forgotten

    Always ask before storing anything sensitive.'''

    agent = Agent(
        tools=[remember, recall],
        system_prompt=SYSTEM_PROMPT,
    )
""")


# =============================================================================
# Example 8: Memory Cleanup
# =============================================================================


def memory_cleanup():
    """Manage memory lifecycle and cleanup."""
    print("\n" + "=" * 60)
    print("8. Memory Cleanup")
    print("=" * 60)

    print("\n  Clean up old or unwanted memories:")
    print("""
    # Delete specific memory
    store.delete(("user_123", "memories"), "old_fact_key")

    # Delete all user memories (GDPR compliance)
    items = store.list(("user_123", "memories"))
    for item in items:
        store.delete(("user_123", "memories"), item.key)

    # Delete old conversations
    conv_memory.delete_user_conversations(user_id="user_123")

    # Use TTL for temporary memories
    store.put(
        namespace=("user_123", "session"),
        key="temp_context",
        value={"data": "..."},
        ttl=3600,  # Auto-delete after 1 hour
    )
""")


# =============================================================================
# Example 9: Memory Hydration
# =============================================================================


def memory_hydration():
    """Pre-load relevant memories into context."""
    print("\n" + "=" * 60)
    print("9. Memory Hydration")
    print("=" * 60)

    print("\n  Pre-load memories into system prompt:")
    print("""
    async def get_personalized_prompt(user_id: str, query: str) -> str:
        # Search for relevant user facts
        facts = store.search_user_memories(
            user_id=user_id,
            query=query,
            limit=5,
        )

        # Search relevant past conversations
        convos = await conv_memory.asearch(
            user_id=user_id,
            query=query,
            limit=3,
        )

        # Build context-aware system prompt
        context_parts = ["You are a helpful assistant."]

        if facts:
            fact_text = "\\n".join(f"- {f.value['fact']}" for f in facts)
            context_parts.append(f"\\nUser facts:\\n{fact_text}")

        if convos:
            convo_text = "\\n".join(f"- {c.context[:100]}..." for c in convos)
            context_parts.append(f"\\nRelevant past discussions:\\n{convo_text}")

        return "\\n".join(context_parts)

    # Use in chat
    system_prompt = await get_personalized_prompt(user_id, message)
    response = await llm.achat(
        [SystemMessage(content=system_prompt), HumanMessage(content=message)]
    )
""")


# =============================================================================
# Example 10: Production Architecture
# =============================================================================


def production_architecture():
    """Production-ready memory architecture."""
    print("\n" + "=" * 60)
    print("10. Production Architecture")
    print("=" * 60)

    print("\n  Complete production setup:")
    print("""
    from ai_infra import Agent, LLM
    from ai_infra.memory import MemoryStore, SummarizationMiddleware
    from ai_infra.llm.tools.custom import ConversationMemory, create_memory_tool

    class ProductionAgent:
        def __init__(self, database_url: str):
            # PostgreSQL for production scalability
            self.fact_store = MemoryStore.postgres(
                database_url,
                embedding_provider="openai",
            )

            self.conv_memory = ConversationMemory(
                backend="postgres",
                connection_string=database_url,
                embedding_provider="openai",
            )

            # Redis for session state (optional)
            # self.session_cache = Redis(...)

            self.agent = self._create_agent()

        def _create_agent(self):
            tools = [
                self._create_remember_tool(),
                self._create_recall_tool(),
                create_memory_tool(self.conv_memory),
            ]

            return Agent(
                tools=tools,
                llm=LLM(model="gpt-4o"),
                middleware=[
                    SummarizationMiddleware(trigger_tokens=6000),
                ],
            )

        # ... tool implementations ...

    # Deployment
    agent = ProductionAgent(os.environ["DATABASE_URL"])
""")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Agent with Full Memory Stack")
    print("=" * 60)
    print("\nBuild agents that remember across sessions.\n")

    agent_with_context()
    agent_with_memory_store()
    agent_with_conversation_recall()
    full_memory_stack()
    memory_chat_loop()
    memory_with_context()
    selective_memory()
    memory_cleanup()
    memory_hydration()
    production_architecture()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nMemory stack components:")
    print("  1. SummarizationMiddleware: Context window management")
    print("  2. MemoryStore: Long-term user facts")
    print("  3. ConversationMemory: Past conversation recall")
    print("  4. Tools: Agent-accessible memory operations")


if __name__ == "__main__":
    main()
