#!/usr/bin/env python
"""Conversation Summarization Example.

This example demonstrates:
- One-shot summarization of old messages
- Rolling summaries for stateless APIs
- fit_context() for automatic handling
- SummarizationMiddleware for agents
- Custom summarization prompts

Use summarization to preserve context when trimming
would lose important information.
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_infra.llm.memory import (
    SummarizationMiddleware,
    count_tokens_approximate,
)

# =============================================================================
# Example 1: Basic Summarization
# =============================================================================


def basic_summarization():
    """Summarize old messages, keep recent ones."""
    print("=" * 60)
    print("1. Basic Summarization")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a Python coding assistant."),
        HumanMessage(content="How do I read a file in Python?"),
        AIMessage(content="Use open() with a context manager: with open('file.txt') as f: ..."),
        HumanMessage(content="What about writing to files?"),
        AIMessage(content="Same pattern, but use 'w' mode: with open('file.txt', 'w') as f: ..."),
        HumanMessage(content="Can I append instead?"),
        AIMessage(content="Yes, use 'a' mode for append: with open('file.txt', 'a') as f: ..."),
        HumanMessage(content="How do I read line by line?"),
        AIMessage(content="Use a for loop: for line in f: print(line.strip())"),
        HumanMessage(content="What about binary files?"),
        AIMessage(content="Use 'rb' or 'wb' mode for binary files."),
        # Recent messages we want to keep
        HumanMessage(content="Can you show me a complete example?"),
    ]

    print(f"\n  Original: {len(messages)} messages")
    print(f"  Tokens: ~{count_tokens_approximate(messages)}")

    # Summarize, keeping last 3 messages
    # Note: This requires an API key for actual summarization
    print("\n  Summarization pattern:")
    print("""
    from ai_infra.memory import summarize_messages

    result = summarize_messages(
        messages,
        keep_last=3,  # Keep 3 most recent
        llm=LLM(),    # LLM for summarization
    )

    print(f"Summary: {result.summary}")
    print(f"Messages: {len(result.messages)}")  # Summary + 3 kept
""")


# =============================================================================
# Example 2: fit_context() - The Primary API
# =============================================================================


def fit_context_example():
    """Use fit_context() for automatic context management."""
    print("\n" + "=" * 60)
    print("2. fit_context() - The Primary API")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Tell me about Python."),
        AIMessage(content="Python is a versatile programming language."),
        HumanMessage(content="What can I build with it?"),
        AIMessage(content="Web apps, AI/ML, automation, games, and more."),
        HumanMessage(content="Is it fast?"),
        AIMessage(content="It's slower than C/Rust but fast enough for most uses."),
        HumanMessage(content="What about typing?"),
    ]

    print(f"\n  Original: {len(messages)} messages")

    # Without summarization (just trims)
    print("\n  Pattern 1: Trim only (no API call)")
    print("""
    result = fit_context(messages, max_tokens=200)

    if result.action == "trimmed":
        print(f"Trimmed from {result.original_count} to {result.final_count}")
    elif result.action == "none":
        print("Already under limit")
""")

    # With summarization
    print("\n  Pattern 2: Summarize if needed")
    print("""
    result = fit_context(
        messages,
        max_tokens=200,
        summarize=True,  # Use LLM to summarize
        keep=3,          # Always keep last 3
        llm=LLM(),
    )

    if result.action == "summarized":
        print(f"Created summary: {result.summary[:50]}...")
""")

    # Rolling summary (stateless APIs)
    print("\n  Pattern 3: Rolling summary (stateless)")
    print("""
    # First request
    result = fit_context(messages, max_tokens=500, summarize=True)
    response = {"answer": ..., "summary": result.summary}

    # Next request - extend existing summary
    result = fit_context(
        new_messages,
        max_tokens=500,
        summarize=True,
        summary=request.summary,  # Pass previous summary
    )
""")


# =============================================================================
# Example 3: Rolling Summaries for REST APIs
# =============================================================================


def rolling_summaries():
    """Maintain rolling summaries in stateless APIs."""
    print("\n" + "=" * 60)
    print("3. Rolling Summaries for REST APIs")
    print("=" * 60)

    print("\n  Pattern for stateless API endpoints:")
    print("""
    from fastapi import FastAPI
    from pydantic import BaseModel
    from ai_infra import LLM
    from ai_infra.memory import fit_context

    app = FastAPI()

    class ChatRequest(BaseModel):
        message: str
        history: list[dict]
        summary: str | None = None  # Client sends back previous summary

    class ChatResponse(BaseModel):
        response: str
        summary: str  # Client stores and sends back next time

    @app.post("/chat")
    async def chat(request: ChatRequest) -> ChatResponse:
        llm = LLM()

        # Add new message
        messages = request.history + [
            {"role": "user", "content": request.message}
        ]

        # Fit context with rolling summary
        result = await afit_context(
            messages,
            max_tokens=3000,
            summarize=True,
            summary=request.summary,  # Extend previous summary
            keep=5,
        )

        # Get response
        response = await llm.achat(result.messages)

        return ChatResponse(
            response=response,
            summary=result.summary,  # Client stores this
        )
""")


# =============================================================================
# Example 4: SummarizationMiddleware for Agents
# =============================================================================


def summarization_middleware():
    """Auto-summarize during agent execution."""
    print("\n" + "=" * 60)
    print("4. SummarizationMiddleware for Agents")
    print("=" * 60)

    print("\n  Attach middleware to Agent for automatic summarization:")
    print("""
    from ai_infra import Agent
    from ai_infra.memory import SummarizationMiddleware

    agent = Agent(
        tools=[...],
        middleware=[
            SummarizationMiddleware(
                trigger_tokens=4000,  # Summarize when over 4000 tokens
                keep_messages=10,     # Always keep last 10 messages
            )
        ]
    )

    # Long conversation - middleware auto-summarizes
    for _ in range(100):
        response = agent.run(user_input)
        # Context is automatically managed
""")

    print("\n  Configuration options:")
    middleware = SummarizationMiddleware(
        trigger_tokens=4000,
        trigger_messages=50,  # Or trigger by message count
        keep_messages=10,
    )
    print(f"    trigger_tokens: {middleware.trigger_tokens}")
    print(f"    trigger_messages: {middleware.trigger_messages}")
    print(f"    keep_messages: {middleware.keep_messages}")


# =============================================================================
# Example 5: Custom Summarization Prompt
# =============================================================================


def custom_prompt():
    """Use a custom prompt for summarization."""
    print("\n" + "=" * 60)
    print("5. Custom Summarization Prompt")
    print("=" * 60)

    print("\n  Customize how summaries are generated:")
    print("""
    from ai_infra.memory import summarize_messages

    # Custom prompt for technical conversations
    TECH_PROMPT = '''Summarize this technical conversation:
    - Focus on: code decisions, bugs found, solutions implemented
    - Note: libraries, versions, and configurations discussed
    - Keep: any command examples or code snippets mentioned

    Conversation:
    {conversation}

    Technical Summary:'''

    result = summarize_messages(
        messages,
        summarize_prompt=TECH_PROMPT,
        keep_last=5,
    )
""")

    print("\n  Domain-specific prompts:")
    prompts = {
        "Support": "Summarize customer issue, steps taken, resolution status",
        "Coding": "Track decisions, bugs, solutions, code patterns",
        "Research": "Key findings, sources cited, questions remaining",
        "Planning": "Goals, decisions made, action items, blockers",
    }

    for domain, focus in prompts.items():
        print(f"    {domain}: {focus}")


# =============================================================================
# Example 6: Preserve System Message
# =============================================================================


def preserve_system():
    """System message handling during summarization."""
    print("\n" + "=" * 60)
    print("6. System Message Handling")
    print("=" * 60)

    print("\n  By default, system message is preserved:")
    print("""
    messages = [
        SystemMessage(content="You are a Python expert."),
        HumanMessage(content="What is a decorator?"),
        AIMessage(content="A function that modifies another function..."),
        # ... more messages
    ]

    result = summarize_messages(
        messages,
        keep_last=3,
        include_system=True,  # Default: preserve system message
    )

    # Result structure:
    # [SystemMessage(original), SystemMessage(summary), ...3 kept messages]
""")

    print("\n  Without preserving system:")
    print("""
    result = summarize_messages(
        messages,
        keep_last=3,
        include_system=False,
    )

    # Result structure:
    # [SystemMessage(summary), ...3 kept messages]
""")


# =============================================================================
# Example 7: Async Summarization
# =============================================================================


async def async_summarization():
    """Async version for web applications."""
    print("\n" + "=" * 60)
    print("7. Async Summarization")
    print("=" * 60)

    print("\n  Use async in web apps:")
    print("""
    from ai_infra.memory import asummarize_messages, afit_context

    # Async summarization
    result = await asummarize_messages(
        messages,
        keep_last=5,
        llm=LLM(),
    )

    # Async fit_context
    result = await afit_context(
        messages,
        max_tokens=3000,
        summarize=True,
    )
""")


# =============================================================================
# Example 8: SummarizeResult Details
# =============================================================================


def result_details():
    """Inspect summarization results."""
    print("\n" + "=" * 60)
    print("8. SummarizeResult Details")
    print("=" * 60)

    print("\n  SummarizeResult fields:")
    print("""
    result = summarize_messages(messages, keep_last=5)

    result.messages       # Processed messages (summary + kept)
    result.summary        # The generated summary text
    result.original_count # Number of input messages
    result.summarized_count  # How many were summarized
    result.kept_count     # How many were kept unchanged
""")

    print("\n  ContextResult fields (from fit_context):")
    print("""
    result = fit_context(messages, max_tokens=3000, summarize=True)

    result.messages       # Processed messages
    result.summary        # Summary if generated
    result.tokens         # Token count of result
    result.action         # "none", "trimmed", or "summarized"
    result.original_count # Input message count
    result.final_count    # Output message count
""")


# =============================================================================
# Example 9: Comparison: Trim vs Summarize
# =============================================================================


def trim_vs_summarize():
    """When to trim vs when to summarize."""
    print("\n" + "=" * 60)
    print("9. Trim vs Summarize")
    print("=" * 60)

    print("\n  Use TRIM when:")
    print("    [OK] Speed is critical (no LLM call)")
    print("    [OK] Only recent context matters")
    print("    [OK] Conversation is short enough")
    print("    [OK] Cost is a concern")

    print("\n  Use SUMMARIZE when:")
    print("    [OK] Historical context is important")
    print("    [OK] Long-running conversations")
    print("    [OK] Need to preserve key decisions/facts")
    print("    [OK] Quality > speed")

    print("\n  Hybrid approach with fit_context:")
    print("""
    result = fit_context(
        messages,
        max_tokens=3000,
        summarize=True,  # Only summarizes if needed
    )

    if result.action == "none":
        # Under limit, nothing done
    elif result.action == "trimmed":
        # Was trimmed (summarize=False or under summarize threshold)
    elif result.action == "summarized":
        # Too long, used LLM to summarize
""")


# =============================================================================
# Example 10: Production Pattern
# =============================================================================


def production_pattern():
    """Complete production pattern for context management."""
    print("\n" + "=" * 60)
    print("10. Production Pattern")
    print("=" * 60)

    print("\n  Complete example with all features:")
    print("""
    from ai_infra import LLM
    from ai_infra.memory import fit_context, afit_context
    from langchain_core.messages import HumanMessage, AIMessage

    class ChatService:
        def __init__(self, max_tokens: int = 4000):
            self.llm = LLM()
            self.max_tokens = max_tokens
            self.summaries: dict[str, str] = {}  # session_id -> summary

        async def chat(
            self,
            session_id: str,
            message: str,
            history: list[dict],
        ) -> str:
            # Build messages
            messages = history + [{"role": "user", "content": message}]

            # Manage context with rolling summary
            result = await afit_context(
                messages,
                max_tokens=self.max_tokens,
                summarize=True,
                summary=self.summaries.get(session_id),
                keep=5,
                llm=self.llm,
            )

            # Store updated summary
            if result.summary:
                self.summaries[session_id] = result.summary

            # Get response
            response = await self.llm.achat(result.messages)

            return response

    # Usage
    service = ChatService()
    response = await service.chat(
        session_id="user_123_session_1",
        message="What did we discuss about decorators?",
        history=[...],
    )
""")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Conversation Summarization Examples")
    print("=" * 60)
    print("\nPreserve context by summarizing instead of trimming.\n")

    basic_summarization()
    fit_context_example()
    rolling_summaries()
    summarization_middleware()
    custom_prompt()
    preserve_system()
    asyncio.run(async_summarization())
    result_details()
    trim_vs_summarize()
    production_pattern()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. fit_context() is the primary API for context management")
    print("  2. summarize=True preserves information vs just trimming")
    print("  3. Rolling summaries work great for stateless APIs")
    print("  4. SummarizationMiddleware auto-manages agent context")
    print("  5. Custom prompts can focus on domain-specific information")


if __name__ == "__main__":
    main()
