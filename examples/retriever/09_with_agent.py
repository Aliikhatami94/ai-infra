#!/usr/bin/env python
"""Retriever with Agent Integration Example.

This example demonstrates:
- Creating a retriever tool for agents
- Using create_retriever_tool()
- Agent with RAG capabilities
- Dynamic knowledge base queries
- Combining retrieval with generation

This is the core pattern for building RAG applications
where an AI agent can search your knowledge base.
"""

import asyncio
import os

from ai_infra import Agent, Retriever, create_retriever_tool

# =============================================================================
# Example 1: Basic Retriever Tool
# =============================================================================


def basic_retriever_tool():
    """Create a simple retriever tool for agents."""
    print("=" * 60)
    print("1. Basic Retriever Tool")
    print("=" * 60)

    # Build knowledge base
    retriever = Retriever()
    retriever.add(
        [
            "Widget Pro costs $99 and includes a 2-year warranty",
            "Widget Pro is available in blue, red, and black colors",
            "Free shipping on all orders over $50",
            "Returns accepted within 30 days with receipt",
            "Customer support available 24/7 at support@widget.com",
        ]
    )

    # Create a tool from the retriever
    search_tool = create_retriever_tool(
        retriever=retriever,
        name="search_knowledge_base",
        description="Search the product knowledge base for information about Widget Pro",
    )

    print("\n  [OK] Created retriever tool")
    print(f"    Name: {search_tool.name}")
    print(f"    Description: {search_tool.description[:50]}...")

    # Test the tool
    result = search_tool("What colors does Widget Pro come in?")
    print("\n  Test query: 'What colors does Widget Pro come in?'")
    print(f"  Result: {result[:100]}...")


# =============================================================================
# Example 2: Agent with Retriever
# =============================================================================


async def agent_with_retriever():
    """Create an agent that can search a knowledge base."""
    print("\n" + "=" * 60)
    print("2. Agent with Retriever")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n  [!] Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this example")
        return

    # Build knowledge base
    retriever = Retriever()
    retriever.add(
        [
            "The Acme Widget Pro is our flagship product, priced at $99",
            "Widget Pro features include: waterproof design, 20-hour battery life",
            "Widget Pro comes with a 2-year manufacturer warranty",
            "Warranty covers defects but not physical damage or water damage",
            "Extended warranty available for $19.99 for an additional year",
        ]
    )

    # Create search tool
    search_tool = create_retriever_tool(
        retriever=retriever,
        name="search_products",
        description="Search product information and policies",
    )

    # Create agent with the tool
    agent = Agent(
        model="gpt-4o-mini",
        tools=[search_tool],
        system_prompt="You are a helpful customer service agent. Use the search_products tool to find information before answering.",
    )

    print("\n  Agent created with search_products tool")

    # Ask a question
    question = "What's covered under the Widget Pro warranty?"
    print(f"\n  Customer: {question}")

    response = await agent.run(question)
    print(f"\n  Agent: {response[:200]}...")


# =============================================================================
# Example 3: Multiple Knowledge Bases
# =============================================================================


def multiple_knowledge_bases():
    """Use multiple specialized knowledge bases."""
    print("\n" + "=" * 60)
    print("3. Multiple Knowledge Bases")
    print("=" * 60)

    # Products knowledge base
    products_retriever = Retriever()
    products_retriever.add(
        [
            "Widget Pro - $99 - Premium widget with all features",
            "Widget Basic - $49 - Entry-level widget for beginners",
            "Widget Max - $199 - Professional-grade widget",
        ]
    )

    # Support knowledge base
    support_retriever = Retriever()
    support_retriever.add(
        [
            "Password reset: Go to Settings > Security > Reset Password",
            "Contact support at support@widget.com or 1-800-WIDGET",
            "Business hours: Monday-Friday 9am-5pm EST",
        ]
    )

    # Create tools for each
    product_tool = create_retriever_tool(
        retriever=products_retriever,
        name="search_products",
        description="Search product catalog for pricing and features",
    )

    support_tool = create_retriever_tool(
        retriever=support_retriever,
        name="search_support",
        description="Search support articles and contact information",
    )

    print("\n  Created two specialized tools:")
    print(f"    1. {product_tool.name}")
    print(f"    2. {support_tool.name}")

    # Test each
    print("\n  Product query: 'cheapest widget'")
    print(f"    -> {product_tool('cheapest widget')[:80]}...")

    print("\n  Support query: 'reset password'")
    print(f"    -> {support_tool('reset password')[:80]}...")


# =============================================================================
# Example 4: Retriever Tool with Metadata
# =============================================================================


def retriever_with_metadata():
    """Create retriever tool that preserves metadata."""
    print("\n" + "=" * 60)
    print("4. Retriever Tool with Metadata")
    print("=" * 60)

    retriever = Retriever()

    # Add documents with source tracking
    docs = [
        ("Return policy: 30 days with receipt", "returns.md"),
        ("Shipping: Free over $50, otherwise $5.99", "shipping.md"),
        ("Gift cards never expire", "gift-cards.md"),
    ]

    for text, source in docs:
        retriever.add_text(text, source=source)

    # Tool with detailed results
    search_tool = create_retriever_tool(
        retriever=retriever,
        name="search_policies",
        description="Search company policies",
        include_sources=True,  # Include source in results
    )

    print("\n  Search with source tracking:")
    result = search_tool("return policy")
    print(f"  Result: {result}")


# =============================================================================
# Example 5: Dynamic Knowledge Base
# =============================================================================


def dynamic_knowledge_base():
    """Update knowledge base while agent is running."""
    print("\n" + "=" * 60)
    print("5. Dynamic Knowledge Base")
    print("=" * 60)

    retriever = Retriever()

    # Initial knowledge
    retriever.add(["Widget Pro is priced at $99"])

    search_tool = create_retriever_tool(
        retriever=retriever,
        name="search_info",
        description="Search current information",
    )

    print("\n  Initial knowledge: Widget Pro is $99")
    print(f"  Query 'price': {search_tool('Widget Pro price')[:50]}...")

    # Add new information
    retriever.add(["SALE: Widget Pro now $79 (20% off)"])

    print("\n  Added: Sale pricing information")
    print(f"  Query 'price': {search_tool('Widget Pro price')[:60]}...")

    print("\n  [OK] Knowledge base updated without recreating agent!")


# =============================================================================
# Example 6: Retriever Tool Configuration
# =============================================================================


def tool_configuration():
    """Configure retriever tool behavior."""
    print("\n" + "=" * 60)
    print("6. Retriever Tool Configuration")
    print("=" * 60)

    retriever = Retriever()
    retriever.add(
        [
            "Feature 1: Automatic syncing",
            "Feature 2: Cloud backup",
            "Feature 3: Multi-device support",
            "Feature 4: Offline mode",
            "Feature 5: Custom themes",
        ]
    )

    # Different configurations
    configs = {
        "Default (k=3)": {"k": 3},
        "More results (k=5)": {"k": 5},
        "Fewer results (k=1)": {"k": 1},
    }

    print("\n  Testing different configurations:")

    for name, config in configs.items():
        tool = create_retriever_tool(
            retriever=retriever,
            name="search",
            description="Search features",
            **config,
        )
        result = tool("features")
        result_count = result.count("Feature")
        print(f"\n    {name}: {result_count} features returned")


# =============================================================================
# Example 7: Full RAG Pipeline
# =============================================================================


async def full_rag_pipeline():
    """Complete RAG pipeline with agent."""
    print("\n" + "=" * 60)
    print("7. Full RAG Pipeline")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n  [!] Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this example")
        print("\n  RAG Pipeline components:")
        print("    1. Retriever with knowledge base")
        print("    2. create_retriever_tool() wraps retriever")
        print("    3. Agent uses tool for search")
        print("    4. Agent generates response with context")
        return

    # 1. Build knowledge base
    retriever = Retriever()
    retriever.add(
        [
            "Company founded in 2020 by Jane Smith",
            "Headquarters in San Francisco, CA",
            "Currently 150 employees worldwide",
            "Main products: Widget Pro, Widget Basic, Widget Enterprise",
            "Raised $50M Series B in 2023",
        ]
    )

    # 2. Create retrieval tool
    search_tool = create_retriever_tool(
        retriever=retriever,
        name="company_search",
        description="Search company information and facts",
        k=3,
    )

    # 3. Create agent
    agent = Agent(
        tools=[search_tool],
        system_prompt="""You are a company information assistant.
Always use the company_search tool to find accurate information before answering.
Be concise and cite the information you find.""",
    )

    # 4. Query and generate
    questions = [
        "Who founded the company and when?",
        "How many employees does the company have?",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        response = await agent.run(q)
        print(f"  A: {response[:150]}...")


# =============================================================================
# Example 8: Error Handling
# =============================================================================


def error_handling():
    """Handle edge cases gracefully."""
    print("\n" + "=" * 60)
    print("8. Error Handling")
    print("=" * 60)

    retriever = Retriever()

    # Empty knowledge base
    search_tool = create_retriever_tool(
        retriever=retriever,
        name="search",
        description="Search knowledge",
    )

    print("\n  Query empty knowledge base:")
    result = search_tool("anything")
    print(f"    Result: {result if result else '(no results)'}")

    # Add some content
    retriever.add(["Now we have content"])

    print("\n  Query after adding content:")
    result = search_tool("content")
    print(f"    Result: {result[:50] if result else '(no results)'}...")

    print("\n  [OK] Handles empty results gracefully")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Agent + Retriever Integration")
    print("=" * 60)
    print("\ncreate_retriever_tool() connects retrieval to agents.\n")

    basic_retriever_tool()
    await agent_with_retriever()
    multiple_knowledge_bases()
    retriever_with_metadata()
    dynamic_knowledge_base()
    tool_configuration()
    await full_rag_pipeline()
    error_handling()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. create_retriever_tool() wraps retriever as agent tool")
    print("  2. Configure with name, description, and k")
    print("  3. Multiple retrievers = specialized search")
    print("  4. Knowledge base can be updated dynamically")
    print("  5. Full RAG: Retrieve -> Augment -> Generate")


if __name__ == "__main__":
    asyncio.run(main())
