#!/usr/bin/env python
"""Subagent Delegation Example.

This example demonstrates:
- Creating subagents for task delegation
- Agent instances as subagents (auto-conversion)
- SubAgent dicts for fine-grained control
- Multi-agent orchestration patterns

Subagents allow a main agent to delegate specialized tasks
to other agents, enabling complex workflows.

Required:
- pip install deepagents (for subagent functionality)
- OPENAI_API_KEY or ANTHROPIC_API_KEY
"""

import asyncio

from ai_infra import Agent
from ai_infra.llm.workspace import Workspace

# =============================================================================
# Sample Tools for Subagents
# =============================================================================


def web_search(query: str) -> str:
    """Search the web for information.

    Args:
        query: Search query string.

    Returns:
        Search results.
    """
    # Simulated search results
    results = {
        "python async": "Python async/await allows concurrent code execution...",
        "machine learning": "Machine learning is a subset of AI that...",
        "web development": "Web development involves building websites...",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"Found results for: {query}"


def write_content(title: str, content: str) -> str:
    """Write content with a title.

    Args:
        title: Content title.
        content: Main content body.

    Returns:
        Confirmation of written content.
    """
    return f"Written: '{title}'\n\n{content}"


def proofread(text: str) -> str:
    """Proofread and improve text.

    Args:
        text: Text to proofread.

    Returns:
        Improved text with suggestions.
    """
    # Simulated proofreading
    return f"Proofread version:\n{text}\n\n[No errors found, improved clarity]"


def code_review(code: str) -> str:
    """Review code for issues.

    Args:
        code: Code to review.

    Returns:
        Review comments.
    """
    return "Code review:\n- Style: ✓ Good\n- Logic: ✓ Sound\n- Security: ✓ No issues"


def run_tests(code: str) -> str:
    """Run tests on code.

    Args:
        code: Code to test.

    Returns:
        Test results.
    """
    return "All tests passed! (5/5)"


# =============================================================================
# Basic Subagent Setup
# =============================================================================


async def basic_subagent_setup():
    """Basic setup with Agent instances as subagents."""
    print("=" * 60)
    print("Basic Subagent Setup")
    print("=" * 60)

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            print("Subagent delegation requires 'deepagents' package.")
            print("Install with: pip install deepagents")
            return
    except ImportError:
        print("Subagent delegation requires 'deepagents' package.")
        return

    # Create specialized agents
    researcher = Agent(
        name="researcher",  # Required for subagent use
        description="Researches topics and gathers information",
        tools=[web_search],
        system="You are a research specialist. Find accurate information.",
    )

    writer = Agent(
        name="writer",
        description="Writes clear and engaging content",
        tools=[write_content, proofread],
        system="You are a content writer. Create clear, engaging content.",
    )

    # Create main agent that can delegate to subagents
    main_agent = Agent(
        deep=True,
        subagents=[researcher, writer],  # Agent instances auto-convert
        workspace=Workspace(".", mode="sandboxed"),
        system="You are a project manager. Delegate research to the researcher "
        "and writing to the writer.",
    )

    print("Main agent created with subagents:")
    print("  - researcher: Research and information gathering")
    print("  - writer: Content creation and proofreading")
    print()

    # The main agent can now delegate tasks
    result = await main_agent.arun(
        "Research Python async programming and write a brief summary about it."
    )
    print(f"Result: {result}")


# =============================================================================
# SubAgent Dicts for Fine-Grained Control
# =============================================================================


async def subagent_dicts():
    """Use SubAgent dicts for more control."""
    print("\n" + "=" * 60)
    print("SubAgent Dicts for Fine-Grained Control")
    print("=" * 60)

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            print("Skipping - deepagents not installed")
            return
    except ImportError:
        print("Skipping - deepagents not installed")
        return

    # Define subagents as dicts for more control
    subagent_configs = [
        {
            "name": "code_reviewer",
            "description": "Reviews code for quality, security, and best practices",
            "system_prompt": "You are an expert code reviewer. Check for bugs, "
            "security issues, and suggest improvements.",
            "tools": [code_review],
            "model": "gpt-4o-mini",  # Can specify different model
        },
        {
            "name": "test_runner",
            "description": "Runs tests and reports results",
            "system_prompt": "You are a QA engineer. Run tests and report results.",
            "tools": [run_tests],
            "model": "gpt-4o-mini",
        },
    ]

    main_agent = Agent(
        deep=True,
        subagents=subagent_configs,
        workspace=Workspace(".", mode="sandboxed"),
        system="You are a tech lead. Use the code_reviewer and test_runner to ensure code quality.",
    )

    print("Subagent dicts allow:")
    print("  - Custom model per subagent")
    print("  - Custom system prompts")
    print("  - Fine-grained tool assignment")
    print()

    result = await main_agent.arun("Review this code: `def add(a, b): return a + b`")
    print(f"Result: {result}")


# =============================================================================
# Mixed Subagent Types
# =============================================================================


async def mixed_subagent_types():
    """Combine Agent instances and SubAgent dicts."""
    print("\n" + "=" * 60)
    print("Mixed Subagent Types")
    print("=" * 60)

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            print("Skipping - deepagents not installed")
            return
    except ImportError:
        print("Skipping - deepagents not installed")
        return

    # Agent instance
    researcher = Agent(
        name="researcher",
        description="Researches topics using web search",
        tools=[web_search],
        system="Find accurate, up-to-date information.",
    )

    # SubAgent dict
    writer_config = {
        "name": "technical_writer",
        "description": "Writes technical documentation",
        "system_prompt": "Write clear, accurate technical documentation.",
        "tools": [write_content],
    }

    # Both can be mixed in subagents list
    _main_agent = Agent(
        deep=True,
        subagents=[researcher, writer_config],
        workspace=Workspace(".", mode="sandboxed"),
        system="Coordinate research and documentation tasks.",
    )

    print("You can mix Agent instances and SubAgent dicts:")
    print("  - Agent instances: Easier to create, auto-converted")
    print("  - SubAgent dicts: More explicit control")
    print()


# =============================================================================
# Specialized Agent Team
# =============================================================================


async def specialized_agent_team():
    """Create a team of specialized agents."""
    print("\n" + "=" * 60)
    print("Specialized Agent Team")
    print("=" * 60)

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            print("Skipping - deepagents not installed")
            return
    except ImportError:
        print("Skipping - deepagents not installed")
        return

    # Frontend specialist
    frontend_dev = Agent(
        name="frontend_dev",
        description="Expert in React, TypeScript, and CSS. Builds user interfaces.",
        system="You are a frontend developer. Create beautiful, responsive UIs.",
    )

    # Backend specialist
    backend_dev = Agent(
        name="backend_dev",
        description="Expert in Python, APIs, and databases. Builds server logic.",
        system="You are a backend developer. Build robust, scalable APIs.",
    )

    # DevOps specialist
    devops = Agent(
        name="devops",
        description="Expert in Docker, Kubernetes, and CI/CD. Handles deployment.",
        system="You are a DevOps engineer. Ensure smooth deployments.",
    )

    # Tech Lead orchestrates the team
    tech_lead = Agent(
        deep=True,
        subagents=[frontend_dev, backend_dev, devops],
        workspace=Workspace(".", mode="sandboxed"),
        system="""You are the tech lead. Your team includes:
        - frontend_dev: React/TypeScript specialist
        - backend_dev: Python/API specialist
        - devops: Docker/Kubernetes specialist

        Delegate tasks to the appropriate specialist.""",
    )

    print("Team structure:")
    print("  Tech Lead (orchestrator)")
    print("  ├── frontend_dev (React, TypeScript)")
    print("  ├── backend_dev (Python, APIs)")
    print("  └── devops (Docker, K8s)")
    print()

    result = await tech_lead.arun(
        "Design a simple user authentication feature. "
        "We need a login form, an API endpoint, and a Docker setup."
    )
    print(f"Result: {result}")


# =============================================================================
# Requirements for Agent as Subagent
# =============================================================================


def subagent_requirements():
    """Show requirements for using Agent as subagent."""
    print("\n" + "=" * 60)
    print("Agent as Subagent Requirements")
    print("=" * 60)

    print("\nWhen using an Agent instance as a subagent, you MUST set:")
    print("  1. name: Unique identifier for the subagent")
    print("  2. description: What the subagent does (for task routing)")
    print()

    # Correct usage
    print("✓ Correct:")
    print("  researcher = Agent(")
    print("      name='researcher',  # Required!")
    print("      description='Researches topics',  # Required!")
    print("      tools=[web_search],")
    print("  )")
    print()

    # Incorrect usage (would raise error)
    print("✗ Incorrect (will raise ValueError):")
    print("  bad_agent = Agent(")
    print("      tools=[web_search],")
    print("      # Missing name and description!")
    print("  )")
    print()

    # Try creating without name/description
    try:
        _unnamed = Agent(tools=[web_search])
        # This works - the agent itself is valid
        print("Agent without name is valid on its own...")

        # But using it as a subagent would fail
        # (We can't actually test this without deep mode)
        print("But would fail when passed to subagents parameter.")
    except ValueError as e:
        print(f"Error: {e}")


# =============================================================================
# Subagent Best Practices
# =============================================================================


def best_practices():
    """Best practices for subagent design."""
    print("\n" + "=" * 60)
    print("Subagent Best Practices")
    print("=" * 60)

    print("\n1. Single Responsibility")
    print("   Each subagent should do one thing well.")
    print("   ✓ Good: 'researcher', 'writer', 'reviewer'")
    print("   ✗ Bad: 'do_everything_agent'")

    print("\n2. Clear Descriptions")
    print("   Help the main agent know when to delegate.")
    print("   ✓ Good: 'Searches the web for academic papers'")
    print("   ✗ Bad: 'Helps with stuff'")

    print("\n3. Appropriate Tools")
    print("   Give each subagent only the tools it needs.")
    print("   ✓ Good: researcher gets search_web, writer gets write_file")
    print("   ✗ Bad: Every agent gets every tool")

    print("\n4. Focused System Prompts")
    print("   Keep subagent prompts focused on their specialty.")
    print("   ✓ Good: 'You are a code reviewer. Focus on security.'")
    print("   ✗ Bad: Generic prompts")

    print("\n5. Consider Model Selection")
    print("   Use cheaper models for simple tasks.")
    print("   ✓ Good: Code review → GPT-4, Search → GPT-3.5")
    print("   ✗ Bad: GPT-4 for everything")


if __name__ == "__main__":
    # Check if deepagents is available
    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        print(f"deepagents installed: {HAS_DEEPAGENTS}")
    except ImportError:
        print("deepagents not installed")
        HAS_DEEPAGENTS = False

    print()

    # Run examples
    asyncio.run(basic_subagent_setup())
    asyncio.run(subagent_dicts())
    asyncio.run(mixed_subagent_types())
    asyncio.run(specialized_agent_team())
    subagent_requirements()
    best_practices()
