#!/usr/bin/env python
"""Deep Mode Agent Example.

This example demonstrates:
- Deep mode for autonomous multi-step task execution
- Built-in file tools (read, write, edit, glob, grep)
- Todo management
- Workspace configuration (virtual, sandboxed, full)

Deep mode agents are designed for complex, autonomous tasks that
require file system operations and multi-step planning.

Required:
- pip install deepagents (for deep=True functionality)
- OPENAI_API_KEY or ANTHROPIC_API_KEY

Note: Some features require the deepagents package.
"""

import asyncio
import os
from pathlib import Path

from ai_infra import Agent
from ai_infra.llm.workspace import Workspace

# =============================================================================
# Basic Deep Mode Agent
# =============================================================================


async def basic_deep_agent():
    """Create a basic deep mode agent with file tools."""
    print("=" * 60)
    print("Basic Deep Mode Agent")
    print("=" * 60)

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            print("Deep mode requires 'deepagents' package.")
            print("Install with: pip install deepagents")
            return
    except ImportError:
        print("Deep mode requires 'deepagents' package.")
        return

    # Create a deep mode agent with sandboxed workspace
    agent = Agent(
        deep=True,
        workspace=Workspace(".", mode="sandboxed"),
        system="You are a helpful assistant that can work with files.",
    )

    print("Deep agent created with built-in file tools:")
    print("  - ls, read_file, write_file, edit_file")
    print("  - glob, grep, execute")
    print("  - Todo management")
    print()

    # Example task
    result = await agent.arun("List the files in the current directory.")
    print(f"Result: {result}")


# =============================================================================
# Workspace Modes
# =============================================================================


def workspace_modes_demo():
    """Demonstrate different workspace modes."""
    print("\n" + "=" * 60)
    print("Workspace Modes")
    print("=" * 60)

    # Mode 1: Virtual (in-memory, no persistence)
    print("\n1. Virtual Mode (in-memory, no persistence)")
    print("   - Files exist only in memory")
    print("   - Safe for untrusted inputs")
    print("   - No changes to real filesystem")
    virtual_workspace = Workspace(mode="virtual")
    print(f"   Config: {virtual_workspace}")

    # Mode 2: Sandboxed (confined to root directory)
    print("\n2. Sandboxed Mode (default, recommended)")
    print("   - Real filesystem access")
    print("   - Confined to workspace root")
    print("   - Cannot escape with ../")
    sandboxed_workspace = Workspace(".", mode="sandboxed")
    print(f"   Config: {sandboxed_workspace}")

    # Mode 3: Full (unrestricted access)
    print("\n3. Full Mode (unrestricted - use with caution!)")
    print("   - Full filesystem access from root")
    print("   - No path restrictions")
    print("   - Only for trusted automation")
    full_workspace = Workspace("/tmp/safe-area", mode="full")
    print(f"   Config: {full_workspace}")


# =============================================================================
# Deep Agent with Custom Tools
# =============================================================================


async def deep_agent_with_custom_tools():
    """Deep agent with additional custom tools."""
    print("\n" + "=" * 60)
    print("Deep Agent with Custom Tools")
    print("=" * 60)

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            print("Skipping - deepagents not installed")
            return
    except ImportError:
        print("Skipping - deepagents not installed")
        return

    # Custom tools that work alongside built-in file tools
    def analyze_code(file_path: str) -> str:
        """Analyze Python code and return a summary.

        Args:
            file_path: Path to Python file to analyze.

        Returns:
            Analysis summary.
        """
        # Placeholder - in real use, would analyze AST
        return f"Analyzed {file_path}: Valid Python file"

    def run_tests(directory: str = ".") -> str:
        """Run tests in a directory.

        Args:
            directory: Directory containing tests.

        Returns:
            Test results summary.
        """
        # Placeholder - would run pytest in real use
        return f"All tests in {directory} passed!"

    def create_git_commit(message: str) -> str:
        """Create a git commit with the given message.

        Args:
            message: Commit message.

        Returns:
            Commit result.
        """
        # Placeholder - would run git in real use
        return f"Created commit: {message}"

    # Create deep agent with custom tools added to built-in tools
    agent = Agent(
        deep=True,
        tools=[analyze_code, run_tests, create_git_commit],  # Added to built-in tools
        workspace=Workspace(".", mode="sandboxed"),
        system="You are a code assistant that can analyze, test, and commit code.",
    )

    print("Deep agent has both:")
    print("  - Built-in file tools (read, write, edit, etc.)")
    print("  - Custom tools (analyze_code, run_tests, create_git_commit)")

    # Example complex task
    result = await agent.arun("List the Python files in the current directory and analyze them.")
    print(f"\nResult: {result}")


# =============================================================================
# Todo Management in Deep Mode
# =============================================================================


async def todo_management():
    """Demonstrate todo management in deep agents."""
    print("\n" + "=" * 60)
    print("Todo Management in Deep Mode")
    print("=" * 60)

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            print("Skipping - deepagents not installed")
            return
    except ImportError:
        print("Skipping - deepagents not installed")
        return

    agent = Agent(
        deep=True,
        workspace=Workspace(".", mode="sandboxed"),
        system="""You are a project manager assistant.
        Use the todo tools to track tasks.
        Break down complex tasks into smaller subtasks.""",
    )

    print("Deep agents include todo management tools:")
    print("  - Create, update, complete todos")
    print("  - Track progress on multi-step tasks")
    print()

    # Complex multi-step task
    result = await agent.arun(
        "Create a plan to refactor a Python module. "
        "Include steps for: analysis, testing, refactoring, and documentation."
    )
    print(f"Result: {result}")


# =============================================================================
# File Operations Example
# =============================================================================


async def file_operations():
    """Demonstrate built-in file operations."""
    print("\n" + "=" * 60)
    print("File Operations in Deep Mode")
    print("=" * 60)

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            print("Skipping - deepagents not installed")
            return
    except ImportError:
        print("Skipping - deepagents not installed")
        return

    # Create a test directory
    test_dir = Path("/tmp/deep_agent_test")
    test_dir.mkdir(exist_ok=True)

    # Create some test files
    (test_dir / "example.py").write_text("def hello():\n    return 'Hello, World!'\n")
    (test_dir / "config.json").write_text('{"name": "test", "version": "1.0"}')

    print(f"Test directory: {test_dir}")
    print("Created files: example.py, config.json\n")

    agent = Agent(
        deep=True,
        workspace=Workspace(str(test_dir), mode="sandboxed"),
        system="You are a file management assistant.",
    )

    # Read files
    result = await agent.arun("List all files and show the contents of example.py")
    print(f"Result:\n{result}")


# =============================================================================
# Recursion Limit for Safety
# =============================================================================


async def recursion_limit_demo():
    """Demonstrate recursion limit for safety."""
    print("\n" + "=" * 60)
    print("Recursion Limit for Safety")
    print("=" * 60)

    print("Deep agents have a recursion_limit to prevent infinite loops:")
    print("  - Default: 50 iterations")
    print("  - Prevents runaway token costs")
    print("  - Critical safety measure")
    print()

    # Low recursion limit for demo
    _agent = Agent(
        deep=True,
        recursion_limit=10,  # Lower limit for demo
        workspace=Workspace(".", mode="sandboxed"),
        system="Complete the task efficiently.",
    )

    print("Agent created with recursion_limit=10")
    print("This prevents the agent from looping indefinitely.")


# =============================================================================
# Deep Agent vs Regular Agent Comparison
# =============================================================================


def comparison():
    """Compare deep mode vs regular agent."""
    print("\n" + "=" * 60)
    print("Deep Mode vs Regular Agent Comparison")
    print("=" * 60)

    print("\nRegular Agent (deep=False, default):")
    print("  ✓ Custom tools only")
    print("  ✓ Simpler, more controlled")
    print("  ✓ Good for: API calls, calculations, simple tasks")
    print("  ✓ Uses: create_react_agent")
    print()

    regular = Agent(
        tools=[lambda x: x],  # Minimal tool
        system="Regular agent",
    )
    print(f"  Example: {regular}")

    print("\nDeep Agent (deep=True):")
    print("  ✓ Built-in file tools (read, write, edit, glob, grep)")
    print("  ✓ Todo management")
    print("  ✓ Subagent delegation")
    print("  ✓ Good for: Code generation, file processing, complex automation")
    print("  ✓ Uses: create_deep_agent from deepagents")
    print()

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if HAS_DEEPAGENTS:
            deep = Agent(
                deep=True,
                workspace=Workspace(".", mode="sandboxed"),
            )
            print(f"  Example: {deep}")
        else:
            print("  (Requires deepagents package)")
    except ImportError:
        print("  (Requires deepagents package)")


# =============================================================================
# Virtual Mode for Cloud/Untrusted
# =============================================================================


async def virtual_mode_demo():
    """Virtual mode for cloud and untrusted environments."""
    print("\n" + "=" * 60)
    print("Virtual Mode for Cloud/Untrusted Environments")
    print("=" * 60)

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            print("Skipping - deepagents not installed")
            return
    except ImportError:
        print("Skipping - deepagents not installed")
        return

    print("Virtual mode is ideal for:")
    print("  - Cloud environments (AWS Lambda, etc.)")
    print("  - Processing untrusted user prompts")
    print("  - Testing without side effects")
    print()

    agent = Agent(
        deep=True,
        workspace=Workspace(mode="virtual"),  # In-memory only
        system="You work with files in a virtual filesystem.",
    )

    # Files created here don't persist to disk
    result = await agent.arun(
        "Create a file called 'test.txt' with the content 'Hello, virtual world!', "
        "then read it back to confirm."
    )
    print(f"Result: {result}")
    print("\n(No files were actually created on disk)")


if __name__ == "__main__":
    # Check if deepagents is available
    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        print(f"deepagents installed: {HAS_DEEPAGENTS}")
    except ImportError:
        print("deepagents not installed")
        HAS_DEEPAGENTS = False

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\nWarning: No API key set. Examples may fail.")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
    print()

    # Run examples
    asyncio.run(basic_deep_agent())
    workspace_modes_demo()
    asyncio.run(deep_agent_with_custom_tools())
    asyncio.run(todo_management())
    asyncio.run(file_operations())
    asyncio.run(recursion_limit_demo())
    comparison()
    asyncio.run(virtual_mode_demo())
