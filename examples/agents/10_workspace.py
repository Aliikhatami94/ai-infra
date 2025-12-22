#!/usr/bin/env python
"""Workspace Configuration Example.

This example demonstrates:
- Workspace modes (virtual, sandboxed, full)
- Configuring file access for agents
- Security implications of each mode
- Integration with deep mode agents

Workspaces control how agents interact with the filesystem,
providing security boundaries for file operations.

Required:
- pip install deepagents (for deep mode with file tools)
- OPENAI_API_KEY or ANTHROPIC_API_KEY
"""

import asyncio
import os
import tempfile
from pathlib import Path

from ai_infra import Agent
from ai_infra.llm.workspace import Workspace, workspace

# =============================================================================
# Workspace Modes Overview
# =============================================================================


def workspace_modes_overview():
    """Overview of the three workspace modes."""
    print("=" * 60)
    print("Workspace Modes Overview")
    print("=" * 60)

    print("""
Workspace controls how agents access the filesystem:

┌──────────────────────────────────────────────────────────────┐
│ Mode        │ File Access    │ Persistence │ Use Case        │
├──────────────────────────────────────────────────────────────┤
│ virtual     │ In-memory only │ None        │ Untrusted input │
│ sandboxed   │ Confined dir   │ Real files  │ Development     │
│ full        │ Unrestricted   │ Real files  │ Trusted automation │
└──────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# Virtual Mode (In-Memory)
# =============================================================================


def virtual_mode():
    """Virtual mode - in-memory filesystem."""
    print("\n" + "=" * 60)
    print("Virtual Mode (In-Memory)")
    print("=" * 60)

    print("""
Virtual mode characteristics:
- Files exist only in memory
- No changes to real filesystem
- Safest for untrusted inputs
- Good for: Cloud functions, untrusted prompts, testing
""")

    # Create virtual workspace
    ws = Workspace(mode="virtual")
    print(f"Created: {ws}")

    # Or use convenience function
    ws = workspace(mode="virtual")
    print(f"Via function: {ws}")


async def virtual_mode_demo():
    """Demo virtual mode with deep agent."""
    print("\nVirtual Mode Demo:")

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            print("  Requires deepagents package")
            return
    except ImportError:
        print("  Requires deepagents package")
        return

    agent = Agent(
        deep=True,
        workspace=Workspace(mode="virtual"),
        system="You work with files in a virtual filesystem.",
    )

    print("  Agent created with virtual filesystem")
    print("  Files created by agent do NOT persist to disk")

    # Example task
    result = await agent.arun("Create a file called 'test.txt' with 'Hello World' content.")
    print(f"  Result: {result}")
    print("  (File exists in memory only, not on disk)")


# =============================================================================
# Sandboxed Mode (Confined to Directory)
# =============================================================================


def sandboxed_mode():
    """Sandboxed mode - confined to a directory."""
    print("\n" + "=" * 60)
    print("Sandboxed Mode (Confined Directory)")
    print("=" * 60)

    print("""
Sandboxed mode characteristics:
- Real filesystem access
- CONFINED to workspace root directory
- Cannot escape with ../ (path traversal blocked)
- Good for: Development, project-specific tasks
""")

    # Create sandboxed workspace for current directory
    ws = Workspace(".", mode="sandboxed")
    print(f"Created: {ws}")

    # Or with a specific directory
    ws = Workspace("/home/user/project", mode="sandboxed")
    print(f"With specific path: {ws}")

    # Path validation demo
    print("\nPath validation:")
    print("  /home/user/project/src/main.py → ✓ Allowed")
    print("  /home/user/project/../secrets.txt → ✗ Blocked (escapes sandbox)")
    print("  /etc/passwd → ✗ Blocked (outside sandbox)")


async def sandboxed_mode_demo():
    """Demo sandboxed mode with deep agent."""
    print("\nSandboxed Mode Demo:")

    try:
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            print("  Requires deepagents package")
            return
    except ImportError:
        print("  Requires deepagents package")
        return

    # Create a temp directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        test_file = Path(tmpdir) / "example.txt"
        test_file.write_text("This is a test file in the sandbox.")

        agent = Agent(
            deep=True,
            workspace=Workspace(tmpdir, mode="sandboxed"),
            system="You work with files in the sandbox directory.",
        )

        print(f"  Sandbox root: {tmpdir}")
        print("  Test file created: example.txt")

        result = await agent.arun("List the files and read example.txt")
        print(f"  Result: {result}")


# =============================================================================
# Full Mode (Unrestricted)
# =============================================================================


def full_mode():
    """Full mode - unrestricted filesystem access."""
    print("\n" + "=" * 60)
    print("Full Mode (Unrestricted)")
    print("=" * 60)

    print("""
⚠️ SECURITY WARNING: Full mode gives unrestricted access!

Full mode characteristics:
- Full filesystem access from root
- NO path restrictions
- Can read/write ANY file the process can access
- ONLY for trusted automation with trusted inputs

Good for:
- System administration scripts
- Build tools with controlled inputs
- Trusted internal automation
""")

    # Create full access workspace (be careful!)
    ws = Workspace("/", mode="full")
    print(f"Created: {ws}")

    print("""
⚠️ Never use full mode with:
- Untrusted user prompts
- External/unvalidated inputs
- Cloud functions handling user requests
""")


# =============================================================================
# Workspace with Agent
# =============================================================================


async def workspace_with_agent():
    """Using workspace with different agent types."""
    print("\n" + "=" * 60)
    print("Workspace with Agent")
    print("=" * 60)

    # Deep agent with workspace
    print("\n1. Deep Agent with Workspace:")
    print("   Agent(deep=True, workspace=Workspace('.', mode='sandboxed'))")
    print("   - Uses workspace for built-in file tools")
    print("   - ls, read_file, write_file, edit_file confined to sandbox")

    # Regular agent with workspace
    print("\n2. Regular Agent with Workspace:")
    print("   Agent(tools=[...], workspace=Workspace('.', mode='sandboxed'))")
    print("   - Configures proj_mgmt tools")
    print("   - file_read, file_write, files_list use sandbox")


# =============================================================================
# Path Resolution
# =============================================================================


def path_resolution():
    """Demonstrate path resolution in workspaces."""
    print("\n" + "=" * 60)
    print("Path Resolution")
    print("=" * 60)

    # Create workspace
    ws = Workspace("/project", mode="sandboxed")

    print(f"Workspace root: {ws.root}")
    print(f"Mode: {ws.mode}")
    print()

    # Show resolved paths
    examples = [
        "src/main.py",
        "./config.json",
        "docs/README.md",
        "../secrets.txt",  # Would be blocked
        "/etc/passwd",  # Would be blocked
    ]

    print("Path resolution examples:")
    for path in examples:
        resolved = (ws.root / path).resolve()
        in_sandbox = resolved.is_relative_to(ws.root)
        status = "✓ Allowed" if in_sandbox else "✗ Blocked"
        print(f"  {path:25} → {status}")


# =============================================================================
# Security Best Practices
# =============================================================================


def security_best_practices():
    """Security best practices for workspace configuration."""
    print("\n" + "=" * 60)
    print("Security Best Practices")
    print("=" * 60)

    print("""
1. DEFAULT TO SANDBOXED
   Always use sandboxed mode unless you need something else.

   ✓ Good:
     workspace = Workspace(".", mode="sandboxed")

   ✗ Avoid:
     workspace = Workspace("/", mode="full")  # Too permissive


2. MINIMIZE WORKSPACE SCOPE
   Give agent access to only what it needs.

   ✓ Good:
     workspace = Workspace("./src", mode="sandboxed")  # Only src/

   ✗ Avoid:
     workspace = Workspace("/home/user", mode="sandboxed")  # Too broad


3. USE VIRTUAL FOR UNTRUSTED INPUT
   If processing user-provided prompts, use virtual mode.

   ✓ Good:
     # User prompt from web form
     agent = Agent(deep=True, workspace=Workspace(mode="virtual"))

   ✗ Dangerous:
     # User prompt with full filesystem access
     agent = Agent(deep=True, workspace=Workspace("/", mode="full"))


4. VALIDATE PATHS IN CALLBACKS
   If you have custom file tools, validate paths.

   def read_file(path: str) -> str:
       resolved = Path(path).resolve()
       if not resolved.is_relative_to(ALLOWED_DIR):
           raise SecurityError("Path outside allowed directory")
       return resolved.read_text()


5. REVIEW AGENT PROMPTS
   Ensure system prompts don't encourage path traversal.

   ✓ Good:
     "Read files from the project directory."

   ✗ Bad:
     "Read any file from the system."
""")


# =============================================================================
# Environment-Based Configuration
# =============================================================================


def environment_based_config():
    """Configure workspace based on environment."""
    print("\n" + "=" * 60)
    print("Environment-Based Configuration")
    print("=" * 60)

    # Get environment
    env = os.getenv("ENVIRONMENT", "development")
    is_production = env == "production"
    is_local = env in ("development", "local")

    print(f"Current environment: {env}")
    print()

    # Configure based on environment
    if is_production:
        # Cloud/production: virtual for safety
        ws = Workspace(mode="virtual")
        print("Production: Using virtual mode (no filesystem)")
    elif is_local:
        # Local development: sandboxed to project
        ws = Workspace(".", mode="sandboxed")
        print("Development: Using sandboxed mode (current directory)")
    else:
        # Testing: virtual for isolation
        ws = Workspace(mode="virtual")
        print("Testing: Using virtual mode")

    print(f"Configured: {ws}")


# =============================================================================
# Workspace Factory Pattern
# =============================================================================


def workspace_factory():
    """Factory pattern for workspace creation."""
    print("\n" + "=" * 60)
    print("Workspace Factory Pattern")
    print("=" * 60)

    def create_workspace(
        project_path: str | None = None,
        *,
        allow_writes: bool = True,
        production: bool = False,
    ) -> Workspace:
        """
        Create workspace with appropriate mode.

        Args:
            project_path: Path to project directory (optional)
            allow_writes: Whether to allow write operations
            production: Whether running in production

        Returns:
            Configured Workspace
        """
        if production:
            # Production: always virtual for safety
            return Workspace(mode="virtual")

        if not allow_writes:
            # Read-only: use virtual (no writes possible)
            return Workspace(mode="virtual")

        if project_path:
            # Project-specific: sandboxed to project
            return Workspace(project_path, mode="sandboxed")

        # Default: sandboxed to current directory
        return Workspace(".", mode="sandboxed")

    # Examples
    print("Factory examples:")

    ws = create_workspace("./my-project")
    print(f"  Project workspace: {ws}")

    ws = create_workspace(production=True)
    print(f"  Production workspace: {ws}")

    ws = create_workspace(allow_writes=False)
    print(f"  Read-only workspace: {ws}")


if __name__ == "__main__":
    # Run examples
    workspace_modes_overview()
    virtual_mode()
    asyncio.run(virtual_mode_demo())
    sandboxed_mode()
    asyncio.run(sandboxed_mode_demo())
    full_mode()
    asyncio.run(workspace_with_agent())
    path_resolution()
    security_best_practices()
    environment_based_config()
    workspace_factory()
