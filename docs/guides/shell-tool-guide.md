# Using Shell Tool in Agents

> A comprehensive guide to integrating the shell tool with AI agents for automated workflows.

## Overview

The shell tool enables AI agents to interact with the operating system, run commands, and automate development workflows. This guide covers integration patterns, best practices, and common use cases.

---

## Getting Started

### Basic Integration

```python
from ai_infra import Agent
from ai_infra.llm.shell import run_shell

# Add shell tool to agent
agent = Agent(
    tools=[run_shell],
    deep=True  # Enable extended reasoning
)

# Agent can now execute commands
result = agent.run("Find all Python files with syntax errors")
```

### With Persistent Sessions

For workflows that require state (like virtual environments), use middleware:

```python
from ai_infra import Agent
from ai_infra.llm.shell import ShellMiddleware

agent = Agent(
    deep=True,
    middleware=[ShellMiddleware(workspace_root="/my/project")]
)

# Environment and directory state persist across commands
result = agent.run("""
    Create a virtual environment, install dependencies,
    and run the test suite
""")
```

---

## Integration Patterns

### Pattern 1: Build and Test Automation

```python
from ai_infra import Agent
from ai_infra.llm.shell import ShellMiddleware

agent = Agent(
    deep=True,
    middleware=[ShellMiddleware(
        workspace_root="/project",
        allowed_commands=("pytest", "npm", "make", "poetry"),
    )]
)

# Automated CI-style workflow
result = agent.run("""
    1. Check for linting errors
    2. Run type checking
    3. Execute the test suite
    4. Report any failures
""")
```

### Pattern 2: Code Analysis

```python
from ai_infra import Agent
from ai_infra.llm.shell import run_shell

agent = Agent(
    tools=[run_shell],
    deep=True
)

# Agent uses shell for code exploration
result = agent.run("""
    Analyze the codebase structure:
    - Count lines of code by language
    - Find the largest files
    - Identify potential dead code
""")
```

### Pattern 3: Environment Setup

```python
from ai_infra import Agent
from ai_infra.llm.shell import ShellMiddleware

agent = Agent(
    deep=True,
    middleware=[ShellMiddleware(workspace_root="/project")]
)

# Automated development environment setup
result = agent.run("""
    Set up the development environment:
    1. Check Python version
    2. Create virtual environment if not exists
    3. Install dependencies from pyproject.toml
    4. Set up pre-commit hooks
    5. Verify the setup by running a simple test
""")
```

### Pattern 4: Multi-Step Debugging

```python
from ai_infra import Agent
from ai_infra.llm.shell import ShellMiddleware

agent = Agent(
    deep=True,
    middleware=[ShellMiddleware(workspace_root="/project")]
)

# Interactive debugging workflow
result = agent.run("""
    Debug the failing test in test_payment.py:
    1. Run the specific test to see the error
    2. Add debug output to understand the state
    3. Identify the root cause
    4. Fix the issue
    5. Verify the fix by running the test again
""")
```

---

## Combining with Other Tools

### Shell + File Tools

```python
from ai_infra import Agent
from ai_infra.llm.shell import run_shell
from ai_infra.llm.tools import read_file, write_file

agent = Agent(
    tools=[run_shell, read_file, write_file],
    deep=True
)

# Agent can read files, modify them, and run tests
result = agent.run("""
    1. Read the failing test file
    2. Understand what it's testing
    3. Fix the implementation
    4. Run the test to verify
""")
```

### Shell + Web Tools

```python
from ai_infra import Agent
from ai_infra.llm.shell import run_shell
from ai_infra.llm.tools import fetch_url

agent = Agent(
    tools=[run_shell, fetch_url],
    deep=True
)

# Agent can fetch documentation and apply changes
result = agent.run("""
    1. Check the current version of the requests library
    2. Fetch the changelog for the latest version
    3. Update the dependency and run tests
""")
```

---

## Session Management

### Understanding Sessions

Sessions maintain state across tool calls:

```python
from ai_infra.llm.shell import ShellSession

async with ShellSession(workspace_root="/project") as session:
    # State persists within the session
    await session.execute("cd src")
    await session.execute("export DEBUG=1")
    await session.execute("python main.py")  # Runs in src/ with DEBUG=1
```

### Session Lifecycle with Middleware

```python
from ai_infra import Agent
from ai_infra.llm.shell import ShellMiddleware

middleware = ShellMiddleware(
    workspace_root="/project",
    session_timeout=3600.0  # 1 hour
)

agent = Agent(deep=True, middleware=[middleware])

# Session starts when agent starts
result = agent.run("Run the build")
# Session ends when agent finishes

# Access session history
for entry in middleware.session.get_history():
    print(f"{entry.command}: exit_code={entry.exit_code}")
```

### Manual Session Control

```python
from ai_infra.llm.shell import ShellSession, create_shell_tool

# Create session manually
session = ShellSession(workspace_root="/project")
await session.start()

# Create tool bound to session
shell_tool = create_shell_tool(session=session)

# Use with agent
agent = Agent(tools=[shell_tool])

try:
    result = agent.run("Build the project")
finally:
    await session.close()
```

---

## Command Patterns

### Project Type Detection

```python
# Agent detects project type and adapts
result = agent.run("""
    Determine the project type and run appropriate tests:
    - If Python: use pytest
    - If Node.js: use npm test
    - If Rust: use cargo test
    - If Go: use go test
""")
```

### Git Workflows

```python
from ai_infra import Agent
from ai_infra.llm.shell import ShellMiddleware

agent = Agent(
    deep=True,
    middleware=[ShellMiddleware(
        workspace_root="/project",
        allowed_commands=("git", "pytest", "make"),
    )]
)

result = agent.run("""
    1. Create a feature branch
    2. Make the requested changes
    3. Run tests to verify
    4. Commit with a descriptive message
""")
```

### Long-Running Commands

```python
from ai_infra.llm.shell import create_shell_tool

# Extend timeout for builds
shell_tool = create_shell_tool(default_timeout=600.0)  # 10 minutes

agent = Agent(tools=[shell_tool])

result = agent.run("Build the Docker image and push to registry")
```

---

## Error Handling in Agents

### Automatic Retry Logic

Agents can be instructed to retry on failure:

```python
result = agent.run("""
    Run the test suite. If tests fail:
    1. Analyze the failure
    2. Attempt to fix the issue
    3. Re-run the tests
    4. Repeat up to 3 times
""")
```

### Graceful Degradation

```python
result = agent.run("""
    Try to run pytest. If pytest is not installed:
    1. Install it with pip
    2. Then run the tests

    If that fails too, try running unittest directly.
""")
```

### Reporting Failures

```python
result = agent.run("""
    Run the full test suite. For any failures:
    1. Capture the error message
    2. Identify the root cause
    3. Suggest a fix
    4. Report a summary of all issues found
""")
```

---

## Best Practices

### 1. Use Specific Working Directories

```python
# Good: Explicit workspace
middleware = ShellMiddleware(workspace_root="/project")

# Avoid: No workspace (runs from current directory)
middleware = ShellMiddleware()
```

### 2. Limit Command Scope

```python
# Good: Restrict to necessary commands
middleware = ShellMiddleware(
    workspace_root="/project",
    allowed_commands=("pytest", "npm", "make"),
)

# Avoid: Unrestricted access in production
middleware = ShellMiddleware(
    workspace_root="/project",
    dangerous_pattern_check=False,  # Don't do this
)
```

### 3. Handle Timeouts

```python
# Set appropriate timeouts for long commands
shell_tool = create_shell_tool(default_timeout=300.0)

# Instruct agent to handle timeouts
result = agent.run("""
    Run the build. If it takes more than 2 minutes,
    check if there's a simpler incremental build option.
""")
```

### 4. Validate Before Destructive Actions

```python
result = agent.run("""
    Before deleting any files:
    1. List what will be deleted
    2. Ask for confirmation
    3. Only then proceed with deletion
""")
```

### 5. Use Structured Output

```python
from ai_infra import Agent
from ai_infra.llm.shell import ShellMiddleware

agent = Agent(
    deep=True,
    middleware=[ShellMiddleware(workspace_root="/project")],
    output_format="structured"  # JSON output
)

result = agent.run("""
    Run tests and report results as:
    - total_tests: number
    - passed: number
    - failed: number
    - failures: list of {test_name, error_message}
""")
```

---

## Debugging

### Enable Verbose Logging

```python
import logging

logging.getLogger("ai_infra.llm.shell").setLevel(logging.DEBUG)

agent = Agent(
    deep=True,
    middleware=[ShellMiddleware(workspace_root="/project")]
)
# Shell commands and outputs are logged
```

### Inspect Session State

```python
middleware = ShellMiddleware(workspace_root="/project")
agent = Agent(deep=True, middleware=[middleware])

result = agent.run("Set up the environment")

# After execution, inspect state
print(f"Current directory: {middleware.session.cwd}")
print(f"Environment: {middleware.session.env}")
print(f"Command count: {len(middleware.session.get_history())}")
```

### Trace Tool Calls

```python
from ai_infra import Agent
from ai_infra.llm.middleware import LoggingMiddleware
from ai_infra.llm.shell import ShellMiddleware

agent = Agent(
    deep=True,
    middleware=[
        LoggingMiddleware(level="DEBUG"),  # Log all tool calls
        ShellMiddleware(workspace_root="/project"),
    ]
)
```

---

## Common Issues

### Issue: Command Not Found

```python
# Agent should check for command availability
result = agent.run("""
    Before running pytest:
    1. Check if pytest is installed: which pytest
    2. If not found, install it: pip install pytest
    3. Then run the tests
""")
```

### Issue: Permission Denied

```python
# Set appropriate working directory
middleware = ShellMiddleware(
    workspace_root="/home/user/project"  # User-writable directory
)
```

### Issue: Environment Variables Not Set

```python
# Use middleware for persistent environment
middleware = ShellMiddleware(workspace_root="/project")

# Without middleware, env vars don't persist
result = await run_shell.ainvoke({"command": "export FOO=bar"})
result = await run_shell.ainvoke({"command": "echo $FOO"})
# FOO is empty without middleware!
```

---

## See Also

- [Shell Tool API](../tools/shell-tool.md) — Function reference
- [ShellMiddleware API](../tools/shell-middleware.md) — Middleware reference
- [Security Best Practices](shell-security.md) — Security guidelines
