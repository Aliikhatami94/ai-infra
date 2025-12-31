# Deep Agent

> Autonomous multi-step agents for complex tasks.

## Quick Start

```python
from ai_infra import DeepAgent

agent = DeepAgent(
    goal="Build a complete REST API for user management",
    tools=[search, read_file, write_file, run_tests],
)

result = await agent.run()
```

---

## Overview

Deep Agent is designed for complex, multi-step tasks that require:
- Planning and breaking down goals into subtasks
- Autonomous execution with minimal human intervention
- Self-correction when approaches fail
- Progress tracking and reporting

---

## Creating a Deep Agent

```python
from ai_infra import DeepAgent

agent = DeepAgent(
    goal="Analyze the codebase and generate documentation",
    tools=[search, read_file, write_file],
    provider="openai",
    model_name="gpt-4o",
    max_iterations=50,
)
```

### Configuration Options

```python
agent = DeepAgent(
    goal="...",
    tools=[...],

    # Model settings
    provider="anthropic",
    model_name="claude-sonnet-4-20250514",
    temperature=0.3,

    # Execution limits
    max_iterations=100,
    max_tokens_per_step=4096,
    timeout=3600,  # 1 hour

    # Behavior
    verbose=True,
    allow_self_correction=True,
    require_approval_for=["delete", "deploy"],
)
```

---

## Running the Agent

### Basic Run

```python
result = await agent.run()

print(f"Status: {result.status}")
print(f"Output: {result.output}")
print(f"Steps taken: {result.steps_count}")
```

### With Progress Callback

```python
def on_progress(progress):
    print(f"Step {progress.step}: {progress.action}")
    print(f"Status: {progress.status}")

result = await agent.run(on_progress=on_progress)
```

### Streaming Progress

```python
async for progress in agent.stream():
    print(f"[{progress.step}] {progress.action}")

    if progress.is_complete:
        print(f"Result: {progress.result}")
```

---

## Goal Decomposition

Deep Agent automatically breaks complex goals into subtasks:

```python
agent = DeepAgent(
    goal="Build a complete user authentication system",
    tools=[...],
)

# Agent internally plans:
# 1. Analyze existing code structure
# 2. Design authentication flow
# 3. Implement user model
# 4. Implement registration endpoint
# 5. Implement login endpoint
# 6. Add password hashing
# 7. Implement JWT tokens
# 8. Write tests
# 9. Update documentation
```

### View Plan

```python
plan = await agent.plan()

for task in plan.tasks:
    print(f"{task.number}. {task.description}")
    print(f"   Dependencies: {task.dependencies}")
    print(f"   Estimated effort: {task.effort}")
```

---

## Human-in-the-Loop

### Approval Gates

```python
agent = DeepAgent(
    goal="Deploy the application",
    tools=[...],
    require_approval_for=["deploy", "delete", "modify_production"],
)

async for progress in agent.stream():
    if progress.requires_approval:
        print(f"Action requires approval: {progress.pending_action}")

        if user_approves():
            await agent.approve()
        else:
            await agent.reject("Don't deploy to production yet")
```

### Interactive Mode

```python
agent = DeepAgent(
    goal="...",
    interactive=True,
)

async for progress in agent.stream():
    if progress.waiting_for_input:
        user_input = input("Agent needs input: ")
        await agent.provide_input(user_input)
```

---

## Self-Correction

Deep Agent can detect and recover from failures:

```python
agent = DeepAgent(
    goal="...",
    allow_self_correction=True,
    max_correction_attempts=3,
)

# Agent automatically:
# 1. Detects when an approach fails
# 2. Analyzes what went wrong
# 3. Tries alternative approaches
# 4. Learns from failures within the session
```

---

## Memory and Context

### Session Memory

```python
agent = DeepAgent(
    goal="...",
    memory=True,  # Maintain context across steps
)

# Agent remembers:
# - Files it has read
# - Actions it has taken
# - Results from previous steps
# - Failed approaches to avoid
```

### Persistent Memory

```python
from ai_infra import DeepAgent, PersistentMemory

memory = PersistentMemory(path="./agent_memory.db")

agent = DeepAgent(
    goal="...",
    memory=memory,
)

# Agent can reference past sessions
```

---

## Progress Tracking

```python
result = await agent.run()

# Execution summary
print(f"Total steps: {result.steps_count}")
print(f"Duration: {result.duration_seconds}s")
print(f"Tokens used: {result.total_tokens}")
print(f"Estimated cost: ${result.estimated_cost:.4f}")

# Step-by-step breakdown
for step in result.steps:
    print(f"\n{step.number}. {step.action}")
    print(f"   Duration: {step.duration_ms}ms")
    print(f"   Status: {step.status}")
    if step.error:
        print(f"   Error: {step.error}")
```

---

## Error Handling

```python
from ai_infra import (
    DeepAgent,
    AgentError,
    AgentTimeoutError,
    AgentMaxIterationsError,
)

try:
    result = await agent.run()
except AgentTimeoutError:
    print("Agent timed out")
    partial_result = agent.get_partial_result()
except AgentMaxIterationsError:
    print("Reached max iterations")
    partial_result = agent.get_partial_result()
except AgentError as e:
    print(f"Agent error: {e}")
```

---

## Workspace Integration

```python
from ai_infra import DeepAgent, Workspace

workspace = Workspace("./project")

agent = DeepAgent(
    goal="Refactor the codebase to use async/await",
    workspace=workspace,
    tools=[...],
)

# Agent has sandboxed access to workspace files
result = await agent.run()
```

---

## Example: Code Review Agent

```python
from ai_infra import DeepAgent, Workspace

workspace = Workspace("./project")

agent = DeepAgent(
    goal="""
    Review all Python files in the codebase:
    1. Check for code quality issues
    2. Identify potential bugs
    3. Suggest improvements
    4. Generate a report in docs/code-review.md
    """,
    workspace=workspace,
    tools=[read_file, write_file, search, run_linter],
    provider="anthropic",
    model_name="claude-sonnet-4-20250514",
    max_iterations=100,
)

result = await agent.run()
print(result.output)
```

---

## Example: Documentation Agent

```python
agent = DeepAgent(
    goal="""
    Generate comprehensive documentation:
    1. Analyze all modules and functions
    2. Extract docstrings and type hints
    3. Generate API reference
    4. Create usage examples
    5. Write getting started guide
    """,
    workspace=workspace,
    tools=[read_file, write_file, search, list_dir],
)

result = await agent.run()
```

---

## Complete Example: All Features

This example shows all DeepAgent features working together:

```python
from ai_infra import DeepAgent, Workspace

# 1. Set up workspace with sandboxed file access
workspace = Workspace("./my-project")

# 2. Define tools the agent can use
def search_codebase(query: str) -> str:
    """Search for code patterns."""
    return workspace.search(query)

def read_file(path: str) -> str:
    """Read a file."""
    return workspace.read(path)

def write_file(path: str, content: str) -> str:
    """Write to a file."""
    workspace.write(path, content)
    return f"Wrote {len(content)} bytes to {path}"

def run_tests() -> str:
    """Run the test suite."""
    # This is a dangerous operation - will require approval
    import subprocess
    result = subprocess.run(["pytest", "-q"], capture_output=True, text=True)
    return result.stdout

# 3. Create agent with all features
agent = DeepAgent(
    goal="Add type hints to all functions in src/ and verify with tests",
    tools=[search_codebase, read_file, write_file, run_tests],
    workspace=workspace,

    # Planning & Execution
    max_iterations=50,
    allow_self_correction=True,  # Retry on failures

    # Human approval for dangerous operations
    require_approval_for=["run_tests", "write_file"],
)

# 4. Run with progress tracking
async def main():
    async for progress in agent.stream():
        # Show progress
        print(f"[Step {progress.step}] {progress.status}")

        # Handle approval requests
        if progress.requires_approval:
            print(f"\n[!]  Approval needed: {progress.pending_action}")
            print(f"   Tool: {progress.tool_name}")
            print(f"   Args: {progress.tool_args}")

            if input("Approve? [y/n]: ").lower() == "y":
                await agent.approve()
            else:
                await agent.reject("User declined")

    # Get final result
    result = agent.get_result()

    # Summary
    print(f"\n[OK] Completed in {result.steps_count} steps")
    print(f"   Duration: {result.duration_seconds:.1f}s")
    print(f"   Tokens: {result.total_tokens}")
    print(f"   Cost: ${result.estimated_cost:.4f}")
    print(f"\nOutput:\n{result.output}")

import asyncio
asyncio.run(main())
```

---

## See Also

- [Agent](../core/agents.md) - Basic agent usage
- [Personas](personas.md) - Agent configuration
- [Workspace](workspace.md) - File operations
- [Replay](replay.md) - Debug workflows
