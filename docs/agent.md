# Agent

The `Agent` class provides a simple API for running LLM agents with tools, including support for sessions, human-in-the-loop approval, streaming, and autonomous deep agent mode.

## Quick Start

```python
from ai_infra.llm import Agent

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

# Create an agent with tools
agent = Agent(tools=[get_weather])

# Run the agent
result = agent.run("What's the weather in NYC?")
print(result)  # "The weather in NYC is Sunny, 72°F"
```

## Basic Usage

### With Provider and Model

```python
agent = Agent(
    provider="anthropic",
    model_name="claude-sonnet-4-20250514",
    tools=[my_tool],
)
```

### With System Prompt

```python
agent = Agent(
    tools=[my_tool],
    system="You are a helpful coding assistant.",
)

result = agent.run("Help me write a Python function", system="Be concise.")
```

### Async Support

```python
result = await agent.arun("What's the weather in NYC?")
```

## Session Memory

Persist conversations across requests using sessions:

```python
from ai_infra.llm import Agent
from ai_infra.llm.session import memory, postgres

# Development: in-memory sessions
agent = Agent(tools=[...], session=memory())

# Production: Postgres sessions
agent = Agent(tools=[...], session=postgres("postgresql://..."))

# Use session IDs to maintain conversations
agent.run("I'm Bob", session_id="user-123")
agent.run("What's my name?", session_id="user-123")  # Knows "Bob"
```

## Human-in-the-Loop (HITL)

### Pause and Resume

```python
agent = Agent(
    tools=[dangerous_tool],
    session=memory(),
    pause_before=["dangerous_tool"],
)

result = agent.run("Delete file.txt", session_id="task-1")

if result.paused:
    print(f"Pending: {result.pending_action}")
    # Get user approval...
    result = agent.resume(session_id="task-1", approved=True)
```

### Approval Handler

```python
agent = Agent(
    tools=[dangerous_tool],
    require_approval=True,  # Console prompt
)

# Or with custom handler
agent = Agent(
    tools=[dangerous_tool],
    require_approval=["dangerous_tool"],
    approval_handler=my_approval_function,
)
```

---

## Deep Mode

Enable `deep=True` to unlock autonomous agent capabilities powered by LangChain's DeepAgents.

### What Deep Mode Adds

- **Filesystem tools**: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- **Execute tool**: Run shell commands
- **Todo management**: `write_todos` for task tracking
- **Subagent orchestration**: Delegate to specialized agents

### Quick Start

```python
from ai_infra import Agent

# Regular agent
agent = Agent(provider="anthropic", tools=[...])

# Deep agent - just add deep=True
agent = Agent(provider="anthropic", tools=[...], deep=True)
```

### All Existing Features Work

Deep mode preserves all existing Agent functionality:

```python
agent = Agent(
    provider="anthropic",           # ✅ Works
    model_name="claude-sonnet-4",   # ✅ Works
    tools=[my_tool],                # ✅ Works (added to built-in tools)
    system="...",                   # ✅ Works
    session=postgres("..."),        # ✅ Works (our abstraction)
    pause_before=["write_file"],    # ✅ Works
    deep=True,
)
```

### Subagents

Create specialized agents that the main agent can delegate to:

```python
from ai_infra.llm import Agent
from ai_infra.llm.session import memory

# Define specialized agents
researcher = Agent(
    name="researcher",
    description="Expert at finding and analyzing information",
    system="You are a thorough researcher.",
    tools=[search_tool],
)

writer = Agent(
    name="writer",
    description="Expert at writing clear documentation",
    system="You are a technical writer.",
)

# Create deep agent with subagents
agent = Agent(
    deep=True,
    session=memory(),
    subagents=[researcher, writer],  # Agents auto-convert
)

result = agent.run("Research our auth module and write docs for it")
```

**Note**: Agents used as subagents must have `name` and `description` set.

### Deep-Only Parameters

These parameters only work with `deep=True`:

| Parameter | Description |
|-----------|-------------|
| `subagents` | List of Agent or SubAgent for delegation |
| `middleware` | Lifecycle hooks for agent execution |
| `response_format` | Structured output format |
| `context_schema` | Schema for agent context |
| `use_longterm_memory` | Enable cross-session memory |

### When to Use Deep Mode

| Task | Regular Agent | Deep Agent |
|------|---------------|------------|
| Chat/Q&A | ✅ | Overkill |
| Simple tool calls | ✅ | Overkill |
| File operations | ❌ | ✅ |
| Multi-step research | Limited | ✅ |
| Code generation | Limited | ✅ |
| Subagent coordination | ❌ | ✅ |

---

## Workspace Configuration

The `workspace` parameter controls how agents interact with the filesystem. It provides a unified configuration that works for both regular agents (using proj_mgmt tools) and deep agents (using built-in file tools).

### Quick Start

```python
from ai_infra.llm import Agent

# Simple: sandbox to current directory
agent = Agent(deep=True, workspace=".")

# Agent can now read/write files in current directory
result = agent.run("List files in src/")
```

### Workspace Modes

| Mode | Filesystem | Persistence | Sandboxing | Use Case |
|------|-----------|-------------|------------|----------|
| `sandboxed` | Real files | ✅ Persists | ✅ Confined to root | Local development (default) |
| `virtual` | In-memory | ❌ Gone when done | N/A | Cloud, untrusted prompts |
| `full` | Real files | ✅ Persists | ❌ No sandboxing | Trusted automation |

### Examples

```python
from ai_infra.llm import Agent, Workspace

# Local development - sandboxed to project (default)
agent = Agent(
    deep=True,
    workspace=Workspace(".", mode="sandboxed"),
)

# Cloud deployment - virtual filesystem (safest)
agent = Agent(
    deep=True,
    workspace=Workspace(mode="virtual"),
)
# Files only exist in memory, safe for untrusted prompts

# Trusted CI/CD - full access (use with caution)
agent = Agent(
    deep=True,
    workspace=Workspace("/", mode="full"),
)
```

### With Regular Agents (proj_mgmt tools)

The workspace also configures proj_mgmt tools like `file_read`, `file_write`:

```python
from ai_infra.llm import Agent
from ai_infra.llm.tools.custom.proj_mgmt import file_read, file_write

agent = Agent(
    tools=[file_read, file_write],
    workspace="/path/to/project",
)
# Tools are now sandboxed to /path/to/project
```

### Migration from set_workspace_root

The old `set_workspace_root()` function is deprecated:

```python
# ❌ Old way (deprecated)
from ai_infra.llm.tools.custom.proj_mgmt import set_workspace_root
set_workspace_root("/path/to/project")
agent = Agent(tools=[file_read])

# ✅ New way
agent = Agent(
    tools=[file_read],
    workspace="/path/to/project",
)
```

---

## Tool Execution Configuration

Control how tools are executed:

```python
agent = Agent(
    tools=[my_tool],
    on_tool_error="return_error",  # "return_error" | "retry" | "abort"
    tool_timeout=30.0,              # Timeout per tool call
    max_tool_retries=3,             # Retries when on_tool_error="retry"
    validate_tool_results=True,     # Validate return types
)
```

## Error Handling

Tool errors are translated to informative messages:

```python
from ai_infra.llm import ToolExecutionError, ToolTimeoutError

try:
    result = agent.run("...")
except ToolTimeoutError as e:
    print(f"Tool {e.tool_name} timed out after {e.timeout}s")
except ToolExecutionError as e:
    print(f"Tool {e.tool_name} failed: {e.message}")
```

## Streaming

```python
async for mode, chunk in agent.arun_agent_stream(messages, provider, model_name):
    if mode == "updates":
        print(chunk)
```

## API Reference

### Agent.__init__

```python
Agent(
    # Basic config
    tools: Optional[List[Any]] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,

    # Agent identity (for subagent use)
    name: Optional[str] = None,
    description: Optional[str] = None,
    system: Optional[str] = None,

    # Tool execution
    on_tool_error: Literal["return_error", "retry", "abort"] = "return_error",
    tool_timeout: Optional[float] = None,
    max_tool_retries: int = 1,
    validate_tool_results: bool = False,

    # Approval
    require_approval: Union[bool, List[str], Callable] = False,
    approval_handler: Optional[Callable] = None,

    # Session
    session: Optional[SessionStorage] = None,
    pause_before: Optional[List[str]] = None,
    pause_after: Optional[List[str]] = None,

    # Deep mode
    deep: bool = False,
    subagents: Optional[List[Union[Agent, SubAgent]]] = None,
    middleware: Optional[Sequence[AgentMiddleware]] = None,
    response_format: Optional[Any] = None,
    context_schema: Optional[Type[Any]] = None,
    use_longterm_memory: bool = False,

    # Workspace (file operations)
    workspace: Optional[Union[str, Path, Workspace]] = None,

    **model_kwargs,
)
```

### Agent.run / Agent.arun

```python
result = agent.run(
    prompt: str,
    provider: Optional[str] = None,      # Override provider
    model_name: Optional[str] = None,    # Override model
    tools: Optional[List[Any]] = None,   # Override tools
    system: Optional[str] = None,        # System message
    session_id: Optional[str] = None,    # For session persistence
    **model_kwargs,
)
```

### Agent.resume / Agent.aresume

```python
result = agent.resume(
    session_id: str,
    approved: bool = True,
    modified_args: Optional[Dict] = None,
    reason: Optional[str] = None,
)
```
