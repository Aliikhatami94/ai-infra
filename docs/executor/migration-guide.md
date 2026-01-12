# Graph Executor Migration Guide

> Migrating from legacy imperative executor to graph-based execution.

## Overview

The graph-based executor replaces the legacy while-loop with a structured state machine built on `ai_infra.graph.Graph`. This guide covers how to migrate your workflows.

## Quick Start

### Opt-in to Graph Mode (Default)

Graph mode is now the default. Simply run:

```bash
ai-infra executor run --roadmap ./ROADMAP.md
```

### Use Legacy Mode (Fallback)

If you need the legacy imperative executor:

```bash
ai-infra executor run --roadmap ./ROADMAP.md --legacy-mode
```

## CLI Options

### New Graph-Specific Options

| Option | Description |
|--------|-------------|
| `--graph-mode` | Use graph-based executor (default) |
| `--legacy-mode` | Fall back to imperative loop |
| `--visualize` | Generate Mermaid diagram of the graph |
| `--interrupt-before <node>` | Pause before specific nodes |
| `--interrupt-after <node>` | Pause after specific nodes |

### Visualize Graph Structure

```bash
ai-infra executor run --roadmap ./ROADMAP.md --visualize
```

This outputs a Mermaid diagram you can render in any Mermaid-compatible viewer.

### Set HITL Interrupt Points

```bash
# Pause before each task execution
ai-infra executor run --roadmap ./ROADMAP.md --interrupt-before execute_task

# Pause after verification for review
ai-infra executor run --roadmap ./ROADMAP.md --interrupt-after verify_task
```

## Known Differences

### Behavioral Changes

| Aspect | Legacy | Graph |
|--------|--------|-------|
| **State Storage** | `.executor/state.json` | `.executor/graph_state.json` |
| **Resume** | From state file | Via thread ID |
| **HITL** | Callback-based | Native graph interrupts |
| **Streaming** | Callback events | Graph node events |
| **Tracing** | Manual spans | Automatic per-node |

### State File Coexistence

Both executors can coexist:
- Legacy: `.executor/state.json`
- Graph: `.executor/graph_state.json`

Switching between modes does not affect the other mode's state.

### Error Handling

Graph mode uses `ExecutorError` typed dict:

```python
{
    "error_type": "execution",  # execution, verification, timeout, etc.
    "message": "Task failed: ...",
    "node": "execute_task",
    "task_id": "task-123",
    "recoverable": True,
    "stack_trace": "..."  # For debugging
}
```

### Retry Behavior

Both modes have the same retry semantics:
- Default max retries: 3
- Exponential backoff: 1s, 2s, 4s
- Non-retryable errors fail immediately

## Troubleshooting

### Graph Executor Won't Start

**Symptom**: Error on startup with graph mode

**Solution**: Check that all required dependencies are installed:

```bash
cd ai-infra
poetry install
```

### Fallback to Legacy Mode

If graph mode fails, it automatically falls back to legacy with a warning:

```
[yellow]Graph executor unavailable, falling back to legacy mode[/yellow]
```

To force legacy mode, use `--legacy-mode`.

### State Corruption

**Symptom**: `Failed to load graph state` error

**Solution**: Clear the graph state and restart:

```bash
rm .executor/graph_state.json
ai-infra executor run --roadmap ./ROADMAP.md
```

### HITL Resume Issues

**Symptom**: Cannot resume after interrupt

**Solution**: Check the HITL state file:

```bash
cat .executor/hitl_state.json
```

If corrupted, clear and restart:

```bash
rm .executor/hitl_state.json
ai-infra executor run --roadmap ./ROADMAP.md
```

### Thread ID Not Found

**Symptom**: `Thread ID not found` when resuming

**Solution**: The thread ID is stored in `.executor/thread_id`. If missing:

```bash
# Start fresh execution
rm .executor/thread_id
ai-infra executor run --roadmap ./ROADMAP.md
```

### Timeout Errors

**Symptom**: Node times out frequently

**Solution**: Timeouts are configured per-node:

| Node | Default Timeout |
|------|-----------------|
| `parse_roadmap` | 30s |
| `build_context` | 60s |
| `execute_task` | 300s |
| `verify_task` | 120s |
| `checkpoint` | 30s |

For long-running tasks, consider:
- Breaking into smaller tasks
- Increasing model response speed
- Checking network connectivity

### Git Conflicts on Rollback

**Symptom**: Rollback fails with git conflict

**Solution**: Manually resolve the conflict:

```bash
git status  # See conflicting files
git checkout --ours <file>  # Keep your changes
# OR
git checkout --theirs <file>  # Use checkpoint version
git add .
```

Then resume execution:

```bash
ai-infra executor run --roadmap ./ROADMAP.md
```

## Programmatic Usage

### Using ExecutorGraph Directly

```python
from ai_infra.executor.graph import ExecutorGraph
from ai_infra import Agent

# Create agent
agent = Agent(model="claude-sonnet-4-20250514")

# Create graph executor
executor = ExecutorGraph(
    agent=agent,
    roadmap_path="./ROADMAP.md",
    max_tasks=10,
    sync_roadmap=True,
)

# Run to completion
result = await executor.arun()
print(f"Completed: {result['completed_count']} tasks")
```

### Streaming Node Events

```python
async for node_name, state in executor.astream():
    print(f"Executed: {node_name}")
    if node_name == "checkpoint":
        print(f"  Completed task: {state.get('current_task', {}).get('title')}")
```

### With Tracing

```python
from ai_infra.executor.graph_tracing import TracingConfig

result = await executor.arun_with_tracing(
    tracing_config=TracingConfig.production(),
    run_id="my-run-123",
)
```

## Rollback to Legacy

If you need to permanently revert to legacy mode:

1. Set `--legacy-mode` flag in all commands
2. Or wait for the automatic fallback on graph errors

Legacy mode remains fully functional and is not deprecated.
