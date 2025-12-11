# Workflow Replay

> Debug and replay agent workflows from recorded traces.

## Quick Start

```python
from ai_infra import Agent, Replay

# 1. Run agent with recording enabled
agent = Agent(tools=[search, write_file], record=True)
result = agent.run("Search for Python tutorials and save notes")

# 2. Get the recorded trace
trace = agent.get_trace()
trace.save("my_workflow.json")

# 3. Later, replay and analyze
replay = Replay.from_file("my_workflow.json")
for step in replay.steps():
    print(f"Step {step.number}: {step.action}")
```

---

## Overview

Replay enables debugging complex agent workflows by:
- Recording all agent actions and tool calls
- Playing back execution step-by-step
- Modifying inputs to test different scenarios
- Identifying where workflows fail

---

## Recording Workflows

### Enable Recording on Agent

```python
from ai_infra import Agent

# Enable recording with record=True
agent = Agent(
    tools=[search, calculate],
    record=True,  # This enables recording
)

result = agent.run("Complete this task")

# Get the recorded trace
trace = agent.get_trace()
trace.save("trace.json")
```

### With Custom Path

```python
agent = Agent(
    tools=[my_tool],
    record=True,
    record_path="./traces/",  # Save traces here
)
```

### Manual Recording

```python
from ai_infra import Replay, RecordingContext

with RecordingContext() as ctx:
    # Run workflow
    result = workflow.run(input_data)

    # Get trace
    trace = ctx.get_trace()
```

---

## Loading Replays

### From File

```python
replay = Replay.from_file("trace.json")
```

### From Trace Object

```python
replay = Replay(trace=trace)
```

### From Trace ID

```python
# Requires tracing backend configured
replay = Replay.from_trace_id("trace_abc123")
```

---

## Stepping Through Replay

### Sequential Steps

```python
replay = Replay.from_file("trace.json")

# Step forward
step = replay.next()
print(f"Action: {step.action}")
print(f"Result: {step.result}")

# Step backward
prev_step = replay.prev()

# Jump to specific step
step = replay.goto(5)
```

### Iterate All Steps

```python
for step in replay.steps():
    print(f"[{step.timestamp}] {step.action}")

    if step.is_tool_call:
        print(f"  Tool: {step.tool_name}")
        print(f"  Args: {step.tool_args}")
        print(f"  Result: {step.tool_result}")

    if step.is_llm_call:
        print(f"  Model: {step.model}")
        print(f"  Tokens: {step.tokens_used}")
```

---

## Step Types

```python
step = replay.next()

# Check step type
if step.is_llm_call:
    # LLM inference step
    print(step.prompt)
    print(step.response)
    print(step.model)
    print(step.tokens_used)

elif step.is_tool_call:
    # Tool execution step
    print(step.tool_name)
    print(step.tool_args)
    print(step.tool_result)
    print(step.duration_ms)

elif step.is_decision:
    # Agent decision step
    print(step.decision)
    print(step.reasoning)
```

---

## Filtering Steps

```python
# Only tool calls
for step in replay.steps(filter="tool_call"):
    print(f"{step.tool_name}: {step.tool_result}")

# Only LLM calls
for step in replay.steps(filter="llm_call"):
    print(f"{step.model}: {step.tokens_used} tokens")

# Custom filter
for step in replay.steps(filter=lambda s: s.duration_ms > 1000):
    print(f"Slow step: {step.action} ({step.duration_ms}ms)")
```

---

## Re-running with Modifications

### Modify Inputs

```python
replay = Replay.from_file("trace.json")

# Modify step input
replay.modify_step(3, input={"data": "new_value"})

# Re-run from modified point
result = replay.rerun_from(3)
```

### Replace Tool Results

```python
# Mock a tool result
replay.mock_tool("search", result={"status": "success"})

# Re-run with mocked tool
result = replay.rerun()
```

---

## Analysis

### Timing Analysis

```python
analysis = replay.analyze()

print(f"Total duration: {analysis.total_duration_ms}ms")
print(f"LLM time: {analysis.llm_time_ms}ms")
print(f"Tool time: {analysis.tool_time_ms}ms")
print(f"Total tokens: {analysis.total_tokens}")

# Slowest steps
for step in analysis.slowest_steps(5):
    print(f"{step.action}: {step.duration_ms}ms")
```

### Token Usage

```python
for step in replay.steps(filter="llm_call"):
    print(f"Model: {step.model}")
    print(f"  Input tokens: {step.input_tokens}")
    print(f"  Output tokens: {step.output_tokens}")
    print(f"  Cost: ${step.estimated_cost:.4f}")
```

### Error Analysis

```python
if replay.has_error:
    error_step = replay.error_step
    print(f"Error at step {error_step.number}")
    print(f"Error: {error_step.error}")
    print(f"Traceback: {error_step.traceback}")

    # Get context before error
    context = replay.get_context_before(error_step.number, steps=3)
```

---

## Export Formats

```python
# JSON export
replay.export_json("replay.json")

# Markdown summary
replay.export_markdown("replay.md")

# HTML visualization
replay.export_html("replay.html")
```

---

## Interactive Debugger

```python
from ai_infra import ReplayDebugger

debugger = ReplayDebugger(replay)

# Interactive CLI
debugger.run_cli()

# Available commands:
# > next - step forward
# > prev - step backward
# > goto 5 - jump to step 5
# > inspect - show current step details
# > tools - list all tool calls
# > llm - list all LLM calls
# > quit - exit debugger
```

---

## Configuration

```python
from ai_infra import AgentConfig

# Configure recording
config = AgentConfig(
    record=True,
    record_path="./traces/",
    record_format="json",
    include_prompts=True,  # Include full prompts
    include_responses=True,  # Include full responses
    max_trace_size_mb=10,  # Limit trace file size
)
```

---

## See Also

- [Tracing](../infrastructure/tracing.md) - Distributed tracing
- [Agent](../core/agents.md) - Agent execution
- [Callbacks](../infrastructure/callbacks.md) - Execution hooks
