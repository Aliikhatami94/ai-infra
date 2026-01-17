# Task Routing

> Phase 16.5.11: Intelligent task routing for executor subagents.

## Overview

The executor routes tasks to specialized subagents based on task semantics. There are two routing mechanisms:

| Mechanism | Status | Description |
|-----------|--------|-------------|
| **OrchestratorAgent** | **Recommended** | LLM-based semantic understanding |
| Keyword Matching | Deprecated | Pattern-based heuristics |

## OrchestratorAgent (Recommended)

The `OrchestratorAgent` uses an LLM to analyze task semantics and context, providing intelligent routing with confidence scores.

### Basic Usage

```python
from ai_infra.executor.agents.orchestrator import OrchestratorAgent, RoutingContext
from ai_infra.executor.todolist import TodoItem
from pathlib import Path

# Create orchestrator
orchestrator = OrchestratorAgent(
    model="gpt-4o-mini",  # Fast, cost-effective model
    confidence_threshold=0.7,
    fallback_to_keywords=True,
)

# Build context
context = RoutingContext(
    workspace=Path("/my/project"),
    completed_tasks=["Create src/user.py with User class"],
    existing_files=["src/user.py", "src/__init__.py"],
    project_type="python",
)

# Route a task
task = TodoItem(id=2, title="Create tests for user.py")
decision = await orchestrator.route(task, context)

print(f"Route to: {decision.agent_type.value}")  # testwriter
print(f"Confidence: {decision.confidence}")       # 0.95
print(f"Reasoning: {decision.reasoning}")
```

### Available Agents

The orchestrator routes tasks to these specialist agents:

| Agent | Purpose | Example Tasks |
|-------|---------|---------------|
| **Coder** | Implements application code | "Create src/user.py", "Add login endpoint" |
| **TestWriter** | Creates test files | "Write tests for user.py", "Add unit tests" |
| **Tester** | Runs existing tests | "Run pytest", "Verify tests pass" |
| **Debugger** | Fixes bugs and errors | "Fix ImportError", "Debug the crash" |
| **Reviewer** | Reviews and refactors code | "Review user.py", "Optimize performance" |
| **Researcher** | Gathers information | "Research best practices", "Find docs for X" |

### Configuration

```python
from ai_infra.executor.graph import ExecutorGraph

# Via ExecutorGraph
graph = ExecutorGraph(
    use_orchestrator=True,           # Enable LLM routing (default)
    orchestrator_model="gpt-4o-mini", # Model for routing
    orchestrator_confidence_threshold=0.7,  # Min confidence
)
```

### CLI Flags

```bash
# Use orchestrator (default)
ai-infra executor run roadmap.md --use-orchestrator

# Disable orchestrator, use keyword fallback
ai-infra executor run roadmap.md --no-orchestrator

# Custom orchestrator model
ai-infra executor run roadmap.md --orchestrator-model gpt-4o

# Custom confidence threshold
ai-infra executor run roadmap.md --orchestrator-threshold 0.8
```

### Benefits

1. **Semantic Understanding**: Routes based on meaning, not just keywords
2. **Context Awareness**: Considers completed tasks and existing files
3. **Confidence Scores**: Indicates routing certainty
4. **Fallback Safety**: Falls back to keywords if LLM fails
5. **Observability**: Full routing history and token tracking

## Keyword-Based Routing (Deprecated)

> **Warning**: Direct use of `SubAgentRegistry.for_task()` is deprecated.
> Use `OrchestratorAgent` instead. Keyword routing remains available
> as a fallback mechanism.

### Migration Guide

```python
# DEPRECATED: Direct keyword routing
from ai_infra.executor.agents import SubAgentRegistry
agent_type = SubAgentRegistry.for_task(task)  # Emits DeprecationWarning

# RECOMMENDED: Use OrchestratorAgent
from ai_infra.executor.agents.orchestrator import OrchestratorAgent
orchestrator = OrchestratorAgent()
decision = await orchestrator.route(task, context)
agent_type = decision.agent_type
```

### Internal Fallback

For internal use (e.g., orchestrator fallback), use the dedicated function:

```python
from ai_infra.executor.agents.registry import keyword_fallback_routing

# Returns (SubAgentType, matched_keyword or None)
agent_type, matched = keyword_fallback_routing(task)
```

## Routing Metrics

The executor tracks routing decisions for observability:

```python
# In state after execution
state["orchestrator_routing_decision"]  # Current task decision
state["orchestrator_routing_history"]   # All decisions this run
state["orchestrator_tokens_total"]      # Total orchestrator tokens
```

### Viewing Routing History

The CLI displays routing history in the summary:

```
╭──────────────── Routing History ────────────────╮
│ Task 1: coder (confidence: 0.92)                │
│ Task 2: testwriter (confidence: 0.95)           │
│ Task 3: tester (confidence: 0.88)               │
╰─────────────────────────────────────────────────╯
```

## Performance

| Metric | Target | Actual |
|--------|--------|--------|
| Routing latency | <2s | ~1.2s |
| Token cost | <500/route | ~200 |
| Accuracy vs keywords | +15% | Measured |

## Troubleshooting

### Low Confidence Routing

If routing confidence is consistently low:

1. Check that task titles are descriptive
2. Ensure context includes completed tasks
3. Consider lowering threshold for exploration

### Fallback to Keywords

The orchestrator falls back to keywords when:

1. LLM response is invalid JSON
2. Confidence is below threshold
3. LLM call times out
4. Unknown agent type returned

Check logs for `"Using keyword fallback"` messages.

### Debugging Routing Decisions

```python
import logging
logging.getLogger("executor.agents.orchestrator").setLevel(logging.DEBUG)
```

This enables detailed logging of routing prompts and decisions.
