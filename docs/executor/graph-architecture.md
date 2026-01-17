# Executor Graph Architecture

> Comprehensive architecture documentation for the graph-based autonomous task executor.

## Overview

The Executor is a **LangGraph-based state machine** that autonomously executes development tasks from a ROADMAP.md file. It features:

- **Orchestrator-based routing** - LLM selects specialist agents per task
- **Multiple repair loops** - Code validation and test repair cycles
- **Adaptive replanning** - Generates new approaches when tasks fail
- **Git checkpointing** - Commits on success, enables rollback
- **Streaming events** - Real-time progress and token streaming

**Key Design Principle**: The executor is **not linear** - it's a directed graph with conditional branches, loops, and multiple terminal states.

---

## High-Level Architecture

```
+---------------------------------------------------------------------------+
|                            ExecutorGraph                                   |
|  (LangGraph-based state machine for autonomous task execution)            |
+---------------------------------------------------------------------------+
                                    |
          +-------------------------+-------------------------+
          v                         v                         v
   +-------------+          +-------------+          +-------------+
   |    Nodes    |          |    Edges    |          |   Agents    |
   |  (Actions)  |          |  (Routing)  |          |(Specialists)|
   +-------------+          +-------------+          +-------------+
```

---

## Complete Graph Flow

```
                                    +-------------------------------------------------------------+
                                    |                         START                                |
                                    +-----------------------------+-------------------------------+
                                                                  |
                                                                  v
                                    +-------------------------------------------------------------+
                                    |                     parse_roadmap                            |
                                    |            (Parse ROADMAP.md -> TodoItems)                   |
                                    +-----------------------------+-------------------------------+
                                                                  |
                                                                  v
                          +-------------------------------------------------------------------------------+
                          |                                  pick_task                                     |
              +---------->|                    (Select next pending task from TodoList)                    |<---------------------+
              |           +-----------------------------------+-------------------------------------------+                      |
              |                                               |                                                                   |
              |                              +----------------+----------------+                                                  |
              |                              |                                 |                                                  |
              |                         NO TASK                           HAS TASK                                                |
              |                              |                                 |                                                  |
              |                              v                                 v                                                  |
              |                     +-------------+              +-------------------------+                                      |
              |                     |     END     |              |  enable_planning=True?  |                                      |
              |                     +-------------+              +------------+------------+                                      |
              |                                                               |                                                   |
              |                                          +--------------------+--------------------+                              |
              |                                          |                                         |                              |
              |                                         YES                                        NO                             |
              |                                          |                                         |                              |
              |                                          v                                         |                              |
              |                           +-------------------------+                              |                              |
              |                           |       plan_task         |                              |                              |
              |                           | (Break into substeps)   |                              |                              |
              |                           +------------+------------+                              |                              |
              |                                        |                                           |                              |
              |                                        +------------------+------------------------+                              |
              |                                                           |                                                       |
              |                                                           v                                                       |
              |                                        +-------------------------------------+                                    |
              |                                        |          build_context              |                                    |
              |                                        |  (Gather project context, files,   |                                    |
              |                                        |   patterns, run_memory)            |                                    |
              |                                        +-----------------+-------------------+                                    |
              |                                                          |                                                        |
              |                                                          v                                                        |
              |                    +---------------------------------------------------------------------+                        |
              |                    |                       execute_task                                  |                        |
              |                    |                                                                     |                        |
              |                    |   +-------------------------------------------------------------+   |                        |
              |                    |   |              OrchestratorAgent.route()                      |   |                        |
              |                    |   |     (LLM analyzes task -> selects specialist)              |   |                        |
              |                    |   +------------------------+--------------------------------+   |                        |
              |                    |                          |                                      |                            |
              |                    |         +----------------+----------------+                     |                            |
              |                    |         v                v                v                     |                            |
              |                    |   +----------+    +----------+    +----------+                  |                            |
              |                    |   |  CODER   |    |TESTWRITER|    |  TESTER  |                  |                            |
              |                    |   +----------+    +----------+    +----------+                  |                            |
              |                    |         v                v                v                     |                            |
              |                    |   +----------+    +----------+    +----------+                  |                            |
              |                    |   | DEBUGGER |    | REVIEWER |    |RESEARCHER|                  |                            |
              |                    |   +----------+    +----------+    +----------+                  |                            |
              |                    |                          |                                      |                            |
              |                    |                          v                                      |                            |
              |                    |              spawn_for_task() -> SubAgentResult                 |                            |
              |                    +--------------------------+--------------------------------------+                            |
              |                                               |                                                                   |
              |                              +----------------+----------------+                                                  |
              |                              |                                 |                                                  |
              |                           ERROR                             SUCCESS                                               |
              |                              |                                 |                                                  |
              |                              v                                 v                                                  |
              |                    +-----------------+              +---------------------+                                       |
              |                    | handle_failure  |              |   validate_code     |                                       |
              |                    |                 |              | (syntax, lint, type)|                                       |
              |                    +--------+--------+              +----------+----------+                                       |
              |                             |                                  |                                                  |
              |                             v                    +-------------+-----------+                                      |
              |              +-------------------------+         |                         |                                      |
              |              |   analyze_failure       |       VALID                    INVALID                                   |
              |              | (Categorize: env, code, |         |                         |                                      |
              |              |  test, timeout, etc.)   |         |                         v                                      |
              |              +------------+------------+         |              +---------------------+                           |
              |                           |                      |              |    repair_code      |<-----+                    |
              |           +---------------+---------------+      |              |  (Fix lint/syntax)  |      |                    |
              |           |               |               |      |              +----------+----------+      |                    |
              |           v               v               v      |                         |                 |                    |
              |     +----------+   +----------+   +----------+   |            +------------+--------+        |                    |
              |     |RETRYABLE |   |  FATAL   |   | REPLAN   |   |            |                     |        |                    |
              |     +----+-----+   +----+-----+   +----+-----+   |          FIXED               STILL BAD    |                    |
              |          |              |              |         |            |                     |        |                    |
              |          |              v              |         |            |      repair_attempts < 2?    |                    |
              |          |    +-----------------+     |         |            |           |         |        |                    |
              |          |    |   decide_next   |     |         |            |          YES        NO       |                    |
              |          |    |  (action=stop)  |     |         |            |           |         |        |                    |
              |          |    +--------+--------+     |         |            |           +---------+--------+                    |
              |          |             |              |         |            |                     |                             |
              |          |             v              |         |            |                     v                             |
              |          |         END (fail)        |         |            |           +-----------------+                      |
              |          |                            |         |            |           |  handle_failure |                      |
              |          |                            |         |            |           +--------+--------+                      |
              |          |                            v         |            |                    |                              |
              |          |              +---------------------+ |            |                    v                              |
              |          |              |    replan_task      | |            |             decide_next                           |
              |          |              |  (Generate new      | |            |                                                   |
              |          |              |   approach)         | |            |                                                   |
              |          |              +----------+----------+ |            |                                                   |
              |          |                         |            |            |                                                   |
              |          |         +---------------+------+     |            |                                                   |
              |          |         |                      |     |            |                                                   |
              |          |    NEW PLAN              GIVE UP     |            |                                                   |
              |          |         |                      |     |            |                                                   |
              |          |         |                      v     |            |                                                   |
              |          |         |            +-------------+ |            |                                                   |
              |          |         |            | decide_next | |            |                                                   |
              |          |         |            |(action=skip)| |            |                                                   |
              |          |         |            +------+------+ |            |                                                   |
              |          |         |                   |        |            |                                                   |
              |          |         v                   |        |            |                                                   |
              |          |   build_context <----------+        |            |                                                   |
              |          |         |                            |            |                                                   |
              |          +---------+----------------------------+            |                                                   |
              |                    |                                         |                                                   |
              |                    |                                         v                                                   |
              |                    |                              +---------------------+                                        |
              |                    |                              |    write_files      |                                        |
              |                    |                              | (Save to disk)      |                                        |
              |                    |                              +----------+----------+                                        |
              |                    |                                         |                                                   |
              |                    |                                         v                                                   |
              |                    |                              +---------------------+                                        |
              |                    |                              |    verify_task      |                                        |
              |                    |                              |  (Run tests, check  |                                        |
              |                    |                              |   assertions)       |                                        |
              |                    |                              +----------+----------+                                        |
              |                    |                                         |                                                   |
              |                    |                    +---------------------+---------------------+                            |
              |                    |                    |                                           |                            |
              |                    |                 PASSED                                      FAILED                          |
              |                    |                    |                                           |                            |
              |                    |                    |                                           v                            |
              |                    |                    |                              +---------------------+                   |
              |                    |                    |                              |   repair_test       |<----+             |
              |                    |                    |                              | (Fix failing test)  |     |             |
              |                    |                    |                              +----------+----------+     |             |
              |                    |                    |                                         |                |             |
              |                    |                    |                        +----------------+--------+       |             |
              |                    |                    |                        |                         |       |             |
              |                    |                    |                      FIXED                    STILL FAILS|             |
              |                    |                    |                        |                         |       |             |
              |                    |                    |                        |              repair_attempts < 2?|             |
              |                    |                    |                        |                    |    |       |             |
              |                    |                    |                        |                   YES   NO      |             |
              |                    |                    |                        |                    |    |       |             |
              |                    |                    |                        |                    +----+-------+             |
              |                    |                    |                        |                         |                     |
              |                    |                    |                        |                         v                     |
              |                    |                    |                        |              +-----------------+              |
              |                    |                    |                        |              |  replan_task    |              |
              |                    |                    |<-----------------------+              +--------+--------+              |
              |                    |                    |                                                |                       |
              |                    |                    v                                   +------------+------------+          |
              |                    |         +---------------------+                        |                         |          |
              |                    |         |    checkpoint       |                   NEW PLAN                   GIVE UP        |
              |                    |         |  (Git commit,       |                        |                         |          |
              |                    |         |   mark complete)    |                        v                         v          |
              |                    |         +----------+----------+                  build_context            decide_next       |
              |                    |                    |                                                      (skip task)       |
              |                    |                    v                                                           |            |
              |                    |         +---------------------+                                                |            |
              |                    |         |    decide_next      |<-----------------------------------------------+            |
              |                    |         |                     |                                                             |
              |                    |         | action = ?          |                                                             |
              |                    |         +----------+----------+                                                             |
              |                    |                    |                                                                        |
              |                    |    +---------------+---------------+-----------------+                                      |
              |                    |    |               |               |                 |                                      |
              |                    | CONTINUE         SKIP           STOP            REPLAN                                      |
              |                    |    |               |               |                 |                                      |
              |                    |    |               |               v                 |                                      |
              |                    |    |               |          +---------+            |                                      |
              |                    |    |               |          |   END   |            |                                      |
              |                    |    |               |          +---------+            |                                      |
              |                    |    |               |                                 |                                      |
              +--------------------+---+---------------+---------------------------------+--------------------------------------+
```

---

## Key Loops and Branches

The executor contains several important cycles and conditional branches:

### 1. Main Task Loop
The primary execution cycle that processes tasks sequentially:

```
pick_task -> execute -> verify -> checkpoint -> decide_next -> pick_task
```

### 2. Validation Repair Loop (max 2 iterations)
Pre-write validation catches syntax/lint errors before committing:

```
validate_code -> [INVALID] -> repair_code -> validate_code
```

### 3. Test Repair Loop (max 2 iterations)
Post-write verification catches failing tests:

```
verify_task -> [FAILED] -> repair_test -> verify_task
```

### 4. Replan Loop
When repairs fail, generates alternative approaches:

```
verify_task -> [FAILED] -> replan_task -> build_context -> execute_task
handle_failure -> analyze_failure -> replan_task -> build_context
```

### 5. Failure Recovery Branches
Different failure types route to different recovery strategies:

```
execute_task -> [ERROR] -> handle_failure -> analyze_failure
                                              |
                         +--------------------+--------------------+
                         v                    v                    v
                    RETRYABLE             FATAL               NEEDS_REPLAN
                         |                    |                    |
                         v                    v                    v
                   execute_task          END (fail)          replan_task
```

---

## Agent Routing System

The executor uses an **OrchestratorAgent** (LLM-based router) to select the appropriate specialist agent for each task:

```
+-----------------------------------------------------------------+
|                      OrchestratorAgent                          |
|     LLM-based semantic routing (gpt-4o-mini / fast model)       |
|                                                                 |
|  Analyzes task title + context -> Routes to specialist agent    |
+-----------------------------------------------------------------+
                            |
       +--------------------+--------------------+
       |                    |                    |
       v                    v                    v
+-------------+     +-------------+     +-------------+
|   CODER     |     | TESTWRITER  |     |   TESTER    |
| claude-4    |     | claude-4    |     | claude-4    |
|             |     |             |     |             |
| Implements  |     | Creates     |     | Runs        |
| features    |     | test files  |     | tests       |
+-------------+     +-------------+     +-------------+

+-------------+     +-------------+     +-------------+
|  DEBUGGER   |     |  REVIEWER   |     | RESEARCHER  |
| claude-4    |     | claude-4    |     | claude-4    |
|             |     |             |     |             |
| Fixes bugs  |     | Reviews &   |     | Gathers     |
| & errors    |     | refactors   |     | information |
+-------------+     +-------------+     +-------------+
```

### SubAgentType Enum

| Type | Purpose | Routing Signals |
|------|---------|-----------------|
| `CODER` | Application code | "Create module", "Implement feature", "Add function" |
| `TESTWRITER` | Test files | "Create tests", "Write tests", "Add test coverage" |
| `TESTER` | Run tests | "Run tests", "Execute tests", "Verify tests pass" |
| `DEBUGGER` | Fix failures | "Fix", "Debug", "Resolve error", "Bug" |
| `REVIEWER` | Code review | "Review", "Refactor", "Optimize", "Improve" |
| `RESEARCHER` | Research/docs | "Research", "Find out", "Look up", "How to" |

---

## Graph Nodes

### Node Definitions

| Node | Purpose | Input | Output | Timeout |
|------|---------|-------|--------|---------|
| `parse_roadmap` | Parse ROADMAP.md to TodoItems | `roadmap_path` | `todos` | 30s |
| `pick_task` | Select next pending task | `todos`, `completed_todos` | `current_task` | 5s |
| `plan_task` | Break complex tasks into steps | `current_task` | `task_plan` | 60s |
| `build_context` | Gather project context | `current_task`, `run_memory` | `context`, `prompt` | 60s |
| `execute_task` | Route to agent, execute | `prompt`, `context` | `agent_result`, `files_modified` | 300s |
| `validate_code` | Syntax/lint/type check | `files_modified` | `valid`, `errors` | 60s |
| `repair_code` | Fix validation errors | `errors` | `repaired_code` | 120s |
| `write_files` | Save files to disk | `agent_result` | `files_written` | 30s |
| `verify_task` | Run tests, assertions | `files_written` | `verified`, `test_results` | 120s |
| `repair_test` | Fix failing tests | `test_results` | `repaired_code` | 120s |
| `checkpoint` | Git commit, sync ROADMAP | `verified=True` | `checkpoint_sha` | 30s |
| `handle_failure` | Process errors | `error` | `failure_category` | 5s |
| `analyze_failure` | Categorize failure type | `error` | `recovery_action` | 10s |
| `replan_task` | Generate new approach | `failure_context` | `new_plan` | 60s |
| `decide_next` | Continue/skip/stop | `state` | `action` | 5s |

---

## Edge Routing Functions

| Route Function | From Node | Conditions | Targets |
|----------------|-----------|------------|---------|
| `route_after_pick` | pick_task | task exists? | plan_task / build_context / END |
| `route_after_execute` | execute_task | error? | verify_task / handle_failure |
| `route_after_validate` | validate_code | valid? repair count? | write_files / repair_code / handle_failure |
| `route_after_verify` | verify_task | passed? repair count? | checkpoint / repair_test / replan_task |
| `route_after_repair_test` | repair_test | fixed? | verify_task / replan_task |
| `route_after_replan` | replan_task | has new plan? | build_context / decide_next |
| `route_after_analyze_failure` | analyze_failure | category? | decide_next / replan_task |
| `route_after_decide` | decide_next | action? | pick_task / END |
| `route_after_write` | write_files | - | verify_task |

---

## State Schema

### ExecutorGraphState

```python
class ExecutorGraphState(TypedDict, total=False):
    # Task Management
    current_task: TodoItem | None
    tasks_completed_count: int
    tasks_failed_count: int
    tasks_skipped_count: int

    # Execution Context
    workspace: str  # Path
    prompt: str
    response: str
    files_created: list[str]
    files_modified: list[str]

    # Verification
    verified: bool
    repair_attempts: int

    # Error Handling
    error: str | None
    failure_category: str | None

    # Subagent Tracking
    agent_type: SubAgentType
    agent_metrics: dict

    # Memory & Context
    run_memory: RunMemory
    project_context: ProjectContext
```

---

## Task State Transitions

```
                    +-------------+
                    |   PENDING   |
                    +------+------+
                           | pick
                           v
                    +-------------+
        +----------|   RUNNING   |----------+
        |          +------+------+          |
        |                 |                 |
     error             verify           timeout
        |                 |                 |
        v                 v                 v
 +-------------+   +-------------+   +-------------+
 |   FAILED    |   |  COMPLETED  |   |   SKIPPED   |
 +-------------+   +-------------+   +-------------+
        |                                   ^
        |         replan                    |
        +-----------------------------------+
```

---

## Data Flow

```
ROADMAP.md
    |
    v
+-----------------+
| RoadmapParser   | -> Roadmap (phases, tasks)
+--------+--------+
         |
         v
+-----------------+
| TodoListManager | -> TodoItem[] (with dependencies)
+--------+--------+
         |
         v
+-----------------+
|ExecutionContext | -> workspace, files, patterns
+--------+--------+
         |
         v
+-----------------+
|OrchestratorAgent| -> RoutingDecision (agent_type)
+--------+--------+
         |
         v
+-----------------+
| SubAgent.execute| -> SubAgentResult (files, output)
+--------+--------+
         |
         v
+-----------------+
| TaskVerifier    | -> verified: bool
+--------+--------+
         |
         v
+-----------------+
| Checkpointer    | -> Git commit
+-----------------+
```

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/ai_infra/executor/graph.py` | Main ExecutorGraph class |
| `src/ai_infra/executor/state.py` | ExecutorGraphState TypedDict |
| `src/ai_infra/executor/agents/orchestrator.py` | LLM-based task routing |
| `src/ai_infra/executor/agents/registry.py` | SubAgentRegistry & SubAgentType |
| `src/ai_infra/executor/agents/spawner.py` | spawn_for_task() function |
| `src/ai_infra/executor/agents/base.py` | SubAgent base class |
| `src/ai_infra/executor/agents/coder.py` | CoderAgent implementation |
| `src/ai_infra/executor/agents/testwriter.py` | TestWriterAgent implementation |
| `src/ai_infra/executor/agents/tester.py` | TesterAgent implementation |
| `src/ai_infra/executor/agents/debugger.py` | DebuggerAgent implementation |
| `src/ai_infra/executor/agents/reviewer.py` | ReviewerAgent implementation |
| `src/ai_infra/executor/agents/researcher.py` | ResearcherAgent implementation |
| `src/ai_infra/executor/nodes/execute.py` | Task execution node |
| `src/ai_infra/executor/nodes/verify.py` | Task verification node |
| `src/ai_infra/executor/nodes/repair.py` | Code repair node |
| `src/ai_infra/executor/edges/routes.py` | Graph edge routing functions |

---

## Execution Flow Summary

1. **Parse** - Load ROADMAP.md -> TodoItems
2. **Pick** - Select next pending task
3. **Route** - OrchestratorAgent determines specialist (CODER/TESTWRITER/etc.)
4. **Execute** - Spawn specialist agent, run task
5. **Validate** - Check syntax, lint errors (repair loop: max 2)
6. **Write** - Save files to disk
7. **Verify** - Run tests, check assertions (repair loop: max 2)
8. **Checkpoint** - Git commit on success
9. **Decide** - Continue to next task, skip, or stop
10. **Loop** - Back to Pick for next task

---

## Configuration Options

```python
ExecutorGraph(
    agent=Agent(...),                    # Base agent for execution
    roadmap_path="ROADMAP.md",           # Path to roadmap file
    checkpointer=Checkpointer(...),      # Git checkpointer
    verifier=TaskVerifier(...),          # Verification component
    project_context=ProjectContext(...), # Project analyzer
    run_memory=RunMemory(...),           # Cross-task memory
    todo_manager=TodoListManager(...),   # Task management
    callbacks=ExecutorCallbacks(...),    # Event callbacks
    check_level=CheckLevel.STRICT,       # Verification strictness
    use_llm_normalization=True,          # Normalize LLM outputs
    sync_roadmap=True,                   # Update ROADMAP checkboxes
    enable_planning=False,               # Enable task planning
    dry_run=False,                       # Preview without executing
    max_tasks=0,                         # Limit tasks (0 = unlimited)
    recursion_limit=100,                 # Max graph transitions
)
```

---

## Related Documentation

- [Task Routing](task-routing.md) - Detailed routing documentation
- [Streaming Consolidation](streaming-consolidation.md) - Event streaming
- [Shell Integration](../llm/shell.md) - Shell tool usage
