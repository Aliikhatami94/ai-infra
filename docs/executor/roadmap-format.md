# ROADMAP Format Specification

> Machine-readable format for autonomous task execution by the Executor module.

---

## Overview

This specification defines how `ROADMAP.md` files should be structured so the Executor can:
1. Parse tasks into structured `Task` objects
2. Track execution state (pending, in-progress, completed, failed)
3. Extract file hints for context building
4. Understand task dependencies and ordering

The format is designed to be:
- **Human-readable**: Standard Markdown that looks good in GitHub/GitLab
- **Machine-parseable**: Structured enough for reliable extraction
- **Backwards-compatible**: Existing ROADMAPs can be incrementally adopted

---

## Document Structure

A ROADMAP file consists of:

```
# Project Title (optional)

Introduction text (ignored by parser)

## Phase N: Phase Name

> **Goal**: What this phase achieves
> **Priority**: HIGH | MEDIUM | LOW  
> **Effort**: Time estimate
> **Prerequisite**: Requirements (optional)

Phase description (becomes phase.description)

### N.M Section Title

**Files**: `path/to/file.py`, `path/to/other.py`

Section description text.

- [ ] **Task title**
  Task description and context.
  ```python
  # Code examples
  ```
  - [ ] Sub-task 1
  - [ ] Sub-task 2

- [x] **Completed task**
```

---

## Elements Reference

### Phase Header

```markdown
## Phase N: Name
```

**Rules**:
- Must start with `## Phase ` (case-insensitive)
- `N` can be any identifier: `0`, `1`, `2`, `0.1`, `A`, etc.
- Name is everything after the colon
- Resets section counter for this phase

**Examples**:
```markdown
## Phase 0: Foundation
## Phase 1: Core Features
## Phase 2.5: Bug Fixes
## Phase A: Experimental
```

**Parsed as**:
```python
Phase(
    id="0",
    name="Foundation",
    ...
)
```

---

### Phase Metadata Block

```markdown
> **Goal**: What this phase achieves
> **Priority**: HIGH | MEDIUM | LOW
> **Effort**: Time estimate
> **Prerequisite**: Requirements
```

**Rules**:
- Must immediately follow phase header
- Uses blockquote syntax (`>`)
- Each line is `**Key**: Value` format
- All fields are optional

**Recognized Fields**:
| Field | Values | Default |
|-------|--------|---------|
| `Goal` | Free text | Empty |
| `Priority` | `HIGH`, `MEDIUM`, `LOW` | `MEDIUM` |
| `Effort` | Free text (e.g., "1 week") | Empty |
| `Prerequisite` | Free text or phase reference | Empty |

**Examples**:
```markdown
> **Goal**: Implement core parsing logic
> **Priority**: HIGH
> **Effort**: 3 days
> **Prerequisite**: Phase 0 complete
```

---

### Section Header

```markdown
### N.M Section Title
```

**Rules**:
- Must start with `### `
- `N.M` should match parent phase (not enforced)
- Title is everything after the identifier
- Creates a new section context

**Examples**:
```markdown
### 1.1 Parser Implementation
### 1.2 State Management
### 2.1.1 Nested Section
```

**Parsed as**:
```python
Section(
    id="1.1",
    title="Parser Implementation",
    phase_id="1",
    ...
)
```

---

### File Hints

```markdown
**Files**: `path/to/file.py`, `path/to/other.py`
```

**Rules**:
- Must be on its own line
- Files enclosed in backticks
- Multiple files separated by commas
- Paths are relative to ROADMAP location
- Applies to all tasks in the section until next `**Files**:`

**Alternative Syntax**:
```markdown
**File**: `single/file.py`
```

**Examples**:
```markdown
**Files**: `src/ai_infra/executor/parser.py`, `tests/executor/test_parser.py`
**File**: `src/main.py`
```

---

### Task Checkbox

```markdown
- [ ] **Task title**
```

**Rules**:
- Must start with `- [ ]` or `- [x]` (also `* [ ]`, `* [x]`)
- Title should be bold (`**title**`)
- `[ ]` = pending, `[x]` = completed
- Everything until the next task or section becomes task context

**Status Mapping**:
| Checkbox | TaskStatus |
|----------|------------|
| `- [ ]` | `PENDING` |
| `- [x]` | `COMPLETED` |

**Examples**:
```markdown
- [ ] **Implement RoadmapParser class**
- [x] **Define data models**
- [ ] **Add error handling**
```

**Parsed as**:
```python
Task(
    id="1.1.1",  # Auto-generated: phase.section.task_num
    title="Implement RoadmapParser class",
    status=TaskStatus.PENDING,
    ...
)
```

---

### Task Context

Everything between a task checkbox and the next task/section becomes context:

```markdown
- [ ] **Task title**
  Description paragraph becomes task.description.

  Additional paragraphs also included.

  ```python
  # Code blocks become task.code_context
  def example():
      pass
  ```

  - [ ] Sub-task 1  <!-- Becomes task.subtasks -->
  - [ ] Sub-task 2

  More description after subtasks.
```

**Rules**:
- Leading whitespace is trimmed
- Code blocks are preserved with language info
- Nested checkboxes become subtasks (rolled into context)
- Context ends at next `- [ ]`, `###`, or `##`

---

### Sub-tasks

```markdown
- [ ] **Parent task**
  - [ ] Sub-task 1
  - [ ] Sub-task 2
  - [x] Completed sub-task
```

**Rules**:
- Nested under parent task (2+ space indent)
- Become `task.subtasks` list
- Also merged into `task.context` for agent
- Sub-task completion tracked separately

**Parsed as**:
```python
Task(
    id="1.1.1",
    title="Parent task",
    subtasks=[
        Subtask(id="1.1.1.1", title="Sub-task 1", completed=False),
        Subtask(id="1.1.1.2", title="Sub-task 2", completed=False),
        Subtask(id="1.1.1.3", title="Completed sub-task", completed=True),
    ],
    context="...\n- [ ] Sub-task 1\n- [ ] Sub-task 2\n..."
)
```

---

## Task ID Generation

Task IDs are auto-generated based on position:

```
{phase_id}.{section_num}.{task_num}
```

**Examples**:
| Location | Generated ID |
|----------|--------------|
| Phase 1, Section 1, Task 1 | `1.1.1` |
| Phase 1, Section 1, Task 2 | `1.1.2` |
| Phase 1, Section 2, Task 1 | `1.2.1` |
| Phase 2, Section 1, Task 1 | `2.1.1` |
| Phase 0.5, Section 1, Task 3 | `0.5.1.3` |

**Sub-task IDs**:
```
{task_id}.{subtask_num}
```

Example: `1.1.1.1`, `1.1.1.2`

---

## Complete Example

```markdown
# Project ROADMAP

This document tracks development progress.

---

## Phase 0: Foundation

> **Goal**: Establish core infrastructure
> **Priority**: HIGH
> **Effort**: 1 week

Build the foundational components needed for the project.

### 0.1 Project Setup

**Files**: `pyproject.toml`, `src/__init__.py`

Initial project configuration.

- [x] **Initialize Poetry project**
  Set up pyproject.toml with dependencies.

- [x] **Configure linting**
  Set up ruff, mypy, and pre-commit hooks.

### 0.2 Core Models

**Files**: `src/models.py`

- [ ] **Define data models**
  Create Pydantic models for core entities.

  ```python
  from pydantic import BaseModel

  class User(BaseModel):
      id: str
      name: str
  ```

  - [ ] Add User model
  - [ ] Add Project model
  - [ ] Add Task model

- [ ] **Add serialization helpers**
  JSON/YAML serialization utilities.

---

## Phase 1: Core Features

> **Goal**: Implement main functionality
> **Priority**: HIGH
> **Effort**: 2 weeks
> **Prerequisite**: Phase 0 complete

### 1.1 API Implementation

**Files**: `src/api.py`, `src/routes/`

- [ ] **Create FastAPI app**
  Initialize the application with proper configuration.

- [ ] **Implement CRUD endpoints**
  - [ ] GET /users
  - [ ] POST /users
  - [ ] PUT /users/{id}
  - [ ] DELETE /users/{id}
```

---

## Parsing Behavior

### What Gets Extracted

| Element | Extracted To |
|---------|--------------|
| Phase header | `Phase.id`, `Phase.name` |
| Phase metadata | `Phase.goal`, `Phase.priority`, `Phase.effort` |
| Section header | `Section.id`, `Section.title` |
| File hints | `Task.file_hints` (inherited by tasks) |
| Task checkbox | `Task.id`, `Task.title`, `Task.status` |
| Task description | `Task.description` |
| Code blocks | `Task.code_context` |
| Sub-tasks | `Task.subtasks` |

### What Gets Ignored

- Top-level headings (`# Title`)
- Horizontal rules (`---`)
- HTML comments (`<!-- comment -->`)
- Regular paragraphs outside task context
- Images, links (preserved in context but not extracted)

### Graceful Degradation

The parser handles malformed input gracefully:

| Issue | Behavior |
|-------|----------|
| Missing phase header | Tasks go into "default" phase |
| Missing section header | Tasks go into "default" section |
| Malformed checkbox | Treated as regular list item (context) |
| Deeply nested lists | Flattened into context |
| Mixed checkbox styles | All recognized (`-`, `*`, `+`) |
| Unclosed code blocks | Best-effort extraction |

---

## State Tracking

The Executor maintains state in a sidecar file:

```
project/
├── ROADMAP.md
└── .executor/
    ├── state.json      # Task states, timestamps
    ├── failures.json   # Failure records
    └── checkpoints/    # Rollback points
```

### State File Format

```json
{
  "roadmap_path": "ROADMAP.md",
  "roadmap_hash": "sha256:abc123...",
  "last_updated": "2026-01-06T12:00:00Z",
  "tasks": {
    "1.1.1": {
      "status": "completed",
      "started_at": "2026-01-06T10:00:00Z",
      "completed_at": "2026-01-06T10:30:00Z",
      "files_modified": ["src/parser.py"],
      "agent_run_id": "run_abc123"
    },
    "1.1.2": {
      "status": "failed",
      "started_at": "2026-01-06T11:00:00Z",
      "failed_at": "2026-01-06T11:15:00Z",
      "error": "SyntaxError in generated code",
      "failure_category": "syntax_error",
      "attempts": 2
    }
  }
}
```

---

## Best Practices

### For ROADMAP Authors

1. **Use consistent formatting**: Stick to the patterns shown above
2. **Add file hints**: Help the agent find relevant code
3. **Keep tasks atomic**: One clear deliverable per task
4. **Include code examples**: Show expected patterns
5. **Order by dependency**: Tasks should be executable in order

### Task Granularity

| Too Large | Right Size | Too Small |
|-----------|------------|-----------|
| "Implement the API" | "Add GET /users endpoint" | "Add import statement" |
| "Write all tests" | "Test user creation flow" | "Test line 42" |
| "Refactor codebase" | "Extract auth into module" | "Rename variable x" |

### File Hints

```markdown
<!-- Good: Specific files -->
**Files**: `src/api/users.py`, `tests/api/test_users.py`

<!-- Okay: Directory hint -->
**Files**: `src/api/`

<!-- Avoid: Too broad -->
**Files**: `src/`
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-06 | Initial specification |
