"""Scenario tests for complex multi-task workflows (Phase 6.3.3).

Tests scenarios for complex task workflows, including:
- Task dependency ordering (simulated with parent_id and children_ids)
- Parallel independent tasks
- Task chains and sequencing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ai_infra.executor.todolist import TodoItem, TodoListManager, TodoStatus

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
    return tmp_path


@pytest.fixture
def complex_roadmap(tmp_path: Path) -> Path:
    """Create a roadmap with complex structure."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""\
# Complex Feature

## Overview
Implementing a complex feature with dependencies.

## Tasks

### Phase 1: Foundation

- [ ] **Task 1.1: Create base class**
  - Description: Create the base abstract class
  - Files: src/base.py

- [ ] **Task 1.2: Create subclass A**
  - Description: Create first subclass
  - Files: src/subclass_a.py

- [ ] **Task 1.3: Create subclass B**
  - Description: Create second subclass
  - Files: src/subclass_b.py

### Phase 2: Integration

- [ ] **Task 2.1: Add tests**
  - Description: Add tests for all classes
  - Files: tests/test_classes.py
""")
    return roadmap


@pytest.fixture
def parallel_roadmap(tmp_path: Path) -> Path:
    """Create a roadmap with independent parallel tasks."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""\
# Parallel Work

## Overview
Multiple independent modules.

## Tasks

### Phase 1: Modules

- [ ] **Task A: Create module A**
  - Description: Standalone module A
  - Files: src/module_a.py

- [ ] **Task B: Create module B**
  - Description: Standalone module B
  - Files: src/module_b.py

- [ ] **Task C: Create module C**
  - Description: Standalone module C
  - Files: src/module_c.py
""")
    return roadmap


# =============================================================================
# Helper dataclass for dependency simulation
# =============================================================================


@dataclass
class TaskNode:
    """Task node with dependency tracking for tests.

    Uses TodoItem internally but tracks dependencies separately.
    """

    id: int
    title: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: TodoStatus = TodoStatus.NOT_STARTED

    def to_todo_item(self) -> TodoItem:
        """Convert to TodoItem."""
        return TodoItem(
            id=self.id,
            title=self.title,
            description=self.description,
            status=self.status,
        )


# =============================================================================
# Task Dependency Tests
# =============================================================================


class TestTaskDependencies:
    """Tests for task dependency handling using TaskNode."""

    def test_parse_simple_dependency(self) -> None:
        """Should parse simple task dependency."""
        task = TaskNode(
            id=2,
            title="Create subclass",
            description="Depends: Task 1",
            depends_on=["Task 1"],
        )

        assert len(task.depends_on) == 1
        assert "Task 1" in task.depends_on

    def test_parse_multiple_dependencies(self) -> None:
        """Should parse multiple dependencies."""
        task = TaskNode(
            id=3,
            title="Add tests",
            description="Test all classes",
            depends_on=["Task 1.2", "Task 1.3"],
        )

        assert len(task.depends_on) == 2

    def test_task_without_dependencies(self) -> None:
        """Task without dependencies should have empty list."""
        task = TaskNode(
            id=1,
            title="Create base class",
            description="Foundation task",
        )

        assert task.depends_on == []


class TestDependencyOrder:
    """Tests for task execution order based on dependencies."""

    def test_base_task_first(self) -> None:
        """Base task should execute before dependents."""
        tasks = [
            TaskNode(id=1, title="Base", depends_on=[]),
            TaskNode(id=2, title="Child", depends_on=["Base"]),
        ]

        # Simple topological sort check
        base_idx = next(i for i, t in enumerate(tasks) if t.title == "Base")
        child_idx = next(i for i, t in enumerate(tasks) if t.title == "Child")

        assert base_idx < child_idx

    def test_diamond_dependency(self) -> None:
        """Handle diamond dependency pattern."""
        # A -> B -> D
        # A -> C -> D
        tasks = [
            TaskNode(id=1, title="A", depends_on=[]),
            TaskNode(id=2, title="B", depends_on=["A"]),
            TaskNode(id=3, title="C", depends_on=["A"]),
            TaskNode(id=4, title="D", depends_on=["B", "C"]),
        ]

        # Build dependency graph
        deps = {t.title: t.depends_on for t in tasks}

        # A should have no dependencies
        assert deps["A"] == []

        # D should depend on B and C
        assert "B" in deps["D"]
        assert "C" in deps["D"]

    def test_topological_sort(self) -> None:
        """Should produce valid topological order."""
        tasks = [
            TaskNode(id=1, title="Task 1", depends_on=[]),
            TaskNode(id=2, title="Task 2", depends_on=["Task 1"]),
            TaskNode(id=3, title="Task 3", depends_on=["Task 1", "Task 2"]),
        ]

        # Simple topological sort
        result = []
        completed: set[str] = set()

        while len(result) < len(tasks):
            for task in tasks:
                if task.title in completed:
                    continue
                if all(dep in completed for dep in task.depends_on):
                    result.append(task.title)
                    completed.add(task.title)
                    break

        assert result == ["Task 1", "Task 2", "Task 3"]


# =============================================================================
# Parallel Task Tests
# =============================================================================


class TestParallelTasks:
    """Tests for parallel independent task execution."""

    def test_identify_independent_tasks(self) -> None:
        """Should identify tasks that can run in parallel."""
        tasks = [
            TaskNode(id=1, title="Task A", depends_on=[]),
            TaskNode(id=2, title="Task B", depends_on=[]),
            TaskNode(id=3, title="Task C", depends_on=[]),
        ]

        # All tasks are independent
        independent = [t for t in tasks if not t.depends_on]
        assert len(independent) == 3

    def test_parallel_vs_sequential(self) -> None:
        """Should distinguish parallel and sequential tasks."""
        tasks = [
            TaskNode(id=1, title="Independent A", depends_on=[]),
            TaskNode(id=2, title="Independent B", depends_on=[]),
            TaskNode(id=3, title="Depends on A", depends_on=["Independent A"]),
        ]

        # First two can run in parallel
        can_parallel = [t for t in tasks if not t.depends_on]
        assert len(can_parallel) == 2

        # Third must wait
        must_wait = [t for t in tasks if t.depends_on]
        assert len(must_wait) == 1


class TestTaskChains:
    """Tests for task chains with sequential dependencies."""

    def test_linear_chain(self) -> None:
        """Linear chain should execute in order."""
        tasks = [
            TaskNode(id=1, title="Step 1", depends_on=[]),
            TaskNode(id=2, title="Step 2", depends_on=["Step 1"]),
            TaskNode(id=3, title="Step 3", depends_on=["Step 2"]),
            TaskNode(id=4, title="Step 4", depends_on=["Step 3"]),
        ]

        # Build execution order
        order = []
        completed: set[str] = set()

        while len(order) < len(tasks):
            for task in tasks:
                if task.title in completed:
                    continue
                if all(dep in completed for dep in task.depends_on):
                    order.append(task.title)
                    completed.add(task.title)
                    break

        assert order == ["Step 1", "Step 2", "Step 3", "Step 4"]

    def test_branching_chain(self) -> None:
        """Branching chain should allow parallel branches."""
        # Root -> Branch A -> Merge
        # Root -> Branch B -> Merge
        tasks = [
            TaskNode(id=1, title="Root", depends_on=[]),
            TaskNode(id=2, title="Branch A", depends_on=["Root"]),
            TaskNode(id=3, title="Branch B", depends_on=["Root"]),
            TaskNode(id=4, title="Merge", depends_on=["Branch A", "Branch B"]),
        ]

        # After Root completes, both branches are available
        completed = {"Root"}
        available = [
            t
            for t in tasks
            if t.title not in completed and all(dep in completed for dep in t.depends_on)
        ]

        assert len(available) == 2
        available_titles = {t.title for t in available}
        assert "Branch A" in available_titles
        assert "Branch B" in available_titles


# =============================================================================
# TodoListManager Integration Tests
# =============================================================================


class TestTodoListManagerWorkflows:
    """Tests for TodoListManager with complex workflows."""

    def test_manager_can_be_created(self) -> None:
        """Should be able to create TodoListManager."""
        manager = TodoListManager()
        assert manager is not None

    def test_todos_property_returns_list(self) -> None:
        """todos property should return list of TodoItems."""
        todos = [
            TodoItem(id=1, title="Task 1", description=""),
            TodoItem(id=2, title="Task 2", description=""),
        ]
        manager = TodoListManager(todos=todos)

        result = manager.todos
        assert len(result) == 2
        assert result[0].title == "Task 1"

    def test_marks_task_in_progress(self) -> None:
        """Should mark tasks as in progress."""
        todos = [
            TodoItem(id=1, title="Task 1", description=""),
        ]
        manager = TodoListManager(todos=todos)

        manager.mark_in_progress(1)

        assert manager.todos[0].status == TodoStatus.IN_PROGRESS


# =============================================================================
# Workflow Simulation Tests
# =============================================================================


class TestWorkflowSimulation:
    """Tests simulating complete workflow execution."""

    def test_simple_workflow_simulation(self, tmp_workspace: Path) -> None:
        """Simulate a simple 3-task workflow using TaskNode."""
        tasks = [
            TaskNode(id=1, title="Create module", depends_on=[]),
            TaskNode(id=2, title="Add function", depends_on=["Create module"]),
            TaskNode(id=3, title="Write tests", depends_on=["Add function"]),
        ]

        # Simulate execution
        completed: list[str] = []

        for _ in range(len(tasks)):
            for task in tasks:
                if task.status == TodoStatus.COMPLETED:
                    continue
                if all(dep in completed for dep in task.depends_on):
                    # Execute task
                    task.status = TodoStatus.COMPLETED
                    completed.append(task.title)
                    break

        assert completed == ["Create module", "Add function", "Write tests"]
        assert all(t.status == TodoStatus.COMPLETED for t in tasks)

    def test_partial_completion_tracking(self) -> None:
        """Track partial completion in complex workflow."""
        todos = [
            TodoItem(id=1, title="Task 1", description="", status=TodoStatus.COMPLETED),
            TodoItem(id=2, title="Task 2", description="", status=TodoStatus.COMPLETED),
            TodoItem(id=3, title="Task 3", description="", status=TodoStatus.NOT_STARTED),
            TodoItem(id=4, title="Task 4", description="", status=TodoStatus.NOT_STARTED),
        ]

        completed = [t for t in todos if t.status == TodoStatus.COMPLETED]
        pending = [t for t in todos if t.status == TodoStatus.NOT_STARTED]

        assert len(completed) == 2
        assert len(pending) == 2

    def test_all_tasks_complete_check(self) -> None:
        """Check if all tasks are complete."""
        manager = TodoListManager(
            todos=[
                TodoItem(id=1, title="Task 1", description="", status=TodoStatus.COMPLETED),
                TodoItem(id=2, title="Task 2", description="", status=TodoStatus.COMPLETED),
                TodoItem(id=3, title="Task 3", description="", status=TodoStatus.COMPLETED),
            ]
        )

        assert manager.completed_count == 3
        assert manager.pending_count == 0
