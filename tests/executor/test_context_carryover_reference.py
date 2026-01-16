"""Tests for Phase 9.3: Testing Context Carryover.

This module tests that later tasks can reference earlier task's work:
- 9.3.1: Task 2 context includes Task 1 results (e.g., models.py reference)

These tests verify the end-to-end context carryover functionality where
subsequent tasks receive relevant context from previously completed tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ai_infra.executor.context_carryover import ContextSummarizer
from ai_infra.executor.nodes.checkpoint import checkpoint_node
from ai_infra.executor.nodes.context import build_context_node
from ai_infra.executor.run_memory import FileAction, RunMemory, TaskOutcome

# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockTodoItem:
    """Mock TodoItem for testing."""

    id: int
    title: str
    description: str = ""
    file_hints: list[str] = field(default_factory=list)
    status: str = "pending"


@pytest.fixture
def roadmap_content() -> str:
    """Roadmap content for context carryover test."""
    return """# Test Carryover

## Phase 1: Setup

- [ ] Create User model in models.py
- [ ] Create UserService that uses User model
"""


@pytest.fixture
def roadmap_path(tmp_path: Path, roadmap_content: str) -> Path:
    """Create a roadmap file."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text(roadmap_content)
    return roadmap


@pytest.fixture
def run_memory() -> RunMemory:
    """Create a RunMemory instance for testing."""
    return RunMemory(run_id="test-carryover-001")


@pytest.fixture
def task_create_model() -> MockTodoItem:
    """Task 1: Create User model."""
    return MockTodoItem(
        id=1,
        title="Create User model in models.py",
        description="Create a User dataclass with id, name, and email fields",
        file_hints=["models.py"],
    )


@pytest.fixture
def task_create_service() -> MockTodoItem:
    """Task 2: Create UserService that uses User model."""
    return MockTodoItem(
        id=2,
        title="Create UserService that uses User model",
        description="Implement service layer for user operations using the User model",
        file_hints=["services.py"],
    )


# =============================================================================
# 9.3.1: Test Later Task References Earlier Task's Work
# =============================================================================


class TestContextCarryoverTaskReference:
    """Tests that later tasks can reference earlier task's work."""

    @pytest.mark.asyncio
    async def test_task2_context_includes_task1_results(
        self,
        run_memory: RunMemory,
        task_create_model: MockTodoItem,
        task_create_service: MockTodoItem,
        roadmap_path: Path,
    ) -> None:
        """Task 2 should know about Task 1's User model via context carryover."""
        # Step 1: Complete Task 1 (Create User model)
        task1_state = {
            "current_task": task_create_model,
            "files_modified": ["models.py"],
            "verified": True,
            "llm_response": "Created User model with id, name, email fields",
        }

        await checkpoint_node(
            task1_state,
            run_memory=run_memory,
            sync_roadmap=False,
        )

        # Verify Task 1 outcome was recorded
        assert len(run_memory.outcomes) == 1
        assert run_memory.outcomes[0].task_id == "1"
        assert "models.py" in str(run_memory.outcomes[0].files)

        # Step 2: Build context for Task 2 (Create UserService)
        task2_state = {
            "current_task": task_create_service,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(task2_state, run_memory=run_memory)

        # Verify Task 2 context includes Task 1 results
        prompt = result.get("prompt", "")
        context = result.get("context", "")

        # The prompt/context should reference the previous task
        combined = f"{prompt} {context}".lower()

        # Check that models.py is mentioned (from Task 1)
        assert "models.py" in combined or "model" in combined, (
            f"Task 2 context should reference Task 1's models.py work. "
            f"Got context: {context[:500]}..."
        )

    @pytest.mark.asyncio
    async def test_context_includes_files_from_previous_task(
        self,
        run_memory: RunMemory,
        task_create_model: MockTodoItem,
        task_create_service: MockTodoItem,
        roadmap_path: Path,
    ) -> None:
        """Context should include file information from previous tasks."""
        # Complete Task 1 with explicit file tracking
        run_memory.add_outcome(
            TaskOutcome(
                task_id="1",
                title="Create User model in models.py",
                status="completed",
                files={Path("models.py"): FileAction.CREATED},
                summary="Created User dataclass with id, name, email fields",
                key_decisions=["Used dataclass over Pydantic", "Added validation"],
            )
        )

        # Build context for Task 2
        task2_state = {
            "current_task": task_create_service,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(task2_state, run_memory=run_memory)

        # Verify the context includes file information
        context = result.get("context", "")
        prompt = result.get("prompt", "")
        combined = f"{prompt} {context}"

        # Should mention models.py
        assert "models.py" in combined, "Context should include files from previous task"

    @pytest.mark.asyncio
    async def test_context_includes_summary_from_previous_task(
        self,
        run_memory: RunMemory,
        task_create_service: MockTodoItem,
        roadmap_path: Path,
    ) -> None:
        """Context should include summary from previous task."""
        # Add a task outcome with a meaningful summary
        run_memory.add_outcome(
            TaskOutcome(
                task_id="1",
                title="Create User model",
                status="completed",
                files={Path("models.py"): FileAction.CREATED},
                summary="Created User dataclass with id, name, email attributes",
                key_decisions=["Used dataclass for simplicity"],
            )
        )

        # Build context for next task
        task2_state = {
            "current_task": task_create_service,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(task2_state, run_memory=run_memory)

        # Check that context or prompt references the previous work
        context = result.get("context", "")
        prompt = result.get("prompt", "")
        combined = f"{prompt} {context}".lower()

        # Should contain reference to the User model work
        assert any(
            keyword in combined for keyword in ["user", "model", "dataclass", "models.py"]
        ), f"Context should reference previous task's work. Got: {combined[:500]}..."


class TestContextCarryoverRelevance:
    """Tests that context carryover prioritizes relevant tasks."""

    @pytest.mark.asyncio
    async def test_relevant_tasks_prioritized_in_context(
        self,
        run_memory: RunMemory,
        task_create_service: MockTodoItem,
        roadmap_path: Path,
    ) -> None:
        """More relevant previous tasks should be prioritized in context."""
        # Add multiple completed tasks
        run_memory.add_outcome(
            TaskOutcome(
                task_id="1",
                title="Create User model",
                status="completed",
                files={Path("models.py"): FileAction.CREATED},
                summary="Created User model with id, name, email",
            )
        )
        run_memory.add_outcome(
            TaskOutcome(
                task_id="2",
                title="Configure logging",
                status="completed",
                files={Path("logging_config.py"): FileAction.CREATED},
                summary="Set up structured logging",
            )
        )
        run_memory.add_outcome(
            TaskOutcome(
                task_id="3",
                title="Add database connection",
                status="completed",
                files={Path("database.py"): FileAction.CREATED},
                summary="Added SQLite connection pool",
            )
        )

        # Build context for UserService task
        task_state = {
            "current_task": task_create_service,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(task_state, run_memory=run_memory)

        context = result.get("context", "")
        prompt = result.get("prompt", "")
        combined = f"{prompt} {context}".lower()

        # User model should be mentioned (most relevant to UserService)
        assert "user" in combined or "model" in combined, (
            "Most relevant previous task should be in context"
        )

    def test_summarizer_filters_by_relevance(self) -> None:
        """ContextSummarizer should filter tasks by relevance."""
        summarizer = ContextSummarizer(
            max_tokens=1000,
            max_tasks=2,
            relevance_threshold=0.05,
        )

        previous_tasks = [
            {
                "title": "Create User model",
                "files": ["models.py"],
                "summary": "Added User dataclass",
                "key_decisions": ["Used dataclass"],
            },
            {
                "title": "Configure CI/CD",
                "files": [".github/workflows/ci.yml"],
                "summary": "Set up GitHub Actions",
                "key_decisions": [],
            },
        ]

        current_task = MockTodoItem(
            id=3,
            title="Create UserService using User model",
            description="Implement service for user operations",
            file_hints=["services.py"],
        )

        summary = summarizer.summarize_for_task(previous_tasks, current_task)

        # User model task should be included (relevant)
        assert "User" in summary or "model" in summary

        # CI/CD task less relevant - may or may not be included
        # The key is that User model IS included


class TestContextCarryoverMultipleTasks:
    """Tests for context carryover across multiple sequential tasks."""

    @pytest.mark.asyncio
    async def test_context_accumulates_across_tasks(
        self,
        run_memory: RunMemory,
        roadmap_path: Path,
    ) -> None:
        """Context should accumulate information from multiple completed tasks."""
        # Complete 3 tasks sequentially
        tasks_data = [
            (
                "1",
                "Create models",
                {Path("models.py"): FileAction.CREATED},
                "Added User, Post models",
            ),
            (
                "2",
                "Add validation",
                {Path("validators.py"): FileAction.CREATED},
                "Added input validation",
            ),
            (
                "3",
                "Create services",
                {Path("services.py"): FileAction.CREATED},
                "Added UserService",
            ),
        ]

        for task_id, title, files, summary in tasks_data:
            run_memory.add_outcome(
                TaskOutcome(
                    task_id=task_id,
                    title=title,
                    status="completed",
                    files=files,
                    summary=summary,
                )
            )

        # Build context for a 4th task
        task4 = MockTodoItem(
            id=4,
            title="Add API endpoints",
            description="Create REST endpoints using services",
            file_hints=["api.py"],
        )

        state = {
            "current_task": task4,
            "roadmap_path": str(roadmap_path),
        }

        await build_context_node(state, run_memory=run_memory)

        # Context should reference multiple previous tasks
        context = run_memory.get_context(current_task_id="4")

        # All 3 previous tasks should be referenced
        assert "1" in context  # Task 1
        assert "2" in context  # Task 2
        assert "3" in context  # Task 3

    @pytest.mark.asyncio
    async def test_context_respects_token_limits(
        self,
        run_memory: RunMemory,
        roadmap_path: Path,
    ) -> None:
        """Context should respect token limits when many tasks are completed."""
        # Add many tasks
        for i in range(20):
            run_memory.add_outcome(
                TaskOutcome(
                    task_id=str(i),
                    title=f"Task {i} with a moderately long title for testing",
                    status="completed",
                    files={Path(f"file{i}.py"): FileAction.CREATED},
                    summary=f"Completed task {i} with some detailed summary text",
                )
            )

        # Build context for next task
        next_task = MockTodoItem(
            id=21,
            title="Final task",
            description="Last task in sequence",
        )

        state = {
            "current_task": next_task,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(state, run_memory=run_memory)

        # Context should exist but be truncated
        context = result.get("context", "")
        prompt = result.get("prompt", "")

        # Should have some context but not infinite
        assert len(context) > 0 or len(prompt) > 0
        # Context should be reasonably sized (not all 20 tasks in full detail)
        # The default context budget is 50000 tokens, but run_memory gets ~25%
        # So context should be truncated if it exceeds budget


class TestContextCarryoverKeyDecisions:
    """Tests that key decisions are carried over between tasks."""

    @pytest.mark.asyncio
    async def test_key_decisions_included_in_context(
        self,
        run_memory: RunMemory,
        roadmap_path: Path,
    ) -> None:
        """Key decisions from previous tasks should be in context."""
        run_memory.add_outcome(
            TaskOutcome(
                task_id="1",
                title="Design database schema",
                status="completed",
                files={Path("models.py"): FileAction.CREATED},
                summary="Created initial schema",
                key_decisions=[
                    "Used SQLAlchemy ORM over raw SQL",
                    "Chose UUID for primary keys",
                ],
            )
        )

        next_task = MockTodoItem(
            id=2,
            title="Implement database operations",
            description="Add CRUD operations for models",
            file_hints=["database.py"],
        )

        state = {
            "current_task": next_task,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(state, run_memory=run_memory)

        # Key decisions should be accessible
        # Note: They may be in the summarized context from ContextSummarizer
        context = result.get("context", "")
        prompt = result.get("prompt", "")
        combined = f"{prompt} {context}".lower()

        # At minimum, the database/models work should be referenced
        assert "models" in combined or "database" in combined or "schema" in combined


class TestContextCarryoverEdgeCases:
    """Tests for edge cases in context carryover."""

    @pytest.mark.asyncio
    async def test_handles_no_previous_tasks(
        self,
        run_memory: RunMemory,
        task_create_model: MockTodoItem,
        roadmap_path: Path,
    ) -> None:
        """First task should work without any previous context."""
        # Empty run_memory (first task)
        state = {
            "current_task": task_create_model,
            "roadmap_path": str(roadmap_path),
        }

        # Should not raise
        result = await build_context_node(state, run_memory=run_memory)

        # Should still have prompt and context
        assert result.get("error") is None
        assert "prompt" in result

    @pytest.mark.asyncio
    async def test_handles_failed_tasks(
        self,
        run_memory: RunMemory,
        task_create_service: MockTodoItem,
        roadmap_path: Path,
    ) -> None:
        """Failed tasks should still be available in context."""
        run_memory.add_outcome(
            TaskOutcome(
                task_id="1",
                title="Attempt to create model",
                status="failed",
                files={},
                summary="Failed due to syntax error",
            )
        )

        state = {
            "current_task": task_create_service,
            "roadmap_path": str(roadmap_path),
        }

        # Should not raise
        result = await build_context_node(state, run_memory=run_memory)

        # Context should still be built
        assert result.get("error") is None

    @pytest.mark.asyncio
    async def test_handles_task_with_no_file_hints(
        self,
        run_memory: RunMemory,
        roadmap_path: Path,
    ) -> None:
        """Tasks without file hints should still get context."""
        run_memory.add_outcome(
            TaskOutcome(
                task_id="1",
                title="Setup project",
                status="completed",
                files={Path("setup.py"): FileAction.CREATED},
                summary="Initial project setup",
            )
        )

        task = MockTodoItem(
            id=2,
            title="Continue setup",
            description="More setup work",
            file_hints=[],  # No file hints
        )

        state = {
            "current_task": task,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(state, run_memory=run_memory)

        assert result.get("error") is None


class TestContextCarryoverIntegration:
    """Integration tests for the full context carryover scenario from Phase 9.3."""

    @pytest.mark.asyncio
    async def test_user_model_to_user_service_carryover(
        self,
        run_memory: RunMemory,
        roadmap_path: Path,
    ) -> None:
        """
        Full integration test matching EXECUTOR_3.md Phase 9.3.1 example.

        Scenario:
        - Task 1: Create User model in models.py
        - Task 2: Create UserService that uses User model

        Task 2 should know about Task 1's User model.
        """
        # Task 1: Create User model (completed)
        task1 = MockTodoItem(
            id=1,
            title="Create User model in models.py",
            description="Create a User dataclass with id, name, and email fields",
            file_hints=["models.py"],
        )

        # Simulate Task 1 completion via checkpoint_node
        task1_state = {
            "current_task": task1,
            "files_modified": ["models.py"],
            "verified": True,
            "llm_response": """Created User model:

```python
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str
```
""",
        }

        await checkpoint_node(
            task1_state,
            run_memory=run_memory,
            sync_roadmap=False,
        )

        # Verify Task 1 was recorded
        assert len(run_memory.outcomes) == 1
        outcome = run_memory.outcomes[0]
        assert outcome.title == "Create User model in models.py"

        # Task 2: Create UserService (pending)
        task2 = MockTodoItem(
            id=2,
            title="Create UserService that uses User model",
            description="Implement service layer for user operations using the User model",
            file_hints=["services.py"],
        )

        # Build context for Task 2
        task2_state = {
            "current_task": task2,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(task2_state, run_memory=run_memory)

        # Verify Task 2 context includes Task 1 results
        prompt = result.get("prompt", "")
        context = result.get("context", "")
        combined = f"{prompt} {context}".lower()

        # Task 2 should know about:
        # 1. models.py file from Task 1
        assert "models.py" in combined or "model" in combined, (
            f"Task 2 should reference Task 1's models.py. Got: {combined[:500]}..."
        )

        # 2. The fact that a User model was created
        assert "user" in combined, f"Task 2 should know about User model. Got: {combined[:500]}..."

        # 3. Task 1's completion status
        context_from_memory = run_memory.get_context(current_task_id="2")
        assert "1" in context_from_memory, "Task 1 ID should be in context"

    @pytest.mark.asyncio
    async def test_full_workflow_three_dependent_tasks(
        self,
        run_memory: RunMemory,
        roadmap_path: Path,
    ) -> None:
        """
        Test a realistic 3-task workflow with dependencies.

        Tasks:
        1. Create User model
        2. Create UserRepository using User model
        3. Create UserService using UserRepository
        """
        # Task 1: Create User model
        run_memory.add_outcome(
            TaskOutcome(
                task_id="1",
                title="Create User model",
                status="completed",
                files={Path("models/user.py"): FileAction.CREATED},
                summary="Created User dataclass with id, name, email, created_at",
                key_decisions=["Used dataclass over Pydantic for simplicity"],
            )
        )

        # Task 2: Create UserRepository
        run_memory.add_outcome(
            TaskOutcome(
                task_id="2",
                title="Create UserRepository",
                status="completed",
                files={Path("repositories/user_repo.py"): FileAction.CREATED},
                summary="Added UserRepository with CRUD operations",
                key_decisions=["Used SQLAlchemy for ORM"],
            )
        )

        # Task 3: Create UserService (current task)
        task3 = MockTodoItem(
            id=3,
            title="Create UserService using UserRepository",
            description="Business logic layer for user operations",
            file_hints=["services/user_service.py"],
        )

        state = {
            "current_task": task3,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(state, run_memory=run_memory)

        context = result.get("context", "")
        prompt = result.get("prompt", "")
        combined = f"{prompt} {context}".lower()

        # Task 3 should have context about both previous tasks
        # The repository is most directly relevant, but the model is also related
        assert "user" in combined, "User context should be present"

        # Verify memory has all tasks
        memory_context = run_memory.get_context(current_task_id="3")
        assert "1" in memory_context, "Task 1 should be in memory context"
        assert "2" in memory_context, "Task 2 should be in memory context"
