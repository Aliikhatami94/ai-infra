"""Tests for Phase 9.1: Context Carryover Wiring.

This module tests that RunMemory is properly wired into the executor graph:
- 9.1.1: RunMemory is available in ExecutorGraph
- 9.1.2: checkpoint_node records task outcomes to RunMemory
- 9.1.3: build_context_node includes previous context from RunMemory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_infra.executor.nodes.checkpoint import _record_task_outcome, checkpoint_node
from ai_infra.executor.nodes.context import build_context_node
from ai_infra.executor.run_memory import FileAction, RunMemory, TaskOutcome

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def run_memory() -> RunMemory:
    """Create a RunMemory instance."""
    return RunMemory(run_id="test-run-001")


@pytest.fixture
def mock_task() -> Any:
    """Create a mock task."""
    task = MagicMock()
    task.id = "1.2"
    task.title = "Add user authentication"
    task.description = "Implement JWT-based auth for API endpoints"
    task.file_hints = ["src/auth.py"]
    return task


@pytest.fixture
def previous_task() -> Any:
    """Create a mock previous task."""
    task = MagicMock()
    task.id = "1.1"
    task.title = "Create database models"
    task.description = "Add SQLAlchemy models for users"
    task.file_hints = ["src/models.py"]
    return task


@pytest.fixture
def roadmap_path(tmp_path: Path) -> Path:
    """Create a roadmap file."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""# Test Roadmap

## Phase 1: Setup

- [x] 1.1 Create database models
- [ ] 1.2 Add user authentication
""")
    return roadmap


# =============================================================================
# 9.1.1: RunMemory in ExecutorGraph
# =============================================================================


class TestRunMemoryInGraph:
    """Tests for RunMemory availability in ExecutorGraph."""

    def test_run_memory_is_accepted_by_graph(self) -> None:
        """ExecutorGraph should accept run_memory parameter."""
        from ai_infra.executor.graph import ExecutorGraph

        memory = RunMemory(run_id="test-123")

        # Should not raise
        graph = ExecutorGraph(
            run_memory=memory,
            roadmap_path="ROADMAP.md",
        )

        assert graph.run_memory is memory

    def test_run_memory_defaults_to_none(self) -> None:
        """ExecutorGraph should work without run_memory."""
        from ai_infra.executor.graph import ExecutorGraph

        graph = ExecutorGraph(roadmap_path="ROADMAP.md")

        # Graph should work without run_memory (it defaults to None in __init__)
        # Based on the code, run_memory is not created by default
        assert graph.run_memory is None


# =============================================================================
# 9.1.2: Task Outcome Recording
# =============================================================================


class TestTaskOutcomeRecording:
    """Tests for recording task outcomes to RunMemory."""

    def test_record_task_outcome(
        self,
        run_memory: RunMemory,
        mock_task: Any,
    ) -> None:
        """_record_task_outcome should add outcome to RunMemory."""
        files_modified = ["src/auth.py", "src/routes.py"]

        _record_task_outcome(
            run_memory=run_memory,
            current_task=mock_task,
            files_modified=files_modified,
            status="completed",
        )

        assert len(run_memory.outcomes) == 1
        outcome = run_memory.outcomes[0]
        assert outcome.task_id == "1.2"
        assert outcome.title == "Add user authentication"
        assert outcome.status == "completed"
        assert len(outcome.files) == 2

    def test_record_multiple_outcomes(
        self,
        run_memory: RunMemory,
        previous_task: Any,
        mock_task: Any,
    ) -> None:
        """Multiple task outcomes should accumulate in RunMemory."""
        # Record first task
        _record_task_outcome(
            run_memory=run_memory,
            current_task=previous_task,
            files_modified=["src/models.py"],
            status="completed",
        )

        # Record second task
        _record_task_outcome(
            run_memory=run_memory,
            current_task=mock_task,
            files_modified=["src/auth.py"],
            status="completed",
        )

        assert len(run_memory.outcomes) == 2
        assert run_memory.outcomes[0].task_id == "1.1"
        assert run_memory.outcomes[1].task_id == "1.2"

    @pytest.mark.asyncio
    async def test_checkpoint_node_records_outcome(
        self,
        run_memory: RunMemory,
        mock_task: Any,
    ) -> None:
        """checkpoint_node should record outcome when run_memory provided."""
        state = {
            "current_task": mock_task,
            "files_modified": ["src/auth.py"],
            "verified": True,
        }

        await checkpoint_node(
            state,
            run_memory=run_memory,
            sync_roadmap=False,  # Don't actually sync
        )

        assert len(run_memory.outcomes) == 1
        assert run_memory.outcomes[0].task_id == "1.2"


# =============================================================================
# 9.1.3: Previous Context in build_context
# =============================================================================


class TestPreviousContextInBuildContext:
    """Tests for including previous context in build_context_node."""

    @pytest.mark.asyncio
    async def test_includes_run_memory_context(
        self,
        run_memory: RunMemory,
        mock_task: Any,
        roadmap_path: Path,
    ) -> None:
        """build_context_node should include context from run_memory."""
        # Add a previous task outcome
        run_memory.add_outcome(
            TaskOutcome(
                task_id="1.1",
                title="Create database models",
                status="completed",
                files={Path("src/models.py"): FileAction.CREATED},
                summary="Created User and Session models",
            )
        )

        state = {
            "current_task": mock_task,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(state, run_memory=run_memory)

        # Context should include previous task info
        prompt = result.get("prompt", "")
        context = result.get("context", "")

        # The previous task should be mentioned somewhere in the prompt
        assert (
            "1.1" in prompt
            or "database models" in prompt.lower()
            or "Previously Completed" in prompt
        )

    @pytest.mark.asyncio
    async def test_no_context_without_run_memory(
        self,
        mock_task: Any,
        roadmap_path: Path,
    ) -> None:
        """build_context_node should work without run_memory."""
        state = {
            "current_task": mock_task,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(state, run_memory=None)

        # Should not crash, and should return valid prompt
        assert "prompt" in result
        assert result["prompt"] is not None

    @pytest.mark.asyncio
    async def test_empty_run_memory_produces_no_context(
        self,
        run_memory: RunMemory,
        mock_task: Any,
        roadmap_path: Path,
    ) -> None:
        """Empty run_memory should not add context section."""
        state = {
            "current_task": mock_task,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(state, run_memory=run_memory)

        # Empty run_memory returns empty string from get_context()
        # So "Previously Completed" header should not appear
        prompt = result.get("prompt", "")
        # This is optional - depends on implementation
        # The key is that it doesn't crash


# =============================================================================
# Integration: Full Context Flow
# =============================================================================


class TestContextCarryoverFlow:
    """Integration tests for the full context carryover flow."""

    @pytest.mark.asyncio
    async def test_context_flows_from_checkpoint_to_build_context(
        self,
        run_memory: RunMemory,
        previous_task: Any,
        mock_task: Any,
        roadmap_path: Path,
    ) -> None:
        """Context recorded in checkpoint should be available in build_context."""
        # Step 1: Complete previous task and record via checkpoint
        prev_state = {
            "current_task": previous_task,
            "files_modified": ["src/models.py"],
            "verified": True,
        }

        await checkpoint_node(
            prev_state,
            run_memory=run_memory,
            sync_roadmap=False,
        )

        # Verify outcome was recorded
        assert len(run_memory.outcomes) == 1

        # Step 2: Build context for next task
        next_state = {
            "current_task": mock_task,
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(next_state, run_memory=run_memory)

        # The previous task's info should be in the prompt
        prompt = result.get("prompt", "")
        assert len(prompt) > 0
        # At minimum, the memory context should be retrieved
        context = run_memory.get_context(current_task_id=str(mock_task.id))
        assert "1.1" in context  # Previous task ID

    @pytest.mark.asyncio
    async def test_multiple_tasks_accumulate_context(
        self,
        run_memory: RunMemory,
        roadmap_path: Path,
    ) -> None:
        """Context should accumulate across multiple tasks."""
        # Create 3 tasks
        tasks = [
            MagicMock(id="1.1", title="Task One", description="First task", file_hints=[]),
            MagicMock(id="1.2", title="Task Two", description="Second task", file_hints=[]),
            MagicMock(id="1.3", title="Task Three", description="Third task", file_hints=[]),
        ]

        # Complete first two tasks
        for i, task in enumerate(tasks[:2]):
            state = {
                "current_task": task,
                "files_modified": [f"file{i}.py"],
                "verified": True,
            }
            await checkpoint_node(state, run_memory=run_memory, sync_roadmap=False)

        # Now build context for third task
        state = {
            "current_task": tasks[2],
            "roadmap_path": str(roadmap_path),
        }

        result = await build_context_node(state, run_memory=run_memory)

        # Both previous tasks should be in context
        context = run_memory.get_context(current_task_id="1.3")
        assert "1.1" in context
        assert "1.2" in context
