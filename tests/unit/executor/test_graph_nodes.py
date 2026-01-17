"""Tests for executor graph nodes.

Phase 1.2.2: Unit tests for all graph node implementations.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_infra.executor.nodes import (
    build_context_node,
    checkpoint_node,
    decide_next_node,
    execute_task_node,
    handle_failure_node,
    parse_roadmap_node,
    pick_task_node,
    rollback_node,
    verify_task_node,
)
from ai_infra.executor.state import (
    ExecutorGraphState,
    NodeTimeouts,
)
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_todo_item() -> TodoItem:
    """Create a mock TodoItem for testing."""
    return TodoItem(
        id=1,
        title="Implement feature X",
        description="Add the X feature to module Y",
        status=TodoStatus.NOT_STARTED,
        file_hints=["src/module.py"],
    )


@pytest.fixture
def completed_todo_item() -> TodoItem:
    """Create a completed TodoItem."""
    return TodoItem(
        id=2,
        title="Done task",
        description="Already completed",
        status=TodoStatus.COMPLETED,
        file_hints=[],
    )


@pytest.fixture
def base_state(mock_todo_item: TodoItem) -> ExecutorGraphState:
    """Create a base ExecutorGraphState for testing."""
    return ExecutorGraphState(
        roadmap_path="ROADMAP.md",
        todos=[mock_todo_item],
        current_task=None,
        context="",
        prompt="",
        agent_result=None,
        files_modified=[],
        verified=False,
        last_checkpoint_sha=None,
        error=None,
        retry_count=0,
        completed_count=0,
        max_tasks=None,
        should_continue=True,
        interrupt_requested=False,
        run_memory={},
    )


# =============================================================================
# parse_roadmap_node tests
# =============================================================================


class TestParseRoadmapNode:
    """Tests for parse_roadmap_node."""

    @pytest.mark.asyncio
    async def test_file_not_found(self, base_state: ExecutorGraphState):
        """Test error handling when ROADMAP doesn't exist."""
        state = {**base_state, "roadmap_path": "/nonexistent/ROADMAP.md"}

        result = await parse_roadmap_node(state, agent=None, use_llm_normalization=False)

        assert result["error"] is not None
        assert result["error"]["error_type"] == "parse"
        assert "not found" in result["error"]["message"]
        assert result["should_continue"] is False

    @pytest.mark.asyncio
    async def test_parse_with_regex(self, base_state: ExecutorGraphState, tmp_path):
        """Test parsing with regex (no LLM)."""
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text(
            """# Test ROADMAP

## Phase 1

- [ ] Task 1 - Do something
- [x] Task 2 - Already done
- [ ] Task 3 - Another task
"""
        )
        state = {**base_state, "roadmap_path": str(roadmap)}

        result = await parse_roadmap_node(state, agent=None, use_llm_normalization=False)

        assert result["error"] is None
        assert len(result["todos"]) > 0


# =============================================================================
# pick_task_node tests
# =============================================================================


class TestPickTaskNode:
    """Tests for pick_task_node."""

    def test_picks_first_not_started_task(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test selecting the first NOT_STARTED task."""
        state = {**base_state, "todos": [mock_todo_item]}

        result = pick_task_node(state)

        assert result["current_task"] == mock_todo_item
        assert result["retry_count"] == 0

    def test_skips_completed_tasks(
        self,
        base_state: ExecutorGraphState,
        mock_todo_item: TodoItem,
        completed_todo_item: TodoItem,
    ):
        """Test skipping completed tasks."""
        state = {**base_state, "todos": [completed_todo_item, mock_todo_item]}

        result = pick_task_node(state)

        assert result["current_task"] == mock_todo_item

    def test_no_tasks_remaining(self, base_state: ExecutorGraphState):
        """Test when no tasks remain."""
        state = {**base_state, "todos": []}

        result = pick_task_node(state)

        assert result["current_task"] is None
        assert result["should_continue"] is False

    def test_respects_max_tasks_limit(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test max_tasks limit is respected."""
        state = {
            **base_state,
            "todos": [mock_todo_item],
            "max_tasks": 5,
            "tasks_completed_count": 5,
        }

        result = pick_task_node(state)

        assert result["current_task"] is None
        assert result["should_continue"] is False


# =============================================================================
# build_context_node tests
# =============================================================================


class TestBuildContextNode:
    """Tests for build_context_node."""

    @pytest.mark.asyncio
    async def test_no_current_task(self, base_state: ExecutorGraphState):
        """Test error when no current task."""
        state = {**base_state, "current_task": None}

        result = await build_context_node(state)

        assert result["error"] is not None
        assert result["error"]["error_type"] == "context"

    @pytest.mark.asyncio
    async def test_builds_context_and_prompt(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test building context and prompt."""
        state = {**base_state, "current_task": mock_todo_item}

        result = await build_context_node(state)

        assert result["error"] is None
        assert result["context"] != ""
        assert result["prompt"] != ""
        assert mock_todo_item.title in result["prompt"]

    @pytest.mark.asyncio
    async def test_includes_run_memory_context(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Phase 2.1.1: Test that RunMemory context is included in prompt."""
        from pathlib import Path

        from ai_infra.executor.run_memory import FileAction, RunMemory, TaskOutcome

        # Create run memory with a completed task
        run_memory = RunMemory(run_id="test-run")
        run_memory.add_outcome(
            TaskOutcome(
                task_id="0",
                title="Setup project structure",
                status="completed",
                files={Path("src/main.py"): FileAction.CREATED},
                summary="Created main.py with entry point",
            )
        )

        state = {**base_state, "current_task": mock_todo_item}

        result = await build_context_node(state, run_memory=run_memory)

        assert result["error"] is None
        # Context should include run memory information
        assert "Run Memory" in result["context"] or "Setup project structure" in result["context"]
        # RunMemory uses summary in context line, not title
        assert "Created main.py with entry point" in result["context"]
        assert "main.py" in result["context"]

    @pytest.mark.asyncio
    async def test_includes_project_memory_context(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem, tmp_path
    ):
        """Phase 2.1.2: Test that ProjectMemory context is included in prompt."""
        from ai_infra.executor.project_memory import ProjectMemory

        # Create project memory with some data
        project_memory = ProjectMemory(project_root=tmp_path)
        project_memory.set_project_type("python")
        project_memory.add_framework("fastapi")
        project_memory.add_convention("Use type hints for all functions")

        state = {**base_state, "current_task": mock_todo_item}

        result = await build_context_node(state, project_memory=project_memory)

        assert result["error"] is None
        # Context should include project memory information
        assert "Project Context" in result["context"]
        assert "python" in result["context"]
        assert "fastapi" in result["context"]
        assert "type hints" in result["context"]


# =============================================================================
# execute_task_node tests
# =============================================================================


class TestExecuteTaskNode:
    """Tests for execute_task_node."""

    @pytest.mark.asyncio
    async def test_no_prompt(self, base_state: ExecutorGraphState, mock_todo_item: TodoItem):
        """Test error when no prompt."""
        state = {**base_state, "current_task": mock_todo_item, "prompt": ""}

        result = await execute_task_node(state)

        assert result["error"] is not None
        assert result["error"]["error_type"] == "execution"

    @pytest.mark.asyncio
    async def test_no_agent(self, base_state: ExecutorGraphState, mock_todo_item: TodoItem):
        """Test error when no agent."""
        state = {**base_state, "current_task": mock_todo_item, "prompt": "Test prompt"}

        result = await execute_task_node(state, agent=None)

        assert result["error"] is not None
        assert "No agent" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_successful_execution(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test successful execution."""
        mock_agent = AsyncMock()
        mock_agent.arun = AsyncMock(return_value={"files_modified": ["src/test.py"]})

        state = {**base_state, "current_task": mock_todo_item, "prompt": "Test prompt"}

        result = await execute_task_node(state, agent=mock_agent)

        assert result["error"] is None
        assert result["agent_result"] is not None

    @pytest.mark.asyncio
    async def test_timeout_handling(self, base_state: ExecutorGraphState, mock_todo_item: TodoItem):
        """Test timeout handling."""
        mock_agent = AsyncMock()
        mock_agent.arun = AsyncMock(side_effect=TimeoutError())

        state = {**base_state, "current_task": mock_todo_item, "prompt": "Test prompt"}

        with patch.object(NodeTimeouts, "EXECUTE_TASK", 0.001):
            result = await execute_task_node(state, agent=mock_agent)

        assert result["error"] is not None
        assert result["error"]["error_type"] == "timeout"


# =============================================================================
# verify_task_node tests
# =============================================================================


class TestVerifyTaskNode:
    """Tests for verify_task_node."""

    @pytest.mark.asyncio
    async def test_no_current_task(self, base_state: ExecutorGraphState):
        """Test error when no current task."""
        state = {**base_state, "current_task": None}

        result = await verify_task_node(state)

        assert result["error"] is not None
        assert result["verified"] is False

    @pytest.mark.asyncio
    async def test_auto_pass_without_verifier(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test auto-pass when no verifier provided."""
        state = {**base_state, "current_task": mock_todo_item}

        result = await verify_task_node(state, verifier=None)

        assert result["verified"] is True
        assert result["error"] is None


# =============================================================================
# checkpoint_node tests
# =============================================================================


class TestCheckpointNode:
    """Tests for checkpoint_node."""

    @pytest.mark.asyncio
    async def test_no_current_task(self, base_state: ExecutorGraphState):
        """Test error when no current task."""
        state = {**base_state, "current_task": None}

        result = await checkpoint_node(state)

        assert result["error"] is not None
        assert result["error"]["error_type"] == "checkpoint"

    @pytest.mark.asyncio
    async def test_no_files_modified(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test skip checkpoint when no files modified."""
        state = {**base_state, "current_task": mock_todo_item, "files_modified": []}

        result = await checkpoint_node(state)

        assert result["error"] is None  # No error, just skipped

    @pytest.mark.asyncio
    async def test_skip_without_checkpointer(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test skip when no checkpointer provided."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "files_modified": ["src/test.py"],
        }

        result = await checkpoint_node(state, checkpointer=None)

        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_records_outcome_to_run_memory(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Phase 2.1.1: Test that checkpoint records outcome to RunMemory."""
        from ai_infra.executor.run_memory import RunMemory

        run_memory = RunMemory(run_id="test-run")
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "files_modified": ["src/module.py", "src/test.py"],
        }

        result = await checkpoint_node(state, run_memory=run_memory)

        # Should record outcome even without checkpointer
        assert result["error"] is None
        assert len(run_memory.outcomes) == 1

        outcome = run_memory.outcomes[0]
        assert outcome.task_id == str(mock_todo_item.id)
        assert outcome.title == mock_todo_item.title
        assert outcome.status == "completed"
        assert len(outcome.files) == 2

    @pytest.mark.asyncio
    async def test_updates_project_memory(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem, tmp_path
    ):
        """Phase 2.1.2: Test that checkpoint updates ProjectMemory."""

        from ai_infra.executor.project_memory import ProjectMemory
        from ai_infra.executor.run_memory import RunMemory

        # Create project memory
        project_memory = ProjectMemory(project_root=tmp_path)

        # Create run memory with this task already recorded
        run_memory = RunMemory(run_id="test-run")

        state = {
            **base_state,
            "current_task": mock_todo_item,
            "files_modified": ["src/module.py", "src/test.py"],
        }

        result = await checkpoint_node(state, run_memory=run_memory, project_memory=project_memory)

        # Should update project memory
        assert result["error"] is None

        # Project memory should have been updated with run info
        assert len(project_memory.run_history) == 1
        assert project_memory.run_history[0].run_id == "test-run"


# =============================================================================
# rollback_node tests
# =============================================================================


class TestRollbackNode:
    """Tests for rollback_node."""

    @pytest.mark.asyncio
    async def test_no_checkpoint_to_rollback(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test when no checkpoint to rollback to."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "last_checkpoint_sha": None,
        }

        result = await rollback_node(state)

        # Should just clear state without error
        assert result["files_modified"] == []
        assert result["agent_result"] is None

    @pytest.mark.asyncio
    async def test_rollback_without_checkpointer(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test rollback without checkpointer."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "last_checkpoint_sha": "abc123",
        }

        result = await rollback_node(state, checkpointer=None)

        assert result["files_modified"] == []


# =============================================================================
# handle_failure_node tests
# =============================================================================


class TestHandleFailureNode:
    """Tests for handle_failure_node."""

    def test_no_error(self, base_state: ExecutorGraphState):
        """Test when called without error."""
        state = {**base_state, "error": None}

        result = handle_failure_node(state)

        assert result == state  # No change

    def test_increments_test_repair_count_on_test_failure(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Phase 2.2: Test repair count incremented on test failure."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "test_repair_count": 0,
            "error": {
                "error_type": "test_failure",
                "message": "Test failed",
                "node": "verify_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = handle_failure_node(state)

        assert result["test_repair_count"] == 1
        assert result["error"] is None  # Error cleared for repair

    def test_stops_on_max_test_repairs(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Phase 2.2: Stops when max test repairs (2) exceeded."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "test_repair_count": 2,  # Already at max
            "error": {
                "error_type": "test_failure",
                "message": "Test failed",
                "node": "verify_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = handle_failure_node(state)

        assert result["should_continue"] is False
        assert result["error"]["recoverable"] is False

    def test_stops_on_non_retryable_error(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test stops on non-retryable error type."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "test_repair_count": 0,
            "error": {
                "error_type": "AuthenticationError",  # Non-retryable
                "message": "authentication failed",  # Matches NonRetryableErrors.PATTERNS
                "node": "execute_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = handle_failure_node(state)

        assert result["should_continue"] is False

    def test_increments_repair_count_on_validation_error(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Phase 2.2: Validation errors use repair_count."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "repair_count": 0,
            "error": {
                "error_type": "validation_error",
                "message": "SyntaxError at line 5",
                "node": "validate_code",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = handle_failure_node(state)

        # Should increment repair_count for validation errors
        assert result["repair_count"] == 1
        assert result["error"] is None

    def test_stops_on_max_validation_repairs(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Phase 2.2: Stops when max validation repairs (2) exceeded."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "repair_count": 2,  # Already at max
            "error": {
                "error_type": "validation_error",
                "message": "SyntaxError at line 5",
                "node": "validate_code",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = handle_failure_node(state)

        # Should stop since 2 >= 2
        assert result["should_continue"] is False
        assert result["error"]["recoverable"] is False
        assert "max repairs exceeded" in result["error"]["message"]


# =============================================================================
# decide_next_node tests
# =============================================================================


class TestDecideNextNode:
    """Tests for decide_next_node."""

    def test_continues_with_remaining_tasks(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test continues when tasks remain."""
        another_task = TodoItem(
            id="2",
            title="Another task",
            description="",
            status=TodoStatus.NOT_STARTED,
            file_hints=[],
        )
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "todos": [mock_todo_item, another_task],
        }

        result = decide_next_node(state)

        assert result["should_continue"] is True
        assert result["tasks_completed_count"] == 1

    def test_stops_on_interrupt(self, base_state: ExecutorGraphState, mock_todo_item: TodoItem):
        """Test stops when interrupt requested."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "todos": [mock_todo_item],
            "interrupt_requested": True,
        }

        result = decide_next_node(state)

        assert result["should_continue"] is False

    def test_stops_at_max_tasks(self, base_state: ExecutorGraphState, mock_todo_item: TodoItem):
        """Test stops when max_tasks reached."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "todos": [mock_todo_item],
            "completed_count": 4,
            "max_tasks": 5,
        }

        result = decide_next_node(state)

        # After incrementing, completed_count = 5, which equals max_tasks
        assert result["should_continue"] is False

    def test_stops_when_all_complete(
        self, base_state: ExecutorGraphState, mock_todo_item: TodoItem
    ):
        """Test stops when all tasks complete."""
        state = {
            **base_state,
            "current_task": mock_todo_item,
            "todos": [mock_todo_item],  # Only one task, which we're completing
        }

        result = decide_next_node(state)

        # After marking current task done, no more NOT_STARTED
        assert result["should_continue"] is False
