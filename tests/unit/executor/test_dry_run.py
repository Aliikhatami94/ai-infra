"""Tests for dry run mode in graph executor.

Phase 2.3.2: Unit tests for dry run functionality.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.nodes import execute_task_node
from ai_infra.executor.state import ExecutorGraphState
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
def base_state(mock_todo_item: TodoItem) -> ExecutorGraphState:
    """Create a base state for testing."""
    return ExecutorGraphState(
        roadmap_path="ROADMAP.md",
        todos=[mock_todo_item],
        current_task=mock_todo_item,
        context="File context here",
        prompt="Execute task: Implement feature X\n\nContext:\nFile content...",
        agent_result=None,
        files_modified=[],
        verified=False,
        last_checkpoint_sha=None,
        error=None,
        retry_count=0,
        should_continue=True,
        interrupt_requested=False,
        run_memory={},
        dry_run=False,
    )


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent."""
    agent = MagicMock()
    agent.arun = AsyncMock(return_value={"success": True})
    agent.tool_calls = []
    return agent


# =============================================================================
# Test: Dry Run Mode via Parameter
# =============================================================================


class TestDryRunViaParameter:
    """Tests for dry run mode passed as parameter to execute_task_node."""

    @pytest.mark.asyncio
    async def test_dry_run_does_not_execute_agent(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test that dry_run=True skips agent execution."""
        result = await execute_task_node(base_state, agent=mock_agent, dry_run=True)

        # Agent should not be called
        mock_agent.arun.assert_not_called()

        # Result should indicate dry run
        assert result["agent_result"]["dry_run"] is True

    @pytest.mark.asyncio
    async def test_dry_run_returns_preview_result(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test that dry run returns a preview of what would be done."""
        result = await execute_task_node(base_state, agent=mock_agent, dry_run=True)

        agent_result = result["agent_result"]
        assert agent_result["dry_run"] is True
        assert agent_result["task_id"] == "1"
        assert agent_result["task_title"] == "Implement feature X"
        assert agent_result["prompt_length"] > 0
        assert "prompt_preview" in agent_result

    @pytest.mark.asyncio
    async def test_dry_run_returns_empty_files_modified(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test that dry run returns empty files_modified list."""
        result = await execute_task_node(base_state, agent=mock_agent, dry_run=True)

        assert result["files_modified"] == []

    @pytest.mark.asyncio
    async def test_dry_run_sets_verified_true(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test that dry run skips verification by setting verified=True."""
        result = await execute_task_node(base_state, agent=mock_agent, dry_run=True)

        assert result["verified"] is True

    @pytest.mark.asyncio
    async def test_dry_run_clears_error(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test that dry run clears any existing error."""
        # Start with an error state
        base_state["error"] = {
            "error_type": "previous_error",
            "message": "Some error",
            "node": "test",
            "task_id": "1",
            "recoverable": True,
            "stack_trace": None,
        }

        result = await execute_task_node(base_state, agent=mock_agent, dry_run=True)

        assert result["error"] is None


# =============================================================================
# Test: Dry Run Mode via State
# =============================================================================


class TestDryRunViaState:
    """Tests for dry run mode passed via state['dry_run']."""

    @pytest.mark.asyncio
    async def test_dry_run_via_state_does_not_execute_agent(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test that state['dry_run']=True skips agent execution."""
        base_state["dry_run"] = True

        result = await execute_task_node(base_state, agent=mock_agent)

        mock_agent.arun.assert_not_called()
        assert result["agent_result"]["dry_run"] is True

    @pytest.mark.asyncio
    async def test_parameter_overrides_state(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test that dry_run parameter can override state."""
        base_state["dry_run"] = False

        result = await execute_task_node(base_state, agent=mock_agent, dry_run=True)

        mock_agent.arun.assert_not_called()
        assert result["agent_result"]["dry_run"] is True


# =============================================================================
# Test: Normal Execution (dry_run=False)
# =============================================================================


class TestNormalExecution:
    """Tests to ensure dry_run=False still executes normally."""

    @pytest.mark.asyncio
    async def test_normal_execution_calls_agent(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test that dry_run=False calls the agent."""
        await execute_task_node(base_state, agent=mock_agent, dry_run=False)

        mock_agent.arun.assert_called_once()

    @pytest.mark.asyncio
    async def test_normal_execution_with_no_dry_run_parameter(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test that omitting dry_run parameter defaults to normal execution."""
        # Ensure state also has dry_run=False
        base_state["dry_run"] = False

        await execute_task_node(base_state, agent=mock_agent)

        mock_agent.arun.assert_called_once()


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestDryRunEdgeCases:
    """Edge case tests for dry run mode."""

    @pytest.mark.asyncio
    async def test_dry_run_with_no_current_task(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test dry run handles missing current_task gracefully."""
        base_state["current_task"] = None

        result = await execute_task_node(base_state, agent=mock_agent, dry_run=True)

        agent_result = result["agent_result"]
        assert agent_result["dry_run"] is True
        assert agent_result["task_id"] == "unknown"
        assert agent_result["task_title"] == "Unknown task"

    @pytest.mark.asyncio
    async def test_dry_run_with_empty_prompt(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test dry run with empty prompt still works."""
        base_state["prompt"] = ""

        result = await execute_task_node(base_state, agent=mock_agent, dry_run=True)

        agent_result = result["agent_result"]
        assert agent_result["dry_run"] is True
        assert agent_result["prompt_length"] == 0
        assert agent_result["prompt_preview"] == ""

    @pytest.mark.asyncio
    async def test_dry_run_with_long_prompt_truncates_preview(
        self, base_state: ExecutorGraphState, mock_agent: MagicMock
    ) -> None:
        """Test that long prompts are truncated in preview."""
        long_prompt = "A" * 5000
        base_state["prompt"] = long_prompt

        result = await execute_task_node(base_state, agent=mock_agent, dry_run=True)

        agent_result = result["agent_result"]
        assert agent_result["prompt_length"] == 5000
        assert len(agent_result["prompt_preview"]) == 1000  # Truncated

    @pytest.mark.asyncio
    async def test_dry_run_does_not_require_agent(self, base_state: ExecutorGraphState) -> None:
        """Test that dry run works even with agent=None."""
        result = await execute_task_node(base_state, agent=None, dry_run=True)

        assert result["agent_result"]["dry_run"] is True
        assert result["error"] is None


# =============================================================================
# Test: ExecutorGraph Integration
# =============================================================================


class TestExecutorGraphDryRun:
    """Tests for ExecutorGraph with dry_run parameter."""

    def test_executor_graph_accepts_dry_run_parameter(self) -> None:
        """Test that ExecutorGraph accepts dry_run parameter."""
        from ai_infra.executor.graph import ExecutorGraph

        graph = ExecutorGraph(
            roadmap_path="ROADMAP.md",
            dry_run=True,
        )

        assert graph.dry_run is True

    def test_executor_graph_defaults_dry_run_to_false(self) -> None:
        """Test that ExecutorGraph defaults dry_run to False."""
        from ai_infra.executor.graph import ExecutorGraph

        graph = ExecutorGraph(roadmap_path="ROADMAP.md")

        assert graph.dry_run is False

    def test_executor_graph_initial_state_includes_dry_run(self) -> None:
        """Test that initial state includes dry_run field."""
        from ai_infra.executor.graph import ExecutorGraph

        graph = ExecutorGraph(
            roadmap_path="ROADMAP.md",
            dry_run=True,
        )

        initial_state = graph.get_initial_state()

        assert initial_state["dry_run"] is True

    def test_executor_graph_initial_state_dry_run_false(self) -> None:
        """Test that initial state has dry_run=False when not set."""
        from ai_infra.executor.graph import ExecutorGraph

        graph = ExecutorGraph(roadmap_path="ROADMAP.md")

        initial_state = graph.get_initial_state()

        assert initial_state["dry_run"] is False


# =============================================================================
# Test: State Field
# =============================================================================


class TestDryRunStateField:
    """Tests for dry_run field in ExecutorGraphState."""

    def test_state_accepts_dry_run_field(self) -> None:
        """Test that ExecutorGraphState accepts dry_run field."""
        state = ExecutorGraphState(
            roadmap_path="ROADMAP.md",
            dry_run=True,
        )

        assert state["dry_run"] is True

    def test_state_dry_run_default_is_optional(self) -> None:
        """Test that dry_run is optional in state."""
        state = ExecutorGraphState(roadmap_path="ROADMAP.md")

        # Should not raise, dry_run is optional (TypedDict total=False)
        assert state.get("dry_run") is None
