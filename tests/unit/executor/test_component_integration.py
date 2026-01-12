"""Integration tests for executor graph with existing components.

Tests verify that graph nodes integrate correctly with:
- Git checkpointer (commit/rollback)
- TodoListManager (ROADMAP sync)
- TaskVerifier (completion verification)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.nodes.checkpoint import checkpoint_node
from ai_infra.executor.nodes.pick import pick_task_node
from ai_infra.executor.nodes.verify import verify_task_node
from ai_infra.executor.state import ExecutorGraphState
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_checkpointer():
    """Create a mock git checkpointer."""
    checkpointer = MagicMock()
    checkpointer.checkpoint.return_value = MagicMock(sha="abc123def456")
    return checkpointer


@pytest.fixture
def mock_todo_manager():
    """Create a mock TodoListManager."""
    manager = MagicMock()
    manager.mark_completed.return_value = 1  # 1 checkbox updated
    return manager


@pytest.fixture
def mock_verifier():
    """Create a mock TaskVerifier with proper VerificationResult mock."""
    verifier = MagicMock()
    # Mock VerificationResult with 'overall' property and 'get_failures()' method
    mock_result = MagicMock()
    mock_result.overall = True
    mock_result.get_failures.return_value = []
    verifier.verify = AsyncMock(return_value=mock_result)
    return verifier


@pytest.fixture
def sample_todo() -> TodoItem:
    """Create a sample todo item."""
    return TodoItem(
        id=1,
        title="Test task",
        description="Test description",
        status=TodoStatus.NOT_STARTED,
        source_task_ids=["task-1"],
    )


@pytest.fixture
def sample_state(sample_todo: TodoItem) -> ExecutorGraphState:
    """Create a sample graph state."""
    return {
        "roadmap_path": "/project/ROADMAP.md",
        "run_id": "run-123",
        "todos": [sample_todo],
        "current_task": sample_todo,
        "context": "sample context",
        "prompt": "sample prompt",
        "agent_result": None,
        "files_modified": ["src/app.py", "src/utils.py"],
        "verified": True,
        "error": None,
        "retry_count": 0,
        "max_retries": 3,
        "completed_todos": [],
        "failed_todos": [],
        "last_checkpoint_sha": None,
        "should_continue": True,
        "tasks_completed_count": 0,
        "max_tasks": None,
        "run_memory": {},
        "iteration": 0,
        "max_iterations": 100,
    }


# =============================================================================
# Tests: 1.4.1 Git Checkpointer Integration
# =============================================================================


class TestGitCheckpointerIntegration:
    """Tests for git checkpointer integration in checkpoint_node."""

    @pytest.mark.asyncio
    async def test_checkpoint_creates_git_commit(
        self,
        sample_state: ExecutorGraphState,
        mock_checkpointer: MagicMock,
    ) -> None:
        """Test that checkpoint_node creates a git commit."""
        result = await checkpoint_node(
            sample_state,
            checkpointer=mock_checkpointer,
        )

        # Verify checkpointer was called
        mock_checkpointer.checkpoint.assert_called_once()
        call_args = mock_checkpointer.checkpoint.call_args

        # Check commit message includes task info
        assert "Test task" in call_args.kwargs["message"]
        assert "src/app.py" in call_args.kwargs["files"]

        # Check state updated with sha
        assert result["last_checkpoint_sha"] == "abc123def456"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_checkpoint_without_checkpointer_skips(
        self,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test that checkpoint skips when no checkpointer provided."""
        result = await checkpoint_node(
            sample_state,
            checkpointer=None,
        )

        # Should not error, just skip
        assert result["error"] is None
        assert result.get("last_checkpoint_sha") is None

    @pytest.mark.asyncio
    async def test_checkpoint_no_files_skips(
        self,
        sample_state: ExecutorGraphState,
        mock_checkpointer: MagicMock,
    ) -> None:
        """Test that checkpoint skips when no files modified."""
        sample_state["files_modified"] = []

        result = await checkpoint_node(
            sample_state,
            checkpointer=mock_checkpointer,
        )

        # Checkpointer should not be called
        mock_checkpointer.checkpoint.assert_not_called()
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_checkpoint_handles_git_error(
        self,
        sample_state: ExecutorGraphState,
        mock_checkpointer: MagicMock,
    ) -> None:
        """Test that checkpoint handles git errors gracefully."""
        mock_checkpointer.checkpoint.side_effect = Exception("Git error")

        result = await checkpoint_node(
            sample_state,
            checkpointer=mock_checkpointer,
        )

        # Should have error in state
        assert result["error"] is not None
        assert result["error"]["error_type"] == "checkpoint"
        assert "Git error" in result["error"]["message"]


# =============================================================================
# Tests: 1.4.2 TodoListManager Integration
# =============================================================================


class TestTodoListManagerIntegration:
    """Tests for TodoListManager integration."""

    @pytest.mark.asyncio
    async def test_checkpoint_syncs_to_roadmap(
        self,
        sample_state: ExecutorGraphState,
        mock_checkpointer: MagicMock,
        mock_todo_manager: MagicMock,
    ) -> None:
        """Test that checkpoint_node syncs completion to ROADMAP."""
        await checkpoint_node(
            sample_state,
            checkpointer=mock_checkpointer,
            todo_manager=mock_todo_manager,
            sync_roadmap=True,
        )

        # Verify todo manager was called
        mock_todo_manager.mark_completed.assert_called_once()
        call_args = mock_todo_manager.mark_completed.call_args

        assert call_args.kwargs["todo_id"] == 1
        assert call_args.kwargs["sync_roadmap"] is True
        assert "src/app.py" in call_args.kwargs["files_created"]

    @pytest.mark.asyncio
    async def test_checkpoint_json_only_mode(
        self,
        sample_state: ExecutorGraphState,
        mock_checkpointer: MagicMock,
        mock_todo_manager: MagicMock,
    ) -> None:
        """Test checkpoint with sync_roadmap=False (JSON-only mode)."""
        await checkpoint_node(
            sample_state,
            checkpointer=mock_checkpointer,
            todo_manager=mock_todo_manager,
            sync_roadmap=False,
        )

        # Verify called with sync_roadmap=False
        call_args = mock_todo_manager.mark_completed.call_args
        assert call_args.kwargs["sync_roadmap"] is False

    @pytest.mark.asyncio
    async def test_checkpoint_without_todo_manager(
        self,
        sample_state: ExecutorGraphState,
        mock_checkpointer: MagicMock,
    ) -> None:
        """Test checkpoint works without TodoListManager."""
        result = await checkpoint_node(
            sample_state,
            checkpointer=mock_checkpointer,
            todo_manager=None,
        )

        # Should complete successfully
        assert result["error"] is None
        assert result["last_checkpoint_sha"] == "abc123def456"

    def test_pick_task_reads_from_state_todos(
        self,
        sample_state: ExecutorGraphState,
        sample_todo: TodoItem,
    ) -> None:
        """Test that pick_task reads todos from state (sourced from TodoListManager)."""
        # Reset current task to pick fresh
        sample_state["current_task"] = None
        sample_state["completed_todos"] = []

        result = pick_task_node(sample_state)

        # Should pick the todo from state
        assert result["current_task"] is not None
        assert result["current_task"].id == sample_todo.id
        assert result["current_task"].title == sample_todo.title

    def test_pick_task_skips_completed(
        self,
        sample_state: ExecutorGraphState,
        sample_todo: TodoItem,
    ) -> None:
        """Test that pick_task skips already completed todos."""
        sample_state["current_task"] = None
        sample_state["completed_todos"] = ["1"]  # Mark as completed

        result = pick_task_node(sample_state)

        # Should not pick completed task
        assert result["current_task"] is None
        assert result["should_continue"] is False


# =============================================================================
# Tests: 1.4.3 TaskVerifier Integration
# =============================================================================


class TestTaskVerifierIntegration:
    """Tests for TaskVerifier integration in verify_task_node."""

    @pytest.mark.asyncio
    async def test_verifier_success_sets_verified(
        self,
        sample_state: ExecutorGraphState,
        mock_verifier: MagicMock,
    ) -> None:
        """Test that successful verification sets verified=True."""
        result = await verify_task_node(
            sample_state,
            verifier=mock_verifier,
        )

        # Verify verifier was called
        mock_verifier.verify.assert_called_once()

        # Check state
        assert result["verified"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_verifier_failure_sets_error(
        self,
        sample_state: ExecutorGraphState,
        mock_verifier: MagicMock,
    ) -> None:
        """Test that verification failure sets error in state."""
        # Mock VerificationResult with failure
        mock_failure = MagicMock()
        mock_failure.message = "Syntax error"
        mock_result = MagicMock()
        mock_result.overall = False
        mock_result.get_failures.return_value = [mock_failure]
        mock_verifier.verify = AsyncMock(return_value=mock_result)

        result = await verify_task_node(
            sample_state,
            verifier=mock_verifier,
        )

        assert result["verified"] is False
        assert result["error"] is not None
        assert result["error"]["error_type"] == "verification"
        assert "Syntax error" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_verifier_result_stored_in_state(
        self,
        sample_state: ExecutorGraphState,
        mock_verifier: MagicMock,
    ) -> None:
        """Test that verification result is stored in state for routing."""
        result = await verify_task_node(
            sample_state,
            verifier=mock_verifier,
        )

        # Result stored in state for route_after_verify
        assert "verified" in result
        assert result["verified"] is True

    @pytest.mark.asyncio
    async def test_verifier_unchanged_api(
        self,
        sample_state: ExecutorGraphState,
        mock_verifier: MagicMock,
        sample_todo: TodoItem,
    ) -> None:
        """Test that verifier API is called with task and levels."""
        from ai_infra.executor.verifier import CheckLevel

        await verify_task_node(
            sample_state,
            verifier=mock_verifier,
            check_level=CheckLevel.SYNTAX,
        )

        # Verify called with expected args (uses 'levels' list, not 'files' and 'level')
        call_args = mock_verifier.verify.call_args
        assert call_args.kwargs["task"] == sample_todo
        assert call_args.kwargs["levels"] == [CheckLevel.SYNTAX]

    @pytest.mark.asyncio
    async def test_verifier_without_verifier_auto_passes(
        self,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test that missing verifier auto-passes (dev mode)."""
        result = await verify_task_node(
            sample_state,
            verifier=None,
        )

        # Should auto-pass
        assert result["verified"] is True
        assert result["error"] is None


# =============================================================================
# Integration Tests: Full Flow
# =============================================================================


class TestIntegrationFlow:
    """Integration tests for Phase 1.4 component integration."""

    @pytest.mark.asyncio
    async def test_full_checkpoint_flow(
        self,
        sample_state: ExecutorGraphState,
        mock_checkpointer: MagicMock,
        mock_todo_manager: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Test full flow: verify -> checkpoint with all integrations."""
        # Step 1: Verify task
        verified_state = await verify_task_node(
            sample_state,
            verifier=mock_verifier,
        )
        assert verified_state["verified"] is True

        # Step 2: Checkpoint with git + ROADMAP sync
        final_state = await checkpoint_node(
            verified_state,
            checkpointer=mock_checkpointer,
            todo_manager=mock_todo_manager,
            sync_roadmap=True,
        )

        # Verify both integrations worked
        assert final_state["last_checkpoint_sha"] == "abc123def456"
        mock_todo_manager.mark_completed.assert_called_once()

    def test_executor_graph_accepts_all_components(self) -> None:
        """Test that ExecutorGraph accepts all integrated components."""
        # Should not raise - just verify constructor signature
        # We don't actually build since we don't have real components
        import inspect

        from ai_infra.executor.graph import ExecutorGraph

        sig = inspect.signature(ExecutorGraph.__init__)
        params = list(sig.parameters.keys())

        # Phase 1.4 components should be in signature
        assert "checkpointer" in params  # 1.4.1
        assert "todo_manager" in params  # 1.4.2
        assert "verifier" in params  # 1.4.3
        assert "sync_roadmap" in params  # 1.4.2 option
