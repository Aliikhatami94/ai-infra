"""Tests for Phase 1.5: Human-in-the-Loop Implementation.

Tests cover:
- 1.5.1: Interrupt points configuration
- 1.5.2: Resume logic with approval/rejection
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ai_infra.executor.hitl import (
    HITLDecision,
    HITLManager,
    HITLState,
    InterruptConfig,
    InterruptPoint,
    get_interrupt_lists,
)
from ai_infra.executor.state import ExecutorGraphState
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hitl_manager(temp_project_dir: Path) -> HITLManager:
    """Create a HITLManager instance."""
    return HITLManager(temp_project_dir)


@pytest.fixture
def sample_todo() -> TodoItem:
    """Create a sample todo item."""
    return TodoItem(
        id=1,
        title="Implement feature X",
        description="Add the new feature to the system",
        status=TodoStatus.IN_PROGRESS,
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
        "files_modified": ["src/feature.py", "tests/test_feature.py"],
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
# Tests: 1.5.1 Interrupt Points Configuration
# =============================================================================


class TestInterruptConfig:
    """Tests for InterruptConfig class."""

    def test_approval_mode(self) -> None:
        """Test approval mode creates correct config."""
        config = InterruptConfig.approval_mode()

        assert "execute_task" in config.interrupt_before
        assert config.require_approval is True
        assert config.interrupt_after == []

    def test_review_mode(self) -> None:
        """Test review mode creates correct config."""
        config = InterruptConfig.review_mode()

        assert "verify_task" in config.interrupt_after
        assert config.require_approval is True
        assert config.interrupt_before == []

    def test_full_control_mode(self) -> None:
        """Test full control mode creates correct config."""
        config = InterruptConfig.full_control_mode()

        assert "execute_task" in config.interrupt_before
        assert "verify_task" in config.interrupt_after
        assert config.require_approval is True

    def test_no_interrupt(self) -> None:
        """Test no interrupt mode."""
        config = InterruptConfig.no_interrupt()

        assert config.interrupt_before == []
        assert config.interrupt_after == []
        assert config.require_approval is False

    def test_get_interrupt_lists(self) -> None:
        """Test getting interrupt lists for Graph constructor."""
        config = InterruptConfig(
            interrupt_before=["execute_task"],
            interrupt_after=["verify_task"],
        )

        before, after = get_interrupt_lists(config)

        assert before == ["execute_task"]
        assert after == ["verify_task"]


class TestInterruptPoint:
    """Tests for InterruptPoint enum."""

    def test_interrupt_points_defined(self) -> None:
        """Test that all interrupt points are defined."""
        assert InterruptPoint.BEFORE_EXECUTE.value == "execute_task"
        assert InterruptPoint.AFTER_VERIFY.value == "verify_task"
        assert InterruptPoint.BEFORE_CHECKPOINT.value == "checkpoint"
        assert InterruptPoint.AFTER_FAILURE.value == "handle_failure"

    def test_interrupt_point_usable_in_config(self) -> None:
        """Test that InterruptPoint enum can be used in config."""
        config = InterruptConfig(
            interrupt_before=[InterruptPoint.BEFORE_EXECUTE.value],
        )

        assert "execute_task" in config.interrupt_before


# =============================================================================
# Tests: 1.5.2 Resume Logic
# =============================================================================


class TestHITLState:
    """Tests for HITLState dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        state = HITLState(
            thread_id="executor-abc123",
            interrupt_node="execute_task",
            interrupt_type="before",
            task_id="1",
            task_title="Test task",
            interrupted_at="2025-01-08T12:00:00Z",
            context_summary="Task: Test task",
        )

        data = state.to_dict()

        assert data["thread_id"] == "executor-abc123"
        assert data["interrupt_node"] == "execute_task"
        assert data["interrupt_type"] == "before"
        assert data["task_id"] == "1"

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "thread_id": "executor-abc123",
            "interrupt_node": "execute_task",
            "interrupt_type": "before",
            "task_id": "1",
            "task_title": "Test task",
            "interrupted_at": "2025-01-08T12:00:00Z",
            "context_summary": "Task: Test task",
            "decision": "approve",
            "decision_at": "2025-01-08T12:05:00Z",
            "notes": "Looks good",
        }

        state = HITLState.from_dict(data)

        assert state.thread_id == "executor-abc123"
        assert state.decision == "approve"
        assert state.notes == "Looks good"


class TestHITLManager:
    """Tests for HITLManager class."""

    def test_save_interrupt_state(
        self,
        hitl_manager: HITLManager,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test saving interrupt state."""
        state = hitl_manager.save_interrupt_state(
            thread_id="executor-abc123",
            interrupt_node="execute_task",
            interrupt_type="before",
            graph_state=sample_state,
        )

        assert state.thread_id == "executor-abc123"
        assert state.interrupt_node == "execute_task"
        assert state.task_id == "1"
        assert state.task_title == "Implement feature X"
        assert "Implement feature X" in state.context_summary

    def test_has_pending_interrupt(
        self,
        hitl_manager: HITLManager,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test checking for pending interrupt."""
        # Initially no pending interrupt
        assert hitl_manager.has_pending_interrupt() is False

        # Save interrupt state
        hitl_manager.save_interrupt_state(
            thread_id="executor-abc123",
            interrupt_node="execute_task",
            interrupt_type="before",
            graph_state=sample_state,
        )

        # Now should have pending interrupt
        assert hitl_manager.has_pending_interrupt() is True

    def test_apply_decision(
        self,
        hitl_manager: HITLManager,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test applying a decision."""
        hitl_manager.save_interrupt_state(
            thread_id="executor-abc123",
            interrupt_node="execute_task",
            interrupt_type="before",
            graph_state=sample_state,
        )

        state = hitl_manager.apply_decision(
            HITLDecision.APPROVE,
            notes="Reviewed and approved",
        )

        assert state.decision == "approve"
        assert state.notes == "Reviewed and approved"
        assert state.decision_at is not None

    def test_apply_decision_no_pending(
        self,
        hitl_manager: HITLManager,
    ) -> None:
        """Test applying decision when no pending interrupt."""
        with pytest.raises(ValueError, match="No pending interrupt"):
            hitl_manager.apply_decision(HITLDecision.APPROVE)

    def test_get_resume_config(
        self,
        hitl_manager: HITLManager,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test getting resume config."""
        hitl_manager.save_interrupt_state(
            thread_id="executor-abc123",
            interrupt_node="execute_task",
            interrupt_type="before",
            graph_state=sample_state,
        )

        config = hitl_manager.get_resume_config()

        assert config["configurable"]["thread_id"] == "executor-abc123"

    def test_should_continue_approve(
        self,
        hitl_manager: HITLManager,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test should_continue returns True for approve."""
        hitl_manager.save_interrupt_state(
            thread_id="executor-abc123",
            interrupt_node="execute_task",
            interrupt_type="before",
            graph_state=sample_state,
        )
        hitl_manager.apply_decision(HITLDecision.APPROVE)

        assert hitl_manager.should_continue_after_decision() is True

    def test_should_continue_abort(
        self,
        hitl_manager: HITLManager,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test should_continue returns False for abort."""
        hitl_manager.save_interrupt_state(
            thread_id="executor-abc123",
            interrupt_node="execute_task",
            interrupt_type="before",
            graph_state=sample_state,
        )
        hitl_manager.apply_decision(HITLDecision.ABORT)

        assert hitl_manager.should_continue_after_decision() is False

    def test_get_decision_action(
        self,
        hitl_manager: HITLManager,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test getting action from decision."""
        hitl_manager.save_interrupt_state(
            thread_id="executor-abc123",
            interrupt_node="execute_task",
            interrupt_type="before",
            graph_state=sample_state,
        )

        # Approve -> continue
        hitl_manager.apply_decision(HITLDecision.APPROVE)
        assert hitl_manager.get_decision_action() == "continue"

    def test_clear_interrupt_state(
        self,
        hitl_manager: HITLManager,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test clearing interrupt state."""
        hitl_manager.save_interrupt_state(
            thread_id="executor-abc123",
            interrupt_node="execute_task",
            interrupt_type="before",
            graph_state=sample_state,
        )

        assert hitl_manager.has_pending_interrupt() is True

        hitl_manager.clear_interrupt_state()

        assert hitl_manager.has_pending_interrupt() is False

    def test_context_summary_includes_files(
        self,
        hitl_manager: HITLManager,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test context summary includes files modified."""
        state = hitl_manager.save_interrupt_state(
            thread_id="executor-abc123",
            interrupt_node="verify_task",
            interrupt_type="after",
            graph_state=sample_state,
        )

        assert "Files modified: 2" in state.context_summary
        assert "src/feature.py" in state.context_summary

    def test_context_summary_includes_error(
        self,
        hitl_manager: HITLManager,
        sample_state: ExecutorGraphState,
    ) -> None:
        """Test context summary includes error info."""
        sample_state["error"] = {
            "error_type": "execution",
            "message": "Failed to compile",
        }

        state = hitl_manager.save_interrupt_state(
            thread_id="executor-abc123",
            interrupt_node="handle_failure",
            interrupt_type="after",
            graph_state=sample_state,
        )

        assert "Failed to compile" in state.context_summary


class TestHITLDecision:
    """Tests for HITLDecision enum."""

    def test_all_decisions_defined(self) -> None:
        """Test all decision types are defined."""
        assert HITLDecision.APPROVE.value == "approve"
        assert HITLDecision.REJECT.value == "reject"
        assert HITLDecision.ABORT.value == "abort"
        assert HITLDecision.RETRY.value == "retry"
        assert HITLDecision.MODIFY.value == "modify"


# =============================================================================
# Tests: ExecutorGraph HITL Integration
# =============================================================================


class TestExecutorGraphHITL:
    """Tests for ExecutorGraph HITL integration."""

    def test_create_executor_with_hitl_approval(self) -> None:
        """Test creating executor with approval mode."""
        # Just verify the factory function works
        import inspect

        from ai_infra.executor.graph import create_executor_with_hitl

        sig = inspect.signature(create_executor_with_hitl)
        params = list(sig.parameters.keys())

        assert "agent" in params
        assert "roadmap_path" in params
        assert "hitl_config" in params

    def test_executor_graph_has_resume_methods(self) -> None:
        """Test that ExecutorGraph has resume methods."""
        from ai_infra.executor.graph import ExecutorGraph

        # Check methods exist
        assert hasattr(ExecutorGraph, "aresume")
        assert hasattr(ExecutorGraph, "resume")
        assert hasattr(ExecutorGraph, "has_pending_interrupt")

    def test_executor_graph_accepts_interrupt_params(self) -> None:
        """Test that ExecutorGraph accepts interrupt_before/after."""
        import inspect

        from ai_infra.executor.graph import ExecutorGraph

        sig = inspect.signature(ExecutorGraph.__init__)
        params = list(sig.parameters.keys())

        assert "interrupt_before" in params
        assert "interrupt_after" in params


# =============================================================================
# Tests: Via ExecutorConfig Option
# =============================================================================


class TestExecutorConfigIntegration:
    """Tests for ExecutorConfig HITL integration."""

    def test_create_hitl_config_from_executor_config(self) -> None:
        """Test creating HITL config from ExecutorConfig."""
        from ai_infra.executor.hitl import create_hitl_config_from_executor_config

        # Mock ExecutorConfig
        class MockConfig:
            hitl_mode = "approval"
            require_approval = True

        config = create_hitl_config_from_executor_config(MockConfig())

        assert "execute_task" in config.interrupt_before
        assert config.require_approval is True

    def test_hitl_mode_review(self) -> None:
        """Test review mode from ExecutorConfig."""
        from ai_infra.executor.hitl import create_hitl_config_from_executor_config

        class MockConfig:
            hitl_mode = "review"
            require_approval = True

        config = create_hitl_config_from_executor_config(MockConfig())

        assert "verify_task" in config.interrupt_after

    def test_hitl_mode_full(self) -> None:
        """Test full control mode from ExecutorConfig."""
        from ai_infra.executor.hitl import create_hitl_config_from_executor_config

        class MockConfig:
            hitl_mode = "full"
            require_approval = True

        config = create_hitl_config_from_executor_config(MockConfig())

        assert "execute_task" in config.interrupt_before
        assert "verify_task" in config.interrupt_after

    def test_no_hitl_mode(self) -> None:
        """Test no HITL when not configured."""
        from ai_infra.executor.hitl import create_hitl_config_from_executor_config

        class MockConfig:
            hitl_mode = None
            require_approval = False

        config = create_hitl_config_from_executor_config(MockConfig())

        assert config.interrupt_before == []
        assert config.interrupt_after == []
        assert config.require_approval is False
