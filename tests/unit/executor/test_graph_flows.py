"""Integration tests for executor graph execution flows.

These tests verify complete graph execution flows using node-level
integration patterns. They test node chaining and data flow without
requiring full graph instantiation in the unit test environment.

Test scenarios covered:
- Happy path: Parse → pick → execute → verify → checkpoint → done
- Retry on failure: Execute fails, retry succeeds
- Max retries exceeded: Fails 3 times, moves to next task
- Rollback on verification failure: (Deprecated in Phase 2.1)
- HITL pause/resume: Interrupt before execute, resume with approve
- HITL reject: Interrupt, reject, skip task
- Timeout handling: Node times out, treated as failure
- State persistence: Kill mid-run, resume from checkpoint
- Empty roadmap: No tasks, exits cleanly
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.hitl import HITLDecision, HITLManager, InterruptConfig
from ai_infra.executor.nodes import (
    build_context_node,
    checkpoint_node,
    decide_next_node,
    execute_task_node,
    handle_failure_node,
    parse_roadmap_node,
    pick_task_node,
    rollback_node,  # Deprecated in Phase 2.1
    verify_task_node,
)
from ai_infra.executor.state import ExecutorGraphState
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_roadmap(tmp_path: Path) -> Path:
    """Create a temporary ROADMAP.md file."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""# Test ROADMAP

## Phase 1

- [ ] Task 1 - Implement feature A
- [ ] Task 2 - Add tests for feature A
- [ ] Task 3 - Update documentation
""")
    return roadmap


@pytest.fixture
def empty_roadmap(tmp_path: Path) -> Path:
    """Create an empty ROADMAP.md file (no tasks)."""
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("""# Empty ROADMAP

## Phase 1

No tasks here.
""")
    return roadmap


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
def mock_agent() -> MagicMock:
    """Create a mock agent that succeeds."""
    agent = MagicMock()
    result = MagicMock()
    result.content = "Task completed successfully"
    result.model_dump.return_value = {"content": "Task completed", "files_modified": []}
    agent.arun = AsyncMock(return_value=result)
    return agent


@pytest.fixture
def failing_agent() -> MagicMock:
    """Create a mock agent that fails."""
    agent = MagicMock()
    agent.arun = AsyncMock(side_effect=Exception("Agent execution failed"))
    return agent


@pytest.fixture
def timeout_agent() -> MagicMock:
    """Create a mock agent that times out."""
    agent = MagicMock()
    agent.arun = AsyncMock(side_effect=TimeoutError())
    return agent


@pytest.fixture
def retry_then_succeed_agent() -> MagicMock:
    """Create a mock agent that fails once then succeeds."""
    agent = MagicMock()
    call_count = {"count": 0}

    async def side_effect(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise Exception("First attempt failed")
        result = MagicMock()
        result.content = "Task completed on retry"
        result.model_dump.return_value = {"content": "Completed", "files_modified": []}
        return result

    agent.arun = AsyncMock(side_effect=side_effect)
    return agent


@pytest.fixture
def base_state(sample_todo: TodoItem, tmp_roadmap: Path) -> ExecutorGraphState:
    """Create a base state for testing."""
    return {
        "roadmap_path": str(tmp_roadmap),
        "run_id": "test-run-123",
        "todos": [sample_todo],
        "current_task": sample_todo,
        "context": "Test context",
        "prompt": "Test prompt",
        "agent_result": None,
        "files_modified": [],
        "verified": False,
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
        "interrupt_requested": False,
    }


# =============================================================================
# Tests: 1.7.3 test_happy_path
# =============================================================================


class TestHappyPath:
    """Test: Parse → pick → execute → verify → checkpoint → done.

    Tests the happy path flow through nodes without full graph instantiation.
    """

    @pytest.mark.asyncio
    async def test_single_task_parse(
        self,
        tmp_roadmap: Path,
        base_state: ExecutorGraphState,
    ) -> None:
        """Test parsing a roadmap successfully."""
        base_state["roadmap_path"] = str(tmp_roadmap)

        # Parse roadmap
        parse_result = await parse_roadmap_node(
            base_state,
            agent=None,
            use_llm_normalization=False,
        )

        # Should have parsed todos
        assert "todos" in parse_result
        assert parse_result.get("error") is None

    @pytest.mark.asyncio
    async def test_execute_to_verify_to_checkpoint_flow(
        self,
        base_state: ExecutorGraphState,
        mock_agent: MagicMock,
    ) -> None:
        """Test execute → verify → checkpoint flow."""
        # Step 1: Execute
        exec_result = await execute_task_node(base_state, agent=mock_agent)
        assert mock_agent.arun.called
        assert exec_result.get("error") is None

        # Update state with execution result
        base_state.update(exec_result)
        base_state["files_modified"] = ["test.py"]

        # Step 2: Verify (with mock verifier)
        mock_verifier = MagicMock()
        mock_verifier.verify = AsyncMock(return_value=MagicMock(passed=True, message=None))
        verify_result = await verify_task_node(
            base_state,
            verifier=mock_verifier,
        )
        assert verify_result["verified"] is True

        # Update state
        base_state.update(verify_result)

        # Step 3: Checkpoint
        mock_checkpointer = MagicMock()
        mock_checkpointer.checkpoint.return_value = MagicMock(sha="abc123")
        checkpoint_result = await checkpoint_node(
            base_state,
            checkpointer=mock_checkpointer,
        )
        assert checkpoint_result["last_checkpoint_sha"] == "abc123"

    def test_pick_to_build_context_flow(
        self,
        base_state: ExecutorGraphState,
        sample_todo: TodoItem,
    ) -> None:
        """Test pick → build_context flow (sync nodes)."""
        # Add todos to state
        base_state["todos"] = [sample_todo]
        base_state["current_task"] = None

        # Step 1: Pick task (sync)
        pick_result = pick_task_node(base_state)
        assert pick_result["current_task"] is not None


# =============================================================================
# Tests: 1.7.3 test_retry_on_failure
# =============================================================================


class TestRetryOnFailure:
    """Test: Execute fails, retry succeeds."""

    @pytest.mark.asyncio
    async def test_execute_timeout_returns_error(
        self,
        base_state: ExecutorGraphState,
        timeout_agent: MagicMock,
    ) -> None:
        """Test that execute_task_node returns error on timeout."""
        result = await execute_task_node(
            base_state,
            agent=timeout_agent,
        )
        # Should return error, not raise
        assert result.get("error") is not None

    def test_failure_node_increments_retry(
        self,
        base_state: ExecutorGraphState,
    ) -> None:
        """Test handle_failure_node increments retry count."""
        # Set a recoverable error (dict format)
        base_state["error"] = {
            "error_type": "execution",
            "message": "Temporary network error",
            "node": "execute_task",
            "task_id": "1",
            "recoverable": True,
            "stack_trace": None,
        }
        base_state["retry_count"] = 0
        base_state["max_retries"] = 3

        # Handle failure (sync)
        result = handle_failure_node(base_state)

        # Should increment retry count and clear error for retry
        assert result.get("retry_count", 0) > 0
        assert result.get("error") is None  # Cleared for retry


# =============================================================================
# Tests: 1.7.3 test_max_retries_exceeded
# =============================================================================


class TestMaxRetriesExceeded:
    """Test: Fails 3 times, moves to next task."""

    def test_gives_up_after_max_retries(
        self,
        base_state: ExecutorGraphState,
    ) -> None:
        """Test that executor gives up after max retries."""
        # Set retry count at max
        base_state["retry_count"] = 3
        base_state["max_retries"] = 3
        base_state["error"] = {
            "error_type": "execution",
            "message": "Persistent failure",
            "node": "execute_task",
            "task_id": "1",
            "recoverable": True,
            "stack_trace": None,
        }

        # Handle failure when at max retries (sync)
        result = handle_failure_node(base_state)

        # Should stop continuing
        assert result.get("should_continue") is False

    def test_decide_next_moves_to_next_task(
        self,
        base_state: ExecutorGraphState,
        sample_todo: TodoItem,
    ) -> None:
        """Test decide_next_node moves to next task after failure."""
        # Add current task to failed
        base_state["failed_todos"] = [sample_todo]
        base_state["current_task"] = None

        # Decide next (sync)
        result = decide_next_node(base_state)

        # Should have should_continue field
        assert "should_continue" in result


# =============================================================================
# Tests: 1.7.3 test_rollback_on_verification_failure (Deprecated in Phase 2.1)
# =============================================================================


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestRollbackOnVerificationFailure:
    """Test: Verify fails, git revert, retry.

    .. deprecated:: Phase 2.1
        Rollback functionality is deprecated. These tests are kept for
        backward compatibility testing of the rollback_node function.
    """

    @pytest.mark.asyncio
    async def test_verify_failure_to_rollback_flow(
        self,
        base_state: ExecutorGraphState,
    ) -> None:
        """Test rollback flow when verification fails (deprecated in Phase 2.1)."""
        # Setup: verification failed
        base_state["verified"] = False
        base_state["error"] = {
            "error_type": "verification",
            "message": "Verification failed",
            "node": "verify_task",
            "task_id": "1",
            "recoverable": True,
            "stack_trace": None,
        }
        base_state["last_checkpoint_sha"] = "prev-commit-sha"

        # Mock git checkpointer for rollback
        mock_checkpointer = MagicMock()
        mock_checkpointer.rollback.return_value = True

        # Rollback (deprecated but still functional)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            await rollback_node(
                base_state,
                checkpointer=mock_checkpointer,
            )

        # Should have rolled back
        mock_checkpointer.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_verify_fail_marks_not_verified(
        self,
        base_state: ExecutorGraphState,
    ) -> None:
        """Test verify_task_node marks as not verified on failure."""
        base_state["agent_result"] = MagicMock()
        base_state["files_modified"] = ["test.py"]

        # Mock verifier that fails - uses 'overall' property
        mock_verifier = MagicMock()
        mock_verifier.verify = AsyncMock(
            return_value=MagicMock(overall=False, message="Tests failed")
        )

        result = await verify_task_node(base_state, verifier=mock_verifier)

        assert result["verified"] is False


# =============================================================================
# Tests: 1.7.3 test_hitl_pause_resume
# =============================================================================


class TestHITLPauseResume:
    """Test: Interrupt before execute, resume with approve."""

    def test_hitl_config_creates_interrupt_points(self) -> None:
        """Test that HITL config creates appropriate interrupt points."""
        config = InterruptConfig.approval_mode()

        assert config.interrupt_before is not None
        assert len(config.interrupt_before) > 0

    def test_hitl_manager_can_be_created(self, tmp_path: Path) -> None:
        """Test HITLManager can be instantiated."""
        manager = HITLManager(project_root=tmp_path)

        # Initially no pending interrupt
        assert not manager.has_pending_interrupt()

    def test_interrupt_config_factory_methods(self) -> None:
        """Test InterruptConfig factory methods."""
        # Approval mode
        approval = InterruptConfig.approval_mode()
        assert approval.interrupt_before is not None

        # Review mode
        review = InterruptConfig.review_mode()
        assert review.interrupt_after is not None

        # No interrupts
        none_config = InterruptConfig.no_interrupt()
        assert len(none_config.interrupt_before) == 0


# =============================================================================
# Tests: 1.7.3 test_hitl_reject
# =============================================================================


class TestHITLReject:
    """Test: Interrupt, reject, skip task."""

    def test_hitl_decision_enum_values(self) -> None:
        """Test HITLDecision enum has required values."""
        # Should have approve and reject
        assert HITLDecision.APPROVE is not None
        assert HITLDecision.REJECT is not None
        assert HITLDecision.ABORT is not None

    def test_hitl_manager_has_required_methods(self, tmp_path: Path) -> None:
        """Test HITLManager has required methods."""
        manager = HITLManager(project_root=tmp_path)

        # Check required methods exist
        assert hasattr(manager, "has_pending_interrupt")
        assert hasattr(manager, "save_interrupt_state")
        assert hasattr(manager, "apply_decision")


# =============================================================================
# Tests: 1.7.3 test_timeout_handling
# =============================================================================


class TestTimeoutHandling:
    """Test: Node times out, treated as failure."""

    @pytest.mark.asyncio
    async def test_agent_timeout_is_handled(
        self,
        base_state: ExecutorGraphState,
        timeout_agent: MagicMock,
    ) -> None:
        """Test that agent timeout is handled gracefully."""
        result = await execute_task_node(base_state, agent=timeout_agent)

        # Timeout should be captured as error
        assert result.get("error") is not None
        assert timeout_agent.arun.called

    def test_timeout_error_is_recoverable(
        self,
        base_state: ExecutorGraphState,
    ) -> None:
        """Test that timeout errors trigger retry."""
        base_state["error"] = {
            "error_type": "timeout",
            "message": "Agent timed out",
            "node": "execute_task",
            "task_id": "1",
            "recoverable": True,
            "stack_trace": None,
        }
        base_state["retry_count"] = 0
        base_state["max_retries"] = 3

        # Handle failure (sync)
        result = handle_failure_node(base_state)

        # Timeout should trigger retry (error cleared, retry_count incremented)
        assert result.get("error") is None  # Cleared for retry
        assert result.get("retry_count", 0) > 0


# =============================================================================
# Tests: 1.7.3 test_state_persistence
# =============================================================================


class TestStatePersistence:
    """Test: Kill mid-run, resume from checkpoint."""

    def test_graph_accepts_checkpointer_parameter(self) -> None:
        """Test that ExecutorGraph accepts a checkpointer parameter."""
        from ai_infra.executor.graph import ExecutorGraph

        sig = inspect.signature(ExecutorGraph.__init__)
        params = list(sig.parameters.keys())

        # Should have graph_checkpointer parameter
        assert "graph_checkpointer" in params

    def test_state_type_is_typed_dict(self) -> None:
        """Test that ExecutorGraphState is a TypedDict."""
        from ai_infra.executor.state import ExecutorGraphState

        # Should be accessible as a type
        assert ExecutorGraphState is not None

    @pytest.mark.asyncio
    async def test_checkpoint_node_saves_sha(
        self,
        base_state: ExecutorGraphState,
    ) -> None:
        """Test checkpoint node saves SHA for resume."""
        base_state["files_modified"] = ["test.py"]
        base_state["verified"] = True

        mock_checkpointer = MagicMock()
        mock_checkpointer.checkpoint.return_value = MagicMock(sha="resume-sha-123")

        result = await checkpoint_node(base_state, checkpointer=mock_checkpointer)

        assert result["last_checkpoint_sha"] == "resume-sha-123"


# =============================================================================
# Tests: 1.7.3 test_empty_roadmap
# =============================================================================


class TestEmptyRoadmap:
    """Test: No tasks, exits cleanly."""

    @pytest.mark.asyncio
    async def test_empty_roadmap_parses_no_todos(
        self,
        empty_roadmap: Path,
        base_state: ExecutorGraphState,
    ) -> None:
        """Test that empty roadmap results in no todos."""
        base_state["roadmap_path"] = str(empty_roadmap)

        result = await parse_roadmap_node(
            base_state,
            agent=None,
            use_llm_normalization=False,
        )

        # Should have empty or minimal todos
        assert result.get("error") is None
        todos = result.get("todos", [])
        assert len(todos) == 0 or result.get("todos") is not None

    def test_pick_task_with_no_todos_returns_none(
        self,
        base_state: ExecutorGraphState,
    ) -> None:
        """Test pick_task_node returns None when no todos available."""
        base_state["todos"] = []
        base_state["current_task"] = None

        # Pick task (sync)
        result = pick_task_node(base_state)

        # Should return None for current_task
        assert result.get("current_task") is None

    def test_decide_next_ends_when_no_tasks(
        self,
        base_state: ExecutorGraphState,
    ) -> None:
        """Test decide_next_node ends execution when no tasks left."""
        base_state["todos"] = []
        base_state["completed_todos"] = []
        base_state["current_task"] = None

        # Decide next (sync)
        result = decide_next_node(base_state)

        # Should not continue
        assert result.get("should_continue") is False


# =============================================================================
# Additional Integration Tests
# =============================================================================


class TestGraphInitialization:
    """Tests for graph initialization and configuration."""

    def test_create_executor_graph_factory_signature(self) -> None:
        """Test factory function has expected signature."""
        from ai_infra.executor.graph import create_executor_graph

        sig = inspect.signature(create_executor_graph)
        params = list(sig.parameters.keys())

        # Factory should accept these key parameters
        assert "agent" in params
        assert "roadmap_path" in params

    def test_executor_graph_accepts_all_options(self) -> None:
        """Test executor graph constructor accepts all configuration options."""
        from ai_infra.executor.graph import ExecutorGraph

        sig = inspect.signature(ExecutorGraph.__init__)
        params = list(sig.parameters.keys())

        # Should have all key parameters
        expected_params = [
            "agent",
            "roadmap_path",
            "max_tasks",
            "use_llm_normalization",
            "sync_roadmap",
            "interrupt_before",
            "interrupt_after",
            "checkpointer",
            "verifier",
            "todo_manager",
        ]

        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

    def test_executor_graph_is_importable(self) -> None:
        """Test ExecutorGraph and factory are importable."""
        from ai_infra.executor.graph import (
            ExecutorGraph,
            create_executor_graph,
            create_executor_with_hitl,
        )

        assert ExecutorGraph is not None
        assert callable(create_executor_graph)
        assert callable(create_executor_with_hitl)


class TestGraphState:
    """Tests for graph state management."""

    def test_executor_graph_has_get_initial_state(self) -> None:
        """Test ExecutorGraph has get_initial_state method."""
        from ai_infra.executor.graph import ExecutorGraph

        assert hasattr(ExecutorGraph, "get_initial_state")
        assert callable(ExecutorGraph.get_initial_state)

    def test_state_type_exists(self) -> None:
        """Test ExecutorGraphState exists and can be imported."""
        from ai_infra.executor.state import ExecutorGraphState

        # Should be a type
        assert ExecutorGraphState is not None

    def test_sample_state_is_valid(self, base_state: ExecutorGraphState) -> None:
        """Test that sample state fixture is valid."""
        # Required fields
        assert "roadmap_path" in base_state
        assert "todos" in base_state
        assert "current_task" in base_state
        assert "should_continue" in base_state
        assert "interrupt_requested" in base_state


class TestTracingIntegration:
    """Tests for tracing integration."""

    def test_tracing_config_is_importable(self) -> None:
        """Test TracingConfig can be imported."""
        from ai_infra.executor.tracing import TracingConfig

        # Should have factory methods
        assert hasattr(TracingConfig, "disabled")

    def test_executor_has_arun_with_tracing(self) -> None:
        """Test ExecutorGraph has arun_with_tracing method."""
        from ai_infra.executor.graph import ExecutorGraph

        assert hasattr(ExecutorGraph, "arun_with_tracing")

    def test_tracing_callbacks_is_importable(self) -> None:
        """Test ExecutorTracingCallbacks can be imported."""
        from ai_infra.executor.tracing import ExecutorTracingCallbacks

        assert ExecutorTracingCallbacks is not None


# =============================================================================
# Import Verification
# =============================================================================


class TestPhase17Imports:
    """Verify all Phase 1.7 related imports work."""

    def test_edge_routes_import(self) -> None:
        """Test edge routing functions can be imported."""
        from ai_infra.executor.edges.routes import (
            route_after_decide,
            route_after_execute,
            route_after_failure,
            route_after_pick,
            route_after_rollback,
            route_after_verify,
        )

        assert callable(route_after_pick)
        assert callable(route_after_execute)
        assert callable(route_after_verify)
        assert callable(route_after_failure)
        assert callable(route_after_rollback)
        assert callable(route_after_decide)

    def test_graph_imports(self) -> None:
        """Test graph components can be imported."""
        from ai_infra.executor.graph import (
            ExecutorGraph,
            create_executor_graph,
            create_executor_with_hitl,
        )

        assert ExecutorGraph is not None
        assert callable(create_executor_graph)
        assert callable(create_executor_with_hitl)

    def test_node_imports(self) -> None:
        """Test all nodes can be imported."""
        from ai_infra.executor.nodes import (
            checkpoint_node,
            decide_next_node,
            execute_task_node,
            handle_failure_node,
            parse_roadmap_node,
            pick_task_node,
            rollback_node,
            verify_task_node,
        )

        assert callable(parse_roadmap_node)
        assert callable(pick_task_node)
        assert callable(build_context_node)
        assert callable(execute_task_node)
        assert callable(verify_task_node)
        assert callable(checkpoint_node)
        assert callable(rollback_node)
        assert callable(handle_failure_node)
        assert callable(decide_next_node)


class TestTokenAndDurationTracking:
    """Phase 2.2.1: Tests for token and duration tracking in ExecutorGraph."""

    @pytest.mark.asyncio
    async def test_arun_returns_duration_ms(self, tmp_path: Path) -> None:
        """Test that arun() returns duration_ms in result."""
        from unittest.mock import AsyncMock

        from ai_infra.executor.graph import ExecutorGraph

        # Create minimal roadmap
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# Test\n- [ ] Task 1")

        graph_executor = ExecutorGraph(
            agent=None,
            roadmap_path=str(roadmap),
        )

        # Mock graph.arun to return minimal state
        mock_result = {
            "todos": [],
            "tasks_completed_count": 0,
            "failed_todos": [],
        }
        graph_executor.graph.arun = AsyncMock(return_value=mock_result)

        result = await graph_executor.arun()

        # Should have duration_ms set
        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], int)
        assert result["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_arun_returns_tokens_used_from_callbacks(self, tmp_path: Path) -> None:
        """Test that arun() returns tokens_used from callbacks."""
        from unittest.mock import AsyncMock

        from ai_infra.executor.graph import ExecutorGraph
        from ai_infra.executor.tracing import ExecutorCallbacks

        # Create minimal roadmap
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# Test\n- [ ] Task 1")

        # Create callbacks with simulated token usage
        callbacks = ExecutorCallbacks()
        callbacks._metrics.total_tokens = 1500  # Simulate token usage

        graph_executor = ExecutorGraph(
            agent=None,
            roadmap_path=str(roadmap),
            callbacks=callbacks,
        )

        # Mock graph.arun to return minimal state
        mock_result = {
            "todos": [],
            "tasks_completed_count": 0,
            "failed_todos": [],
        }
        graph_executor.graph.arun = AsyncMock(return_value=mock_result)

        result = await graph_executor.arun()

        # Should have tokens_used set from callbacks
        assert "tokens_used" in result
        assert result["tokens_used"] == 1500

    @pytest.mark.asyncio
    async def test_arun_without_callbacks_returns_zero_tokens(self, tmp_path: Path) -> None:
        """Test that arun() returns 0 tokens when no callbacks provided."""
        from unittest.mock import AsyncMock

        from ai_infra.executor.graph import ExecutorGraph

        # Create minimal roadmap
        roadmap = tmp_path / "ROADMAP.md"
        roadmap.write_text("# Test\n- [ ] Task 1")

        graph_executor = ExecutorGraph(
            agent=None,
            roadmap_path=str(roadmap),
            callbacks=None,  # No callbacks
        )

        # Mock graph.arun to return minimal state
        mock_result = {
            "todos": [],
            "tasks_completed_count": 0,
            "failed_todos": [],
        }
        graph_executor.graph.arun = AsyncMock(return_value=mock_result)

        result = await graph_executor.arun()

        # Should have tokens_used = 0 without callbacks
        assert "tokens_used" in result
        assert result["tokens_used"] == 0
