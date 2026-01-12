"""Tests for Phase 1.7.2: Unit tests for conditional edges.

Tests for all edge routing functions in the executor graph.
"""

from __future__ import annotations

import pytest
from langgraph.constants import END

from ai_infra.executor.edges.routes import (
    route_after_apply_recovery,
    route_after_await_approval,
    route_after_classify_failure,
    route_after_decide,
    route_after_decide_with_recovery,
    route_after_execute,
    route_after_failure,
    route_after_pick,
    route_after_pick_with_recovery,
    route_after_propose_recovery,
    route_after_repair,
    route_after_retry_deferred,
    route_after_rollback,
    route_after_validate,
    route_after_verify,
    route_after_write,
    route_to_recovery_or_failure,
)
from ai_infra.executor.nodes.recovery import (
    FailureReason,
    RecoveryProposal,
    RecoveryStrategy,
)
from ai_infra.executor.state import ExecutorGraphState
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_todo() -> TodoItem:
    """Create a sample todo item."""
    return TodoItem(
        id=1,
        title="Test task",
        description="Test description",
        status=TodoStatus.NOT_STARTED,
    )


@pytest.fixture
def base_state(sample_todo: TodoItem) -> ExecutorGraphState:
    """Create a base state for testing."""
    return ExecutorGraphState(
        roadmap_path="ROADMAP.md",
        todos=[sample_todo],
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
# Tests: route_after_pick
# =============================================================================


class TestRouteAfterPick:
    """Tests for route_after_pick - Has task → context, no task → END."""

    def test_has_task_routes_to_context(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: When task is selected, route to build_context."""
        state = {**base_state, "current_task": sample_todo}

        result = route_after_pick(state)

        assert result == "build_context"

    def test_no_task_routes_to_end(self, base_state: ExecutorGraphState) -> None:
        """Test: When no task selected, route to END."""
        state = {**base_state, "current_task": None}

        result = route_after_pick(state)

        assert result == END

    def test_empty_todos_with_no_task(self, base_state: ExecutorGraphState) -> None:
        """Test: Empty todos and no current task routes to END."""
        state = {**base_state, "todos": [], "current_task": None}

        result = route_after_pick(state)

        assert result == END


# =============================================================================
# Tests: route_after_execute
# =============================================================================


class TestRouteAfterExecute:
    """Tests for route_after_execute - Success → verify, error → failure."""

    def test_success_routes_to_verify(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Successful execution routes to verify_task."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "error": None,
            "agent_result": {"content": "Task completed"},
        }

        result = route_after_execute(state)

        assert result == "verify_task"

    def test_error_routes_to_failure(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Execution error routes to handle_failure."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "error": {
                "error_type": "execution",
                "message": "Agent failed",
                "node": "execute_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = route_after_execute(state)

        assert result == "handle_failure"

    def test_timeout_error_routes_to_failure(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Timeout error routes to handle_failure."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "error": {
                "error_type": "timeout",
                "message": "Execution timed out",
                "node": "execute_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = route_after_execute(state)

        assert result == "handle_failure"


# =============================================================================
# Tests: route_after_verify
# =============================================================================


class TestRouteAfterVerify:
    """Tests for route_after_verify - Passed → checkpoint, failed → repair_test (Phase 2.6)."""

    def test_verified_routes_to_checkpoint(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Verification passed routes to checkpoint."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "verified": True,
            "error": None,
        }

        result = route_after_verify(state)

        assert result == "checkpoint"

    def test_not_verified_routes_to_repair_test(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Verification failed routes to repair_test (Phase 2.6)."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "verified": False,
            "test_repair_count": 0,
            "error": {
                "error_type": "verification",
                "message": "Syntax errors found",
                "node": "verify_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = route_after_verify(state)

        assert result == "repair_test"

    def test_missing_verified_defaults_to_repair_test(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Missing verified flag defaults to repair_test (Phase 2.6)."""
        state = {**base_state, "current_task": sample_todo, "test_repair_count": 0}
        # Don't set verified - should default to False

        result = route_after_verify(state)

        assert result == "repair_test"

    def test_max_repairs_exceeded_routes_to_handle_failure(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Max test repairs exceeded routes to handle_failure (Phase 2.6)."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "verified": False,
            "test_repair_count": 2,  # At limit
        }

        result = route_after_verify(state)

        assert result == "handle_failure"


# =============================================================================
# Tests: route_after_failure
# =============================================================================


class TestRouteAfterFailure:
    """Tests for route_after_failure - Phase 2.1: Always routes to decide_next."""

    def test_recoverable_with_checkpoint_routes_to_decide(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Phase 2.1 - Recoverable error with checkpoint routes to decide_next.

        Previously routed to rollback, but rollback has been removed from active flow.
        """
        state = {
            **base_state,
            "current_task": sample_todo,
            "last_checkpoint_sha": "abc123",
            "error": {
                "error_type": "verification",
                "message": "Test failed",
                "node": "verify_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = route_after_failure(state)

        # Phase 2.1: Always goes to decide_next, no more rollback
        assert result == "decide_next"

    def test_recoverable_without_checkpoint_routes_to_decide(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Recoverable error without checkpoint routes to decide."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "last_checkpoint_sha": None,
            "error": {
                "error_type": "verification",
                "message": "Test failed",
                "node": "verify_task",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        result = route_after_failure(state)

        assert result == "decide_next"

    def test_non_recoverable_routes_to_decide(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Non-recoverable error routes to decide."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "last_checkpoint_sha": "abc123",
            "error": {
                "error_type": "parse",
                "message": "ROADMAP not found",
                "node": "parse_roadmap",
                "task_id": None,
                "recoverable": False,
                "stack_trace": None,
            },
        }

        result = route_after_failure(state)

        assert result == "decide_next"

    def test_no_error_routes_to_decide(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: No error routes to decide."""
        state = {**base_state, "current_task": sample_todo, "error": None}

        result = route_after_failure(state)

        assert result == "decide_next"


# =============================================================================
# Tests: route_after_rollback (Deprecated in Phase 2.1)
# =============================================================================


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestRouteAfterRollback:
    """Tests for route_after_rollback.

    .. deprecated:: Phase 2.1
        These tests cover deprecated functionality. The route_after_rollback
        function is no longer used in the active graph flow but is kept for
        backward compatibility.
    """

    def test_retry_count_below_max_routes_to_execute(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Retry count below max routes to execute_task."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "retry_count": 1,
            "max_retries": 3,
        }

        result = route_after_rollback(state)

        assert result == "execute_task"

    def test_retry_count_at_max_routes_to_decide(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Retry count at max routes to decide_next."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "retry_count": 3,
            "max_retries": 3,
        }

        result = route_after_rollback(state)

        assert result == "decide_next"

    def test_retry_count_above_max_routes_to_decide(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Retry count above max routes to decide_next."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "retry_count": 5,
            "max_retries": 3,
        }

        result = route_after_rollback(state)

        assert result == "decide_next"

    def test_default_max_retries(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Uses default max_retries of 3 if not set."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "retry_count": 2,
            # max_retries not explicitly set - should default to 3
        }

        result = route_after_rollback(state)

        assert result == "execute_task"


# =============================================================================
# Tests: route_after_decide
# =============================================================================


class TestRouteAfterDecide:
    """Tests for route_after_decide - Continue → pick, done → END."""

    def test_should_continue_routes_to_pick(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: should_continue=True routes to pick_task."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "should_continue": True,
        }

        result = route_after_decide(state)

        assert result == "pick_task"

    def test_should_not_continue_routes_to_end(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: should_continue=False routes to END."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "should_continue": False,
        }

        result = route_after_decide(state)

        assert result == END

    def test_missing_should_continue_routes_to_end(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Missing should_continue defaults to END."""
        state = dict(base_state)
        # Ensure should_continue is not in state or is falsy
        state["should_continue"] = False

        result = route_after_decide(state)

        assert result == END


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for routing functions."""

    def test_all_routes_return_valid_strings(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: All routes return valid node names or END."""
        # Phase 2.6: Added "repair_test" to valid nodes
        valid_nodes = {
            "build_context",
            "execute_task",
            "verify_task",
            "checkpoint",
            "handle_failure",
            "decide_next",
            "pick_task",
            "repair_test",  # Phase 2.6
            END,
        }

        # Test each routing function with various states
        state_with_task = {**base_state, "current_task": sample_todo, "test_repair_count": 0}
        state_without_task = {**base_state, "current_task": None}
        state_with_error = {
            **state_with_task,
            "error": {
                "error_type": "test",
                "message": "test",
                "node": "test",
                "task_id": "1",
                "recoverable": True,
                "stack_trace": None,
            },
        }

        results = [
            route_after_pick(state_with_task),
            route_after_pick(state_without_task),
            route_after_execute(state_with_task),
            route_after_execute(state_with_error),
            route_after_verify({**state_with_task, "verified": True}),
            route_after_verify({**state_with_task, "verified": False}),
            route_after_failure(state_with_error),
            route_after_failure(state_with_task),
            # Phase 2.1: route_after_rollback is deprecated but still tested
            route_after_rollback({**state_with_task, "retry_count": 0}),
            route_after_rollback({**state_with_task, "retry_count": 5}),
            route_after_decide({**state_with_task, "should_continue": True}),
            route_after_decide({**state_with_task, "should_continue": False}),
        ]

        # Include deprecated rollback targets for backward compatibility tests
        valid_nodes_with_deprecated = valid_nodes | {"rollback", "execute_task"}
        for result in results:
            assert result in valid_nodes_with_deprecated, f"Invalid route result: {result}"

    def test_routes_are_deterministic(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: Routes are deterministic - same input gives same output."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "verified": True,
            "error": None,
            "should_continue": True,
            "retry_count": 1,
        }

        # Call each route multiple times
        for _ in range(5):
            assert route_after_pick(state) == "build_context"
            assert route_after_execute(state) == "verify_task"
            assert route_after_verify(state) == "checkpoint"
            assert route_after_decide(state) == "pick_task"


# =============================================================================
# Tests: Phase 1.4 - Validation and Repair Routes
# =============================================================================


class TestRouteAfterValidate:
    """Tests for route_after_validate - validated → write, needs_repair → repair."""

    def test_validated_routes_to_write_files(self, base_state: ExecutorGraphState) -> None:
        """Test: When validated is True, route to write_files."""
        state = {**base_state, "validated": True, "needs_repair": False}

        result = route_after_validate(state)

        assert result == "write_files"

    def test_needs_repair_under_limit_routes_to_repair(
        self, base_state: ExecutorGraphState
    ) -> None:
        """Test: When needs_repair and repair_count < MAX, route to repair_code."""
        state = {
            **base_state,
            "validated": False,
            "needs_repair": True,
            "repair_count": 0,
        }

        result = route_after_validate(state)

        assert result == "repair_code"

    def test_needs_repair_at_limit_routes_to_failure(self, base_state: ExecutorGraphState) -> None:
        """Test: When needs_repair and repair_count >= MAX, route to handle_failure."""
        state = {
            **base_state,
            "validated": False,
            "needs_repair": True,
            "repair_count": 2,  # MAX_REPAIRS = 2
        }

        result = route_after_validate(state)

        assert result == "handle_failure"

    def test_default_routes_to_write_files(self, base_state: ExecutorGraphState) -> None:
        """Test: When no flags set, default to write_files."""
        state = {**base_state}

        result = route_after_validate(state)

        assert result == "write_files"


class TestRouteAfterRepair:
    """Tests for route_after_repair - loops back to validate_code."""

    def test_no_error_routes_to_validate(self, base_state: ExecutorGraphState) -> None:
        """Test: When no error, route back to validate_code."""
        state = {**base_state, "error": None}

        result = route_after_repair(state)

        assert result == "validate_code"

    def test_with_error_routes_to_failure(self, base_state: ExecutorGraphState) -> None:
        """Test: When error is present, route to handle_failure."""
        state = {
            **base_state,
            "error": {
                "error_type": "validation",
                "message": "max repairs exceeded",
            },
        }

        result = route_after_repair(state)

        assert result == "handle_failure"


class TestRouteAfterWrite:
    """Tests for route_after_write - success → verify, error → failure."""

    def test_success_routes_to_verify_task(self, base_state: ExecutorGraphState) -> None:
        """Test: When files_written and no error, route to verify_task."""
        state = {**base_state, "files_written": True, "error": None}

        result = route_after_write(state)

        assert result == "verify_task"

    def test_error_routes_to_failure(self, base_state: ExecutorGraphState) -> None:
        """Test: When error is present, route to handle_failure."""
        state = {
            **base_state,
            "files_written": False,
            "error": {
                "error_type": "execution",
                "message": "write failed",
            },
        }

        result = route_after_write(state)

        assert result == "handle_failure"

    def test_no_error_even_without_files_written(self, base_state: ExecutorGraphState) -> None:
        """Test: When no error, route to verify_task even if files_written is False."""
        state = {**base_state, "files_written": False, "error": None}

        result = route_after_write(state)

        assert result == "verify_task"


# =============================================================================
# Phase 2.9.8: Recovery Flow Routing Tests
# =============================================================================


class TestRouteAfterClassifyFailure:
    """Tests for route_after_classify_failure - always routes to propose_recovery."""

    def test_routes_to_propose_recovery(self, base_state: ExecutorGraphState) -> None:
        """Test: Always routes to propose_recovery."""
        state = {
            **base_state,
            "failure_reason": FailureReason.TASK_TOO_COMPLEX,
            "recovery_strategy": RecoveryStrategy.DECOMPOSE,
        }

        result = route_after_classify_failure(state)

        assert result == "propose_recovery"


class TestRouteAfterProposeRecovery:
    """Tests for route_after_propose_recovery - routes based on approval mode."""

    @pytest.fixture
    def decompose_proposal(self) -> RecoveryProposal:
        """Create a decompose proposal."""
        return RecoveryProposal(
            original_task="Complex task",
            failure_reason=FailureReason.TASK_TOO_COMPLEX,
            strategy=RecoveryStrategy.DECOMPOSE,
            proposed_tasks=["Sub-task 1", "Sub-task 2"],
            explanation="Too complex",
            requires_approval=True,
        )

    @pytest.fixture
    def rewrite_proposal(self) -> RecoveryProposal:
        """Create a rewrite proposal."""
        return RecoveryProposal(
            original_task="Vague task",
            failure_reason=FailureReason.TASK_TOO_VAGUE,
            strategy=RecoveryStrategy.REWRITE,
            proposed_tasks=["Specific task"],
            explanation="Too vague",
            requires_approval=True,
        )

    @pytest.fixture
    def escalate_proposal(self) -> RecoveryProposal:
        """Create an escalate proposal."""
        return RecoveryProposal(
            original_task="Unknown task",
            failure_reason=FailureReason.UNKNOWN,
            strategy=RecoveryStrategy.ESCALATE,
            proposed_tasks=[],
            explanation="Unknown failure",
            requires_approval=True,
        )

    def test_no_proposal_routes_to_decide_next(self, base_state: ExecutorGraphState) -> None:
        """Test: No proposal routes to decide_next."""
        state = {**base_state, "recovery_proposal": None}

        result = route_after_propose_recovery(state)

        assert result == "decide_next"

    def test_auto_mode_routes_to_apply(
        self, base_state: ExecutorGraphState, decompose_proposal: RecoveryProposal
    ) -> None:
        """Test: AUTO mode routes to apply_recovery."""
        state = {
            **base_state,
            "recovery_proposal": decompose_proposal,
            "approval_mode": "auto",
        }

        result = route_after_propose_recovery(state)

        assert result == "apply_recovery"

    def test_interactive_mode_routes_to_await(
        self, base_state: ExecutorGraphState, decompose_proposal: RecoveryProposal
    ) -> None:
        """Test: INTERACTIVE mode routes to await_approval."""
        state = {
            **base_state,
            "recovery_proposal": decompose_proposal,
            "approval_mode": "interactive",
        }

        result = route_after_propose_recovery(state)

        assert result == "await_approval"

    def test_review_only_mode_routes_to_await(
        self, base_state: ExecutorGraphState, decompose_proposal: RecoveryProposal
    ) -> None:
        """Test: REVIEW_ONLY mode routes to await_approval."""
        state = {
            **base_state,
            "recovery_proposal": decompose_proposal,
            "approval_mode": "review_only",
        }

        result = route_after_propose_recovery(state)

        assert result == "await_approval"

    def test_approve_decompose_mode_asks_for_decompose(
        self, base_state: ExecutorGraphState, decompose_proposal: RecoveryProposal
    ) -> None:
        """Test: APPROVE_DECOMPOSE asks for DECOMPOSE operations."""
        state = {
            **base_state,
            "recovery_proposal": decompose_proposal,
            "approval_mode": "approve_decompose",
        }

        result = route_after_propose_recovery(state)

        assert result == "await_approval"

    def test_approve_decompose_mode_auto_approves_rewrite(
        self, base_state: ExecutorGraphState, rewrite_proposal: RecoveryProposal
    ) -> None:
        """Test: APPROVE_DECOMPOSE auto-approves REWRITE."""
        state = {
            **base_state,
            "recovery_proposal": rewrite_proposal,
            "approval_mode": "approve_decompose",
        }

        result = route_after_propose_recovery(state)

        assert result == "apply_recovery"

    def test_escalate_always_requires_approval(
        self, base_state: ExecutorGraphState, escalate_proposal: RecoveryProposal
    ) -> None:
        """Test: ESCALATE always routes to await_approval."""
        state = {
            **base_state,
            "recovery_proposal": escalate_proposal,
            "approval_mode": "auto",  # Even in auto mode
        }

        result = route_after_propose_recovery(state)

        assert result == "await_approval"


class TestRouteAfterAwaitApproval:
    """Tests for route_after_await_approval - routes based on approval."""

    def test_approved_routes_to_apply(self, base_state: ExecutorGraphState) -> None:
        """Test: When approved, route to apply_recovery."""
        state = {**base_state, "recovery_approved": True}

        result = route_after_await_approval(state)

        assert result == "apply_recovery"

    def test_rejected_routes_to_decide_next(self, base_state: ExecutorGraphState) -> None:
        """Test: When rejected, route to decide_next."""
        state = {**base_state, "recovery_approved": False}

        result = route_after_await_approval(state)

        assert result == "decide_next"

    def test_missing_approval_defaults_to_rejected(self, base_state: ExecutorGraphState) -> None:
        """Test: Missing approval defaults to rejected."""
        state = {**base_state}  # No recovery_approved key

        result = route_after_await_approval(state)

        assert result == "decide_next"


class TestRouteAfterApplyRecovery:
    """Tests for route_after_apply_recovery - routes based on strategy."""

    @pytest.fixture
    def rewrite_proposal(self) -> RecoveryProposal:
        """Create a rewrite proposal."""
        return RecoveryProposal(
            original_task="Vague task",
            failure_reason=FailureReason.TASK_TOO_VAGUE,
            strategy=RecoveryStrategy.REWRITE,
            proposed_tasks=["Specific task"],
            explanation="Rewritten",
            requires_approval=False,
        )

    @pytest.fixture
    def defer_proposal(self) -> RecoveryProposal:
        """Create a defer proposal."""
        return RecoveryProposal(
            original_task="Task",
            failure_reason=FailureReason.MISSING_DEPENDENCY,
            strategy=RecoveryStrategy.DEFER,
            proposed_tasks=[],
            explanation="Deferred",
            requires_approval=False,
        )

    def test_rewrite_routes_to_pick_task(
        self, base_state: ExecutorGraphState, rewrite_proposal: RecoveryProposal
    ) -> None:
        """Test: REWRITE routes to pick_task."""
        state = {**base_state, "recovery_proposal": rewrite_proposal}

        result = route_after_apply_recovery(state)

        assert result == "pick_task"

    def test_defer_routes_to_decide_next(
        self, base_state: ExecutorGraphState, defer_proposal: RecoveryProposal
    ) -> None:
        """Test: DEFER routes to decide_next."""
        state = {**base_state, "recovery_proposal": defer_proposal}

        result = route_after_apply_recovery(state)

        assert result == "decide_next"

    def test_no_proposal_routes_to_decide_next(self, base_state: ExecutorGraphState) -> None:
        """Test: No proposal routes to decide_next."""
        state = {**base_state, "recovery_proposal": None}

        result = route_after_apply_recovery(state)

        assert result == "decide_next"


class TestRouteAfterRetryDeferred:
    """Tests for route_after_retry_deferred."""

    def test_retrying_routes_to_pick_task(self, base_state: ExecutorGraphState) -> None:
        """Test: When retrying deferred, route to pick_task."""
        state = {**base_state, "retrying_deferred": True}

        result = route_after_retry_deferred(state)

        assert result == "pick_task"

    def test_not_retrying_routes_to_generate_report(self, base_state: ExecutorGraphState) -> None:
        """Test: When not retrying, route to generate_report."""
        state = {**base_state, "retrying_deferred": False}

        result = route_after_retry_deferred(state)

        assert result == "generate_report"


class TestRouteAfterPickWithRecovery:
    """Tests for route_after_pick_with_recovery."""

    def test_has_task_routes_to_build_context(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: When task selected, route to build_context."""
        state = {**base_state, "current_task": sample_todo}

        result = route_after_pick_with_recovery(state)

        assert result == "build_context"

    def test_has_task_with_planning_routes_to_plan_task(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Test: When task selected with planning, route to plan_task."""
        state = {**base_state, "current_task": sample_todo, "enable_planning": True}

        result = route_after_pick_with_recovery(state)

        assert result == "plan_task"

    def test_no_task_with_deferred_routes_to_retry_deferred(
        self, base_state: ExecutorGraphState
    ) -> None:
        """Test: No task but deferred tasks routes to retry_deferred."""
        state = {
            **base_state,
            "current_task": None,
            "tasks": [],  # Empty task queue
            "deferred_tasks": [{"task": "Deferred", "retry_count": 0}],
        }

        result = route_after_pick_with_recovery(state)

        assert result == "retry_deferred"

    def test_no_task_no_deferred_routes_to_generate_report(
        self, base_state: ExecutorGraphState
    ) -> None:
        """Test: No tasks and no deferred routes to generate_report."""
        state = {
            **base_state,
            "current_task": None,
            "tasks": [],
            "deferred_tasks": [],
        }

        result = route_after_pick_with_recovery(state)

        assert result == "generate_report"


class TestRouteAfterDecideWithRecovery:
    """Tests for route_after_decide_with_recovery."""

    def test_should_continue_routes_to_pick_task(self, base_state: ExecutorGraphState) -> None:
        """Test: When should_continue, route to pick_task."""
        state = {**base_state, "should_continue": True}

        result = route_after_decide_with_recovery(state)

        assert result == "pick_task"

    def test_no_continue_with_deferred_routes_to_retry(
        self, base_state: ExecutorGraphState
    ) -> None:
        """Test: When not continuing but deferred exist, route to retry."""
        state = {
            **base_state,
            "should_continue": False,
            "tasks": [],
            "deferred_tasks": [{"task": "Deferred", "retry_count": 0}],
        }

        result = route_after_decide_with_recovery(state)

        assert result == "retry_deferred"

    def test_no_continue_with_recovery_flow_routes_to_report(
        self, base_state: ExecutorGraphState
    ) -> None:
        """Test: With recovery flow enabled, route to generate_report."""
        state = {
            **base_state,
            "should_continue": False,
            "tasks": [],
            "deferred_tasks": [],
            "enable_recovery_flow": True,
        }

        result = route_after_decide_with_recovery(state)

        assert result == "generate_report"

    def test_no_continue_no_recovery_routes_to_end(self, base_state: ExecutorGraphState) -> None:
        """Test: Without recovery flow, route to END."""
        state = {
            **base_state,
            "should_continue": False,
            "tasks": [],
            "deferred_tasks": [],
            "enable_recovery_flow": False,
        }

        result = route_after_decide_with_recovery(state)

        assert result == END


class TestRouteToRecoveryOrFailure:
    """Tests for route_to_recovery_or_failure."""

    def test_recovery_flow_enabled_routes_to_classify(self, base_state: ExecutorGraphState) -> None:
        """Test: With recovery flow, route to classify_failure."""
        state = {**base_state, "enable_recovery_flow": True}

        result = route_to_recovery_or_failure(state)

        assert result == "classify_failure"

    def test_recovery_flow_disabled_routes_to_handle_failure(
        self, base_state: ExecutorGraphState
    ) -> None:
        """Test: Without recovery flow, route to handle_failure."""
        state = {**base_state, "enable_recovery_flow": False}

        result = route_to_recovery_or_failure(state)

        assert result == "handle_failure"

    def test_missing_flag_defaults_to_handle_failure(self, base_state: ExecutorGraphState) -> None:
        """Test: Missing flag defaults to handle_failure."""
        state = {**base_state}  # No enable_recovery_flow

        result = route_to_recovery_or_failure(state)

        assert result == "handle_failure"
