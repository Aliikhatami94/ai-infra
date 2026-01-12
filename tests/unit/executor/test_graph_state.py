"""Tests for executor graph state types and edge routing.

Phase 1.1: Design Phase tests.
"""

from __future__ import annotations

from ai_infra.executor.edges import (
    END,
    route_after_decide,
    route_after_execute,
    route_after_failure,
    route_after_pick,
    route_after_rollback,
    route_after_verify,
)
from ai_infra.executor.state import (
    EdgeTargets,
    ExecutorError,
    ExecutorErrorType,
    ExecutorGraphState,
    ExecutorNodes,
    NodeTimeouts,
    NonRetryableErrors,
    RetryPolicy,
)


class TestExecutorGraphState:
    """Tests for ExecutorGraphState TypedDict."""

    def test_state_can_be_created_empty(self) -> None:
        """State can be created as empty dict (total=False)."""
        state: ExecutorGraphState = {}
        assert state == {}

    def test_state_with_required_fields(self) -> None:
        """State can be created with typical fields."""
        state: ExecutorGraphState = {
            "roadmap_path": "/path/to/ROADMAP.md",
            "max_retries": 3,
            "max_tasks": 10,
            "todos": [],
            "current_task": None,
            "completed_todos": [],
            "failed_todos": [],
            "tasks_completed_count": 0,
            "run_memory": {},
            "error": None,
            "retry_count": 0,
            "should_continue": True,
            "interrupt_requested": False,
            "verified": False,
            "last_checkpoint_sha": None,
            "thread_id": None,
        }
        assert state["roadmap_path"] == "/path/to/ROADMAP.md"
        assert state["max_retries"] == 3
        assert state["should_continue"] is True

    def test_state_with_error(self) -> None:
        """State can contain an error."""
        error: ExecutorError = {
            "error_type": "execution",
            "message": "Agent failed to complete task",
            "node": "execute_task",
            "task_id": "1.1",
            "recoverable": True,
            "stack_trace": None,
        }
        state: ExecutorGraphState = {
            "error": error,
            "retry_count": 1,
        }
        assert state["error"]["error_type"] == "execution"
        assert state["error"]["recoverable"] is True


class TestExecutorError:
    """Tests for ExecutorError TypedDict."""

    def test_error_types(self) -> None:
        """All error types are valid."""
        valid_types = ["execution", "verification", "timeout", "rollback", "parse", "context"]
        for error_type in valid_types:
            error: ExecutorError = {
                "error_type": error_type,  # type: ignore
                "message": f"Test {error_type} error",
                "node": "test_node",
                "recoverable": True,
            }
            assert error["error_type"] == error_type

    def test_error_type_constants(self) -> None:
        """ExecutorErrorType constants match expected values."""
        assert ExecutorErrorType.EXECUTION == "execution"
        assert ExecutorErrorType.VERIFICATION == "verification"
        assert ExecutorErrorType.TIMEOUT == "timeout"
        assert ExecutorErrorType.ROLLBACK == "rollback"
        assert ExecutorErrorType.PARSE == "parse"
        assert ExecutorErrorType.CONTEXT == "context"


class TestNodeTimeouts:
    """Tests for NodeTimeouts configuration."""

    def test_all_timeouts_are_positive(self) -> None:
        """All timeouts should be positive values."""
        assert NodeTimeouts.PARSE_ROADMAP > 0
        assert NodeTimeouts.PICK_TASK > 0
        assert NodeTimeouts.BUILD_CONTEXT > 0
        assert NodeTimeouts.EXECUTE_TASK > 0
        assert NodeTimeouts.VERIFY_TASK > 0
        assert NodeTimeouts.CHECKPOINT > 0
        assert NodeTimeouts.ROLLBACK > 0
        assert NodeTimeouts.HANDLE_FAILURE > 0
        assert NodeTimeouts.DECIDE_NEXT > 0

    def test_execute_task_has_longest_timeout(self) -> None:
        """Execute task should have the longest timeout."""
        all_timeouts = [
            NodeTimeouts.PARSE_ROADMAP,
            NodeTimeouts.PICK_TASK,
            NodeTimeouts.BUILD_CONTEXT,
            NodeTimeouts.EXECUTE_TASK,
            NodeTimeouts.VERIFY_TASK,
            NodeTimeouts.CHECKPOINT,
            NodeTimeouts.ROLLBACK,
            NodeTimeouts.HANDLE_FAILURE,
            NodeTimeouts.DECIDE_NEXT,
        ]
        assert max(all_timeouts) == NodeTimeouts.EXECUTE_TASK

    def test_expected_timeout_values(self) -> None:
        """Verify expected timeout values from spec."""
        assert NodeTimeouts.PARSE_ROADMAP == 30.0
        assert NodeTimeouts.BUILD_CONTEXT == 60.0
        assert NodeTimeouts.EXECUTE_TASK == 300.0
        assert NodeTimeouts.VERIFY_TASK == 120.0
        assert NodeTimeouts.CHECKPOINT == 30.0


class TestRetryPolicy:
    """Tests for RetryPolicy configuration."""

    def test_default_values(self) -> None:
        """Verify default retry policy values."""
        assert RetryPolicy.MAX_RETRIES == 3
        assert RetryPolicy.BASE_DELAY == 1.0
        assert RetryPolicy.MULTIPLIER == 2.0
        assert RetryPolicy.JITTER_PERCENT == 0.1

    def test_get_delay_first_attempt(self) -> None:
        """First retry should have base delay."""
        delay = RetryPolicy.get_delay(1)
        # With 10% jitter, delay should be between 0.9 and 1.1
        assert 0.9 <= delay <= 1.1

    def test_get_delay_exponential(self) -> None:
        """Delays should increase exponentially."""
        delay1 = RetryPolicy.get_delay(1)
        delay2 = RetryPolicy.get_delay(2)
        delay3 = RetryPolicy.get_delay(3)

        # Base values (without jitter): 1, 2, 4
        # With jitter: ranges are [0.9, 1.1], [1.8, 2.2], [3.6, 4.4]
        assert 0.9 <= delay1 <= 1.1
        assert 1.8 <= delay2 <= 2.2
        assert 3.6 <= delay3 <= 4.4


class TestNonRetryableErrors:
    """Tests for NonRetryableErrors configuration."""

    def test_authentication_not_retryable(self) -> None:
        """Authentication failures should not be retried."""
        assert NonRetryableErrors.is_non_retryable("Authentication failed for API key")
        assert NonRetryableErrors.is_non_retryable("Invalid API Key provided")

    def test_rate_limit_not_retryable(self) -> None:
        """Rate limit errors should not be retried."""
        assert NonRetryableErrors.is_non_retryable("Rate limit exceeded, try again later")

    def test_git_conflict_not_retryable(self) -> None:
        """Git conflicts should not be retried."""
        assert NonRetryableErrors.is_non_retryable("Git conflict detected in file.py")
        assert NonRetryableErrors.is_non_retryable("Merge conflict in src/main.py")

    def test_permission_denied_not_retryable(self) -> None:
        """Permission errors should not be retried."""
        assert NonRetryableErrors.is_non_retryable("Permission denied: /etc/passwd")

    def test_roadmap_not_found_not_retryable(self) -> None:
        """Missing ROADMAP should not be retried."""
        assert NonRetryableErrors.is_non_retryable("ROADMAP not found at /path/to/ROADMAP.md")

    def test_invalid_roadmap_not_retryable(self) -> None:
        """Invalid ROADMAP format should not be retried."""
        assert NonRetryableErrors.is_non_retryable("Invalid ROADMAP format: missing phases")

    def test_regular_errors_are_retryable(self) -> None:
        """Regular errors should be retryable."""
        assert not NonRetryableErrors.is_non_retryable("Connection timeout")
        assert not NonRetryableErrors.is_non_retryable("LLM returned empty response")
        assert not NonRetryableErrors.is_non_retryable("File not found: src/foo.py")

    def test_case_insensitive_matching(self) -> None:
        """Error matching should be case-insensitive."""
        assert NonRetryableErrors.is_non_retryable("AUTHENTICATION FAILED")
        assert NonRetryableErrors.is_non_retryable("Rate Limit Exceeded")


class TestExecutorNodes:
    """Tests for ExecutorNodes constants."""

    def test_all_node_names(self) -> None:
        """Verify all node names are defined."""
        assert ExecutorNodes.PARSE_ROADMAP == "parse_roadmap"
        assert ExecutorNodes.PICK_TASK == "pick_task"
        assert ExecutorNodes.BUILD_CONTEXT == "build_context"
        assert ExecutorNodes.EXECUTE_TASK == "execute_task"
        assert ExecutorNodes.VERIFY_TASK == "verify_task"
        assert ExecutorNodes.CHECKPOINT == "checkpoint"
        assert ExecutorNodes.ROLLBACK == "rollback"
        assert ExecutorNodes.HANDLE_FAILURE == "handle_failure"
        assert ExecutorNodes.DECIDE_NEXT == "decide_next"

    def test_all_method_returns_all_nodes(self) -> None:
        """all() should return all 9 nodes."""
        all_nodes = ExecutorNodes.all()
        assert len(all_nodes) == 9
        assert "parse_roadmap" in all_nodes
        assert "execute_task" in all_nodes
        assert "checkpoint" in all_nodes


class TestEdgeTargets:
    """Tests for EdgeTargets constants."""

    def test_edge_targets_defined(self) -> None:
        """Verify edge target constants."""
        assert EdgeTargets.CONTEXT == "build_context"
        assert EdgeTargets.END == "__end__"
        assert EdgeTargets.VERIFY == "verify_task"
        assert EdgeTargets.FAILURE == "handle_failure"
        assert EdgeTargets.CHECKPOINT == "checkpoint"
        assert EdgeTargets.ROLLBACK == "rollback"
        assert EdgeTargets.DECIDE == "decide_next"
        assert EdgeTargets.EXECUTE == "execute_task"
        assert EdgeTargets.PICK == "pick_task"


# =============================================================================
# Edge Routing Tests
# =============================================================================


class TestRouteAfterPick:
    """Tests for route_after_pick edge function."""

    def test_routes_to_context_when_task_selected(self) -> None:
        """Should route to build_context when a task is selected."""
        # Create a mock task (just needs to be truthy)
        state: ExecutorGraphState = {
            "current_task": {"id": 1, "title": "Test task"},  # type: ignore
        }
        assert route_after_pick(state) == "build_context"

    def test_routes_to_end_when_no_task(self) -> None:
        """Should route to END when no task is selected."""
        state: ExecutorGraphState = {"current_task": None}
        assert route_after_pick(state) == END

    def test_routes_to_end_when_current_task_missing(self) -> None:
        """Should route to END when current_task key is missing."""
        state: ExecutorGraphState = {}
        assert route_after_pick(state) == END


class TestRouteAfterExecute:
    """Tests for route_after_execute edge function."""

    def test_routes_to_verify_on_success(self) -> None:
        """Should route to verify_task when no error."""
        state: ExecutorGraphState = {"error": None}
        assert route_after_execute(state) == "verify_task"

    def test_routes_to_failure_on_error(self) -> None:
        """Should route to handle_failure when error present."""
        state: ExecutorGraphState = {
            "error": {
                "error_type": "execution",
                "message": "Agent failed",
                "node": "execute_task",
                "recoverable": True,
            }
        }
        assert route_after_execute(state) == "handle_failure"

    def test_routes_to_verify_when_error_missing(self) -> None:
        """Should route to verify_task when error key is missing."""
        state: ExecutorGraphState = {}
        assert route_after_execute(state) == "verify_task"


class TestRouteAfterVerify:
    """Tests for route_after_verify edge function (Phase 2.6: routes to repair_test)."""

    def test_routes_to_checkpoint_when_verified(self) -> None:
        """Should route to checkpoint when verification passed."""
        state: ExecutorGraphState = {"verified": True}
        assert route_after_verify(state) == "checkpoint"

    def test_routes_to_repair_test_when_not_verified(self) -> None:
        """Phase 2.6: Should route to repair_test when verification failed."""
        state: ExecutorGraphState = {"verified": False, "test_repair_count": 0}
        assert route_after_verify(state) == "repair_test"

    def test_routes_to_repair_test_when_verified_missing(self) -> None:
        """Phase 2.6: Should route to repair_test when verified key is missing."""
        state: ExecutorGraphState = {"test_repair_count": 0}
        assert route_after_verify(state) == "repair_test"

    def test_routes_to_handle_failure_when_max_repairs_exceeded(self) -> None:
        """Phase 2.6: Should route to handle_failure when max test repairs exceeded."""
        state: ExecutorGraphState = {"verified": False, "test_repair_count": 2}
        assert route_after_verify(state) == "handle_failure"


class TestRouteAfterFailure:
    """Tests for route_after_failure edge function.

    Phase 2.1: Rollback node removed. All failures now route to decide_next.
    """

    def test_routes_to_decide_next_always(self) -> None:
        """Phase 2.1: Should always route to decide_next (rollback removed)."""
        state: ExecutorGraphState = {
            "error": {
                "error_type": "execution",
                "message": "Agent failed",
                "node": "execute_task",
                "recoverable": True,
            },
            "last_checkpoint_sha": "abc123",
        }
        assert route_after_failure(state) == "decide_next"

    def test_routes_to_decide_when_recoverable_no_checkpoint(self) -> None:
        """Should route to decide_next when recoverable but no checkpoint."""
        state: ExecutorGraphState = {
            "error": {
                "error_type": "execution",
                "message": "Agent failed",
                "node": "execute_task",
                "recoverable": True,
            },
            "last_checkpoint_sha": None,
        }
        assert route_after_failure(state) == "decide_next"

    def test_routes_to_decide_when_not_recoverable(self) -> None:
        """Should route to decide_next when error is not recoverable."""
        state: ExecutorGraphState = {
            "error": {
                "error_type": "execution",
                "message": "Authentication failed",
                "node": "execute_task",
                "recoverable": False,
            },
            "last_checkpoint_sha": "abc123",
        }
        assert route_after_failure(state) == "decide_next"

    def test_routes_to_decide_when_no_error(self) -> None:
        """Should route to decide_next when no error (shouldn't happen but handle)."""
        state: ExecutorGraphState = {"error": None}
        assert route_after_failure(state) == "decide_next"


class TestRouteAfterRollback:
    """Tests for route_after_rollback edge function."""

    def test_routes_to_execute_when_under_max_retries(self) -> None:
        """Should route to execute_task when retry count < max."""
        state: ExecutorGraphState = {
            "retry_count": 1,
            "max_retries": 3,
        }
        assert route_after_rollback(state) == "execute_task"

    def test_routes_to_decide_when_max_retries_reached(self) -> None:
        """Should route to decide_next when max retries reached."""
        state: ExecutorGraphState = {
            "retry_count": 3,
            "max_retries": 3,
        }
        assert route_after_rollback(state) == "decide_next"

    def test_routes_to_decide_when_over_max_retries(self) -> None:
        """Should route to decide_next when retry count exceeds max."""
        state: ExecutorGraphState = {
            "retry_count": 5,
            "max_retries": 3,
        }
        assert route_after_rollback(state) == "decide_next"

    def test_uses_default_max_retries(self) -> None:
        """Should use default max_retries of 3 when not specified."""
        state: ExecutorGraphState = {"retry_count": 2}
        assert route_after_rollback(state) == "execute_task"

        state2: ExecutorGraphState = {"retry_count": 3}
        assert route_after_rollback(state2) == "decide_next"


class TestRouteAfterDecide:
    """Tests for route_after_decide edge function."""

    def test_routes_to_pick_when_should_continue(self) -> None:
        """Should route to pick_task when should_continue is True."""
        state: ExecutorGraphState = {"should_continue": True}
        assert route_after_decide(state) == "pick_task"

    def test_routes_to_end_when_should_not_continue(self) -> None:
        """Should route to END when should_continue is False."""
        state: ExecutorGraphState = {"should_continue": False}
        assert route_after_decide(state) == END

    def test_routes_to_end_when_should_continue_missing(self) -> None:
        """Should route to END when should_continue key is missing."""
        state: ExecutorGraphState = {}
        assert route_after_decide(state) == END
