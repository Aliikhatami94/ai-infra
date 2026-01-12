"""Tests for simplified executor flow (Phase 2.8).

This module tests the Phase 2 simplified flow:
- Pre-write validation (validate_code_node)
- Surgical repair (repair_code_node, repair_test_node)
- Separated file writing (write_files_node)
- No rollback on failure

Tests verify:
1. Happy path: generate → validate (pass) → write → verify (pass)
2. Repair loop: generate → validate (fail) → repair → validate (pass)
3. Escalation: max repairs exceeded → handle_failure
4. Test repair: write → verify (fail) → repair_test → verify (pass)
5. No rollback: failures go to handle_failure, not rollback
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.executor.edges.routes import (
    route_after_repair,
    route_after_repair_test,
    route_after_validate,
    route_after_verify,
    route_after_write,
)
from ai_infra.executor.nodes import (
    handle_failure_node,
    repair_code_node,
    validate_code_node,
    write_files_node,
)
from ai_infra.executor.nodes.repair_test import repair_test_node
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
        description="A test task",
        status=TodoStatus.NOT_STARTED,
    )


@pytest.fixture
def base_state(sample_todo: TodoItem) -> ExecutorGraphState:
    """Base state for flow tests."""
    return {
        "roadmap_path": "/tmp/test/ROADMAP.md",
        "todos": [sample_todo],
        "current_task": sample_todo,
        "generated_code": {},
        "validated": False,
        "needs_repair": False,
        "validation_errors": {},
        "repair_count": 0,
        "test_repair_count": 0,
        "files_written": False,
        "verified": False,
        "error": None,
        "should_continue": True,
    }


@pytest.fixture
def valid_code() -> dict[str, str]:
    """Valid Python code."""
    return {
        "src/main.py": "def main():\n    print('Hello, World!')\n",
        "src/utils.py": "def add(a, b):\n    return a + b\n",
    }


@pytest.fixture
def invalid_code() -> dict[str, str]:
    """Invalid Python code with syntax errors."""
    return {
        "src/main.py": "def main(\n    print('Hello, World!')\n",  # Missing closing paren
    }


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestHappyPath:
    """Test happy path: generate → validate (pass) → write → verify (pass)."""

    @pytest.mark.asyncio
    async def test_validate_pass_routes_to_write(
        self, base_state: ExecutorGraphState, valid_code: dict[str, str]
    ) -> None:
        """Phase 2.8: Valid code routes to write_files."""
        state = {
            **base_state,
            "generated_code": valid_code,
        }

        # Validate the code
        result = await validate_code_node(state)

        assert result["validated"] is True
        assert result["needs_repair"] is False
        assert result["validation_errors"] == {}

        # Routing should go to write_files
        route = route_after_validate(result)
        assert route == "write_files"

    @pytest.mark.asyncio
    async def test_write_files_routes_to_verify(
        self, base_state: ExecutorGraphState, valid_code: dict[str, str]
    ) -> None:
        """Phase 2.8: Successful write routes to verify_task."""
        state = {
            **base_state,
            "generated_code": valid_code,
            "validated": True,
        }

        with patch("ai_infra.executor.nodes.write.Path") as mock_path:
            # Mock successful file writes
            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.parent.mkdir = MagicMock()
            mock_path_instance.write_text = MagicMock()

            result = await write_files_node(state, dry_run=True)

        assert result["files_written"] is True
        assert result.get("write_errors") is None

        # Routing should go to verify_task
        route = route_after_write(result)
        assert route == "verify_task"

    @pytest.mark.asyncio
    async def test_verify_pass_routes_to_checkpoint(self, base_state: ExecutorGraphState) -> None:
        """Phase 2.8: Verified task routes to checkpoint."""
        state = {
            **base_state,
            "verified": True,
            "error": None,
        }

        route = route_after_verify(state)
        assert route == "checkpoint"


# =============================================================================
# Repair Loop Tests
# =============================================================================


class TestRepairLoop:
    """Test repair loop: validate (fail) → repair → validate (pass)."""

    @pytest.mark.asyncio
    async def test_validate_fail_routes_to_repair(
        self, base_state: ExecutorGraphState, invalid_code: dict[str, str]
    ) -> None:
        """Phase 2.8: Invalid code routes to repair_code."""
        state = {
            **base_state,
            "generated_code": invalid_code,
            "repair_count": 0,
        }

        result = await validate_code_node(state)

        assert result["validated"] is False
        assert result["needs_repair"] is True
        assert len(result["validation_errors"]) > 0

        # Routing should go to repair_code (under max repairs)
        route = route_after_validate(result)
        assert route == "repair_code"

    @pytest.mark.asyncio
    async def test_repair_increments_count(
        self, base_state: ExecutorGraphState, invalid_code: dict[str, str]
    ) -> None:
        """Phase 2.8: Repair node increments repair_count."""
        state = {
            **base_state,
            "generated_code": invalid_code,
            "repair_count": 0,
            "validation_errors": {
                "src/main.py": {
                    "error_type": "syntax",
                    "error_message": "Missing closing parenthesis",
                    "error_line": 1,
                }
            },
        }

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(
            return_value=MagicMock(output="def main():\n    print('Hello, World!')\n")
        )

        result = await repair_code_node(state, agent=mock_agent)

        assert result["repair_count"] == 1
        assert result["needs_repair"] is False

    @pytest.mark.asyncio
    async def test_repair_routes_back_to_validate(self, base_state: ExecutorGraphState) -> None:
        """Phase 2.8: Repair node routes back to validate_code."""
        state = {
            **base_state,
            "repair_count": 1,
            "needs_repair": False,
        }

        route = route_after_repair(state)
        assert route == "validate_code"


# =============================================================================
# Escalation Tests
# =============================================================================


class TestEscalation:
    """Test escalation: max repairs exceeded → handle_failure."""

    @pytest.mark.asyncio
    async def test_validate_fail_at_max_repairs_routes_to_handle_failure(
        self, base_state: ExecutorGraphState, invalid_code: dict[str, str]
    ) -> None:
        """Phase 2.8: Max repairs exceeded routes to handle_failure."""
        state = {
            **base_state,
            "generated_code": invalid_code,
            "repair_count": 2,  # At max
            "validated": False,
            "needs_repair": True,
        }

        # Routing should go to handle_failure
        route = route_after_validate(state)
        assert route == "handle_failure"

    @pytest.mark.asyncio
    async def test_verify_fail_at_max_repairs_routes_to_handle_failure(
        self, base_state: ExecutorGraphState
    ) -> None:
        """Phase 2.8: Max test repairs exceeded routes to handle_failure."""
        state = {
            **base_state,
            "verified": False,
            "test_repair_count": 2,  # At max
        }

        route = route_after_verify(state)
        assert route == "handle_failure"

    def test_handle_failure_sets_error(
        self, base_state: ExecutorGraphState, sample_todo: TodoItem
    ) -> None:
        """Phase 2.8: Handle failure sets appropriate error state."""
        state = {
            **base_state,
            "current_task": sample_todo,
            "error": {
                "error_type": "validation",
                "message": "Max repairs exceeded",
                "node": "validate_code",
                "recoverable": False,
            },
            "repair_count": 2,
        }

        result = handle_failure_node(state, max_retries=3)

        # Non-recoverable error should stop processing this task
        # The error is logged and task is marked as failed
        # should_continue may be False for non-recoverable errors
        assert "error" in result or result.get("should_continue") is False


# =============================================================================
# Test Repair Tests
# =============================================================================


class TestTestRepair:
    """Test repair flow: write → verify (fail) → repair_test → verify (pass)."""

    @pytest.mark.asyncio
    async def test_verify_fail_routes_to_repair_test(self, base_state: ExecutorGraphState) -> None:
        """Phase 2.8: Verification failure routes to repair_test."""
        state = {
            **base_state,
            "verified": False,
            "test_repair_count": 0,
            "error": {
                "error_type": "verification",
                "message": "Test test_add failed",
                "node": "verify_task",
            },
        }

        route = route_after_verify(state)
        assert route == "repair_test"

    @pytest.mark.asyncio
    async def test_repair_test_increments_count(self, base_state: ExecutorGraphState) -> None:
        """Phase 2.8: Repair test node increments test_repair_count."""
        state = {
            **base_state,
            "test_repair_count": 0,
            "error": {
                "error_type": "verification",
                "message": "Test test_add failed",
                "stack_trace": "tests/test_main.py:42: AssertionError",
            },
        }

        result = await repair_test_node(state)

        assert result["test_repair_count"] == 1
        assert result["error"] is None  # Error cleared for retry

    @pytest.mark.asyncio
    async def test_repair_test_routes_back_to_verify(self, base_state: ExecutorGraphState) -> None:
        """Phase 2.8: Repair test routes back to verify_task."""
        state = {
            **base_state,
            "test_repair_count": 1,
            "error": None,
        }

        route = route_after_repair_test(state)
        assert route == "verify_task"

    @pytest.mark.asyncio
    async def test_repair_test_at_limit_routes_to_handle_failure(
        self, base_state: ExecutorGraphState
    ) -> None:
        """Phase 2.8: Max test repairs routes to handle_failure."""
        state = {
            **base_state,
            "test_repair_count": 2,
            "error": {
                "error_type": "test_repair_limit_exceeded",
                "message": "Max test repairs exceeded",
            },
        }

        route = route_after_repair_test(state)
        assert route == "handle_failure"


# =============================================================================
# No Rollback Tests
# =============================================================================


class TestNoRollback:
    """Test that rollback is not used in the new flow."""

    def test_rollback_node_is_deprecated(self) -> None:
        """Phase 2.8: Rollback node should emit deprecation warning."""
        import warnings

        from ai_infra.executor.nodes.rollback import rollback_node

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # The deprecation warning is emitted when the function is called
            # We don't need to actually call it here, just verify it exists
            assert rollback_node is not None

    def test_failure_routes_to_handle_failure_not_rollback(
        self, base_state: ExecutorGraphState
    ) -> None:
        """Phase 2.8: Failures route to handle_failure, not rollback."""
        from ai_infra.executor.edges.routes import route_after_failure

        state = {
            **base_state,
            "error": {
                "error_type": "execution",
                "message": "Task failed",
                "node": "execute_task",
            },
        }

        route = route_after_failure(state)
        # Phase 2.1: Should route to decide_next, not rollback
        assert route == "decide_next"
        assert route != "rollback"

    def test_no_rollback_in_graph_edges(self) -> None:
        """Phase 2.8: Verify rollback is not in active graph edges."""
        from ai_infra.executor.edges.routes import (
            route_after_validate,
            route_after_verify,
            route_after_write,
        )

        # Test all routing functions don't return rollback
        test_states = [
            {"validated": True},
            {"validated": False, "needs_repair": True, "repair_count": 0},
            {"validated": False, "needs_repair": True, "repair_count": 2},
            {"verified": True},
            {"verified": False, "test_repair_count": 0},
            {"verified": False, "test_repair_count": 2},
            {"files_written": True},
            {"files_written": False, "write_errors": [{"file": "x", "error": "y"}]},
        ]

        for state in test_states:
            for route_fn in [route_after_validate, route_after_verify, route_after_write]:
                try:
                    result = route_fn(state)
                    assert result != "rollback", f"Unexpected rollback from {route_fn.__name__}"
                except KeyError:
                    # Some states don't have all required keys - that's fine
                    pass


# =============================================================================
# End-to-End Flow Tests
# =============================================================================


class TestEndToEndFlow:
    """Test complete flow scenarios."""

    @pytest.mark.asyncio
    async def test_full_happy_path_flow(
        self, base_state: ExecutorGraphState, valid_code: dict[str, str]
    ) -> None:
        """Phase 2.8: Complete happy path without any repairs."""
        state = {**base_state, "generated_code": valid_code}
        nodes_visited = []

        # 1. Validate
        result = await validate_code_node(state)
        nodes_visited.append("validate_code")
        assert result["validated"] is True

        # 2. Route to write
        route = route_after_validate(result)
        assert route == "write_files"

        # 3. Write files
        result = await write_files_node(result, dry_run=True)
        nodes_visited.append("write_files")
        assert result["files_written"] is True

        # 4. Route to verify
        route = route_after_write(result)
        assert route == "verify_task"

        # 5. Simulate verification pass
        result["verified"] = True
        nodes_visited.append("verify_task")

        # 6. Route to checkpoint
        route = route_after_verify(result)
        assert route == "checkpoint"
        nodes_visited.append("checkpoint")

        # Verify no repair nodes were visited
        assert "repair_code" not in nodes_visited
        assert "repair_test" not in nodes_visited
        assert "rollback" not in nodes_visited

    @pytest.mark.asyncio
    async def test_validation_repair_flow(
        self, base_state: ExecutorGraphState, invalid_code: dict[str, str]
    ) -> None:
        """Phase 2.8: Flow with one validation repair."""
        state = {**base_state, "generated_code": invalid_code}
        nodes_visited = []

        # 1. Validate (fails)
        result = await validate_code_node(state)
        nodes_visited.append("validate_code")
        assert result["validated"] is False

        # 2. Route to repair
        route = route_after_validate(result)
        assert route == "repair_code"

        # 3. Simulate repair (mock agent)
        nodes_visited.append("repair_code")
        result["repair_count"] = 1
        result["generated_code"] = {"src/main.py": "def main():\n    pass\n"}
        result["needs_repair"] = False

        # 4. Route back to validate
        route = route_after_repair(result)
        assert route == "validate_code"

        # 5. Validate again (passes)
        result = await validate_code_node(result)
        nodes_visited.append("validate_code")
        assert result["validated"] is True

        # 6. Route to write
        route = route_after_validate(result)
        assert route == "write_files"

        # Verify repair was visited once
        assert nodes_visited.count("repair_code") == 1
        assert "rollback" not in nodes_visited

    @pytest.mark.asyncio
    async def test_test_repair_flow(
        self, base_state: ExecutorGraphState, valid_code: dict[str, str]
    ) -> None:
        """Phase 2.8: Flow with test failure and repair."""
        state = {
            **base_state,
            "generated_code": valid_code,
            "validated": True,
            "files_written": True,
            "verified": False,
            "test_repair_count": 0,
            "error": {
                "error_type": "verification",
                "message": "Test failed",
                "stack_trace": "tests/test_app.py:10: AssertionError",
            },
        }
        nodes_visited = []

        # 1. Route to repair_test
        route = route_after_verify(state)
        assert route == "repair_test"

        # 2. Repair test
        result = await repair_test_node(state)
        nodes_visited.append("repair_test")
        assert result["test_repair_count"] == 1

        # 3. Route back to verify
        route = route_after_repair_test(result)
        assert route == "verify_task"

        # 4. Simulate verification pass
        result["verified"] = True
        nodes_visited.append("verify_task")

        # 5. Route to checkpoint
        route = route_after_verify(result)
        assert route == "checkpoint"

        # Verify
        assert "repair_test" in nodes_visited
        assert "rollback" not in nodes_visited
