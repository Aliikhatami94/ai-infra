import pytest

from ai_infra.executor.nodes.repair_test import TestFailure, repair_test_node


@pytest.fixture
def base_state():
    return {
        "test_repair_count": 0,
        "test_repair_results": {},
        "error": None,
        "current_task": None,
    }


def make_failure():
    return TestFailure(
        test_name="test_add",
        file_path="src/app.py",
        line_number=42,
        expected="5",
        actual="3",
        error_type="AssertionError",
        traceback="Traceback (most recent call last)...",
    )


@pytest.mark.asyncio
async def test_repair_test_node_increments_count(base_state):
    """Phase 2.6: Test that repair_test_node increments test_repair_count."""
    failure = make_failure()
    state = base_state.copy()
    result = await repair_test_node(state, test_failure=failure)
    assert result["test_repair_count"] == 1
    assert result["test_repair_results"]["src/app.py"]["status"] == "attempted"
    assert result["error"] is None


@pytest.mark.asyncio
async def test_repair_test_node_stops_at_limit(base_state):
    """Phase 2.6: Test that repair_test_node stops when max repairs exceeded."""
    failure = make_failure()
    state = base_state.copy()
    state["test_repair_count"] = 2
    result = await repair_test_node(state, test_failure=failure)
    assert result["should_continue"] is True  # Continue to handle_failure
    assert result["error"]["error_type"] == "test_repair_limit_exceeded"


@pytest.mark.asyncio
async def test_repair_test_node_extracts_failure_from_state(base_state):
    """Phase 2.6: Test that repair_test_node can extract failure from state error."""
    state = base_state.copy()
    state["error"] = {
        "error_type": "test_failure",
        "message": "Test failed: test_add in tests/test_app.py:42",
        "stack_trace": "tests/test_app.py:42: AssertionError\n  assert 3 == 5",
    }
    result = await repair_test_node(state)
    assert result["test_repair_count"] == 1
    assert result["error"] is None  # Error cleared for retry
