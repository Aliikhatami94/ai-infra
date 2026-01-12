"""Repair test node for executor graph.

Phase 2.3: Surgically fix code when tests fail.
Phase 2.6: Wired into graph, extracts test failure from state.
This node is called after test failures (logic errors, not syntax).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_infra.executor.state import ExecutorGraphState
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.agent import Agent

logger = get_logger("executor.nodes.repair_test")


MAX_TEST_REPAIRS = 2  # Maximum test repair attempts before escalation


@dataclass
class TestFailure:
    """Parsed test failure information."""

    test_name: str
    file_path: str
    line_number: int
    expected: str | None
    actual: str | None
    error_type: str  # AssertionError, TypeError, etc.
    traceback: str


def _extract_test_failure_from_state(state: ExecutorGraphState) -> TestFailure | None:
    """Extract test failure information from graph state.

    Parses the error from verify_task_node to build a TestFailure.
    """
    error = state.get("error")
    if not error:
        return None

    # Extract test failure info from error
    message = error.get("message", "")
    stack_trace = error.get("stack_trace", "")

    # Parse test name and file from stack trace or message
    # This is a heuristic; real implementation would parse pytest output
    test_name = "unknown_test"
    file_path = "unknown_file.py"
    line_number = 0
    error_type = error.get("error_type", "test_failure")

    # Try to extract from stack trace
    import re

    # Match patterns like "tests/test_foo.py:42: AssertionError"
    match = re.search(r"([\w/]+\.py):(\d+):", stack_trace or message)
    if match:
        file_path = match.group(1)
        line_number = int(match.group(2))

    # Match test function name
    test_match = re.search(r"(test_\w+)", stack_trace or message)
    if test_match:
        test_name = test_match.group(1)

    return TestFailure(
        test_name=test_name,
        file_path=file_path,
        line_number=line_number,
        expected=None,  # Would be parsed from assertion message
        actual=None,
        error_type=error_type,
        traceback=stack_trace or message,
    )


async def repair_test_node(
    state: ExecutorGraphState,
    *,
    repair_agent: Agent | None = None,
    test_failure: TestFailure | None = None,
    max_test_repairs: int = MAX_TEST_REPAIRS,
) -> ExecutorGraphState:
    """Attempt to repair code based on test failure info.

    Phase 2.6: Graph node that extracts test failure from state and repairs.

    Args:
        state: Current graph state.
        repair_agent: Agent to use for repair (optional, for future LLM repair).
        test_failure: Parsed test failure info (optional, extracted from state if None).
        max_test_repairs: Maximum allowed test repairs (default: 2).

    Returns:
        Updated state with incremented test_repair_count and repair results.
    """
    test_repair_count = state.get("test_repair_count", 0)

    # Check if we've exceeded the limit
    if test_repair_count >= max_test_repairs:
        logger.error(f"Max test repairs ({max_test_repairs}) exceeded for task.")
        return {
            **state,
            "should_continue": True,  # Continue to handle_failure
            "error": {
                "error_type": "test_repair_limit_exceeded",
                "message": f"Max test repairs ({max_test_repairs}) exceeded.",
                "node": "repair_test",
                "task_id": state.get("current_task", {}).get("id")
                if state.get("current_task")
                else None,
                "recoverable": False,
                "stack_trace": None,
            },
        }

    # Extract test failure from state if not provided
    if test_failure is None:
        test_failure = _extract_test_failure_from_state(state)

    if test_failure is None:
        logger.warning("No test failure found in state, cannot repair")
        return {
            **state,
            "test_repair_count": test_repair_count + 1,
            "error": None,  # Clear error to retry verification
        }

    logger.info(
        f"Repairing test failure: {test_failure.test_name} in "
        f"{test_failure.file_path} at line {test_failure.line_number}"
    )

    # TODO: Implement actual repair logic using repair_agent
    # For now, just increment count and clear error to retry
    # Real implementation would:
    # 1. Read the failing test and source file
    # 2. Build a repair prompt with the test failure context
    # 3. Ask the agent to fix the specific issue
    # 4. Write the repaired code back

    repair_results = dict(state.get("test_repair_results", {}))
    repair_results[test_failure.file_path] = {
        "status": "attempted",
        "test_name": test_failure.test_name,
        "line_number": test_failure.line_number,
        "error_type": test_failure.error_type,
        "traceback": test_failure.traceback,
        "attempt": test_repair_count + 1,
    }

    return {
        **state,
        "test_repair_count": test_repair_count + 1,
        "test_repair_results": repair_results,
        "error": None,  # Clear error to allow retry
        "verified": False,  # Reset verification status
    }
