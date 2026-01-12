"""Verify task node for executor graph.

Phase 1.2.2: Verifies task completion using TaskVerifier.
Phase 1.4.3: Adapts TaskVerifier for graph (unchanged API).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ai_infra.executor.state import ExecutorGraphState, NodeTimeouts
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.verifier import CheckLevel, TaskVerifier

logger = get_logger("executor.nodes.verify")


async def verify_task_node(
    state: ExecutorGraphState,
    *,
    verifier: TaskVerifier | None = None,
    check_level: CheckLevel | None = None,
) -> ExecutorGraphState:
    """Verify that the current task was completed successfully.

    Phase 1.4.3: TaskVerifier integration with graph (unchanged API).

    The verifier runs the same checks as before, but results are stored
    in graph state for routing decisions:
    - verified=True -> route to checkpoint
    - verified=False + error -> route to handle_failure

    This node:
    1. Gets current_task and files_modified from state
    2. Runs TaskVerifier.verify() to validate completion
    3. Sets verified=True on success, error on failure

    Args:
        state: Current graph state with current_task and files_modified.
        verifier: The TaskVerifier instance (Phase 1.4.3).
        check_level: Optional verification level (defaults to SYNTAX).

    Returns:
        Updated state with verified flag or error.
    """
    current_task = state.get("current_task")
    files_modified = state.get("files_modified", [])
    task_id = str(current_task.id) if current_task else "unknown"

    if current_task is None:
        logger.error("No current task to verify")
        return {
            **state,
            "verified": False,
            "error": {
                "error_type": "verification",
                "message": "No current task to verify",
                "node": "verify_task",
                "task_id": None,
                "recoverable": False,
                "stack_trace": None,
            },
        }

    # If no verifier, skip verification (development mode)
    if verifier is None:
        logger.warning(f"No verifier provided, auto-passing task [{task_id}]")
        return {
            **state,
            "verified": True,
            "error": None,
        }

    logger.info(f"Verifying task [{task_id}] with {len(files_modified)} files")

    try:
        # Get default check level if not specified
        if check_level is None:
            from ai_infra.executor.verifier import CheckLevel

            check_level = CheckLevel.SYNTAX

        # Run verification with timeout
        # Note: TaskVerifier.verify() uses 'levels' (list) not 'level' (singular)
        verification_result = await asyncio.wait_for(
            verifier.verify(
                task=current_task,
                levels=[check_level],
            ),
            timeout=NodeTimeouts.VERIFY_TASK,
        )

        # Use 'overall' property which checks all checks passed
        if verification_result.overall:
            logger.info(f"Task [{task_id}] verified successfully")
            return {
                **state,
                "verified": True,
                "error": None,
            }
        else:
            # Verification failed - get failure details from failed checks
            failures = verification_result.get_failures()
            failure_message = failures[0].message if failures else verification_result.summary()
            logger.warning(f"Task [{task_id}] verification failed: {failure_message}")

            return {
                **state,
                "verified": False,
                "error": {
                    "error_type": "verification",
                    "message": failure_message,
                    "node": "verify_task",
                    "task_id": task_id,
                    "recoverable": True,  # Can retry after fixes
                    "stack_trace": None,
                },
            }

    except TimeoutError:
        logger.error(f"Task [{task_id}] verification timed out")
        return {
            **state,
            "verified": False,
            "error": {
                "error_type": "timeout",
                "message": f"Verification timed out after {NodeTimeouts.VERIFY_TASK}s",
                "node": "verify_task",
                "task_id": task_id,
                "recoverable": True,
                "stack_trace": None,
            },
        }

    except Exception as e:
        logger.exception(f"Task [{task_id}] verification error: {e}")

        import traceback

        return {
            **state,
            "verified": False,
            "error": {
                "error_type": "verification",
                "message": str(e),
                "node": "verify_task",
                "task_id": task_id,
                "recoverable": True,
                "stack_trace": traceback.format_exc(),
            },
        }
