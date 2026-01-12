"""Rollback node for executor graph.

.. deprecated:: Phase 2.1
    This module is deprecated and no longer used in the active graph flow.
    Rollback has been removed because:
    - With pre-write validation (Phase 1.1), bad code never reaches disk
    - Rollback + retry rarely produces different results
    - Simpler graph without rollback

    The module is kept for backward compatibility and potential future use
    in advanced recovery scenarios.

Phase 1.2.2: Originally rolled back to last checkpoint on verification failure.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import TYPE_CHECKING

from ai_infra.executor.state import ExecutorGraphState, NodeTimeouts
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.checkpoint import Checkpointer

logger = get_logger("executor.nodes.rollback")


async def rollback_node(
    state: ExecutorGraphState,
    *,
    checkpointer: Checkpointer | None = None,
) -> ExecutorGraphState:
    """Roll back to the last checkpoint after a failure.

    .. deprecated:: Phase 2.1
        This node is deprecated and no longer used in the active graph flow.
        Use pre-write validation (Phase 1.1) instead.

    This node:
    1. Gets last_checkpoint_sha from state
    2. Performs git rollback to that SHA
    3. Clears files_modified and agent_result

    Args:
        state: Current graph state with last_checkpoint_sha.
        checkpointer: The Checkpointer instance for git operations.

    Returns:
        Updated state with cleared execution data.
    """
    warnings.warn(
        "rollback_node is deprecated since Phase 2.1. "
        "Pre-write validation prevents bad code from reaching disk.",
        DeprecationWarning,
        stacklevel=2,
    )

    last_sha = state.get("last_checkpoint_sha")
    current_task = state.get("current_task")
    task_id = str(current_task.id) if current_task else "unknown"

    # If no checkpoint to rollback to, just clear state
    if not last_sha:
        logger.warning(f"No checkpoint to rollback to for task [{task_id}]")
        return {
            **state,
            "files_modified": [],
            "agent_result": None,
            "error": None,
        }

    # If no checkpointer, just clear state
    if checkpointer is None:
        logger.warning(f"No checkpointer provided, clearing state for [{task_id}]")
        return {
            **state,
            "files_modified": [],
            "agent_result": None,
            "error": None,
        }

    logger.info(f"Rolling back task [{task_id}] to checkpoint {last_sha[:8]}")

    try:
        # Run synchronous rollback in executor
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: checkpointer.rollback(sha=last_sha),
            ),
            timeout=NodeTimeouts.ROLLBACK,
        )

        logger.info(f"Rollback complete for task [{task_id}]")

        return {
            **state,
            "files_modified": [],
            "agent_result": None,
            "error": None,
        }

    except TimeoutError:
        logger.error(f"Rollback timed out for task [{task_id}]")
        return {
            **state,
            "error": {
                "error_type": "timeout",
                "message": f"Rollback timed out after {NodeTimeouts.ROLLBACK}s",
                "node": "rollback",
                "task_id": task_id,
                "recoverable": False,  # Manual intervention needed
                "stack_trace": None,
            },
        }

    except Exception as e:
        logger.exception(f"Rollback failed for task [{task_id}]: {e}")

        import traceback

        return {
            **state,
            "error": {
                "error_type": "rollback",
                "message": str(e),
                "node": "rollback",
                "task_id": task_id,
                "recoverable": False,  # Manual intervention needed
                "stack_trace": traceback.format_exc(),
            },
        }
