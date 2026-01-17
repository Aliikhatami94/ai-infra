"""Edge routing functions for the executor graph.

Phase 1.1.3: Conditional edge routing for graph-based executor.
Phase 2.3.1: Added adaptive replanning routing.
Phase 2.4.2: Updated route_after_pick for planning support.
Phase 1.2.3: Added route_after_validate and route_after_repair for pre-write validation.
Phase 16.5.7: Added logging for route_after_verify decisions.

These functions determine the next node to execute based on current state.
All edge routing is done through these pure functions for testability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.constants import END

if TYPE_CHECKING:
    from ai_infra.executor.state import ExecutorGraphState

logger = logging.getLogger(__name__)

# Maximum repair attempts - must match nodes/repair.py
MAX_REPAIRS = 2


def route_after_pick(state: ExecutorGraphState) -> str:
    """Route after pick_task node.

    Phase 2.4.2: Routes to plan_task if enable_planning=True, else build_context.
    The actual target is determined by the ConditionalEdge targets list.

    Decision:
        - If a task was selected -> continue (plan_task or build_context)
        - If no more tasks -> END

    Args:
        state: Current graph state.

    Returns:
        Next node name or END. Returns "continue" which maps to first target.
    """
    if state.get("current_task") is None:
        return END

    # Phase 2.4.2: Check if planning is enabled
    if state.get("enable_planning", False):
        return "plan_task"

    return "build_context"


def route_after_execute(state: ExecutorGraphState) -> str:
    """Route after execute_task node.

    Decision:
        - If execution succeeded (no error) -> verify_task
        - If execution failed -> handle_failure

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    if state.get("error") is not None:
        return "handle_failure"
    return "verify_task"


def route_after_verify(state: ExecutorGraphState) -> str:
    """Route after verify_task node.

    Phase 2.6: Updated to route to repair_test for targeted test repair.
    Phase 16.5.7: Added logging for routing decisions.

    Decision:
        - If verification passed -> checkpoint
        - If verification failed and test_repair_count < 2 -> repair_test
        - If verification failed and test_repair_count >= 2 -> handle_failure

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    if state.get("verified", False):
        logger.debug("Verification passed, routing to checkpoint")
        return "checkpoint"

    # Phase 2.6: Route to repair_test for targeted test repair
    test_repair_count = state.get("test_repair_count", 0)
    max_test_repairs = 2

    if test_repair_count < max_test_repairs:
        logger.debug(
            f"Verification failed, routing to repair_test "
            f"(attempt {test_repair_count + 1}/{max_test_repairs})"
        )
        return "repair_test"

    logger.debug(
        f"Verification failed and max repairs ({max_test_repairs}) reached, "
        "routing to handle_failure"
    )
    return "handle_failure"


def route_after_analyze_failure(state: ExecutorGraphState) -> str:
    """Route after analyze_failure node.

    DEPRECATED (Phase 2.4): This routing function is deprecated along with
    analyze_failure_node and replan_task_node. Use handle_failure_node with
    repair_count and test_repair_count for targeted repair instead.

    Phase 2.3.1: Routes based on failure classification.

    Decision:
        - If FATAL -> decide_next (give up)
        - If WRONG_APPROACH and replans < max -> replan_task
        - If TRANSIENT -> handle_failure (retry)

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    from ai_infra.executor.nodes.failure import FailureClassification

    classification = state.get("failure_classification")
    replan_count = state.get("replan_count", 0)
    max_replans = 2  # Matches MAX_REPLANS in replan.py

    # FATAL errors should stop
    if classification == FailureClassification.FATAL:
        return "decide_next"

    # WRONG_APPROACH should replan (if within limit)
    if classification == FailureClassification.WRONG_APPROACH:
        if replan_count < max_replans:
            return "replan_task"
        # Exceeded replan limit, treat as failure
        return "handle_failure"

    # TRANSIENT errors should retry via handle_failure
    return "handle_failure"


def route_after_replan(state: ExecutorGraphState) -> str:
    """Route after replan_task node.

    DEPRECATED (Phase 2.4): This routing function is deprecated along with
    replan_task_node. Use handle_failure_node with targeted repair instead.

    Phase 2.3.1: Always routes to build_context to retry with new plan.

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    return "build_context"


def route_after_failure(state: ExecutorGraphState) -> str:
    """Route after handle_failure node.

    Phase 2.1: Simplified to always route to decide_next.
    Rollback has been removed from the active flow because:
    - With pre-write validation, bad code never reaches disk
    - Rollback + retry rarely produces different results
    - Simpler graph without rollback

    Args:
        state: Current graph state.

    Returns:
        Next node name (always decide_next).
    """
    # Phase 2.1: Always go to decide_next, no more rollback
    return "decide_next"


def route_after_rollback(state: ExecutorGraphState) -> str:
    """Route after rollback node.

    .. deprecated:: Phase 2.1
        Rollback has been removed from the active flow. This function
        is kept for backward compatibility but is no longer used.

    Decision:
        - If retry count < max retries -> execute_task (retry)
        - Otherwise -> decide_next (give up on this task)

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        return "execute_task"
    return "decide_next"


def route_after_decide(state: ExecutorGraphState) -> str:
    """Route after decide_next node.

    Decision:
        - If should continue -> pick_task (next task)
        - Otherwise -> END

    Args:
        state: Current graph state.

    Returns:
        Next node name or END.
    """
    if state.get("should_continue", False):
        return "pick_task"
    return END


# =============================================================================
# Phase 1.2.3: Validation and Repair Routing
# =============================================================================


def route_after_validate(state: ExecutorGraphState) -> str:
    """Route after validate_code node.

    Phase 1.2.3: Routes based on validation result.

    Decision:
        - If validated (all code passed) -> write_files (proceed)
        - If needs_repair and repair_count < MAX -> repair_code
        - If needs_repair and repair_count >= MAX -> handle_failure

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    # If validation passed, proceed to write files
    if state.get("validated", False):
        return "write_files"

    # If repair is needed
    if state.get("needs_repair", False):
        repair_count = state.get("repair_count", 0)
        if repair_count < MAX_REPAIRS:
            return "repair_code"
        # Max repairs exceeded - escalate to failure handling
        return "handle_failure"

    # Default: proceed to write (shouldn't happen in normal flow)
    return "write_files"


def route_after_repair(state: ExecutorGraphState) -> str:
    """Route after repair_code node.

    Phase 1.2.3: Always routes back to validate_code to re-check.

    The repair node produces updated code that needs to be re-validated.
    This creates the validate -> repair -> validate loop with MAX_REPAIRS limit.

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    # Check if repair hit an error (e.g., max repairs exceeded)
    if state.get("error") is not None:
        return "handle_failure"

    # Always re-validate after repair
    return "validate_code"


def route_after_write(state: ExecutorGraphState) -> str:
    """Route after write_files node.

    Phase 1.3: Routes based on write result.

    Decision:
        - If files_written successfully -> verify_task
        - If write failed with error -> handle_failure

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    if state.get("error") is not None:
        return "handle_failure"

    return "verify_task"


def route_after_repair_test(state: ExecutorGraphState) -> str:
    """Route after repair_test node (Phase 2.3).

    Decision:
        - If test_repair_count < max_test_repairs and error is None -> verify_task (retry verification)
        - If test_repair_count >= max_test_repairs or error is not None -> handle_failure (give up)

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    max_test_repairs = 2
    test_repair_count = state.get("test_repair_count", 0)
    error = state.get("error")
    if error is None and test_repair_count < max_test_repairs:
        return "verify_task"
    return "handle_failure"


# =============================================================================
# Edge Target Constants
# =============================================================================

# Re-export END for convenience
__all__ = [
    "route_after_pick",
    "route_after_execute",
    "route_after_verify",
    "route_after_analyze_failure",
    "route_after_replan",
    "route_after_failure",
    "route_after_rollback",
    "route_after_decide",
    # Phase 1.2.3: Validation and repair routes
    "route_after_validate",
    "route_after_repair",
    # Phase 1.3: Write files route
    "route_after_write",
    # Phase 2.9.8: Recovery routes
    "route_after_classify_failure",
    "route_after_propose_recovery",
    "route_after_await_approval",
    "route_after_apply_recovery",
    "route_after_retry_deferred",
    "END",
]


# =============================================================================
# Phase 2.9.8: Recovery Flow Routing
# =============================================================================


def route_after_classify_failure(state: ExecutorGraphState) -> str:
    """Route after classify_failure node.

    Phase 2.9.8: Determines next step after failure classification.

    Decision:
        - Always routes to propose_recovery to generate a recovery proposal

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    return "propose_recovery"


def route_after_propose_recovery(state: ExecutorGraphState) -> str:
    """Route after propose_recovery node.

    Phase 2.9.8: Routes based on recovery proposal and approval requirements.

    Decision:
        - If proposal requires approval -> await_approval
        - If auto-approved (AUTO mode + non-ESCALATE) -> apply_recovery

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    from ai_infra.executor.nodes.recovery import ApprovalMode, RecoveryStrategy

    proposal = state.get("recovery_proposal")
    if proposal is None:
        return "decide_next"

    # Get approval mode
    mode_value = state.get("approval_mode", "auto")
    try:
        mode = ApprovalMode(mode_value)
    except ValueError:
        mode = ApprovalMode.AUTO

    # ESCALATE always requires approval
    if proposal.strategy == RecoveryStrategy.ESCALATE:
        return "await_approval"

    # Check approval mode
    if mode == ApprovalMode.INTERACTIVE:
        return "await_approval"
    elif mode == ApprovalMode.REVIEW_ONLY:
        return "await_approval"  # Will log and skip
    elif mode == ApprovalMode.APPROVE_DECOMPOSE:
        if proposal.strategy == RecoveryStrategy.DECOMPOSE:
            return "await_approval"
        # Auto-approve REWRITE, DEFER, SKIP
        return "apply_recovery"
    else:
        # AUTO mode - auto-approve
        return "apply_recovery"


def route_after_await_approval(state: ExecutorGraphState) -> str:
    """Route after await_approval node.

    Phase 2.9.8: Routes based on user approval decision.

    Decision:
        - If recovery_approved -> apply_recovery
        - If rejected -> decide_next

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    if state.get("recovery_approved", False):
        return "apply_recovery"
    return "decide_next"


def route_after_apply_recovery(state: ExecutorGraphState) -> str:
    """Route after apply_recovery node.

    Phase 2.9.8: Routes based on recovery strategy.

    Decision:
        - If REWRITE/DECOMPOSE (new tasks added) -> pick_task
        - If DEFER/SKIP/ESCALATE (no new tasks) -> decide_next

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    from ai_infra.executor.nodes.recovery import RecoveryStrategy

    proposal = state.get("recovery_proposal")
    if proposal is None:
        return "decide_next"

    # REWRITE and DECOMPOSE add new tasks to execute
    if proposal.strategy in (RecoveryStrategy.REWRITE, RecoveryStrategy.DECOMPOSE):
        return "pick_task"

    # DEFER, SKIP, ESCALATE don't add new tasks
    return "decide_next"


def route_after_retry_deferred(state: ExecutorGraphState) -> str:
    """Route after retry_deferred node.

    Phase 2.9.8: Routes based on whether deferred tasks were added.

    Decision:
        - If retrying_deferred (tasks added to queue) -> pick_task
        - If no deferred or all exceeded retries -> generate_report

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    if state.get("retrying_deferred", False):
        return "pick_task"
    return "generate_report"


def route_after_pick_with_recovery(state: ExecutorGraphState) -> str:
    """Route after pick_task with recovery flow support.

    Phase 2.9.8: Extended pick_task routing to include deferred retry.

    Decision:
        - If task selected -> plan_task or build_context
        - If no tasks but deferred exist -> retry_deferred
        - If no tasks and no deferred -> generate_report

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    from ai_infra.executor.nodes.recovery import should_retry_deferred

    # If a task was selected, continue normal flow
    if state.get("current_task") is not None:
        if state.get("enable_planning", False):
            return "plan_task"
        return "build_context"

    # No task selected - check for deferred tasks
    if should_retry_deferred(state):
        return "retry_deferred"

    # No tasks and no deferred - generate report and end
    return "generate_report"


def route_after_decide_with_recovery(state: ExecutorGraphState) -> str:
    """Route after decide_next with recovery flow support.

    Phase 2.9.8: Extended decide_next routing.

    Decision:
        - If should_continue -> pick_task
        - If no more tasks but deferred exist -> retry_deferred
        - Otherwise -> END (or generate_report if enabled)

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    from ai_infra.executor.nodes.recovery import should_retry_deferred

    if state.get("should_continue", False):
        return "pick_task"

    # Check for deferred tasks to retry
    if should_retry_deferred(state):
        return "retry_deferred"

    # Generate report before ending (if recovery flow enabled)
    if state.get("enable_recovery_flow", False):
        return "generate_report"

    return END


def route_to_recovery_or_failure(state: ExecutorGraphState) -> str:
    """Route to recovery flow or legacy failure handling.

    Phase 2.9.8: Determines whether to use new recovery flow or legacy path.

    Decision:
        - If enable_recovery_flow -> classify_failure
        - Otherwise -> handle_failure (legacy)

    Args:
        state: Current graph state.

    Returns:
        Next node name.
    """
    if state.get("enable_recovery_flow", False):
        return "classify_failure"
    return "handle_failure"
