"""Replan task node for executor graph.

Phase 2.3.1: Adaptive replanning - generates new approach based on failure analysis.

.. deprecated:: Phase 2.4
    Adaptive replanning has been removed from the active flow in favor of
    targeted repair (repair_code and repair_test). This module is kept for
    backward compatibility but is no longer used in the default graph.

This node is invoked when analyze_failure_node classifies an error as WRONG_APPROACH,
meaning the original approach was fundamentally flawed and needs a different strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_infra.executor.state import ExecutorGraphState
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.agent import Agent

logger = get_logger("executor.nodes.replan")


# Maximum replans per task to prevent infinite loops
MAX_REPLANS = 2


async def replan_task_node(
    state: ExecutorGraphState,
    *,
    planner_agent: Agent | None = None,
) -> ExecutorGraphState:
    """Generate a new approach based on failure analysis.

    .. deprecated:: Phase 2.4
        Use repair_code_node and repair_test_node instead. This function
        is kept for backward compatibility.

    Phase 2.3.1: Creates a revised execution plan when the original approach failed.

    This node:
    1. Gets failure reason and suggested fix from state
    2. Uses LLM to generate a revised approach
    3. Updates state with new execution plan
    4. Increments replan_count to track replanning attempts

    The node routes back to build_context -> execute_task with the new plan.

    Args:
        state: Current graph state with failure analysis.
        planner_agent: Agent for generating revised plan.

    Returns:
        Updated state with execution_plan and incremented replan_count.
    """
    current_task = state.get("current_task")
    task_id = str(current_task.id) if current_task else "unknown"
    task_title = current_task.title if current_task else "Unknown task"
    task_description = (
        current_task.description if current_task and hasattr(current_task, "description") else ""
    )

    failure_reason = state.get("failure_reason", "Unknown failure")
    suggested_fix = state.get("suggested_fix", "")
    previous_prompt = state.get("prompt", "")
    replan_count = state.get("replan_count", 0)

    logger.info(f"Replanning task [{task_id}] (attempt {replan_count + 1}/{MAX_REPLANS})")

    # Build the replan prompt
    replan_prompt = f"""The previous attempt to complete this task failed.

## Task
{task_title}
{task_description}

## What Failed
{failure_reason}

## Suggested Fix
{suggested_fix if suggested_fix else "No specific fix suggested - revise approach"}

## Previous Approach (excerpt)
{previous_prompt[:1000] if previous_prompt else "No previous prompt available"}

## Your Task
Generate a REVISED approach that:
1. Addresses the failure reason
2. Avoids the same mistake
3. Completes the original task

Be specific about what to do differently. If a dependency is missing, include installation.
If a file path was wrong, specify the correct path. If syntax was wrong, show correct syntax.
"""

    # Generate new plan using agent
    if planner_agent is not None:
        try:
            plan_result = await planner_agent.arun(replan_prompt)
            execution_plan = str(plan_result)
            logger.info(f"Generated new plan for task [{task_id}]")
        except Exception as e:
            logger.warning(f"Failed to generate replan: {e}")
            execution_plan = (
                f"Retry with fix: {suggested_fix}" if suggested_fix else "Retry with caution"
            )
    else:
        # No agent - use suggested fix as the plan
        execution_plan = (
            f"Revised approach: {suggested_fix}"
            if suggested_fix
            else "Retry with different approach"
        )
        logger.info(f"No planner agent, using suggested fix for task [{task_id}]")

    # Clear error state for retry with new plan
    return {
        **state,
        "execution_plan": execution_plan,
        "replan_count": replan_count + 1,
        "error": None,  # Clear error for new attempt
        "verified": False,  # Reset verification
        "agent_result": None,  # Clear previous result
        "retry_count": 0,  # Reset retry count for new plan
    }


def should_replan(state: ExecutorGraphState) -> bool:
    """Check if replanning should be attempted.

    Args:
        state: Current graph state.

    Returns:
        True if replanning should be attempted, False otherwise.
    """
    from ai_infra.executor.nodes.failure import FailureClassification

    classification = state.get("failure_classification")
    replan_count = state.get("replan_count", 0)

    # Only replan for WRONG_APPROACH and within replan limit
    if classification == FailureClassification.WRONG_APPROACH:
        if replan_count < MAX_REPLANS:
            return True
        else:
            logger.warning(f"Max replans ({MAX_REPLANS}) exceeded, giving up")

    return False
