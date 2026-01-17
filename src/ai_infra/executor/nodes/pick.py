"""Pick task node for executor graph.

Phase 1.2.2: Selects the next incomplete task from todos.
Phase 1.4.2: References TodoListManager via state["todos"].
Phase 3.2: Adds complexity estimation and auto-decomposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_infra.executor.state import ExecutorGraphState
from ai_infra.executor.task_decomposition import (
    ComplexityEstimator,
    RecommendedAction,
)
from ai_infra.executor.todolist import TodoStatus
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoItem

logger = get_logger("executor.nodes.pick")


# Default threshold for auto-decomposition (complexity score)
DEFAULT_DECOMPOSE_THRESHOLD = 8


def _should_skip_task(
    todo: TodoItem,
    completed: set[str],
    failed: set[str],
) -> bool:
    """Check if a task should be skipped.

    Args:
        todo: The task to check.
        completed: Set of completed task IDs.
        failed: Set of failed task IDs.

    Returns:
        True if the task should be skipped.
    """
    todo_id_str = str(todo.id)

    # Skip completed or failed tasks
    if todo_id_str in completed or todo_id_str in failed:
        return True

    # Skip tasks not in NOT_STARTED status
    if todo.status != TodoStatus.NOT_STARTED:
        return True

    # Skip parent tasks with uncompleted children
    if hasattr(todo, "has_subtasks") and todo.has_subtasks:
        # Children must be handled by the executor
        # This prevents executing the parent before children
        return True

    return False


def _estimate_task_complexity(
    todo: TodoItem,
    context: dict[str, any] | None = None,
) -> tuple[int, str]:
    """Estimate complexity of a task.

    Args:
        todo: The task to analyze.
        context: Additional context for estimation.

    Returns:
        Tuple of (complexity_score, recommended_action).
    """
    estimator = ComplexityEstimator()
    complexity = estimator.estimate(todo, context)
    return complexity.score, complexity.recommended_action.value


def pick_task_node(state: ExecutorGraphState) -> ExecutorGraphState:
    """Select the next incomplete task from todos.

    Phase 1.4.2: TodoListManager integration.
    Phase 3.2: Adds complexity estimation for task analysis.

    The todos list in state is sourced from TodoListManager, which normalizes
    tasks from ROADMAP.md. This node scans those todos to find the next
    pending task.

    This node:
    1. Scans state["todos"] for NOT_STARTED tasks
    2. Skips tasks already in completed_todos or failed_todos
    3. Skips parent tasks with uncompleted children (decomposed tasks)
    4. Estimates complexity and stores it in state
    5. Sets state["current_task"] to the selected task (or None)
    6. Resets repair_count and test_repair_count to 0 for the new task

    Args:
        state: Current graph state with todos list (from TodoListManager).

    Returns:
        Updated state with current_task set and complexity_info populated.
    """
    todos = state.get("todos", [])
    completed = set(state.get("completed_todos", []))
    failed = set(state.get("failed_todos", []))
    max_tasks = state.get("max_tasks")
    tasks_completed_count = state.get("tasks_completed_count", 0)
    auto_decompose = state.get("auto_decompose_enabled", False)
    decompose_threshold = state.get("decompose_threshold", DEFAULT_DECOMPOSE_THRESHOLD)

    # Check if we've hit max_tasks limit
    if max_tasks is not None and max_tasks > 0 and tasks_completed_count >= max_tasks:
        logger.info(f"Reached max_tasks limit: {max_tasks}")
        return {
            **state,
            "current_task": None,
            "should_continue": False,
        }

    # Find next pending task
    for todo in todos:
        # Use helper to check if task should be skipped
        if _should_skip_task(todo, completed, failed):
            continue

        # Estimate complexity
        context = {
            "dependency_count": len(getattr(todo, "source_task_ids", []) or []),
            "file_count_estimate": len(getattr(todo, "file_hints", []) or []),
            "previous_failures": 0,  # Could track this in state
        }
        complexity_score, recommended_action = _estimate_task_complexity(todo, context)

        # Store complexity on the todo for reference
        if hasattr(todo, "complexity_score"):
            todo.complexity_score = complexity_score

        logger.info(
            f"Selected task: [{todo.id}] {todo.title} "
            f"(complexity={complexity_score}, action={recommended_action})"
        )

        # Build complexity info for state
        complexity_info = {
            "score": complexity_score,
            "recommended_action": recommended_action,
            "should_decompose": (
                auto_decompose
                and complexity_score >= decompose_threshold
                and recommended_action == RecommendedAction.DECOMPOSE.value
            ),
        }

        return {
            **state,
            "current_task": todo,
            "complexity_info": complexity_info,
            "retry_count": 0,  # DEPRECATED: Phase 2.2 - kept for backward compat
            "repair_count": 0,  # Phase 2.2: Reset validation repair counter
            "test_repair_count": 0,  # Phase 2.2: Reset test repair counter
            "error": None,  # Clear any previous error
            "verified": False,
            "files_modified": [],
        }

    # No more tasks
    logger.info("No more pending tasks")
    return {
        **state,
        "current_task": None,
        "complexity_info": None,
        "should_continue": False,
    }
