"""Pick task node for executor graph.

Phase 1.2.2: Selects the next incomplete task from todos.
Phase 1.4.2: References TodoListManager via state["todos"].
"""

from __future__ import annotations

from ai_infra.executor.state import ExecutorGraphState
from ai_infra.executor.todolist import TodoStatus
from ai_infra.logging import get_logger

logger = get_logger("executor.nodes.pick")


def pick_task_node(state: ExecutorGraphState) -> ExecutorGraphState:
    """Select the next incomplete task from todos.

    Phase 1.4.2: TodoListManager integration.

    The todos list in state is sourced from TodoListManager, which normalizes
    tasks from ROADMAP.md. This node scans those todos to find the next
    pending task.

    This node:
    1. Scans state["todos"] for NOT_STARTED tasks
    2. Skips tasks already in completed_todos or failed_todos
    3. Sets state["current_task"] to the selected task (or None)
    4. Resets repair_count and test_repair_count to 0 for the new task

    Args:
        state: Current graph state with todos list (from TodoListManager).

    Returns:
        Updated state with current_task set.
    """
    todos = state.get("todos", [])
    completed = set(state.get("completed_todos", []))
    failed = set(state.get("failed_todos", []))
    max_tasks = state.get("max_tasks")
    tasks_completed_count = state.get("tasks_completed_count", 0)

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
        todo_id_str = str(todo.id)

        # Skip completed or failed tasks
        if todo_id_str in completed or todo_id_str in failed:
            continue

        # Skip tasks not in NOT_STARTED status
        if todo.status != TodoStatus.NOT_STARTED:
            continue

        logger.info(f"Selected task: [{todo.id}] {todo.title}")
        return {
            **state,
            "current_task": todo,
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
        "should_continue": False,
    }
