"""Decide next node for executor graph.

Phase 1.2.2: Decides whether to continue to next task or end execution.
Phase 2.3.4: Saves todos.json on every decision (fixes bug where failed tasks didn't persist).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_infra.executor.state import ExecutorGraphState
from ai_infra.executor.todolist import TodoStatus
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoListManager

logger = get_logger("executor.nodes.decide")


def decide_next_node(
    state: ExecutorGraphState,
    *,
    todo_manager: TodoListManager | None = None,
) -> ExecutorGraphState:
    """Decide whether to continue to the next task or end execution.

    Phase 2.3.4: Now also saves todos.json on every decision to ensure
    state is persisted even for failed tasks.

    This node:
    1. Checks interrupt_requested flag
    2. Checks max_tasks limit
    3. Checks if there are remaining tasks
    4. Sets should_continue flag
    5. Saves todos.json (Phase 2.3.4)

    Args:
        state: Current graph state.
        todo_manager: TodoListManager for persisting state (Phase 2.3.4).

    Returns:
        Updated state with should_continue flag.
    """
    current_task = state.get("current_task")
    todos = state.get("todos", [])
    tasks_completed_count = state.get("tasks_completed_count", 0)
    max_tasks = state.get("max_tasks")
    interrupt_requested = state.get("interrupt_requested", False)

    task_id = str(current_task.id) if current_task else "unknown"

    # Mark current task as completed
    if current_task is not None:
        updated_todos = _mark_task_completed(todos, current_task)
        tasks_completed_count += 1
    else:
        updated_todos = todos

    logger.info(f"Task [{task_id}] completed. Total completed: {tasks_completed_count}")

    # Determine if we should continue
    should_continue = True
    stop_reason = None

    if interrupt_requested:
        should_continue = False
        stop_reason = "Interrupt requested"
    elif max_tasks is not None and tasks_completed_count >= max_tasks:
        should_continue = False
        stop_reason = f"Max tasks limit ({max_tasks}) reached"
    else:
        remaining = _count_remaining_tasks(updated_todos)
        if remaining == 0:
            should_continue = False
            stop_reason = "All tasks completed"
        else:
            logger.info(f"{remaining} tasks remaining, continuing execution")

    if stop_reason:
        logger.info(f"{stop_reason}, stopping execution")

    # Phase 2.3.4: Always save todos.json to persist state (even for failed tasks)
    _save_todos_json(todo_manager, updated_todos)

    return {
        **state,
        "todos": updated_todos,
        "tasks_completed_count": tasks_completed_count,
        "current_task": None,
        "should_continue": should_continue,
    }


def _save_todos_json(
    todo_manager: TodoListManager | None,
    todos: list,
) -> None:
    """Save todos to .executor/todos.json.

    Phase 2.3.4: Ensures state is persisted on every decision,
    including for failed tasks that bypass the checkpoint node.

    Args:
        todo_manager: TodoListManager instance.
        todos: Current list of todos.
    """
    if todo_manager is None:
        return

    try:
        # Update the manager's internal state with current todos
        if hasattr(todo_manager, "_todos"):
            todo_manager._todos = todos

        saved_path = todo_manager.save_to_json(create_if_missing=True)
        if saved_path:
            logger.info(f"Saved todos state to {saved_path}")
    except Exception as e:
        logger.warning(f"Failed to save todos.json: {e}")


def _mark_task_completed(todos: list, task: object) -> list:
    """Mark a task as completed in the todos list.

    Args:
        todos: List of TodoItem objects.
        task: The task to mark as completed.

    Returns:
        Updated todos list with task marked as DONE.
    """
    task_id = getattr(task, "id", None)
    if task_id is None:
        return todos

    updated = []
    for todo in todos:
        if getattr(todo, "id", None) == task_id:
            # Create new todo with COMPLETED status
            if hasattr(todo, "_replace"):
                # NamedTuple
                updated.append(todo._replace(status=TodoStatus.COMPLETED))
            elif hasattr(todo, "status"):
                # Dataclass or similar - create copy
                import copy

                new_todo = copy.copy(todo)
                new_todo.status = TodoStatus.COMPLETED
                updated.append(new_todo)
            else:
                updated.append(todo)
        else:
            updated.append(todo)

    return updated


def _count_remaining_tasks(todos: list) -> int:
    """Count remaining NOT_STARTED tasks.

    Args:
        todos: List of TodoItem objects.

    Returns:
        Number of tasks with NOT_STARTED status.
    """
    count = 0
    for todo in todos:
        status = getattr(todo, "status", None)
        if status == TodoStatus.NOT_STARTED:
            count += 1
    return count
