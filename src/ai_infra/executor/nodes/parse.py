"""Parse roadmap node for executor graph.

Phase 1.2.2: Parses ROADMAP.md and populates todos list.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from ai_infra.executor.state import ExecutorGraphState, NodeTimeouts
from ai_infra.executor.todolist import TodoListManager, TodoStatus
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra import Agent

logger = get_logger("executor.nodes.parse")


async def parse_roadmap_node(
    state: ExecutorGraphState,
    *,
    agent: Agent | None = None,
    use_llm_normalization: bool = False,
    todo_manager: TodoListManager | None = None,
) -> ExecutorGraphState:
    """Parse ROADMAP.md and populate todos list.

    This node:
    1. Reads the ROADMAP.md file from state["roadmap_path"]
    2. Uses TodoListManager to parse and normalize tasks
    3. Populates state["todos"] with TodoItem list
    4. Initializes tracking fields (completed_todos, failed_todos, etc.)
    5. Saves initial state to .executor/todos.json

    Args:
        state: Current graph state with roadmap_path.
        agent: Optional agent for LLM-based normalization.
        use_llm_normalization: If True, use LLM to normalize ROADMAP.
        todo_manager: Optional TodoListManager for JSON persistence.

    Returns:
        Updated state with todos populated.

    Raises:
        FileNotFoundError: If ROADMAP.md doesn't exist.
        ValueError: If ROADMAP format is invalid.
    """
    roadmap_path = Path(state["roadmap_path"])

    if not roadmap_path.exists():
        logger.error(f"ROADMAP not found: {roadmap_path}")
        return {
            **state,
            "todos": [],
            "error": {
                "error_type": "parse",
                "message": f"ROADMAP not found: {roadmap_path}",
                "node": "parse_roadmap",
                "task_id": None,
                "recoverable": False,
                "stack_trace": None,
            },
            "should_continue": False,
        }

    try:
        if use_llm_normalization and agent is not None:
            # Use LLM-based normalization (Phase 5.13)
            manager = await asyncio.wait_for(
                TodoListManager.from_roadmap_llm(
                    roadmap_path,
                    agent,
                    force_renormalize=False,
                ),
                timeout=NodeTimeouts.PARSE_ROADMAP,
            )
        else:
            # Use regex-based parsing (faster, standard format only)
            from ai_infra.executor.parser import RoadmapParser

            parser = RoadmapParser()
            roadmap = parser.parse(roadmap_path)
            manager = TodoListManager.from_roadmap(
                roadmap,
                roadmap_path=roadmap_path,
                group_strategy="smart",
            )

        todos = manager.todos
        pending_count = sum(1 for t in todos if t.status == TodoStatus.NOT_STARTED)

        logger.info(f"Parsed ROADMAP: {len(todos)} todos ({pending_count} pending)")

        # Phase 2.3.4: Save initial todos.json immediately after parsing
        # This ensures todos.json exists before any task execution
        if todo_manager is not None:
            try:
                # Update the manager's internal todos list
                if hasattr(todo_manager, "_todos"):
                    todo_manager._todos = todos
                saved_path = todo_manager.save_to_json(create_if_missing=True)
                if saved_path:
                    logger.info(f"Created initial todos.json: {saved_path}")
            except Exception as e:
                logger.warning(f"Failed to save initial todos.json: {e}")

        return {
            **state,
            "todos": todos,
            "completed_todos": state.get("completed_todos", []),
            "failed_todos": state.get("failed_todos", []),
            "tasks_completed_count": state.get("tasks_completed_count", 0),
            "run_memory": state.get("run_memory", {}),
            "error": None,
        }

    except TimeoutError:
        logger.error(f"Timeout parsing ROADMAP: {roadmap_path}")
        return {
            **state,
            "todos": [],
            "error": {
                "error_type": "timeout",
                "message": f"Timeout parsing ROADMAP after {NodeTimeouts.PARSE_ROADMAP}s",
                "node": "parse_roadmap",
                "task_id": None,
                "recoverable": False,
                "stack_trace": None,
            },
            "should_continue": False,
        }
    except Exception as e:
        logger.exception(f"Failed to parse ROADMAP: {e}")
        return {
            **state,
            "todos": [],
            "error": {
                "error_type": "parse",
                "message": str(e),
                "node": "parse_roadmap",
                "task_id": None,
                "recoverable": False,
                "stack_trace": None,
            },
            "should_continue": False,
        }
