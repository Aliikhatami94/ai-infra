"""Write files node for executor graph.

Phase 1.3: Separate file writing from code generation for cleaner flow.

This module provides:
- write_files_node(): Graph node for writing validated code to disk

The new flow is:
    execute_task (generate only) → validate_code → repair_code? → write_files → verify_task

Benefits:
- Validation happens BEFORE any disk writes
- Failed validation = no disk write = no wasted I/O
- Cleaner separation of concerns
- Easier testing and debugging
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.executor.state import ExecutorGraphState
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    pass


logger = get_logger("executor.nodes.write")


async def write_files_node(
    state: ExecutorGraphState,
    *,
    dry_run: bool = False,
) -> ExecutorGraphState:
    """Write validated code to disk.

    Phase 1.3: This node only runs AFTER validation passes.
    It writes all generated files atomically (as much as possible).

    The node:
    1. Gets generated_code from state (populated by execute_task)
    2. Resolves file paths relative to project_root
    3. Creates directories as needed
    4. Writes each file to disk
    5. Tracks files_modified for checkpointing

    Args:
        state: Current state with generated_code.
        dry_run: If True, skip actual file writes (for testing).

    Returns:
        Updated state with files_modified and files_written=True.

    Examples:
        >>> state = {"generated_code": {"src/app.py": "print('hello')"}}
        >>> result = await write_files_node(state)
        >>> result["files_written"]
        True
    """
    generated_code: dict[str, str] = state.get("generated_code", {})

    # Get project root from state or roadmap path
    project_root_value = state.get("project_root")
    if project_root_value:
        project_root = str(project_root_value)
    else:
        roadmap_path = state.get("roadmap_path")
        if roadmap_path:
            project_root = str(Path(str(roadmap_path)).parent)
        else:
            project_root = os.getcwd()

    # Get task ID for logging
    current_task = state.get("current_task")
    task_id = str(current_task.id) if current_task else "unknown"

    # Check if there's anything to write
    if not generated_code:
        logger.info(f"Task [{task_id}] has no generated code to write")
        return {
            **state,
            "files_modified": [],
            "files_written": True,
        }

    # Check if already validated
    if not state.get("validated", False):
        logger.warning(
            f"Task [{task_id}] code not validated, skipping write. "
            "This indicates a graph routing issue."
        )
        return {
            **state,
            "files_written": False,
            "error": {
                "error_type": "validation",
                "message": "Cannot write files: code not validated",
                "node": "write_files",
                "task_id": task_id,
                "recoverable": False,
                "stack_trace": None,
            },
        }

    files_modified: list[str] = []
    write_errors: list[dict[str, Any]] = []

    for file_path, code in generated_code.items():
        # Resolve full path
        if os.path.isabs(file_path):
            full_path = Path(file_path)
        else:
            full_path = Path(project_root) / file_path

        try:
            if dry_run:
                logger.info(f"[DRY RUN] Would write {len(code)} bytes to {file_path}")
            else:
                # Ensure directory exists
                full_path.parent.mkdir(parents=True, exist_ok=True)

                # Write file
                full_path.write_text(code, encoding="utf-8")
                logger.info(f"Wrote {len(code)} bytes to {file_path}")

            files_modified.append(str(file_path))

        except OSError as e:
            logger.error(f"Failed to write {file_path}: {e}")
            write_errors.append(
                {
                    "file": str(file_path),
                    "error": str(e),
                }
            )

    # If any writes failed, report error but continue
    if write_errors:
        error_files = [e["file"] for e in write_errors]
        logger.warning(f"Task [{task_id}] had {len(write_errors)} write failures: {error_files}")
        return {
            **state,
            "files_modified": files_modified,
            "files_written": len(files_modified) > 0,
            "write_errors": write_errors,
            "error": {
                "error_type": "execution",
                "message": f"Failed to write {len(write_errors)} files: {error_files}",
                "node": "write_files",
                "task_id": task_id,
                "recoverable": True,  # Could retry
                "stack_trace": None,
            },
        }

    logger.info(f"Task [{task_id}] wrote {len(files_modified)} files successfully")

    return {
        **state,
        "files_modified": files_modified,
        "files_written": True,
        "write_errors": None,  # Clear any previous errors
    }


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
