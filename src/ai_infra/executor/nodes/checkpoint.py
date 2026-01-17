"""Checkpoint node for executor graph.

Phase 1.2.2: Creates git checkpoints after successful task verification.
Phase 1.4.1: Integrates git checkpointer with graph.
Phase 1.4.2: Integrates TodoListManager for ROADMAP sync.
Phase 8.3: Extracts skills from successful task completions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.executor.state import ExecutorGraphState, NodeTimeouts
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.checkpoint import Checkpointer
    from ai_infra.executor.project_memory import ProjectMemory
    from ai_infra.executor.run_memory import RunMemory
    from ai_infra.executor.skills.database import SkillsDatabase
    from ai_infra.executor.skills.extractor import SkillExtractor
    from ai_infra.executor.todolist import TodoListManager

logger = get_logger("executor.nodes.checkpoint")


async def checkpoint_node(
    state: ExecutorGraphState,
    *,
    checkpointer: Checkpointer | None = None,
    todo_manager: TodoListManager | None = None,
    run_memory: RunMemory | None = None,
    project_memory: ProjectMemory | None = None,
    sync_roadmap: bool = True,
    # Phase 8.3: Skills extraction (EXECUTOR_3.md)
    skills_db: SkillsDatabase | None = None,
    skill_extractor: SkillExtractor | None = None,
    enable_learning: bool = True,
) -> ExecutorGraphState:
    """Create a git checkpoint after successful task verification.

    Phase 1.4.1: Git checkpointer integration.
    Phase 1.4.2: TodoListManager integration for ROADMAP sync.
    Phase 2.1.1: Records task outcome to RunMemory for context injection.
    Phase 2.1.2: Updates ProjectMemory with run outcomes for cross-run insights.
    Phase 8.3: Extracts skills from successful task completions (EXECUTOR_3.md).

    This node:
    1. Gets current_task and files_modified from state
    2. Creates a git commit with descriptive message
    3. Updates last_checkpoint_sha in state
    4. Marks todo as completed and syncs to ROADMAP
    5. Records task outcome to RunMemory for subsequent tasks
    6. Updates ProjectMemory after task completion (Phase 2.1.2)
    7. Extracts skills from verified successful tasks (Phase 8.3)

    Note: Graph state persistence is handled by decide_next node (after todos updated).

    Args:
        state: Current graph state with current_task and files_modified.
        checkpointer: The Checkpointer instance for git operations.
        todo_manager: The TodoListManager for ROADMAP sync.
        run_memory: RunMemory instance for recording task outcomes.
        project_memory: ProjectMemory instance for cross-run insights (Phase 2.1.2).
        sync_roadmap: Whether to sync completion to ROADMAP.md.
        skills_db: SkillsDatabase for storing extracted skills (Phase 8.3).
        skill_extractor: SkillExtractor for extracting skills (Phase 8.3).
        enable_learning: Whether to extract skills (Phase 8.3, default: True).

    Returns:
        Updated state with last_checkpoint_sha.
    """
    current_task = state.get("current_task")
    files_modified = state.get("files_modified", [])
    task_id = str(current_task.id) if current_task else "unknown"

    if current_task is None:
        logger.error("No current task for checkpoint")
        return {
            **state,
            "error": {
                "error_type": "checkpoint",
                "message": "No current task for checkpoint",
                "node": "checkpoint",
                "task_id": None,
                "recoverable": False,
                "stack_trace": None,
            },
        }

    # Phase 1.4.2: Always sync todo completion, regardless of checkpointer
    # This ensures ROADMAP checkboxes are updated even without git
    if todo_manager is not None:
        _sync_todo_completion(
            todo_manager=todo_manager,
            current_task=current_task,
            files_modified=files_modified,
            sync_roadmap=sync_roadmap,
        )

    # Phase 2.1.1: Record task outcome to RunMemory for context injection
    # This happens regardless of git checkpointing
    if run_memory is not None:
        _record_task_outcome(
            run_memory=run_memory,
            current_task=current_task,
            files_modified=files_modified,
            status="completed",
        )

        # Phase 2.1.2: Update ProjectMemory with run outcomes for cross-run insights
        # This happens after task outcome is recorded to run_memory
        if project_memory is not None:
            _update_project_memory(
                project_memory=project_memory,
                run_memory=run_memory,
                current_task=current_task,
            )

    # Phase 8.3: Extract skills from verified successful tasks (EXECUTOR_3.md)
    # Only extract when:
    # 1. Learning is enabled
    # 2. Task was verified successful
    # 3. Files were modified (something meaningful happened)
    verified = state.get("verified", False)
    if enable_learning and verified and files_modified:
        await _extract_skill_from_task(
            current_task=current_task,
            files_modified=files_modified,
            agent_result=state.get("agent_result"),
            skills_db=skills_db,
            skill_extractor=skill_extractor,
            state=state,
        )

    # If no checkpointer, skip git checkpoint (development mode)
    if checkpointer is None:
        logger.warning(f"No checkpointer provided, skipping git checkpoint for [{task_id}]")
        return {**state, "error": None}

    # If no files modified, nothing to git checkpoint
    if not files_modified:
        logger.info(f"No files modified for task [{task_id}], skipping git checkpoint")
        return {**state, "error": None}

    logger.info(f"Creating checkpoint for task [{task_id}] ({len(files_modified)} files)")

    try:
        # Build commit message
        commit_message = _build_commit_message(current_task, files_modified)

        # Create checkpoint with timeout
        # Run synchronous checkpoint in executor to avoid blocking
        loop = asyncio.get_event_loop()
        commit_info = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: checkpointer.checkpoint(
                    message=commit_message,
                    files=files_modified,
                ),
            ),
            timeout=NodeTimeouts.CHECKPOINT,
        )

        sha = commit_info.sha if commit_info else None

        if sha:
            logger.info(f"Checkpoint created for task [{task_id}]: {sha[:8]}")
            return {**state, "last_checkpoint_sha": sha, "error": None}
        else:
            # Checkpoint returned None (nothing to commit)
            logger.info(f"No changes to checkpoint for task [{task_id}]")
            return {**state, "error": None}

    except TimeoutError:
        logger.error(f"Checkpoint timed out for task [{task_id}]")
        return {
            **state,
            "error": {
                "error_type": "timeout",
                "message": f"Checkpoint timed out after {NodeTimeouts.CHECKPOINT}s",
                "node": "checkpoint",
                "task_id": task_id,
                "recoverable": True,
                "stack_trace": None,
            },
        }

    except Exception as e:
        logger.exception(f"Checkpoint failed for task [{task_id}]: {e}")

        import traceback

        return {
            **state,
            "error": {
                "error_type": "checkpoint",
                "message": str(e),
                "node": "checkpoint",
                "task_id": task_id,
                "recoverable": True,  # Can retry
                "stack_trace": traceback.format_exc(),
            },
        }


def _sync_todo_completion(
    todo_manager: TodoListManager | None,
    current_task: object,
    files_modified: list[str],
    sync_roadmap: bool,
) -> None:
    """Sync todo completion to TodoListManager and optionally ROADMAP.

    Phase 1.4.2: TodoListManager integration.
    Phase 1.3.1: Also saves to .executor/todos.json.

    Args:
        todo_manager: The TodoListManager instance.
        current_task: The completed task.
        files_modified: List of modified file paths.
        sync_roadmap: Whether to sync to ROADMAP.md.
    """
    if todo_manager is None:
        return

    todo_id = getattr(current_task, "id", None)
    if todo_id is None:
        return

    try:
        # Mark todo as completed with files created
        todo_manager.mark_completed(
            todo_id=todo_id,
            files_created=files_modified,
            sync_roadmap=sync_roadmap,
        )
        logger.info(f"Todo [{todo_id}] marked completed, sync_roadmap={sync_roadmap}")

        # Phase 1.3.1: Also save to .executor/todos.json
        saved_path = todo_manager.save_to_json(create_if_missing=True)
        if saved_path:
            logger.debug(f"Todo state saved to {saved_path}")
    except Exception as e:
        # Log but don't fail - git checkpoint already succeeded
        logger.warning(f"Failed to sync todo completion: {e}")


def _build_commit_message(task: object, files_modified: list[str]) -> str:
    """Build a descriptive commit message for the checkpoint.

    Args:
        task: The completed task.
        files_modified: List of modified file paths.

    Returns:
        Formatted commit message.
    """
    task_id = getattr(task, "id", "?")
    title = getattr(task, "title", "Task completed")

    # Truncate title if too long
    if len(title) > 50:
        title = title[:47] + "..."

    lines = [
        f"[executor] {title}",
        "",
        f"Task ID: {task_id}",
        "",
        "Files modified:",
    ]

    # Add files (limit to 10)
    for f in files_modified[:10]:
        lines.append(f"  - {f}")

    if len(files_modified) > 10:
        lines.append(f"  ... and {len(files_modified) - 10} more")

    return "\n".join(lines)


def _record_task_outcome(
    run_memory: RunMemory,
    current_task: object,
    files_modified: list[str],
    status: str = "completed",
) -> None:
    """Record task outcome to RunMemory for context injection.

    Phase 2.1.1: Records completed tasks to RunMemory so subsequent tasks
    can see what was done before.

    Args:
        run_memory: The RunMemory instance to record to.
        current_task: The completed task.
        files_modified: List of modified file paths.
        status: Task completion status ("completed", "failed", "skipped").
    """
    from ai_infra.executor.run_memory import FileAction, TaskOutcome

    task_id = getattr(current_task, "id", "?")
    title = getattr(current_task, "title", "Unknown task")
    description = getattr(current_task, "description", "")

    try:
        # Build files dict with actions (assume all are created/modified)
        files: dict[Path, FileAction] = {}
        for f in files_modified:
            path = Path(f)
            # Simple heuristic: if it's a new file we created, mark as CREATED
            # In practice, both CREATED and MODIFIED are useful context
            files[path] = FileAction.MODIFIED

        # Create outcome
        outcome = TaskOutcome(
            task_id=str(task_id),
            title=title,
            status=status,
            files=files,
            key_decisions=[],  # Could be extracted from agent response in future
            summary=description if description else f"Completed task: {title}",
            duration_seconds=0.0,  # TODO: Track in Phase 2.2.1
            tokens_used=0,  # TODO: Track in Phase 2.2.1
        )

        # Record to run memory
        run_memory.add_outcome(outcome)

        logger.info(
            f"Recorded outcome for task [{task_id}] to run memory "
            f"({len(files_modified)} files, status={status})"
        )

    except Exception as e:
        # Log but don't fail - this is supplementary context
        logger.warning(f"Failed to record task outcome to run memory: {e}")


def _update_project_memory(
    project_memory: ProjectMemory,
    run_memory: RunMemory,
    current_task: object,
) -> None:
    """Update ProjectMemory with run outcomes for cross-run insights.

    Phase 2.1.2: Updates project memory after each task completion so that
    subsequent runs benefit from accumulated knowledge.

    Args:
        project_memory: The ProjectMemory instance to update.
        run_memory: The current RunMemory with task outcomes.
        current_task: The completed task.
    """
    task_id = getattr(current_task, "id", "?")

    try:
        # Update project memory from the current run
        # This incrementally saves key files, lessons learned, and run history
        project_memory.update_from_run(run_memory)

        logger.info(
            f"Updated project memory after task [{task_id}]: "
            f"{len(project_memory.key_files)} key files tracked"
        )

    except Exception as e:
        # Log but don't fail - this is supplementary context
        logger.warning(f"Failed to update project memory: {e}")


async def _extract_skill_from_task(
    current_task: Any,
    files_modified: list[str],
    agent_result: dict[str, Any] | None,
    skills_db: SkillsDatabase | None,
    skill_extractor: SkillExtractor | None,
    state: ExecutorGraphState,
) -> None:
    """Extract a skill from a successful task completion.

    Phase 8.3: Extracts skills from verified successful tasks using heuristics.
    Skills are stored in SkillsDatabase for future task context injection.

    Args:
        current_task: The completed task.
        files_modified: List of modified file paths.
        agent_result: The agent execution result.
        skills_db: SkillsDatabase for storing extracted skills.
        skill_extractor: SkillExtractor for skill extraction.
        state: Current graph state for context.
    """
    # Need both skills_db and extractor to extract skills
    if skills_db is None:
        return

    task_id = getattr(current_task, "id", "?")
    task_title = getattr(current_task, "title", "Unknown")

    try:
        # Create skill extractor if not provided
        extractor = skill_extractor
        if extractor is None:
            from ai_infra.executor.skills.extractor import SkillExtractor

            extractor = SkillExtractor(db=skills_db, llm=None)

        # Build context for skill matching
        from ai_infra.executor.skills.models import SkillContext

        # Infer language from files modified
        language = _infer_language_from_files(files_modified)

        context = SkillContext(
            language=language,
            framework=state.get("project_framework"),
            task_title=task_title,
            task_description=getattr(current_task, "description", "") or "",
            file_hints=files_modified,
        )

        # Build task result from agent result
        from ai_infra.executor.skills.extractor import TaskResult

        task_result = TaskResult(
            success=True,
            files_modified=files_modified,
            files_created=[f for f in files_modified if _is_likely_new_file(f, agent_result)],
            actions_summary=_extract_actions_summary(agent_result),
            diff_summary=_extract_diff_summary(agent_result, files_modified),
        )

        # Extract skill using heuristic-based extraction (no LLM)
        skill = await extractor.extract_from_success(
            task=current_task,
            result=task_result,
            context=context,
        )

        if skill:
            logger.info(
                f"Extracted skill from task [{task_id}]: {skill.title}",
                extra={"skill_id": skill.id, "type": skill.type.value},
            )

    except Exception as e:
        # Don't fail checkpoint on skill extraction error
        logger.warning(f"Failed to extract skill from task [{task_id}]: {e}")


def _infer_language_from_files(files_modified: list[str]) -> str:
    """Infer programming language from modified files.

    Args:
        files_modified: List of modified file paths.

    Returns:
        Inferred language (default: "python").
    """
    if not files_modified:
        return "python"

    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
    }

    for filepath in files_modified:
        for ext, lang in extension_map.items():
            if filepath.endswith(ext):
                return lang

    return "python"


def _is_likely_new_file(filepath: str, agent_result: dict[str, Any] | None) -> bool:
    """Check if file was likely created (vs modified).

    Args:
        filepath: Path to the file.
        agent_result: Agent execution result.

    Returns:
        True if file was likely created.
    """
    # Heuristic: check if file was mentioned in creation context
    if agent_result is None:
        return False

    # Check for create/write patterns in agent messages
    messages = agent_result.get("messages", [])
    for msg in messages:
        content = str(msg.get("content", "")) if isinstance(msg, dict) else str(msg)
        if filepath in content and any(
            kw in content.lower() for kw in ["create", "creating", "new file"]
        ):
            return True

    return False


def _extract_actions_summary(agent_result: dict[str, Any] | None) -> str:
    """Extract actions summary from agent result.

    Args:
        agent_result: Agent execution result.

    Returns:
        Summary of actions taken.
    """
    if agent_result is None:
        return ""

    # Try to extract from agent response
    messages = agent_result.get("messages", [])
    if messages:
        # Get last assistant message as summary
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "assistant" and content:
                    # Truncate to reasonable length
                    return content[:500]

    return ""


def _extract_diff_summary(agent_result: dict[str, Any] | None, files_modified: list[str]) -> str:
    """Extract diff summary from agent result.

    Args:
        agent_result: Agent execution result.
        files_modified: List of modified file paths.

    Returns:
        Summary of code changes.
    """
    if agent_result is None or not files_modified:
        return ""

    # Build simple summary from files list
    summary_parts = [f"Modified {len(files_modified)} file(s):"]
    for f in files_modified[:5]:
        summary_parts.append(f"  - {f}")
    if len(files_modified) > 5:
        summary_parts.append(f"  ... and {len(files_modified) - 5} more")

    return "\n".join(summary_parts)
