"""Run memory for tracking task execution within a single run.

This module provides ephemeral memory for the current execution run,
tracking completed tasks, files modified, and key decisions made by
the agent. This context is injected into subsequent task prompts to
help the agent maintain awareness of what has been accomplished.

Part of Phase 5.8.1: Execution Memory Architecture.

Example:
    ```python
    from ai_infra.executor import RunMemory, TaskOutcome, FileAction

    # Initialize at run start
    memory = RunMemory(run_id="run_abc123")

    # Record task completions
    memory.add_outcome(TaskOutcome(
        task_id="1.1",
        title="Create utils module",
        status="completed",
        files={Path("src/utils.js"): FileAction.CREATED},
        key_decisions=["Used ES6 module syntax"],
        summary="Created utils.js with formatName() function",
        duration_seconds=12.5,
        tokens_used=1500,
    ))

    # Get context for next task's prompt
    context = memory.get_context(current_task_id="1.2")
    # Returns formatted markdown with completed tasks and files
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class FileAction(str, Enum):
    """Type of file operation performed during task execution."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class TaskOutcome:
    """Record of a completed task's execution.

    Attributes:
        task_id: The task identifier (e.g., "1.1", "2.3").
        title: Brief description of the task.
        status: Execution status ("completed", "failed", "skipped").
        files: Map of file paths to the action performed on them.
        key_decisions: List of key decisions made by the agent.
        summary: One-line summary of what was accomplished.
        duration_seconds: How long the task took to execute.
        tokens_used: Total tokens consumed (input + output).
    """

    task_id: str
    title: str
    status: str
    files: dict[Path, FileAction] = field(default_factory=dict)
    key_decisions: list[str] = field(default_factory=list)
    summary: str = ""
    duration_seconds: float = 0.0
    tokens_used: int = 0

    def to_context_line(self) -> str:
        """Format task outcome for inclusion in prompt context.

        Returns a markdown-formatted single line summarizing this task.
        Files are truncated to show at most 3, with a count of remaining.

        Returns:
            Formatted string like "- **1.1**: Created utils module | Files: utils.js (created)"
        """
        # Build files string, truncating if more than 3
        if self.files:
            file_parts = [f"{p.name} ({a.value})" for p, a in list(self.files.items())[:3]]
            files_str = ", ".join(file_parts)
            if len(self.files) > 3:
                files_str += f" (+{len(self.files) - 3} more)"
        else:
            files_str = "none"

        # Use summary if available, otherwise title
        description = self.summary if self.summary else self.title

        return f"- **{self.task_id}**: {description} | Files: {files_str}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status,
            "files": {str(p): a.value for p, a in self.files.items()},
            "key_decisions": self.key_decisions,
            "summary": self.summary,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskOutcome:
        """Deserialize from dictionary.

        Args:
            data: Dictionary from to_dict() or JSON.

        Returns:
            Reconstructed TaskOutcome instance.
        """
        files = {Path(p): FileAction(a) for p, a in data.get("files", {}).items()}
        return cls(
            task_id=data["task_id"],
            title=data["title"],
            status=data["status"],
            files=files,
            key_decisions=data.get("key_decisions", []),
            summary=data.get("summary", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            tokens_used=data.get("tokens_used", 0),
        )


def _count_tokens_simple(text: str) -> int:
    """Simple token estimation for context budgeting.

    Uses the standard heuristic of ~4 characters per token.
    This is intentionally conservative (overestimates) to stay within limits.

    Args:
        text: Text to count tokens for.

    Returns:
        Estimated token count.
    """
    return (len(text) + 3) // 4


@dataclass
class RunMemory:
    """Ephemeral memory for tracking the current execution run.

    RunMemory provides task-to-task context within a single executor run.
    It tracks completed tasks, files modified, and can generate a context
    summary to inject into subsequent task prompts.

    This memory is ephemeral - it only exists in memory during the run
    and is not persisted. For persistent memory, see ProjectMemory.

    Attributes:
        run_id: Unique identifier for this execution run.
        started_at: ISO timestamp when the run started.
        outcomes: List of task outcomes recorded during the run.
        roadmap_path: Optional path to the ROADMAP being executed.
    """

    run_id: str
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    outcomes: list[TaskOutcome] = field(default_factory=list)
    roadmap_path: str = ""
    _max_context_tokens: int = 4000

    def add_outcome(self, outcome: TaskOutcome) -> None:
        """Record a task completion.

        Args:
            outcome: The TaskOutcome to record.
        """
        self.outcomes.append(outcome)

    def get_outcome(self, task_id: str) -> TaskOutcome | None:
        """Get outcome for a specific task.

        Args:
            task_id: The task identifier to look up.

        Returns:
            TaskOutcome if found, None otherwise.
        """
        for outcome in self.outcomes:
            if outcome.task_id == task_id:
                return outcome
        return None

    def get_completed_files(self) -> dict[Path, TaskOutcome]:
        """Get map of all files touched to the task that touched them.

        If multiple tasks touched the same file, the most recent task wins.

        Returns:
            Dictionary mapping file paths to the TaskOutcome that last modified them.
        """
        result: dict[Path, TaskOutcome] = {}
        for outcome in self.outcomes:
            for path in outcome.files:
                result[path] = outcome
        return result

    def get_files_by_action(self, action: FileAction) -> list[Path]:
        """Get all files with a specific action type.

        Args:
            action: The FileAction to filter by.

        Returns:
            List of file paths that had the specified action.
        """
        result: list[Path] = []
        for outcome in self.outcomes:
            for path, file_action in outcome.files.items():
                if file_action == action:
                    result.append(path)
        return result

    def get_context(
        self,
        current_task_id: str = "",
        max_tokens: int | None = None,
    ) -> str:
        """Generate context for prompt injection.

        Creates a markdown-formatted summary of completed tasks and files
        modified during this run. The context is truncated to fit within
        the token budget, prioritizing recent tasks.

        Args:
            current_task_id: The ID of the task about to be executed (for reference).
            max_tokens: Maximum tokens for context. Uses default if None.

        Returns:
            Markdown-formatted context string, or empty string if no outcomes.
        """
        if not self.outcomes:
            return ""

        budget = max_tokens if max_tokens is not None else self._max_context_tokens

        # Build full context first
        lines = ["## Previously Completed Tasks (This Run)", ""]

        for outcome in self.outcomes:
            lines.append(outcome.to_context_line())

        # Add file map for reference
        files = self.get_completed_files()
        if files:
            lines.extend(["", "### Files Created/Modified"])
            file_items = list(files.items())
            for path, outcome in file_items[:10]:
                action = outcome.files[path].value
                lines.append(f"- `{path}` ({action} by {outcome.task_id})")
            if len(file_items) > 10:
                lines.append(f"- ... and {len(file_items) - 10} more files")

        raw_context = "\n".join(lines)

        # Check if within budget
        current_tokens = _count_tokens_simple(raw_context)
        if current_tokens <= budget:
            return raw_context

        # Over budget - truncate from old tasks
        return self._truncate_context(budget)

    def _truncate_context(self, max_tokens: int) -> str:
        """Truncate context to fit within token budget.

        Strategy: Keep recent tasks, drop oldest ones first.
        Always include the file map as it's most relevant.

        Args:
            max_tokens: Maximum tokens to use.

        Returns:
            Truncated context string.
        """
        if not self.outcomes:
            return ""

        # Reserve tokens for header and file map
        header_reserve = 100
        file_map_reserve = min(500, max_tokens // 4)
        task_budget = max_tokens - header_reserve - file_map_reserve

        # Build task lines from most recent, working backwards
        task_lines: list[str] = []
        tokens_used = 0

        for outcome in reversed(self.outcomes):
            line = outcome.to_context_line()
            line_tokens = _count_tokens_simple(line)

            if tokens_used + line_tokens > task_budget:
                # Add truncation note and stop
                remaining = len(self.outcomes) - len(task_lines)
                if remaining > 0:
                    task_lines.insert(0, f"- ... {remaining} earlier tasks omitted ...")
                break

            task_lines.insert(0, line)
            tokens_used += line_tokens

        # Build result
        lines = ["## Previously Completed Tasks (This Run)", ""]
        lines.extend(task_lines)

        # Add compact file map
        files = self.get_completed_files()
        if files:
            lines.extend(["", "### Files Created/Modified"])
            file_items = list(files.items())
            # Show fewer files when truncating
            for path, outcome in file_items[:5]:
                action = outcome.files[path].value
                lines.append(f"- `{path}` ({action})")
            if len(file_items) > 5:
                lines.append(f"- ... and {len(file_items) - 5} more files")

        return "\n".join(lines)

    def get_stats(self) -> dict[str, int]:
        """Get summary statistics for this run.

        Returns:
            Dictionary with task counts by status.
        """
        completed = sum(1 for o in self.outcomes if o.status == "completed")
        failed = sum(1 for o in self.outcomes if o.status == "failed")
        skipped = sum(1 for o in self.outcomes if o.status == "skipped")
        total_tokens = sum(o.tokens_used for o in self.outcomes)
        total_duration = sum(o.duration_seconds for o in self.outcomes)

        return {
            "total": len(self.outcomes),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "total_tokens": total_tokens,
            "total_duration_seconds": int(total_duration),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize run memory for project memory update.

        Returns:
            Dictionary with run summary suitable for persistence.
        """
        stats = self.get_stats()
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "roadmap_path": self.roadmap_path,
            "task_count": stats["total"],
            "completed": stats["completed"],
            "failed": stats["failed"],
            "skipped": stats["skipped"],
            "total_tokens": stats["total_tokens"],
            "total_duration_seconds": stats["total_duration_seconds"],
            "files_created": [str(p) for p in self.get_files_by_action(FileAction.CREATED)],
            "files_modified": [str(p) for p in self.get_files_by_action(FileAction.MODIFIED)],
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Serialize complete run memory including all outcomes.

        Returns:
            Full dictionary with all task outcomes.
        """
        base = self.to_dict()
        base["outcomes"] = [o.to_dict() for o in self.outcomes]
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMemory:
        """Deserialize from dictionary.

        Args:
            data: Dictionary from to_full_dict() or JSON.

        Returns:
            Reconstructed RunMemory instance.
        """
        outcomes = [TaskOutcome.from_dict(o) for o in data.get("outcomes", [])]
        return cls(
            run_id=data["run_id"],
            started_at=data.get("started_at", datetime.now(UTC).isoformat()),
            outcomes=outcomes,
            roadmap_path=data.get("roadmap_path", ""),
        )
