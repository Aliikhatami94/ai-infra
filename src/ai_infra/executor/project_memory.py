"""Project memory for persistent cross-run project knowledge.

This module provides persistent memory about a project, including:
- Project type and detected frameworks
- Key files and their purposes
- Conventions discovered during execution
- History of past execution runs

The memory is stored at `.executor/project-memory.json` and updated
after each execution run.

Part of Phase 5.8.2: Execution Memory Architecture.

Example:
    ```python
    from pathlib import Path
    from ai_infra.executor import ProjectMemory, RunMemory

    # Load existing or create new project memory
    project_root = Path("/path/to/project")
    memory = ProjectMemory.load(project_root)

    # After a run completes, update project memory
    run_memory = RunMemory(run_id="run_abc123")
    # ... tasks executed ...
    memory.update_from_run(run_memory)

    # Get context for prompts
    context = memory.get_context()
    # Returns formatted markdown with project info, key files, etc.
    ```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_infra.executor.run_memory import RunMemory


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
class FileInfo:
    """Tracked information about a project file.

    Attributes:
        path: Relative path to the file from project root.
        purpose: Description of what the file does (from task summary).
        created_by_task: Task ID that created this file.
        last_modified_by_task: Task ID that last modified this file.
        imports: List of modules this file imports (for future use).
        exports: List of symbols this file exports (for future use).
    """

    path: str
    purpose: str = ""
    created_by_task: str | None = None
    last_modified_by_task: str | None = None
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage.

        Returns:
            Dictionary representation.
        """
        return {
            "path": self.path,
            "purpose": self.purpose,
            "created_by_task": self.created_by_task,
            "last_modified_by_task": self.last_modified_by_task,
            "imports": self.imports,
            "exports": self.exports,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileInfo:
        """Deserialize from dictionary.

        Args:
            data: Dictionary from to_dict() or JSON.

        Returns:
            Reconstructed FileInfo instance.
        """
        return cls(
            path=data["path"],
            purpose=data.get("purpose", ""),
            created_by_task=data.get("created_by_task"),
            last_modified_by_task=data.get("last_modified_by_task"),
            imports=data.get("imports", []),
            exports=data.get("exports", []),
        )


@dataclass
class RunSummary:
    """Summary of a past execution run.

    Attributes:
        run_id: Unique identifier for the run.
        timestamp: ISO timestamp when the run completed.
        roadmap_path: Path to the ROADMAP that was executed.
        tasks_completed: Number of tasks completed successfully.
        tasks_failed: Number of tasks that failed.
        key_files_created: List of important files created.
        lessons_learned: Key takeaways from failures.
    """

    run_id: str
    timestamp: str
    roadmap_path: str = ""
    tasks_completed: int = 0
    tasks_failed: int = 0
    key_files_created: list[str] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage.

        Returns:
            Dictionary representation.
        """
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "roadmap_path": self.roadmap_path,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "key_files_created": self.key_files_created,
            "lessons_learned": self.lessons_learned,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSummary:
        """Deserialize from dictionary.

        Args:
            data: Dictionary from to_dict() or JSON.

        Returns:
            Reconstructed RunSummary instance.
        """
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            roadmap_path=data.get("roadmap_path", ""),
            tasks_completed=data.get("tasks_completed", 0),
            tasks_failed=data.get("tasks_failed", 0),
            key_files_created=data.get("key_files_created", []),
            lessons_learned=data.get("lessons_learned", []),
        )


@dataclass
class ProjectMemory:
    """Persistent memory about a project across execution runs.

    ProjectMemory tracks project-level information that persists between
    executor runs. This includes project type, detected frameworks,
    key files with their purposes, discovered conventions, and a history
    of past runs.

    The memory is stored at `.executor/project-memory.json` within the
    project root directory.

    Attributes:
        project_root: Absolute path to the project root.
        project_type: Detected project type (python, node, rust, etc.).
        detected_frameworks: List of detected frameworks (fastapi, react, etc.).
        key_files: Map of file paths to FileInfo objects.
        conventions: List of discovered project conventions.
        run_history: List of past run summaries.
    """

    project_root: Path
    project_type: str = "unknown"
    detected_frameworks: list[str] = field(default_factory=list)
    key_files: dict[str, FileInfo] = field(default_factory=dict)
    conventions: list[str] = field(default_factory=list)
    run_history: list[RunSummary] = field(default_factory=list)
    _max_history: int = 10
    _max_key_files: int = 50
    _max_context_tokens: int = 2000

    # Storage location relative to project root
    STORAGE_DIR = ".executor"
    STORAGE_FILE = "project-memory.json"

    @classmethod
    def load(cls, project_root: Path) -> ProjectMemory:
        """Load project memory from storage or create new.

        Args:
            project_root: Absolute path to project root.

        Returns:
            Loaded ProjectMemory or new empty instance.
        """
        memory_path = project_root / cls.STORAGE_DIR / cls.STORAGE_FILE
        if memory_path.exists():
            try:
                data = json.loads(memory_path.read_text(encoding="utf-8"))
                return cls._from_dict(data, project_root)
            except (json.JSONDecodeError, KeyError, TypeError):
                # Corrupted file, start fresh
                pass
        return cls(project_root=project_root)

    def save(self) -> Path:
        """Persist project memory to storage.

        Creates the .executor directory if it doesn't exist.

        Returns:
            Path to the saved file.
        """
        memory_path = self.project_root / self.STORAGE_DIR / self.STORAGE_FILE
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(
            json.dumps(self._to_dict(), indent=2),
            encoding="utf-8",
        )
        return memory_path

    def update_from_run(self, run_memory: RunMemory) -> None:
        """Update project memory after a run completes.

        Updates:
        - Adds run to history (trimmed to max)
        - Updates key files map with files touched
        - Extracts lessons from failures

        Args:
            run_memory: The completed RunMemory to incorporate.
        """
        # Create run summary
        stats = run_memory.get_stats()
        summary = RunSummary(
            run_id=run_memory.run_id,
            timestamp=datetime.now(UTC).isoformat(),
            roadmap_path=run_memory.roadmap_path,
            tasks_completed=stats["completed"],
            tasks_failed=stats["failed"],
            key_files_created=self._get_key_files_from_run(run_memory),
            lessons_learned=self._extract_lessons(run_memory),
        )
        self.run_history.append(summary)

        # Trim old history
        if len(self.run_history) > self._max_history:
            self.run_history = self.run_history[-self._max_history :]

        # Update key files
        self._update_key_files(run_memory)

        # Save changes
        self.save()

    def _get_key_files_from_run(self, run_memory: RunMemory) -> list[str]:
        """Extract key file paths from a run.

        Args:
            run_memory: The run to extract files from.

        Returns:
            List of file paths (limited to 10).
        """
        files = run_memory.get_completed_files()
        return [str(p) for p in list(files.keys())[:10]]

    def _update_key_files(self, run_memory: RunMemory) -> None:
        """Update key files map from run memory.

        Args:
            run_memory: The run to extract files from.
        """
        from ai_infra.executor.run_memory import FileAction

        for path, outcome in run_memory.get_completed_files().items():
            # Get relative path if possible
            try:
                rel_path = str(path.relative_to(self.project_root))
            except ValueError:
                rel_path = str(path)

            action = outcome.files.get(path)

            if rel_path not in self.key_files:
                # New file
                self.key_files[rel_path] = FileInfo(
                    path=rel_path,
                    purpose=outcome.summary or outcome.title,
                    created_by_task=outcome.task_id if action == FileAction.CREATED else None,
                    last_modified_by_task=outcome.task_id,
                )
            else:
                # Existing file - update
                self.key_files[rel_path].last_modified_by_task = outcome.task_id
                # Update purpose if we have a better summary
                if outcome.summary:
                    self.key_files[rel_path].purpose = outcome.summary

        # Trim to max key files (keep most recently modified)
        if len(self.key_files) > self._max_key_files:
            # This is a simple trim - could be smarter
            excess = len(self.key_files) - self._max_key_files
            keys_to_remove = list(self.key_files.keys())[:excess]
            for key in keys_to_remove:
                del self.key_files[key]

    def _extract_lessons(self, run_memory: RunMemory) -> list[str]:
        """Extract lessons from failures in a run.

        Args:
            run_memory: The run to extract lessons from.

        Returns:
            List of lesson strings (limited to 5).
        """
        lessons: list[str] = []
        for outcome in run_memory.outcomes:
            if outcome.status == "failed" and outcome.key_decisions:
                # Take first decision as the lesson
                decision = outcome.key_decisions[0]
                lessons.append(f"Task {outcome.task_id}: {decision[:100]}")
        return lessons[:5]

    def get_context(
        self,
        task_title: str = "",
        max_tokens: int | None = None,
    ) -> str:
        """Get project context for prompt injection.

        Creates a markdown-formatted summary of project information,
        key files, conventions, and recent run history.

        Args:
            task_title: Current task title (for relevance filtering).
            max_tokens: Maximum tokens for context. Uses default if None.

        Returns:
            Markdown-formatted context string, or empty string if no data.
        """
        if not self.key_files and not self.run_history and not self.conventions:
            return ""

        budget = max_tokens if max_tokens is not None else self._max_context_tokens

        lines = ["## Project Context", ""]

        # Project type and frameworks
        lines.append(f"**Type**: {self.project_type}")
        if self.detected_frameworks:
            lines.append(f"**Frameworks**: {', '.join(self.detected_frameworks)}")

        # Conventions
        if self.conventions:
            lines.extend(["", "### Conventions"])
            for conv in self.conventions[:5]:
                lines.append(f"- {conv}")

        # Key files
        if self.key_files:
            lines.extend(["", "### Key Files"])
            # Sort by most recently modified (those with last_modified_by_task)
            sorted_files = sorted(
                self.key_files.items(),
                key=lambda x: x[1].last_modified_by_task or "",
                reverse=True,
            )
            for path, info in sorted_files[:15]:
                purpose = info.purpose[:80] if info.purpose else "no description"
                lines.append(f"- `{path}`: {purpose}")
            if len(self.key_files) > 15:
                lines.append(f"- ... and {len(self.key_files) - 15} more files")

        # Recent runs
        if self.run_history:
            lines.extend(["", "### Recent Runs"])
            for run in self.run_history[-3:]:
                date = run.timestamp[:10]
                lines.append(
                    f"- {date}: {run.tasks_completed} completed, {run.tasks_failed} failed"
                )

        raw_context = "\n".join(lines)

        # Check if within budget
        current_tokens = _count_tokens_simple(raw_context)
        if current_tokens <= budget:
            return raw_context

        # Over budget - truncate
        return self._truncate_context(budget)

    def _truncate_context(self, max_tokens: int) -> str:
        """Truncate context to fit within token budget.

        Strategy: Reduce key files and conventions first.

        Args:
            max_tokens: Maximum tokens to use.

        Returns:
            Truncated context string.
        """
        lines = ["## Project Context", ""]
        lines.append(f"**Type**: {self.project_type}")

        if self.detected_frameworks:
            lines.append(f"**Frameworks**: {', '.join(self.detected_frameworks[:3])}")

        # Limited conventions
        if self.conventions:
            lines.extend(["", "### Conventions"])
            for conv in self.conventions[:3]:
                lines.append(f"- {conv[:60]}")

        # Limited key files
        if self.key_files:
            lines.extend(["", "### Key Files"])
            sorted_files = sorted(
                self.key_files.items(),
                key=lambda x: x[1].last_modified_by_task or "",
                reverse=True,
            )
            for path, info in sorted_files[:8]:
                purpose = info.purpose[:50] if info.purpose else ""
                lines.append(f"- `{path}`: {purpose}")
            if len(self.key_files) > 8:
                lines.append(f"- ... and {len(self.key_files) - 8} more")

        # Only last run
        if self.run_history:
            lines.extend(["", "### Recent Runs"])
            run = self.run_history[-1]
            lines.append(
                f"- {run.timestamp[:10]}: {run.tasks_completed} completed, "
                f"{run.tasks_failed} failed"
            )

        return "\n".join(lines)

    def add_convention(self, convention: str) -> None:
        """Add a discovered convention.

        Args:
            convention: The convention to add.
        """
        if convention not in self.conventions:
            self.conventions.append(convention)
            # Keep max 20 conventions
            if len(self.conventions) > 20:
                self.conventions = self.conventions[-20:]

    def set_project_type(self, project_type: str) -> None:
        """Set the project type.

        Args:
            project_type: Type like "python", "node", "rust", etc.
        """
        self.project_type = project_type

    def add_framework(self, framework: str) -> None:
        """Add a detected framework.

        Args:
            framework: Framework name like "fastapi", "react", etc.
        """
        if framework not in self.detected_frameworks:
            self.detected_frameworks.append(framework)

    def get_file_info(self, path: str) -> FileInfo | None:
        """Get info about a specific file.

        Args:
            path: Relative path to the file.

        Returns:
            FileInfo if found, None otherwise.
        """
        return self.key_files.get(path)

    def get_files_by_task(self, task_id: str) -> list[FileInfo]:
        """Get all files created or modified by a task.

        Args:
            task_id: The task ID to look up.

        Returns:
            List of FileInfo objects.
        """
        result: list[FileInfo] = []
        for info in self.key_files.values():
            if info.created_by_task == task_id or info.last_modified_by_task == task_id:
                result.append(info)
        return result

    def get_last_run(self) -> RunSummary | None:
        """Get the most recent run summary.

        Returns:
            RunSummary if any runs exist, None otherwise.
        """
        return self.run_history[-1] if self.run_history else None

    def get_stats(self) -> dict[str, int]:
        """Get summary statistics for this project memory.

        Returns:
            Dictionary with counts.
        """
        total_completed = sum(r.tasks_completed for r in self.run_history)
        total_failed = sum(r.tasks_failed for r in self.run_history)

        return {
            "run_count": len(self.run_history),
            "key_file_count": len(self.key_files),
            "convention_count": len(self.conventions),
            "framework_count": len(self.detected_frameworks),
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
        }

    def clear(self) -> None:
        """Clear all project memory data.

        Does not delete the file - call save() after to persist.
        """
        self.project_type = "unknown"
        self.detected_frameworks = []
        self.key_files = {}
        self.conventions = []
        self.run_history = []

    def delete(self) -> bool:
        """Delete the project memory file.

        Returns:
            True if file was deleted, False if it didn't exist.
        """
        memory_path = self.project_root / self.STORAGE_DIR / self.STORAGE_FILE
        if memory_path.exists():
            memory_path.unlink()
            return True
        return False

    def _to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage.

        Returns:
            Dictionary representation.
        """
        return {
            "version": 1,
            "project_type": self.project_type,
            "detected_frameworks": self.detected_frameworks,
            "key_files": {path: info.to_dict() for path, info in self.key_files.items()},
            "conventions": self.conventions,
            "run_history": [run.to_dict() for run in self.run_history],
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any], project_root: Path) -> ProjectMemory:
        """Deserialize from dictionary.

        Args:
            data: Dictionary from _to_dict() or JSON.
            project_root: Absolute path to project root.

        Returns:
            Reconstructed ProjectMemory instance.
        """
        key_files = {
            path: FileInfo.from_dict(info) for path, info in data.get("key_files", {}).items()
        }
        run_history = [RunSummary.from_dict(run) for run in data.get("run_history", [])]

        return cls(
            project_root=project_root,
            project_type=data.get("project_type", "unknown"),
            detected_frameworks=data.get("detected_frameworks", []),
            key_files=key_files,
            conventions=data.get("conventions", []),
            run_history=run_history,
        )
