"""TodoList manager for the Executor module.

This module provides intelligent task normalization and tracking:
- Groups related bullet points into logical tasks
- Handles various ROADMAP structures (subtasks, nested bullets, etc.)
- Tracks completion and syncs to ROADMAP immediately
- Creates its own normalized todo list from parsed ROADMAP
- Supports LLM-based normalization for format-agnostic parsing
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from ai_infra.executor.models import Task, TaskStatus
from ai_infra.executor.roadmap import ParsedTask, Roadmap, Section
from ai_infra.logging import get_logger

logger = get_logger("executor.todolist")

# Type alias for grouping strategies
GroupStrategy = Literal["none", "section", "smart"]

# Type alias for normalized todo status (from LLM)
NormalizedStatus = Literal["pending", "completed", "skipped"]

# Version for normalized todos JSON format
NORMALIZED_TODOS_VERSION = "1.0"


class TodoStatus(str, Enum):
    """Status for todo items.

    Values align with NormalizedStatus for consistency between
    graph_state.json and todos.json.
    """

    NOT_STARTED = "pending"  # Aligns with NormalizedStatus
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    FAILED = "skipped"  # Aligns with NormalizedStatus


# =============================================================================
# LLM-Based Normalization Schema (Phase 5.13)
# =============================================================================


@dataclass
class NormalizedTodo:
    """A normalized todo extracted by LLM from any ROADMAP format.

    This dataclass represents a task extracted from a ROADMAP file by an LLM,
    enabling format-agnostic parsing. The LLM can identify tasks regardless of
    whether they use checkboxes, emojis, bullets, or prose.

    Attributes:
        id: Unique identifier for this todo (1-based sequential).
        title: Short actionable title extracted by LLM.
        description: Optional longer description or context.
        status: Current status (pending, completed, skipped).
        source_line: Line number in original ROADMAP (for sync-back).
        source_text: Original text from ROADMAP (preserved verbatim).
        file_hints: File paths mentioned or implied by the task.
        dependencies: IDs of other todos this depends on.
        subtasks: Nested subtasks (can be flattened for execution).
    """

    id: int
    title: str
    description: str | None = None
    status: NormalizedStatus = "pending"

    # Original line references (for optional sync-back)
    source_line: int | None = None
    source_text: str | None = None

    # Extracted hints
    file_hints: list[str] = field(default_factory=list)
    dependencies: list[int] = field(default_factory=list)

    # Subtasks (can be nested or flattened)
    subtasks: list[NormalizedTodo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "source_line": self.source_line,
            "source_text": self.source_text,
            "file_hints": self.file_hints,
            "dependencies": self.dependencies,
            "subtasks": [st.to_dict() for st in self.subtasks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedTodo:
        """Create from dictionary (e.g., from JSON)."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description"),
            status=data.get("status", "pending"),
            source_line=data.get("source_line"),
            source_text=data.get("source_text"),
            file_hints=data.get("file_hints", []),
            dependencies=data.get("dependencies", []),
            subtasks=[cls.from_dict(st) for st in data.get("subtasks", [])],
        )

    def is_pending(self) -> bool:
        """Check if this todo is pending execution."""
        return self.status == "pending"

    def is_completed(self) -> bool:
        """Check if this todo is completed."""
        return self.status == "completed"


@dataclass
class NormalizedTodoFile:
    """Container for the .executor/todos.json file format.

    This represents the cached normalized todos extracted by LLM from a ROADMAP.
    The source_hash allows detecting when the ROADMAP has changed and needs
    re-normalization.

    Attributes:
        version: Schema version for forward compatibility.
        source_file: Path to the original ROADMAP file.
        source_hash: SHA-256 hash of ROADMAP content for cache invalidation.
        normalized_at: Timestamp when normalization was performed.
        todos: List of normalized todos extracted by LLM.
    """

    version: str
    source_file: str
    source_hash: str
    normalized_at: datetime
    todos: list[NormalizedTodo]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "version": self.version,
            "source_file": self.source_file,
            "source_hash": self.source_hash,
            "normalized_at": self.normalized_at.isoformat(),
            "todos": [todo.to_dict() for todo in self.todos],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedTodoFile:
        """Create from dictionary (e.g., from JSON)."""
        return cls(
            version=data["version"],
            source_file=data["source_file"],
            source_hash=data["source_hash"],
            normalized_at=datetime.fromisoformat(data["normalized_at"]),
            todos=[NormalizedTodo.from_dict(t) for t in data.get("todos", [])],
        )

    def save(self, path: Path) -> None:
        """Save to JSON file.

        Args:
            path: Path to save the JSON file (typically .executor/todos.json).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug(f"Saved normalized todos to {path}")

    @classmethod
    def load(cls, path: Path) -> NormalizedTodoFile:
        """Load from JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            NormalizedTodoFile instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def create(
        cls,
        source_file: Path,
        todos: list[NormalizedTodo],
    ) -> NormalizedTodoFile:
        """Create a new NormalizedTodoFile from source ROADMAP.

        Computes the source hash automatically from the file content.

        Args:
            source_file: Path to the source ROADMAP.md file.
            todos: List of normalized todos extracted by LLM.

        Returns:
            NormalizedTodoFile instance ready to save.
        """
        content = source_file.read_text(encoding="utf-8")
        source_hash = hashlib.sha256(content.encode()).hexdigest()

        return cls(
            version=NORMALIZED_TODOS_VERSION,
            source_file=str(source_file),
            source_hash=source_hash,
            normalized_at=datetime.now(UTC),
            todos=todos,
        )

    def is_stale(self, source_file: Path) -> bool:
        """Check if the cached todos are stale (ROADMAP changed).

        Args:
            source_file: Path to the source ROADMAP file.

        Returns:
            True if the ROADMAP has changed since normalization.
        """
        if not source_file.exists():
            return True

        content = source_file.read_text(encoding="utf-8")
        current_hash = hashlib.sha256(content.encode()).hexdigest()
        return current_hash != self.source_hash

    def get_pending_todos(self) -> list[NormalizedTodo]:
        """Get all pending todos (flat, including subtasks if flattened)."""
        return [todo for todo in self.todos if todo.is_pending()]

    def mark_completed(self, todo_id: int) -> bool:
        """Mark a todo as completed by ID.

        Args:
            todo_id: The ID of the todo to mark completed.

        Returns:
            True if the todo was found and marked, False otherwise.
        """
        for todo in self.todos:
            if todo.id == todo_id:
                todo.status = "completed"
                return True
            # Check subtasks
            for subtask in todo.subtasks:
                if subtask.id == todo_id:
                    subtask.status = "completed"
                    return True
        return False

    def merge_execution_status(
        self,
        cached: NormalizedTodoFile,
        *,
        similarity_threshold: float = 0.6,
    ) -> int:
        """Merge execution status from a cached NormalizedTodoFile.

        When the ROADMAP changes and we re-normalize with LLM, we want to
        preserve the execution status (completed/skipped) of todos that
        were already executed. This method matches todos by title similarity
        and transfers their status.

        Args:
            cached: Previously cached NormalizedTodoFile with execution status.
            similarity_threshold: Minimum title similarity (0-1) to consider a match.
                Default is 0.6 to handle minor LLM title variations.

        Returns:
            Number of todos whose status was preserved from cache.
        """
        preserved_count = 0

        # Build a lookup of cached todos by normalized title
        cached_by_title: dict[str, NormalizedTodo] = {}
        for todo in cached.todos:
            # Normalize title: lowercase, strip, collapse whitespace
            normalized_title = " ".join(todo.title.lower().split())
            cached_by_title[normalized_title] = todo

        for todo in self.todos:
            # Skip if already marked completed (from ROADMAP markers)
            if todo.status != "pending":
                continue

            # Try exact match first
            normalized_title = " ".join(todo.title.lower().split())
            if normalized_title in cached_by_title:
                cached_todo = cached_by_title[normalized_title]
                if cached_todo.status in ("completed", "skipped"):
                    todo.status = cached_todo.status
                    preserved_count += 1
                    logger.debug(
                        f"Preserved status '{cached_todo.status}' for todo {todo.id}: {todo.title}"
                    )
                    continue

            # Try fuzzy match using word overlap
            todo_words = set(normalized_title.split())
            best_match: tuple[float, NormalizedTodo | None] = (0.0, None)

            for cached_title, cached_todo in cached_by_title.items():
                if cached_todo.status not in ("completed", "skipped"):
                    continue

                cached_words = set(cached_title.split())
                if not todo_words or not cached_words:
                    continue

                # Jaccard similarity
                intersection = len(todo_words & cached_words)
                union = len(todo_words | cached_words)
                similarity = intersection / union if union > 0 else 0.0

                if similarity > best_match[0]:
                    best_match = (similarity, cached_todo)

            if best_match[0] >= similarity_threshold and best_match[1]:
                todo.status = best_match[1].status
                preserved_count += 1
                logger.debug(
                    f"Preserved status '{best_match[1].status}' for todo {todo.id}: "
                    f"{todo.title} (matched '{best_match[1].title}' with similarity {best_match[0]:.2f})"
                )

        if preserved_count > 0:
            logger.info(
                f"Smart resume: preserved execution status for {preserved_count} todos from cache"
            )

        return preserved_count

    def verify_completions_against_codebase(
        self,
        project_root: Path,
    ) -> tuple[int, int]:
        """Verify completed todos by checking if their expected files exist.

        This provides an additional layer of verification by checking if files
        mentioned in file_hints actually exist in the project. This helps catch
        cases where the cache says a todo is complete but the files don't exist.

        Args:
            project_root: Root directory of the project to check files against.

        Returns:
            Tuple of (verified_count, reverted_count) - how many todos were
            confirmed vs how many were reverted to pending due to missing files.
        """
        verified = 0
        reverted = 0

        for todo in self.todos:
            if todo.status != "completed":
                continue

            # If no file hints, trust the cached status
            if not todo.file_hints:
                verified += 1
                continue

            # Check if at least one file hint exists
            any_file_exists = False
            for file_hint in todo.file_hints:
                file_path = project_root / file_hint
                if file_path.exists():
                    any_file_exists = True
                    break

            if any_file_exists:
                verified += 1
            else:
                # Files don't exist - revert to pending
                logger.warning(
                    f"Todo {todo.id} marked completed but expected files not found: "
                    f"{todo.file_hints}. Reverting to pending."
                )
                todo.status = "pending"
                reverted += 1

        if reverted > 0:
            logger.info(
                f"Codebase verification: {verified} confirmed, {reverted} reverted to pending"
            )
        elif verified > 0:
            logger.debug(f"Codebase verification: all {verified} completed todos confirmed")

        return verified, reverted


@dataclass
class TodoItem:
    """A normalized todo item that may represent one or more ROADMAP tasks.

    Attributes:
        id: Unique identifier for this todo.
        title: Display title (action-oriented, 3-7 words).
        description: Detailed context, requirements, or implementation notes.
        status: Current status.
        source_task_ids: List of ROADMAP task IDs this todo represents.
        source_titles: Original titles from ROADMAP for checkbox matching.
        file_hints: Files expected to be created/modified (from task descriptions).
        files_created: Files actually created during execution.
        error: Error message if the todo failed.
        started_at: When work started.
        completed_at: When work completed.
    """

    id: int
    title: str
    description: str
    status: TodoStatus = TodoStatus.NOT_STARTED
    source_task_ids: list[str] = field(default_factory=list)
    source_titles: list[str] = field(default_factory=list)
    file_hints: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "source_task_ids": self.source_task_ids,
            "source_titles": self.source_titles,
            "file_hints": self.file_hints,
            "files_created": self.files_created,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TodoItem:
        """Create from dictionary.

        Handles migration from old status values:
        - 'not-started' -> 'pending'
        - 'failed' -> 'skipped'
        """
        raw_status = data.get("status", "pending")
        # Migrate old status values
        if raw_status == "not-started":
            raw_status = "pending"
        elif raw_status == "failed":
            raw_status = "skipped"
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=TodoStatus(raw_status),
            source_task_ids=data.get("source_task_ids", []),
            source_titles=data.get("source_titles", []),
            file_hints=data.get("file_hints", []),
            files_created=data.get("files_created", []),
            error=data.get("error"),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
        )


class TodoListManager:
    """Manages a normalized todo list derived from ROADMAP.

    Creates its own internal todo list by:
    1. Parsing ROADMAP tasks
    2. Grouping related tasks intelligently
    3. Tracking progress independently
    4. Syncing completions back to ROADMAP immediately (or JSON only)

    When created via `from_roadmap_llm()`, status updates are saved to
    `.executor/todos.json` only, leaving the original ROADMAP untouched.
    Use `sync_to_roadmap()` to explicitly sync status back if desired.

    Example:
        >>> manager = TodoListManager.from_roadmap(roadmap)
        >>> todo = manager.next_pending()
        >>> manager.mark_in_progress(todo.id)
        >>> # ... execute todo ...
        >>> manager.mark_completed(todo.id)  # Syncs to ROADMAP automatically
    """

    def __init__(
        self,
        roadmap_path: Path | None = None,
        todos: list[TodoItem] | None = None,
        *,
        use_json_only: bool = False,
    ) -> None:
        """Initialize the todo list manager.

        Args:
            roadmap_path: Path to the source ROADMAP.md file.
            todos: Pre-existing todo items.
            use_json_only: If True, status updates go to JSON only,
                not to ROADMAP. Set automatically by from_roadmap_llm().
        """
        self._roadmap_path = roadmap_path
        self._todos: list[TodoItem] = todos or []
        self._task_id_to_todo: dict[str, int] = {}  # Maps task ID -> todo ID
        self._use_json_only = use_json_only

        # Build reverse mapping
        for todo in self._todos:
            for task_id in todo.source_task_ids:
                self._task_id_to_todo[task_id] = todo.id

    @classmethod
    def from_roadmap(
        cls,
        roadmap: Roadmap,
        roadmap_path: Path | None = None,
        group_strategy: GroupStrategy = "smart",
    ) -> TodoListManager:
        """Create a TodoListManager from a parsed Roadmap.

        Args:
            roadmap: Parsed Roadmap object.
            roadmap_path: Path to the source ROADMAP file.
            group_strategy: How to group tasks:
                - "none": Each task becomes a separate todo
                - "section": Group all tasks in a section into one todo
                - "smart": Intelligently group related tasks (default)

        Returns:
            TodoListManager with normalized todos.
        """
        if group_strategy == "none":
            todos = cls._create_ungrouped_todos(roadmap)
        elif group_strategy == "section":
            todos = cls._create_section_grouped_todos(roadmap)
        else:  # smart
            todos = cls._create_smart_grouped_todos(roadmap)

        manager = cls(roadmap_path=roadmap_path, todos=todos)
        logger.debug(
            f"Created TodoListManager with {len(todos)} todos using '{group_strategy}' strategy"
        )
        return manager

    @classmethod
    async def from_roadmap_llm(
        cls,
        roadmap_path: Path,
        agent: Any,
        *,
        force_renormalize: bool = False,
        timeout: float = 120.0,
        verify_codebase: bool = True,
    ) -> TodoListManager:
        """Create TodoListManager using LLM to normalize any ROADMAP format.

        This uses one LLM call to extract todos from any format (emojis,
        prose, custom bullets, etc.) and stores the normalized result
        in .executor/todos.json. The cached result is reused if the ROADMAP
        content hasn't changed.

        Smart Resume: When ROADMAP changes and re-normalization occurs, execution
        status from the cached todos is preserved by matching todo titles. This
        prevents re-executing already-completed work.

        Args:
            roadmap_path: Path to ROADMAP.md (any format).
            agent: Agent instance (used for its model_name).
            force_renormalize: If True, re-normalize even if cached.
            timeout: Timeout for LLM call in seconds.
            verify_codebase: If True, verify completed todos by checking if
                their expected files exist in the project.

        Returns:
            TodoListManager with normalized todos.

        Raises:
            ValueError: If the LLM response cannot be parsed.
            asyncio.TimeoutError: If the LLM call times out.

        Example:
            >>> agent = Agent(model="gpt-4")
            >>> manager = await TodoListManager.from_roadmap_llm(
            ...     Path("ROADMAP.md"),
            ...     agent,
            ... )
            >>> print(manager.total_count)
            5
        """
        import asyncio

        from ai_infra.executor.prompts import (
            NORMALIZE_ROADMAP_PROMPT,
            NORMALIZE_ROADMAP_SYSTEM_PROMPT,
        )
        from ai_infra.llm import LLM

        executor_dir = roadmap_path.parent / ".executor"
        executor_dir.mkdir(parents=True, exist_ok=True)
        todos_json_path = executor_dir / "todos.json"

        # Read ROADMAP content
        roadmap_content = roadmap_path.read_text(encoding="utf-8")

        # Load cached todos if they exist (for smart resume)
        cached_todos: NormalizedTodoFile | None = None
        if todos_json_path.exists():
            try:
                cached_todos = NormalizedTodoFile.load(todos_json_path)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load cached todos: {e}")
                cached_todos = None

        # Check if we can use cached normalization (hash matches)
        if cached_todos and not force_renormalize:
            if not cached_todos.is_stale(roadmap_path):
                logger.info(f"Using cached normalized todos ({len(cached_todos.todos)} todos)")
                # Verify completed todos against codebase even when using cache
                if verify_codebase:
                    project_root = roadmap_path.parent
                    verified, reverted = cached_todos.verify_completions_against_codebase(
                        project_root
                    )
                    if reverted > 0:
                        # Save updated cache with reverted todos
                        cached_todos.save(todos_json_path)
                        pending_count = len([t for t in cached_todos.todos if t.is_pending()])
                        completed_count = len([t for t in cached_todos.todos if t.is_completed()])
                        logger.info(
                            f"After verification: {completed_count} completed, {pending_count} pending"
                        )
                return cls._from_normalized_todo_file(cached_todos, roadmap_path)
            logger.info("ROADMAP changed, re-normalizing (will preserve execution status)")

        # Normalize with LLM (direct call, not full agent)
        logger.info("Normalizing ROADMAP with LLM...")
        prompt = NORMALIZE_ROADMAP_PROMPT.format(roadmap_content=roadmap_content)

        # Get model from agent - check _default_model_name first (Agent stores it there)
        model_name = (
            getattr(agent, "_default_model_name", None)
            or getattr(agent, "model_name", None)
            or "gpt-5-mini"
        )
        logger.info(f"Using model: {model_name}")
        llm = LLM()

        # Call LLM with timeout (simple chat call, no tools)
        async def _call_llm() -> str:
            response = await llm.achat(
                user_msg=prompt,
                model_name=model_name,
                system=NORMALIZE_ROADMAP_SYSTEM_PROMPT,
            )
            # Extract content from response
            if hasattr(response, "content"):
                return response.content
            return str(response)

        llm_output = await asyncio.wait_for(
            _call_llm(),
            timeout=timeout,
        )

        # Parse LLM response
        normalized_todos = cls._parse_llm_normalization_response(llm_output)
        logger.info(f"LLM extracted {len(normalized_todos)} todos from ROADMAP")

        # Create NormalizedTodoFile
        todo_file = NormalizedTodoFile.create(roadmap_path, normalized_todos)

        # Smart resume: merge execution status from cached todos
        if cached_todos:
            preserved = todo_file.merge_execution_status(cached_todos)
            if preserved > 0:
                pending_count = len([t for t in todo_file.todos if t.is_pending()])
                completed_count = len([t for t in todo_file.todos if t.is_completed()])
                logger.info(f"After merge: {completed_count} completed, {pending_count} pending")

        # Verify completed todos against codebase (check if files exist)
        if verify_codebase:
            project_root = roadmap_path.parent
            verified, reverted = todo_file.verify_completions_against_codebase(project_root)
            if reverted > 0:
                # Re-save if any todos were reverted
                pending_count = len([t for t in todo_file.todos if t.is_pending()])
                completed_count = len([t for t in todo_file.todos if t.is_completed()])
                logger.info(
                    f"After verification: {completed_count} completed, {pending_count} pending"
                )

        # Save the (potentially merged and verified) result
        todo_file.save(todos_json_path)

        return cls._from_normalized_todo_file(todo_file, roadmap_path)

    @classmethod
    def _from_normalized_todo_file(
        cls,
        todo_file: NormalizedTodoFile,
        roadmap_path: Path,
    ) -> TodoListManager:
        """Create TodoListManager from a NormalizedTodoFile.

        Converts NormalizedTodo objects to TodoItem objects for use with
        the existing TodoListManager interface.

        Args:
            todo_file: The normalized todo file.
            roadmap_path: Path to the source ROADMAP file.

        Returns:
            TodoListManager instance.
        """
        todos: list[TodoItem] = []
        for ntodo in todo_file.todos:
            # Map NormalizedTodo status to TodoStatus
            if ntodo.status == "completed":
                status = TodoStatus.COMPLETED
            elif ntodo.status == "skipped":
                status = TodoStatus.FAILED  # Map skipped to failed for tracking
            else:
                status = TodoStatus.NOT_STARTED

            # Collect all titles including subtasks
            source_titles = [ntodo.title]
            for subtask in ntodo.subtasks:
                source_titles.append(subtask.title)

            todo = TodoItem(
                id=ntodo.id,
                title=ntodo.title,
                description=ntodo.description or "",
                status=status,
                source_task_ids=[str(ntodo.id)],
                source_titles=source_titles,
                file_hints=ntodo.file_hints,
            )
            todos.append(todo)

        # Use JSON-only mode for LLM-normalized todos
        manager = cls(roadmap_path=roadmap_path, todos=todos, use_json_only=True)
        logger.debug(f"Created TodoListManager from NormalizedTodoFile with {len(todos)} todos")
        return manager

    @staticmethod
    def _parse_llm_normalization_response(response: str) -> list[NormalizedTodo]:
        """Parse LLM response into NormalizedTodo objects.

        Handles various JSON formats that LLMs might return, including
        markdown code blocks.

        Args:
            response: Raw LLM response text.

        Returns:
            List of NormalizedTodo objects.

        Raises:
            ValueError: If the response cannot be parsed.
        """
        # Clean up response - remove markdown code blocks if present
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Try to find JSON object in the response
        try:
            # First try parsing the whole response
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError(f"Could not find JSON in LLM response: {text[:200]}...")
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse LLM JSON response: {e}\nResponse: {text[:500]}"
                ) from e

        # Extract todos from the parsed data
        if "todos" not in data:
            raise ValueError(f"LLM response missing 'todos' key: {list(data.keys())}")

        todos: list[NormalizedTodo] = []
        for item in data["todos"]:
            todo = NormalizedTodo.from_dict(item)
            todos.append(todo)

        return todos

    # =========================================================================
    # Grouping Strategies
    # =========================================================================

    @staticmethod
    def _collect_titles_with_subtasks(tasks: list) -> list[str]:
        """Collect all titles including subtask titles for checkbox matching.

        When a task has subtasks (nested checkboxes), we need to include
        those titles so they get checked off when the todo completes.

        Args:
            tasks: List of ParsedTask objects.

        Returns:
            List of titles (parent + subtask titles).
        """
        titles: list[str] = []
        for task in tasks:
            titles.append(task.title)
            # Include subtask titles if present
            if hasattr(task, "subtasks") and task.subtasks:
                for subtask in task.subtasks:
                    titles.append(subtask.title)
        return titles

    @staticmethod
    def _create_ungrouped_todos(roadmap: Roadmap) -> list[TodoItem]:
        """Create one todo per ROADMAP task (no grouping)."""
        todos: list[TodoItem] = []
        todo_id = 1

        for task in roadmap.all_tasks():
            if task.status == TaskStatus.COMPLETED:
                status = TodoStatus.COMPLETED
            else:
                status = TodoStatus.NOT_STARTED

            # Include subtask titles for checkbox matching
            all_titles = TodoListManager._collect_titles_with_subtasks([task])

            todos.append(
                TodoItem(
                    id=todo_id,
                    title=TodoListManager._shorten_title(task.title),
                    description=task.context or task.description,
                    status=status,
                    source_task_ids=[task.id],
                    source_titles=all_titles,
                    file_hints=task.file_hints,
                )
            )
            todo_id += 1

        return todos

    @staticmethod
    def _create_section_grouped_todos(roadmap: Roadmap) -> list[TodoItem]:
        """Create one todo per ROADMAP section."""
        todos: list[TodoItem] = []
        todo_id = 1

        for phase in roadmap.phases:
            for section in phase.sections:
                if section.task_count == 0:
                    continue

                # Collect all tasks in section
                task_ids = [t.id for t in section.tasks]
                [t.title for t in section.tasks]
                file_hints: list[str] = []
                for t in section.tasks:
                    for hint in t.file_hints:
                        if hint not in file_hints:
                            file_hints.append(hint)

                # Determine status based on all tasks
                all_completed = all(t.status == TaskStatus.COMPLETED for t in section.tasks)
                any_pending = any(t.status == TaskStatus.PENDING for t in section.tasks)

                if all_completed:
                    status = TodoStatus.COMPLETED
                elif any_pending:
                    status = TodoStatus.NOT_STARTED
                else:
                    status = TodoStatus.IN_PROGRESS

                # Build description from all task contexts
                description_parts = [section.description] if section.description else []
                for task in section.tasks:
                    description_parts.append(f"- {task.title}")
                    if task.description:
                        description_parts.append(f"  {task.description}")

                # Include subtask titles for checkbox matching
                all_titles = TodoListManager._collect_titles_with_subtasks(list(section.tasks))

                todos.append(
                    TodoItem(
                        id=todo_id,
                        title=TodoListManager._shorten_title(section.title),
                        description="\n".join(description_parts),
                        status=status,
                        source_task_ids=task_ids,
                        source_titles=all_titles,
                        file_hints=file_hints,
                    )
                )
                todo_id += 1

        return todos

    @staticmethod
    def _create_smart_grouped_todos(roadmap: Roadmap) -> list[TodoItem]:
        """Intelligently group related tasks into todos.

        Grouping heuristics:
        1. Tasks mentioning the same files are grouped together
        2. Sequential tasks with similar prefixes are grouped
        3. Tasks that are clearly subtasks (indented or numbered .1, .2) are grouped
        4. Single tasks with no relations stay as individual todos
        """
        todos: list[TodoItem] = []
        todo_id = 1

        for phase in roadmap.phases:
            for section in phase.sections:
                if section.task_count == 0:
                    continue

                # Group tasks within this section
                grouped = TodoListManager._group_section_tasks(section)

                for group in grouped:
                    # Build todo from group
                    task_ids = [t.id for t in group]
                    [t.title for t in group]

                    # Collect file hints
                    file_hints: list[str] = []
                    for t in group:
                        for hint in t.file_hints:
                            if hint not in file_hints:
                                file_hints.append(hint)

                    # Determine status
                    all_completed = all(t.status == TaskStatus.COMPLETED for t in group)
                    if all_completed:
                        status = TodoStatus.COMPLETED
                    else:
                        status = TodoStatus.NOT_STARTED

                    # Create title from group
                    if len(group) == 1:
                        title = TodoListManager._shorten_title(group[0].title)
                        description = group[0].context or group[0].description
                    else:
                        # Find common theme or use section title
                        title = TodoListManager._derive_group_title(group, section)
                        description = TodoListManager._build_group_description(group)

                    # Include subtask titles for checkbox matching
                    all_titles = TodoListManager._collect_titles_with_subtasks(group)

                    todos.append(
                        TodoItem(
                            id=todo_id,
                            title=title,
                            description=description,
                            status=status,
                            source_task_ids=task_ids,
                            source_titles=all_titles,
                            file_hints=file_hints,
                        )
                    )
                    todo_id += 1

        return todos

    @staticmethod
    def _group_section_tasks(section: Section) -> list[list[ParsedTask]]:
        """Group tasks within a section based on relationships."""
        tasks = list(section.tasks)
        if not tasks:
            return []

        # Build file-to-task mapping
        file_to_tasks: dict[str, list[ParsedTask]] = {}
        for task in tasks:
            for hint in task.file_hints:
                # Normalize file path
                normalized = hint.strip().lower()
                if normalized not in file_to_tasks:
                    file_to_tasks[normalized] = []
                file_to_tasks[normalized].append(task)

        # Find groups by shared files
        used_tasks: set[str] = set()
        groups: list[list[ParsedTask]] = []

        # First pass: group by shared files
        for file_path, file_tasks in file_to_tasks.items():
            if len(file_tasks) > 1:
                group: list[ParsedTask] = []
                for t in file_tasks:
                    if t.id not in used_tasks:
                        group.append(t)
                        used_tasks.add(t.id)
                if group:
                    groups.append(group)

        # Second pass: check for semantic similarity in remaining tasks
        remaining = [t for t in tasks if t.id not in used_tasks]
        if remaining:
            # Check for Create/Implement patterns
            re.compile(
                r"^(Create|Implement|Add|Set up|Build|Define|Write)\s+",
                re.IGNORECASE,
            )

            # Group tasks that seem to describe the same thing
            similar_groups = TodoListManager._group_by_similarity(remaining)
            for group in similar_groups:
                groups.append(group)
                for t in group:
                    used_tasks.add(t.id)

        # Any remaining tasks become individual todos
        for task in tasks:
            if task.id not in used_tasks:
                groups.append([task])

        # Sort groups by first task's original order
        task_order = {t.id: i for i, t in enumerate(tasks)}
        groups.sort(key=lambda g: min(task_order.get(t.id, 999) for t in g))

        return groups

    @staticmethod
    def _group_by_similarity(tasks: list[ParsedTask]) -> list[list[ParsedTask]]:
        """Group tasks by semantic similarity in their titles.

        Uses two similarity measures:
        1. Entity overlap: Tasks mentioning same backtick-wrapped entities
        2. Word overlap: Tasks with >60% word overlap in titles
        """
        if len(tasks) <= 1:
            return [tasks] if tasks else []

        groups: list[list[ParsedTask]] = []
        used: set[str] = set()

        # Extract key entities from titles (files, classes, etc.)
        entity_pattern = re.compile(r"`([^`]+)`")

        for i, task in enumerate(tasks):
            if task.id in used:
                continue

            group = [task]
            used.add(task.id)

            # Find entities in this task
            task_entities = set(entity_pattern.findall(task.title.lower()))
            task_words = TodoListManager._extract_significant_words(task.title)

            # Look for other tasks with similarity
            for j, other in enumerate(tasks):
                if i == j or other.id in used:
                    continue

                other_entities = set(entity_pattern.findall(other.title.lower()))
                other_words = TodoListManager._extract_significant_words(other.title)

                # Check for entity overlap
                if task_entities & other_entities:
                    group.append(other)
                    used.add(other.id)
                    continue

                # Check for word overlap (>60%)
                if task_words and other_words:
                    overlap = len(task_words & other_words)
                    min_len = min(len(task_words), len(other_words))
                    if min_len > 0 and (overlap / min_len) >= 0.6:
                        group.append(other)
                        used.add(other.id)

            groups.append(group)

        return groups

    @staticmethod
    def _extract_significant_words(title: str) -> set[str]:
        """Extract significant words from a title for similarity comparison.

        Filters out common stop words and short words.
        """
        # Remove backtick-wrapped content (handled separately as entities)
        clean = re.sub(r"`[^`]+`", "", title.lower())

        # Remove punctuation and split
        words = re.findall(r"[a-z]+", clean)

        # Filter out stop words and short words
        stop_words = {
            "the",
            "a",
            "an",
            "to",
            "in",
            "on",
            "at",
            "of",
            "for",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "with",
            "from",
            "this",
            "that",
            "it",
            "as",
            "by",
            "not",
            "but",
            "if",
            "then",
        }

        return {w for w in words if len(w) > 2 and w not in stop_words}

    @staticmethod
    def _shorten_title(title: str, max_words: int = 7) -> str:
        """Shorten a title to be action-oriented and concise."""
        # Remove backticks and path prefixes
        clean = re.sub(r"`([^`]+)`", r"\1", title)
        clean = re.sub(r"\bsrc/\b", "", clean)
        clean = re.sub(r"\btests/\b", "", clean)

        # Remove common filler words
        clean = re.sub(
            r"\b(the|a|an|this|that|with|for|and|to|in|on|at|of)\b",
            "",
            clean,
            flags=re.IGNORECASE,
        )

        # Clean up whitespace
        words = clean.split()
        if len(words) > max_words:
            words = words[:max_words]

        result = " ".join(words).strip()
        return result if result else title[:50]

    @staticmethod
    def _derive_group_title(tasks: list[ParsedTask], section: Section) -> str:
        """Derive a title for a group of tasks."""
        # If all tasks mention same file, use that
        common_files: set[str] = set()
        for task in tasks:
            for hint in task.file_hints:
                if not common_files:
                    common_files.add(hint)
                else:
                    common_files &= {hint}

        if common_files:
            file_name = next(iter(common_files)).split("/")[-1]
            return f"Implement {file_name}"

        # Use section title if short enough
        if section.title and len(section.title.split()) <= 7:
            return section.title

        # Find common prefix in task titles
        titles = [t.title for t in tasks]
        common_prefix = TodoListManager._find_common_prefix(titles)
        if common_prefix and len(common_prefix) > 10:
            return TodoListManager._shorten_title(common_prefix)

        # Fallback: use first task's action
        return TodoListManager._shorten_title(tasks[0].title)

    @staticmethod
    def _find_common_prefix(strings: list[str]) -> str:
        """Find common prefix among strings."""
        if not strings:
            return ""
        if len(strings) == 1:
            return strings[0]

        prefix = []
        for chars in zip(*strings):
            if len(set(chars)) == 1:
                prefix.append(chars[0])
            else:
                break

        return "".join(prefix).strip()

    @staticmethod
    def _build_group_description(tasks: list[ParsedTask]) -> str:
        """Build description from multiple tasks."""
        parts: list[str] = []

        for task in tasks:
            parts.append(f"- {task.title}")
            if task.description:
                # Indent description
                for line in task.description.split("\n"):
                    parts.append(f"  {line}")

        return "\n".join(parts)

    # =========================================================================
    # Todo Operations
    # =========================================================================

    def next_pending(self) -> TodoItem | None:
        """Get the next pending todo item."""
        for todo in self._todos:
            if todo.status == TodoStatus.NOT_STARTED:
                return todo
        return None

    def get_todo(self, todo_id: int) -> TodoItem | None:
        """Get a todo by ID."""
        for todo in self._todos:
            if todo.id == todo_id:
                return todo
        return None

    def get_todo_for_task(self, task_id: str) -> TodoItem | None:
        """Get the todo that contains a specific ROADMAP task."""
        todo_id = self._task_id_to_todo.get(task_id)
        if todo_id is not None:
            return self.get_todo(todo_id)
        return None

    def get_source_tasks(self, todo: TodoItem, roadmap: Roadmap) -> list[ParsedTask]:
        """Get the source ParsedTask objects for a todo.

        Args:
            todo: The todo item to get source tasks for.
            roadmap: The parsed Roadmap to look up tasks in.

        Returns:
            List of ParsedTask objects that this todo represents.
        """
        tasks: list[ParsedTask] = []
        for task_id in todo.source_task_ids:
            task = roadmap.get_task(task_id)
            if task is not None:
                tasks.append(task)
        return tasks

    def create_synthetic_task(self, todo: TodoItem) -> Task:
        """Create a synthetic Task from a TodoItem for LLM-normalized ROADMAPs.

        When using LLM normalization with non-checkbox ROADMAPs (emojis, prose),
        there are no source checkbox tasks. This method creates a Task object
        directly from the TodoItem so it can be executed.

        Args:
            todo: The todo item to convert to a Task.

        Returns:
            A Task object representing this todo.
        """
        return Task(
            id=f"llm-{todo.id}",
            title=todo.title,
            description=todo.description,
            status=TaskStatus.PENDING,
            file_hints=todo.file_hints.copy(),
            dependencies=[],  # Dependencies handled at todo level
            phase="",
            section="",
            metadata={"source": "llm-normalized", "todo_id": todo.id},
        )

    def pending(self) -> list[TodoItem]:
        """Get all pending (not started) todos.

        Returns:
            List of todos with NOT_STARTED status.
        """
        return [t for t in self._todos if t.status == TodoStatus.NOT_STARTED]

    def mark_in_progress(self, todo_id: int) -> None:
        """Mark a todo as in-progress."""
        todo = self.get_todo(todo_id)
        if todo:
            todo.status = TodoStatus.IN_PROGRESS
            todo.started_at = datetime.now(UTC)
            logger.info(f"Todo {todo_id} in-progress: {todo.title}")

            # Save to JSON if in JSON-only mode
            if self._use_json_only:
                self._save_to_json()

    def mark_completed(
        self,
        todo_id: int,
        *,
        files_created: list[str] | None = None,
        sync_roadmap: bool | None = None,
    ) -> int:
        """Mark a todo as completed.

        When the manager was created via `from_roadmap_llm()`, status is saved
        to `.executor/todos.json` only. Otherwise, it syncs to ROADMAP.md.

        Args:
            todo_id: Todo ID to mark completed.
            files_created: List of files created during this todo's execution.
            sync_roadmap: Whether to sync to ROADMAP file. If None, uses the
                default based on how the manager was created (JSON-only for
                LLM mode, ROADMAP sync otherwise).

        Returns:
            Number of ROADMAP checkboxes updated (0 if JSON-only mode).
        """
        todo = self.get_todo(todo_id)
        if not todo:
            return 0

        todo.status = TodoStatus.COMPLETED
        todo.completed_at = datetime.now(UTC)
        if files_created:
            todo.files_created = files_created
        logger.info(f"Todo {todo_id} completed: {todo.title}")

        # Save to JSON if in JSON-only mode
        if self._use_json_only:
            self._save_to_json()

        # Determine if we should sync to ROADMAP
        should_sync = sync_roadmap
        if should_sync is None:
            # Default: sync to ROADMAP unless in JSON-only mode
            should_sync = not self._use_json_only

        # Sync to ROADMAP if requested
        if should_sync and self._roadmap_path:
            return self._sync_todo_to_roadmap(todo)

        return 0

    def mark_failed(self, todo_id: int, error: str | None = None) -> None:
        """Mark a todo as failed.

        Args:
            todo_id: Todo ID to mark as failed.
            error: Error message describing the failure.
        """
        todo = self.get_todo(todo_id)
        if todo:
            todo.status = TodoStatus.FAILED
            todo.error = error
            logger.warning(f"Todo {todo_id} failed: {todo.title} - {error}")

            # Save to JSON if in JSON-only mode
            if self._use_json_only:
                self._save_to_json()

    def mark_task_completed(
        self,
        task_id: str,
        *,
        sync_roadmap: bool = True,
    ) -> int:
        """Mark the todo containing a specific task as completed.

        Args:
            task_id: ROADMAP task ID that was completed.
            sync_roadmap: Whether to sync to ROADMAP file.

        Returns:
            Number of ROADMAP checkboxes updated.
        """
        todo = self.get_todo_for_task(task_id)
        if todo:
            return self.mark_completed(todo.id, sync_roadmap=sync_roadmap)
        return 0

    # =========================================================================
    # JSON Persistence (Phase 5.13.4)
    # =========================================================================

    def save_to_json(self, create_if_missing: bool = True) -> Path | None:
        """Save current todo status to .executor/todos.json.

        Phase 1.3.1: Extended to support graph mode state persistence.

        Args:
            create_if_missing: If True, create the file if it doesn't exist.
                If False, only update an existing file.

        Returns:
            Path to the saved file, or None if save was skipped.
        """
        if not self._roadmap_path:
            return None

        executor_dir = self._roadmap_path.parent / ".executor"
        todos_json_path = executor_dir / "todos.json"

        if not todos_json_path.exists():
            if not create_if_missing:
                logger.debug(f"Skipping save: {todos_json_path} does not exist")
                return None
            # Create new file
            executor_dir.mkdir(parents=True, exist_ok=True)
            todo_file = self._create_normalized_file()
            todo_file.save(todos_json_path)
            logger.info(f"Created {todos_json_path} with {len(self._todos)} todos")
            return todos_json_path

        try:
            # Load existing file to preserve metadata
            todo_file = NormalizedTodoFile.load(todos_json_path)

            # Update todos with current status
            for todo in self._todos:
                for ntodo in todo_file.todos:
                    if ntodo.id == todo.id:
                        # Map TodoStatus to NormalizedStatus
                        if todo.status == TodoStatus.COMPLETED:
                            ntodo.status = "completed"
                        elif todo.status == TodoStatus.FAILED:
                            ntodo.status = "skipped"
                        else:
                            ntodo.status = "pending"
                        break

            # Save updated file
            todo_file.save(todos_json_path)
            logger.debug(f"Saved todo status to {todos_json_path}")
            return todos_json_path

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to save to JSON: {e}")
            return None

    def _create_normalized_file(self) -> NormalizedTodoFile:
        """Create a NormalizedTodoFile from current todos.

        Returns:
            NormalizedTodoFile with current todo data.
        """
        from datetime import datetime

        roadmap_content = ""
        if self._roadmap_path and self._roadmap_path.exists():
            roadmap_content = self._roadmap_path.read_text(encoding="utf-8")

        normalized_todos = []
        for todo in self._todos:
            ntodo = NormalizedTodo(
                id=todo.id,
                title=todo.title,
                description=todo.description or "",
                source_line=0,  # Line number not available in TodoItem
                source_text=todo.title,  # Use title as source text
                status=(
                    "completed"
                    if todo.status == TodoStatus.COMPLETED
                    else "skipped"
                    if todo.status == TodoStatus.FAILED
                    else "pending"
                ),
                file_hints=list(todo.file_hints) if todo.file_hints else [],
                dependencies=[],
            )
            normalized_todos.append(ntodo)

        return NormalizedTodoFile(
            version="1.0",
            source_file=str(self._roadmap_path) if self._roadmap_path else "",
            source_hash=hashlib.sha256(roadmap_content.encode()).hexdigest()[:16],
            normalized_at=datetime.now(UTC),
            todos=normalized_todos,
        )

    def _save_to_json(self) -> None:
        """Save current todo status to .executor/todos.json.

        This is used in JSON-only mode (when created via from_roadmap_llm)
        to persist status changes without modifying the original ROADMAP.
        """
        if not self._roadmap_path:
            return

        executor_dir = self._roadmap_path.parent / ".executor"
        todos_json_path = executor_dir / "todos.json"

        if not todos_json_path.exists():
            logger.warning(f"Cannot save to JSON: {todos_json_path} does not exist")
            return

        try:
            # Load existing file to preserve metadata
            todo_file = NormalizedTodoFile.load(todos_json_path)

            # Update todos with current status
            for todo in self._todos:
                for ntodo in todo_file.todos:
                    if ntodo.id == todo.id:
                        # Map TodoStatus to NormalizedStatus
                        if todo.status == TodoStatus.COMPLETED:
                            ntodo.status = "completed"
                        elif todo.status == TodoStatus.FAILED:
                            ntodo.status = "skipped"
                        else:
                            ntodo.status = "pending"
                        break

            # Save updated file
            todo_file.save(todos_json_path)
            logger.debug(f"Saved todo status to {todos_json_path}")

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to save to JSON: {e}")

    @property
    def uses_json_only(self) -> bool:
        """Whether this manager uses JSON-only mode (no ROADMAP sync)."""
        return self._use_json_only

    def sync_all_to_roadmap(self) -> int:
        """Sync all completed todos back to ROADMAP checkboxes.

        This is used to explicitly sync status from JSON back to the
        original ROADMAP file. Only call this when you want to update
        the ROADMAP (e.g., after execution is complete).

        Returns:
            Number of checkboxes updated.

        Example:
            >>> # After execution, optionally sync back
            >>> manager.sync_all_to_roadmap()
            3
        """
        if not self._roadmap_path or not self._roadmap_path.exists():
            logger.warning("Cannot sync: no ROADMAP path set")
            return 0

        total_updated = 0
        for todo in self._todos:
            if todo.status == TodoStatus.COMPLETED:
                updated = self._sync_todo_to_roadmap(todo)
                total_updated += updated

        if total_updated > 0:
            logger.info(f"Synced {total_updated} checkbox(es) to ROADMAP")
        else:
            logger.debug("No checkboxes to sync")

        return total_updated

    @classmethod
    def sync_json_to_roadmap(cls, roadmap_path: Path) -> int:
        """Sync completed todos from .executor/todos.json to ROADMAP.

        This is a standalone method that reads the cached JSON file and
        syncs completed todos back to the ROADMAP. Use this after execution
        when you want to update the original ROADMAP with completion status.

        Args:
            roadmap_path: Path to the ROADMAP.md file.

        Returns:
            Number of checkboxes updated.

        Raises:
            FileNotFoundError: If todos.json does not exist.

        Example:
            >>> # After execution, sync status back
            >>> updated = TodoListManager.sync_json_to_roadmap(Path("ROADMAP.md"))
            >>> print(f"Updated {updated} checkboxes")
        """
        todos_json_path = roadmap_path.parent / ".executor" / "todos.json"

        if not todos_json_path.exists():
            raise FileNotFoundError(
                f"No todos.json found at {todos_json_path}. "
                "Run executor first to create normalized todos."
            )

        # Load the normalized todos
        todo_file = NormalizedTodoFile.load(todos_json_path)

        if not roadmap_path.exists():
            raise FileNotFoundError(f"ROADMAP file not found: {roadmap_path}")

        content = roadmap_path.read_text(encoding="utf-8")
        original_content = content
        updated_count = 0

        # Sync each completed todo
        for todo in todo_file.todos:
            if todo.status != "completed":
                continue

            # Use source_text to find and update the checkbox
            if todo.source_text:
                # Try to find the original line and update it
                escaped = re.escape(todo.source_text)
                # Match unchecked checkbox with this text
                pattern = rf"(\[ \])([^\n]*{escaped})"
                if re.search(pattern, content, re.IGNORECASE):
                    content = re.sub(pattern, r"[x]\2", content, count=1)
                    updated_count += 1
                    continue

            # Fallback: use title to match
            escaped_title = re.escape(todo.title)

            # Try various checkbox formats
            patterns = [
                # Standard checkbox with title
                rf"^(\s*[-*+]\s*)\[ \](\s+{escaped_title})",
                rf"^(\s*[-*+]\s*)\[ \](\s+\*\*{escaped_title}\*\*)",
                rf"^(\s*[-*+]\s*)\[ \](\s+`{escaped_title}`)",
                # Numbered list
                rf"^(\s*\d+\.\s*)\[ \](\s+{escaped_title})",
            ]

            for pattern in patterns:
                match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
                if match:
                    content = re.sub(
                        pattern,
                        r"\1[x]\2",
                        content,
                        count=1,
                        flags=re.MULTILINE | re.IGNORECASE,
                    )
                    updated_count += 1
                    break

        # Write back if changed
        if content != original_content:
            roadmap_path.write_text(content, encoding="utf-8")
            logger.info(f"Synced {updated_count} checkbox(es) to {roadmap_path}")

        return updated_count

    # =========================================================================
    # ROADMAP Synchronization
    # =========================================================================

    def _sync_todo_to_roadmap(self, todo: TodoItem) -> int:
        """Sync a completed todo's tasks to ROADMAP checkboxes.

        Args:
            todo: The completed todo to sync.

        Returns:
            Number of checkboxes updated.
        """
        if not self._roadmap_path or not self._roadmap_path.exists():
            return 0

        content = self._roadmap_path.read_text(encoding="utf-8")
        original_content = content
        updated_count = 0

        # Update checkbox for each source title
        for title in todo.source_titles:
            # Escape regex special chars in title
            escaped_title = re.escape(title)

            # Match various title formats including:
            # - Bold: **title**
            # - Plain: title
            # - Backticks: `title` with ... description
            # - Indented subtasks: any level of indentation
            patterns = [
                # Bold title
                rf"^(\s*[-*+]\s*)\[ \](\s+\*\*{escaped_title}\*\*)",
                # Plain title (full line)
                rf"^(\s*[-*+]\s*)\[ \](\s+{escaped_title})\s*$",
                # Backtick-wrapped title (common for file paths)
                rf"^(\s*[-*+]\s*)\[ \](\s+`{escaped_title}`)",
                # Plain title (partial match for subtasks with descriptions)
                rf"^(\s*[-*+]\s*)\[ \](\s+{escaped_title})",
            ]

            for pattern in patterns:
                match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
                if match:
                    old = match.group(0)
                    new = old.replace("[ ]", "[x]", 1)
                    content = content.replace(old, new, 1)
                    updated_count += 1
                    break

        # Write back if changed
        if content != original_content:
            self._roadmap_path.write_text(content, encoding="utf-8")
            logger.info(f"Updated {updated_count} checkboxes in {self._roadmap_path.name}")

        return updated_count

    def sync_all_completed(self) -> int:
        """Sync all completed todos to ROADMAP.

        Returns:
            Total number of checkboxes updated.
        """
        total = 0
        for todo in self._todos:
            if todo.status == TodoStatus.COMPLETED:
                total += self._sync_todo_to_roadmap(todo)
        return total

    # =========================================================================
    # Query Methods
    # =========================================================================

    @property
    def todos(self) -> list[TodoItem]:
        """Get all todo items."""
        return self._todos.copy()

    @property
    def total_count(self) -> int:
        """Get total number of todos."""
        return len(self._todos)

    @property
    def pending_count(self) -> int:
        """Get number of pending todos."""
        return sum(1 for t in self._todos if t.status == TodoStatus.NOT_STARTED)

    @property
    def completed_count(self) -> int:
        """Get number of completed todos."""
        return sum(1 for t in self._todos if t.status == TodoStatus.COMPLETED)

    @property
    def in_progress_count(self) -> int:
        """Get number of in-progress todos."""
        return sum(1 for t in self._todos if t.status == TodoStatus.IN_PROGRESS)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of todo list state."""
        return {
            "total": self.total_count,
            "pending": self.pending_count,
            "in_progress": self.in_progress_count,
            "completed": self.completed_count,
            "failed": sum(1 for t in self._todos if t.status == TodoStatus.FAILED),
        }

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"TodoListManager(todos={self.total_count}, "
            f"completed={self.completed_count}, pending={self.pending_count})"
        )
