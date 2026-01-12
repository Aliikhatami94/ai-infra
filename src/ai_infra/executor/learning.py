"""Learning and adaptation for the Executor module (Phase 5.3).

Provides persistent learning from task executions:
- Store and retrieve failure patterns
- Track successful prompts and contexts
- Build task-type-specific templates
- Suggest improvements based on history
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ai_infra.logging import get_logger

logger = get_logger("executor.learning")


# =============================================================================
# Data Models
# =============================================================================


class PatternType(Enum):
    """Type of execution pattern."""

    FAILURE = "failure"
    SUCCESS = "success"
    RETRY = "retry"


class TaskType(Enum):
    """Type of task for template matching."""

    CREATE_FILE = "create_file"
    MODIFY_FILE = "modify_file"
    ADD_FEATURE = "add_feature"
    FIX_BUG = "fix_bug"
    REFACTOR = "refactor"
    ADD_TESTS = "add_tests"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class FailurePattern:
    """A recorded failure pattern for learning.

    Attributes:
        pattern_id: Unique identifier for this pattern.
        category: Failure category (e.g., "SYNTAX_ERROR").
        error_signature: Hash of the error message pattern.
        error_message: Original error message.
        task_context: Context about the task that failed.
        file_hints: Files involved in the failure.
        fix_attempts: List of fix attempts and their outcomes.
        occurrence_count: Number of times this pattern occurred.
        first_seen: When this pattern was first seen.
        last_seen: When this pattern was last seen.
        resolved: Whether this pattern has been successfully resolved.
        resolution_strategy: Strategy that resolved this pattern (if resolved).
    """

    pattern_id: str
    category: str
    error_signature: str
    error_message: str
    task_context: str = ""
    file_hints: list[str] = field(default_factory=list)
    fix_attempts: list[dict[str, Any]] = field(default_factory=list)
    occurrence_count: int = 1
    first_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved: bool = False
    resolution_strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pattern_id": self.pattern_id,
            "category": self.category,
            "error_signature": self.error_signature,
            "error_message": self.error_message,
            "task_context": self.task_context,
            "file_hints": self.file_hints,
            "fix_attempts": self.fix_attempts,
            "occurrence_count": self.occurrence_count,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "resolved": self.resolved,
            "resolution_strategy": self.resolution_strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailurePattern:
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            category=data["category"],
            error_signature=data["error_signature"],
            error_message=data["error_message"],
            task_context=data.get("task_context", ""),
            file_hints=data.get("file_hints", []),
            fix_attempts=data.get("fix_attempts", []),
            occurrence_count=data.get("occurrence_count", 1),
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            resolved=data.get("resolved", False),
            resolution_strategy=data.get("resolution_strategy", ""),
        )


@dataclass
class SuccessPattern:
    """A recorded success pattern for learning.

    Attributes:
        pattern_id: Unique identifier for this pattern.
        task_type: Type of task (e.g., "add_feature", "fix_bug").
        task_signature: Hash of task characteristics.
        task_title: Original task title.
        task_description: Task description.
        file_hints: Files involved.
        prompt_template: The prompt that worked.
        context_strategy: How context was built.
        verification_level: What verification level passed.
        execution_time: How long the task took (seconds).
        files_modified: Files that were modified.
        success_count: Number of times this pattern succeeded.
        created_at: When this pattern was first recorded.
        last_used: When this pattern was last used.
    """

    pattern_id: str
    task_type: TaskType
    task_signature: str
    task_title: str
    task_description: str = ""
    file_hints: list[str] = field(default_factory=list)
    prompt_template: str = ""
    context_strategy: str = "default"
    verification_level: str = "tests"
    execution_time: float = 0.0
    files_modified: list[str] = field(default_factory=list)
    success_count: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pattern_id": self.pattern_id,
            "task_type": self.task_type.value,
            "task_signature": self.task_signature,
            "task_title": self.task_title,
            "task_description": self.task_description,
            "file_hints": self.file_hints,
            "prompt_template": self.prompt_template,
            "context_strategy": self.context_strategy,
            "verification_level": self.verification_level,
            "execution_time": self.execution_time,
            "files_modified": self.files_modified,
            "success_count": self.success_count,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuccessPattern:
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            task_type=TaskType(data.get("task_type", "unknown")),
            task_signature=data["task_signature"],
            task_title=data["task_title"],
            task_description=data.get("task_description", ""),
            file_hints=data.get("file_hints", []),
            prompt_template=data.get("prompt_template", ""),
            context_strategy=data.get("context_strategy", "default"),
            verification_level=data.get("verification_level", "tests"),
            execution_time=data.get("execution_time", 0.0),
            files_modified=data.get("files_modified", []),
            success_count=data.get("success_count", 1),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]),
        )


@dataclass
class LearningStats:
    """Statistics about learning patterns.

    Attributes:
        total_failures: Total failure patterns recorded.
        total_successes: Total success patterns recorded.
        resolved_failures: Failures that have been resolved.
        avg_fix_attempts: Average fix attempts before resolution.
        top_failure_categories: Most common failure categories.
        top_task_types: Most successful task types.
        avg_execution_time: Average execution time for successes.
    """

    total_failures: int = 0
    total_successes: int = 0
    resolved_failures: int = 0
    avg_fix_attempts: float = 0.0
    top_failure_categories: list[tuple[str, int]] = field(default_factory=list)
    top_task_types: list[tuple[str, int]] = field(default_factory=list)
    avg_execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "resolved_failures": self.resolved_failures,
            "avg_fix_attempts": self.avg_fix_attempts,
            "top_failure_categories": self.top_failure_categories,
            "top_task_types": self.top_task_types,
            "avg_execution_time": self.avg_execution_time,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Learning Statistics",
            "=" * 40,
            f"Total Failure Patterns: {self.total_failures}",
            f"Total Success Patterns: {self.total_successes}",
            f"Resolved Failures: {self.resolved_failures}",
            f"Avg Fix Attempts: {self.avg_fix_attempts:.1f}",
            f"Avg Execution Time: {self.avg_execution_time:.1f}s",
        ]

        if self.top_failure_categories:
            lines.append("\nTop Failure Categories:")
            for category, count in self.top_failure_categories[:5]:
                lines.append(f"  {category}: {count}")

        if self.top_task_types:
            lines.append("\nTop Success Task Types:")
            for task_type, count in self.top_task_types[:5]:
                lines.append(f"  {task_type}: {count}")

        return "\n".join(lines)


# =============================================================================
# Learning Store
# =============================================================================


class LearningStore:
    """Persistent storage for learning patterns.

    Uses SQLite for local storage with optional semantic search
    via the MemoryStore for finding similar patterns.

    Example:
        >>> from ai_infra.executor.learning import LearningStore
        >>>
        >>> store = LearningStore(Path(".executor/learning"))
        >>>
        >>> # Record a failure
        >>> store.record_failure(
        ...     category="SYNTAX_ERROR",
        ...     error_message="SyntaxError: unexpected EOF",
        ...     task_context="Implementing feature X",
        ...     file_hints=["src/feature.py"],
        ... )
        >>>
        >>> # Find similar failures
        >>> similar = store.find_similar_failures("SyntaxError: invalid syntax")
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        use_semantic_search: bool = False,
        max_patterns: int = 10000,
    ) -> None:
        """Initialize the learning store.

        Args:
            data_dir: Directory to store learning data (None for in-memory).
            use_semantic_search: Enable semantic similarity search.
            max_patterns: Maximum patterns to keep in memory.
        """
        self._data_dir = Path(data_dir) if data_dir else None
        self._use_semantic_search = use_semantic_search
        self._max_patterns = max_patterns

        # Pattern storage
        self._failure_patterns: dict[str, FailurePattern] = {}
        self._success_patterns: dict[str, SuccessPattern] = {}

        # Memory store for semantic search (lazy init)
        self._memory_store = None

        # Load existing patterns
        if self._data_dir:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._load_patterns()

    def _load_patterns(self) -> None:
        """Load existing patterns from disk."""
        if not self._data_dir:
            return

        # Load failure patterns
        failure_file = self._data_dir / "failure_patterns.json"
        if failure_file.exists():
            try:
                with open(failure_file) as f:
                    data = json.load(f)
                for pattern_data in data.get("patterns", []):
                    pattern = FailurePattern.from_dict(pattern_data)
                    self._failure_patterns[pattern.pattern_id] = pattern
                logger.info(f"Loaded {len(self._failure_patterns)} failure patterns")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load failure patterns: {e}")

        # Load success patterns
        success_file = self._data_dir / "success_patterns.json"
        if success_file.exists():
            try:
                with open(success_file) as f:
                    data = json.load(f)
                for pattern_data in data.get("patterns", []):
                    pattern = SuccessPattern.from_dict(pattern_data)
                    self._success_patterns[pattern.pattern_id] = pattern
                logger.info(f"Loaded {len(self._success_patterns)} success patterns")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load success patterns: {e}")

    def save(self) -> None:
        """Save patterns to disk."""
        if not self._data_dir:
            return

        # Save failure patterns
        failure_file = self._data_dir / "failure_patterns.json"
        failure_data = {
            "patterns": [p.to_dict() for p in self._failure_patterns.values()],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        with open(failure_file, "w") as f:
            json.dump(failure_data, f, indent=2)

        # Save success patterns
        success_file = self._data_dir / "success_patterns.json"
        success_data = {
            "patterns": [p.to_dict() for p in self._success_patterns.values()],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        with open(success_file, "w") as f:
            json.dump(success_data, f, indent=2)

        logger.debug(
            f"Saved {len(self._failure_patterns)} failure, "
            f"{len(self._success_patterns)} success patterns"
        )

    @property
    def memory_store(self):
        """Get the memory store for semantic search (lazy init)."""
        if self._memory_store is None and self._use_semantic_search:
            from ai_infra.memory import MemoryStore

            if self._data_dir:
                db_path = self._data_dir / "learning_memory.db"
                self._memory_store = MemoryStore.sqlite(str(db_path))
            else:
                self._memory_store = MemoryStore()
        return self._memory_store

    # =========================================================================
    # Error Signature Generation
    # =========================================================================

    @staticmethod
    def generate_error_signature(error_message: str) -> str:
        """Generate a signature for an error message.

        This normalizes the error to group similar errors together.

        Args:
            error_message: The raw error message.

        Returns:
            A hash signature for the normalized error.
        """
        # Normalize the error message
        normalized = error_message.lower()

        # Remove line numbers (e.g., "line 42")
        import re

        normalized = re.sub(r"line \d+", "line N", normalized)

        # Remove file paths
        normalized = re.sub(r"['\"]?[\w/\\.]+\.(py|js|ts)['\"]?", "FILE", normalized)

        # Remove specific variable names in quotes
        normalized = re.sub(r"['\"][^'\"]+['\"]", "'X'", normalized)

        # Hash the normalized message
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    @staticmethod
    def generate_task_signature(
        task_title: str,
        file_hints: list[str],
    ) -> str:
        """Generate a signature for a task.

        Args:
            task_title: The task title.
            file_hints: Files involved in the task.

        Returns:
            A hash signature for the task.
        """
        # Extract key words from title
        import re

        words = re.findall(r"\b\w+\b", task_title.lower())
        key_words = [w for w in words if len(w) > 3]

        # Extract file extensions
        extensions = set()
        for hint in file_hints:
            if "." in hint:
                ext = hint.rsplit(".", 1)[-1]
                extensions.add(ext)

        # Combine for signature
        signature_parts = sorted(key_words[:5]) + sorted(extensions)
        signature_str = ":".join(signature_parts)

        return hashlib.md5(signature_str.encode()).hexdigest()[:12]

    @staticmethod
    def infer_task_type(task_title: str, task_description: str = "") -> TaskType:
        """Infer the task type from title and description.

        Args:
            task_title: The task title.
            task_description: Optional task description.

        Returns:
            Inferred TaskType.
        """
        text = f"{task_title} {task_description}".lower()

        # Check for keywords
        if any(kw in text for kw in ["create", "new file", "add file", "generate"]):
            return TaskType.CREATE_FILE
        if any(kw in text for kw in ["modify", "update", "change", "edit"]):
            return TaskType.MODIFY_FILE
        if any(kw in text for kw in ["fix", "bug", "error", "issue", "resolve"]):
            return TaskType.FIX_BUG
        if any(kw in text for kw in ["add feature", "implement", "add support"]):
            return TaskType.ADD_FEATURE
        if any(kw in text for kw in ["refactor", "cleanup", "reorganize"]):
            return TaskType.REFACTOR
        if any(kw in text for kw in ["test", "spec", "coverage"]):
            return TaskType.ADD_TESTS
        if any(kw in text for kw in ["doc", "readme", "comment"]):
            return TaskType.DOCUMENTATION
        if any(kw in text for kw in ["config", "setting", "environment"]):
            return TaskType.CONFIGURATION

        return TaskType.UNKNOWN

    # =========================================================================
    # Failure Pattern Operations
    # =========================================================================

    def record_failure(
        self,
        category: str,
        error_message: str,
        *,
        task_context: str = "",
        file_hints: list[str] | None = None,
        fix_attempt: dict[str, Any] | None = None,
    ) -> FailurePattern:
        """Record a failure pattern.

        If a similar pattern exists, it will be updated with increased count.

        Args:
            category: Failure category (e.g., "SYNTAX_ERROR").
            error_message: The error message.
            task_context: Context about the task.
            file_hints: Files involved.
            fix_attempt: Optional fix attempt details.

        Returns:
            The recorded or updated FailurePattern.
        """
        error_signature = self.generate_error_signature(error_message)
        pattern_id = f"fail_{category}_{error_signature}"

        if pattern_id in self._failure_patterns:
            # Update existing pattern
            pattern = self._failure_patterns[pattern_id]
            pattern.occurrence_count += 1
            pattern.last_seen = datetime.now(UTC)
            if fix_attempt:
                pattern.fix_attempts.append(fix_attempt)
            logger.debug(
                f"Updated failure pattern {pattern_id} (count: {pattern.occurrence_count})"
            )
        else:
            # Create new pattern
            pattern = FailurePattern(
                pattern_id=pattern_id,
                category=category,
                error_signature=error_signature,
                error_message=error_message,
                task_context=task_context,
                file_hints=file_hints or [],
                fix_attempts=[fix_attempt] if fix_attempt else [],
            )
            self._failure_patterns[pattern_id] = pattern
            logger.info(f"Recorded new failure pattern: {category} ({pattern_id})")

        # Store in memory for semantic search
        if self.memory_store:
            self.memory_store.put(
                namespace=("executor", "failures"),
                key=pattern_id,
                value={
                    "category": category,
                    "error_message": error_message,
                    "task_context": task_context,
                },
            )

        # Trim old patterns if needed
        self._trim_patterns()

        # Auto-save
        if self._data_dir:
            self.save()

        return pattern

    def mark_failure_resolved(
        self,
        pattern_id: str,
        resolution_strategy: str,
    ) -> bool:
        """Mark a failure pattern as resolved.

        Args:
            pattern_id: The pattern ID to resolve.
            resolution_strategy: Strategy that resolved the failure.

        Returns:
            True if pattern was found and updated.
        """
        if pattern_id not in self._failure_patterns:
            return False

        pattern = self._failure_patterns[pattern_id]
        pattern.resolved = True
        pattern.resolution_strategy = resolution_strategy

        if self._data_dir:
            self.save()

        logger.info(f"Marked failure pattern {pattern_id} as resolved")
        return True

    def find_similar_failures(
        self,
        error_message: str,
        *,
        limit: int = 5,
    ) -> list[FailurePattern]:
        """Find failure patterns similar to the given error.

        Args:
            error_message: The error to match.
            limit: Maximum results.

        Returns:
            List of similar FailurePatterns.
        """
        # First, try exact signature match
        signature = self.generate_error_signature(error_message)
        results: list[FailurePattern] = []

        for pattern in self._failure_patterns.values():
            if pattern.error_signature == signature:
                results.append(pattern)

        # If we have semantic search, use it for fuzzy matching
        if self.memory_store and len(results) < limit:
            try:
                similar = self.memory_store.search(
                    namespace=("executor", "failures"),
                    query=error_message,
                    limit=limit,
                )
                for item in similar:
                    pattern_id = item.key
                    if pattern_id in self._failure_patterns:
                        pattern = self._failure_patterns[pattern_id]
                        if pattern not in results:
                            results.append(pattern)
            except ValueError:
                # Semantic search not available
                pass

        # Sort by occurrence count (most common first)
        results.sort(key=lambda p: p.occurrence_count, reverse=True)
        return results[:limit]

    def get_resolution_strategies(
        self,
        category: str,
    ) -> list[str]:
        """Get resolution strategies that worked for a category.

        Args:
            category: Failure category.

        Returns:
            List of resolution strategies.
        """
        strategies: list[str] = []
        for pattern in self._failure_patterns.values():
            if pattern.category == category and pattern.resolved:
                if pattern.resolution_strategy and pattern.resolution_strategy not in strategies:
                    strategies.append(pattern.resolution_strategy)
        return strategies

    # =========================================================================
    # Success Pattern Operations
    # =========================================================================

    def record_success(
        self,
        task_title: str,
        *,
        task_description: str = "",
        file_hints: list[str] | None = None,
        prompt_template: str = "",
        context_strategy: str = "default",
        verification_level: str = "tests",
        execution_time: float = 0.0,
        files_modified: list[str] | None = None,
    ) -> SuccessPattern:
        """Record a successful task pattern.

        Args:
            task_title: The task title.
            task_description: Task description.
            file_hints: Files involved.
            prompt_template: The prompt that worked.
            context_strategy: How context was built.
            verification_level: What passed.
            execution_time: How long it took.
            files_modified: Files that were modified.

        Returns:
            The recorded or updated SuccessPattern.
        """
        file_hints = file_hints or []
        task_signature = self.generate_task_signature(task_title, file_hints)
        task_type = self.infer_task_type(task_title, task_description)
        pattern_id = f"success_{task_type.value}_{task_signature}"

        if pattern_id in self._success_patterns:
            # Update existing pattern
            pattern = self._success_patterns[pattern_id]
            pattern.success_count += 1
            pattern.last_used = datetime.now(UTC)
            # Update execution time (running average)
            pattern.execution_time = (
                pattern.execution_time * (pattern.success_count - 1) + execution_time
            ) / pattern.success_count
            logger.debug(f"Updated success pattern {pattern_id} (count: {pattern.success_count})")
        else:
            # Create new pattern
            pattern = SuccessPattern(
                pattern_id=pattern_id,
                task_type=task_type,
                task_signature=task_signature,
                task_title=task_title,
                task_description=task_description,
                file_hints=file_hints,
                prompt_template=prompt_template,
                context_strategy=context_strategy,
                verification_level=verification_level,
                execution_time=execution_time,
                files_modified=files_modified or [],
            )
            self._success_patterns[pattern_id] = pattern
            logger.info(f"Recorded new success pattern: {task_type.value} ({pattern_id})")

        # Store in memory for semantic search
        if self.memory_store:
            self.memory_store.put(
                namespace=("executor", "successes"),
                key=pattern_id,
                value={
                    "task_type": task_type.value,
                    "task_title": task_title,
                    "task_description": task_description,
                },
            )

        # Trim old patterns if needed
        self._trim_patterns()

        # Auto-save
        if self._data_dir:
            self.save()

        return pattern

    def find_similar_successes(
        self,
        task_title: str,
        *,
        file_hints: list[str] | None = None,
        limit: int = 5,
    ) -> list[SuccessPattern]:
        """Find success patterns similar to the given task.

        Args:
            task_title: The task title.
            file_hints: Files involved.
            limit: Maximum results.

        Returns:
            List of similar SuccessPatterns.
        """
        file_hints = file_hints or []

        # Try signature match first
        signature = self.generate_task_signature(task_title, file_hints)
        task_type = self.infer_task_type(task_title)

        results: list[SuccessPattern] = []

        # First, find patterns with same signature
        for pattern in self._success_patterns.values():
            if pattern.task_signature == signature:
                results.append(pattern)

        # Then, find patterns with same task type
        if len(results) < limit:
            for pattern in self._success_patterns.values():
                if pattern.task_type == task_type and pattern not in results:
                    results.append(pattern)

        # Use semantic search if available
        if self.memory_store and len(results) < limit:
            try:
                similar = self.memory_store.search(
                    namespace=("executor", "successes"),
                    query=task_title,
                    limit=limit,
                )
                for item in similar:
                    pattern_id = item.key
                    if pattern_id in self._success_patterns:
                        pattern = self._success_patterns[pattern_id]
                        if pattern not in results:
                            results.append(pattern)
            except ValueError:
                pass

        # Sort by success count (most successful first)
        results.sort(key=lambda p: p.success_count, reverse=True)
        return results[:limit]

    def get_template_for_task_type(
        self,
        task_type: TaskType,
    ) -> str | None:
        """Get the best prompt template for a task type.

        Args:
            task_type: The task type.

        Returns:
            The most successful template, or None.
        """
        best_pattern: SuccessPattern | None = None
        best_count = 0

        for pattern in self._success_patterns.values():
            if pattern.task_type == task_type and pattern.prompt_template:
                if pattern.success_count > best_count:
                    best_pattern = pattern
                    best_count = pattern.success_count

        return best_pattern.prompt_template if best_pattern else None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> LearningStats:
        """Get learning statistics.

        Returns:
            LearningStats with aggregated data.
        """
        # Count failure categories
        failure_categories: dict[str, int] = {}
        resolved_count = 0
        total_fix_attempts = 0

        for pattern in self._failure_patterns.values():
            failure_categories[pattern.category] = (
                failure_categories.get(pattern.category, 0) + pattern.occurrence_count
            )
            if pattern.resolved:
                resolved_count += 1
            total_fix_attempts += len(pattern.fix_attempts)

        # Count task types
        task_types: dict[str, int] = {}
        total_execution_time = 0.0
        success_count = 0

        for pattern in self._success_patterns.values():
            task_types[pattern.task_type.value] = (
                task_types.get(pattern.task_type.value, 0) + pattern.success_count
            )
            total_execution_time += pattern.execution_time * pattern.success_count
            success_count += pattern.success_count

        return LearningStats(
            total_failures=sum(p.occurrence_count for p in self._failure_patterns.values()),
            total_successes=success_count,
            resolved_failures=resolved_count,
            avg_fix_attempts=total_fix_attempts / len(self._failure_patterns)
            if self._failure_patterns
            else 0.0,
            top_failure_categories=sorted(
                failure_categories.items(), key=lambda x: x[1], reverse=True
            )[:5],
            top_task_types=sorted(task_types.items(), key=lambda x: x[1], reverse=True)[:5],
            avg_execution_time=total_execution_time / success_count if success_count else 0.0,
        )

    # =========================================================================
    # Utilities
    # =========================================================================

    def _trim_patterns(self) -> None:
        """Trim old patterns to stay under max_patterns limit."""
        # Trim failures (keep most recent)
        if len(self._failure_patterns) > self._max_patterns:
            sorted_patterns = sorted(
                self._failure_patterns.values(),
                key=lambda p: p.last_seen,
                reverse=True,
            )
            self._failure_patterns = {
                p.pattern_id: p for p in sorted_patterns[: self._max_patterns]
            }

        # Trim successes (keep most successful)
        if len(self._success_patterns) > self._max_patterns:
            sorted_patterns = sorted(
                self._success_patterns.values(),
                key=lambda p: p.success_count,
                reverse=True,
            )
            self._success_patterns = {
                p.pattern_id: p for p in sorted_patterns[: self._max_patterns]
            }

    def clear(self) -> None:
        """Clear all patterns."""
        self._failure_patterns.clear()
        self._success_patterns.clear()

        if self._data_dir:
            self.save()

        logger.info("Cleared all learning patterns")

    @property
    def failure_count(self) -> int:
        """Get number of failure patterns."""
        return len(self._failure_patterns)

    @property
    def success_count(self) -> int:
        """Get number of success patterns."""
        return len(self._success_patterns)


# =============================================================================
# Prompt Refinement
# =============================================================================


class PromptRefiner:
    """Refine prompts based on learning patterns.

    Uses failure and success patterns to improve prompts
    by adding context about what has worked or failed before.

    Example:
        >>> from ai_infra.executor.learning import PromptRefiner, LearningStore
        >>>
        >>> store = LearningStore(Path(".executor/learning"))
        >>> refiner = PromptRefiner(store)
        >>>
        >>> refined = refiner.refine_prompt(
        ...     original_prompt="Implement the login feature",
        ...     task_title="Add user authentication",
        ...     file_hints=["src/auth.py"],
        ... )
    """

    def __init__(self, learning_store: LearningStore) -> None:
        """Initialize the prompt refiner.

        Args:
            learning_store: The learning store to use.
        """
        self._store = learning_store

    def refine_prompt(
        self,
        original_prompt: str,
        *,
        task_title: str = "",
        file_hints: list[str] | None = None,
        include_failure_hints: bool = True,
        include_success_hints: bool = True,
        max_hints: int = 3,
    ) -> str:
        """Refine a prompt based on learning patterns.

        Args:
            original_prompt: The original prompt.
            task_title: The task title.
            file_hints: Files involved.
            include_failure_hints: Add hints about past failures.
            include_success_hints: Add hints from successes.
            max_hints: Maximum hints to add.

        Returns:
            The refined prompt.
        """
        hints: list[str] = []

        # Add success hints
        if include_success_hints and task_title:
            similar_successes = self._store.find_similar_successes(
                task_title, file_hints=file_hints, limit=max_hints
            )
            for pattern in similar_successes[:max_hints]:
                if pattern.prompt_template:
                    hints.append(
                        f"NOTE: Similar tasks succeeded with approach: "
                        f"{self._summarize_template(pattern.prompt_template)}"
                    )
                    break  # Only use one success hint

        # Add failure hints
        if include_failure_hints:
            # Check for common failure patterns in files
            for hint in file_hints or []:
                related_failures = [
                    p for p in self._store._failure_patterns.values() if hint in p.file_hints
                ]
                if related_failures:
                    # Get most common failure category for this file
                    categories: dict[str, int] = {}
                    for f in related_failures:
                        categories[f.category] = categories.get(f.category, 0) + f.occurrence_count
                    if categories:
                        top_category = max(categories.items(), key=lambda x: x[1])[0]
                        hints.append(
                            f"CAUTION: File {hint} has had {top_category} issues before. "
                            "Double-check your changes."
                        )
                        break  # Only one file warning

        if not hints:
            return original_prompt

        # Prepend hints to prompt
        hints_section = "\n".join(f"- {hint}" for hint in hints)
        return f"## Learning Hints\n{hints_section}\n\n{original_prompt}"

    def _summarize_template(self, template: str) -> str:
        """Summarize a prompt template to key points."""
        # Extract first 100 chars or first sentence
        template = template.strip()
        if len(template) <= 100:
            return template

        # Try to get first sentence
        if "." in template[:100]:
            return template[: template.index(".") + 1]

        return template[:100] + "..."

    def suggest_improvements(
        self,
        failed_prompt: str,
        error_message: str,
        *,
        category: str = "",
    ) -> list[str]:
        """Suggest prompt improvements based on failure.

        Args:
            failed_prompt: The prompt that failed.
            error_message: The error that occurred.
            category: Failure category if known.

        Returns:
            List of improvement suggestions.
        """
        suggestions: list[str] = []

        # Find similar failures
        similar = self._store.find_similar_failures(error_message)

        # Get resolution strategies that worked
        for pattern in similar:
            if pattern.resolved and pattern.resolution_strategy:
                suggestions.append(
                    f"Resolution that worked for similar error: {pattern.resolution_strategy}"
                )
                break

        # Add category-specific suggestions
        if category:
            strategies = self._store.get_resolution_strategies(category)
            for strategy in strategies[:2]:
                if strategy not in suggestions:
                    suggestions.append(f"Strategy for {category}: {strategy}")

        # Generic suggestions based on common patterns
        if not suggestions:
            if "syntax" in error_message.lower():
                suggestions.append(
                    "Check for proper Python syntax, especially indentation and brackets"
                )
            elif "import" in error_message.lower():
                suggestions.append("Verify import paths and ensure modules exist")
            elif "test" in error_message.lower():
                suggestions.append(
                    "Review test expectations and ensure assertions match implementation"
                )

        return suggestions
