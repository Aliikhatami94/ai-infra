"""Pattern recognition for executor learning (Phase 5.2).

This module provides pattern tracking and suggestion capabilities:
- FailurePatternTracker: Track and generalize failure patterns
- FixPatternTracker: Track what fixes work for what errors
- PatternSuggester: Suggest patterns during execution
- AgentsUpdater: Propose AGENTS.md updates with learned patterns

Example:
    ```python
    from ai_infra.executor.patterns import (
        FailurePatternTracker,
        FixPatternTracker,
        PatternSuggester,
        PatternsDatabase,
    )

    # Create database
    db = PatternsDatabase()

    # Track failures and fixes
    failure_tracker = FailurePatternTracker(db)
    fix_tracker = FixPatternTracker(db)

    # Get suggestions
    suggester = PatternSuggester(failure_tracker, fix_tracker)
    suggestion = suggester.on_error(error)
    ```
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class TaskError:
    """Represents a task error for pattern matching.

    Attributes:
        type: Error type (e.g., "SyntaxError", "ImportError").
        message: Full error message.
        file_path: File where error occurred.
        line_number: Line number of error.
        context: Additional context about the error.
    """

    type: str
    message: str
    file_path: str | None = None
    line_number: int | None = None
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskError:
        """Create from dictionary."""
        return cls(
            type=data.get("type", "Unknown"),
            message=data.get("message", ""),
            file_path=data.get("file_path"),
            line_number=data.get("line_number"),
            context=data.get("context", ""),
        )

    @classmethod
    def from_exception(cls, exc: Exception, context: str = "") -> TaskError:
        """Create from an exception."""
        return cls(
            type=type(exc).__name__,
            message=str(exc),
            context=context,
        )


@dataclass
class ExecutionContext:
    """Context about task execution.

    Attributes:
        task_title: Title of the task.
        task_description: Description of the task.
        language: Programming language.
        framework: Framework in use.
        files_involved: Files involved in the task.
        summary: Brief summary of the context.
    """

    task_title: str = ""
    task_description: str = ""
    language: str = ""
    framework: str = ""
    files_involved: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Get a brief summary of the context."""
        parts = [self.task_title]
        if self.language:
            parts.append(f"[{self.language}]")
        if self.framework:
            parts.append(f"({self.framework})")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_title": self.task_title,
            "task_description": self.task_description,
            "language": self.language,
            "framework": self.framework,
            "files_involved": self.files_involved,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionContext:
        """Create from dictionary."""
        return cls(
            task_title=data.get("task_title", ""),
            task_description=data.get("task_description", ""),
            language=data.get("language", ""),
            framework=data.get("framework", ""),
            files_involved=data.get("files_involved", []),
        )


@dataclass
class FailurePattern:
    """A recorded failure pattern.

    Attributes:
        id: Unique identifier.
        error_type: Type of error (e.g., "SyntaxError").
        error_message_pattern: Generalized error message pattern.
        occurrence_count: Number of times this pattern occurred.
        contexts: Contexts where this occurred.
        suggested_fix: Suggested fix for this pattern.
        created_at: When first recorded.
        updated_at: When last updated.
    """

    id: str
    error_type: str
    error_message_pattern: str
    occurrence_count: int = 1
    contexts: list[str] = field(default_factory=list)
    suggested_fix: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "error_type": self.error_type,
            "error_message_pattern": self.error_message_pattern,
            "occurrence_count": self.occurrence_count,
            "contexts": self.contexts[-10:],  # Keep last 10
            "suggested_fix": self.suggested_fix,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailurePattern:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            error_type=data["error_type"],
            error_message_pattern=data["error_message_pattern"],
            occurrence_count=data.get("occurrence_count", 1),
            contexts=data.get("contexts", []),
            suggested_fix=data.get("suggested_fix"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class FixPattern:
    """A recorded fix pattern.

    Attributes:
        id: Unique identifier.
        error_pattern: Pattern of error this fixes.
        fix_approach: Description of the fix approach.
        success_count: Number of successful applications.
        failure_count: Number of failed applications.
        example_diff: Example code diff.
        created_at: When first recorded.
        updated_at: When last updated.
    """

    id: str
    error_pattern: str
    fix_approach: str
    success_count: int = 0
    failure_count: int = 0
    example_diff: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "error_pattern": self.error_pattern,
            "fix_approach": self.fix_approach,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "example_diff": self.example_diff,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixPattern:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            error_pattern=data["error_pattern"],
            fix_approach=data["fix_approach"],
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            example_diff=data.get("example_diff", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class FixAction:
    """An action taken to fix an error.

    Attributes:
        action_type: Type of action (e.g., "edit", "create", "delete").
        file_path: File that was modified.
        description: What was done.
        diff: Code diff if applicable.
    """

    action_type: str
    file_path: str = ""
    description: str = ""
    diff: str = ""


# =============================================================================
# Patterns Database
# =============================================================================


class PatternsDatabase:
    """Persistent storage for patterns.

    Stores failure and fix patterns to JSON files.

    Attributes:
        path: Path to the patterns directory.
    """

    DEFAULT_PATH = Path.home() / ".ai-infra" / "patterns"

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        auto_save: bool = True,
    ) -> None:
        """Initialize patterns database.

        Args:
            path: Path to patterns directory (None for default).
            auto_save: Auto-save after modifications.
        """
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.auto_save = auto_save
        self._failure_patterns: dict[str, FailurePattern] = {}
        self._fix_patterns: dict[str, FixPattern] = {}

        # Ensure directory exists
        self.path.mkdir(parents=True, exist_ok=True)

        # Load existing patterns
        self._load()

    def _load(self) -> None:
        """Load patterns from disk."""
        # Load failure patterns
        failure_file = self.path / "failure_patterns.json"
        if failure_file.exists():
            try:
                with open(failure_file) as f:
                    data = json.load(f)
                for pattern_data in data.get("patterns", []):
                    pattern = FailurePattern.from_dict(pattern_data)
                    self._failure_patterns[pattern.id] = pattern
                logger.debug(f"Loaded {len(self._failure_patterns)} failure patterns")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load failure patterns: {e}")

        # Load fix patterns
        fix_file = self.path / "fix_patterns.json"
        if fix_file.exists():
            try:
                with open(fix_file) as f:
                    data = json.load(f)
                for pattern_data in data.get("patterns", []):
                    pattern = FixPattern.from_dict(pattern_data)
                    self._fix_patterns[pattern.id] = pattern
                logger.debug(f"Loaded {len(self._fix_patterns)} fix patterns")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load fix patterns: {e}")

    def save(self) -> None:
        """Save patterns to disk."""
        # Save failure patterns
        failure_file = self.path / "failure_patterns.json"
        failure_data = {
            "version": "1.0",
            "updated_at": datetime.now(UTC).isoformat(),
            "patterns": [p.to_dict() for p in self._failure_patterns.values()],
        }
        with open(failure_file, "w") as f:
            json.dump(failure_data, f, indent=2)

        # Save fix patterns
        fix_file = self.path / "fix_patterns.json"
        fix_data = {
            "version": "1.0",
            "updated_at": datetime.now(UTC).isoformat(),
            "patterns": [p.to_dict() for p in self._fix_patterns.values()],
        }
        with open(fix_file, "w") as f:
            json.dump(fix_data, f, indent=2)

    def _auto_save(self) -> None:
        """Auto-save if enabled."""
        if self.auto_save:
            self.save()

    # Failure pattern methods

    def get_failure_pattern(self, pattern_id: str) -> FailurePattern | None:
        """Get a failure pattern by ID."""
        return self._failure_patterns.get(pattern_id)

    def find_failure_pattern(self, error_pattern: str) -> FailurePattern | None:
        """Find a failure pattern by error pattern string."""
        for pattern in self._failure_patterns.values():
            if pattern.error_message_pattern == error_pattern:
                return pattern
        return None

    def save_failure_pattern(self, pattern: FailurePattern) -> None:
        """Save or update a failure pattern."""
        pattern.updated_at = datetime.now(UTC)
        self._failure_patterns[pattern.id] = pattern
        self._auto_save()

    def get_all_failure_patterns(self) -> list[FailurePattern]:
        """Get all failure patterns."""
        return list(self._failure_patterns.values())

    # Fix pattern methods

    def get_fix_pattern(self, pattern_id: str) -> FixPattern | None:
        """Get a fix pattern by ID."""
        return self._fix_patterns.get(pattern_id)

    def find_fix_pattern(self, error_pattern: str) -> FixPattern | None:
        """Find a fix pattern by error pattern string."""
        for pattern in self._fix_patterns.values():
            if pattern.error_pattern == error_pattern:
                return pattern
        return None

    def save_fix_pattern(self, pattern: FixPattern) -> None:
        """Save or update a fix pattern."""
        pattern.updated_at = datetime.now(UTC)
        self._fix_patterns[pattern.id] = pattern
        self._auto_save()

    def get_all_fix_patterns(self) -> list[FixPattern]:
        """Get all fix patterns."""
        return list(self._fix_patterns.values())

    def get_best_fixes(
        self,
        error_pattern: str,
        min_success_rate: float = 0.5,
        limit: int = 3,
    ) -> list[FixPattern]:
        """Get best fix patterns for an error.

        Args:
            error_pattern: Error pattern to match.
            min_success_rate: Minimum success rate.
            limit: Maximum fixes to return.

        Returns:
            List of FixPatterns sorted by success rate.
        """
        matches = []
        for pattern in self._fix_patterns.values():
            if pattern.error_pattern == error_pattern:
                if pattern.success_rate >= min_success_rate:
                    matches.append(pattern)

        matches.sort(key=lambda p: p.success_rate, reverse=True)
        return matches[:limit]


# =============================================================================
# Failure Pattern Tracker
# =============================================================================


class FailurePatternTracker:
    """Track common failure patterns.

    Phase 5.2.1: Records failures and generalizes error messages
    to create reusable patterns.

    Example:
        ```python
        tracker = FailurePatternTracker(db)

        # Record a failure
        tracker.record_failure(error, context)

        # Get fix suggestion
        suggestion = tracker.get_fix_suggestion(error)
        ```
    """

    # Patterns to generalize in error messages
    GENERALIZATION_PATTERNS = [
        (r"'[^']+\.py'", "'*.py'"),
        (r"'[^']+\.ts'", "'*.ts'"),
        (r"'[^']+\.js'", "'*.js'"),
        (r"line \d+", "line N"),
        (r"column \d+", "column N"),
        (r"at 0x[0-9a-fA-F]+", "at 0xXXXX"),
        (r"/[\w/.-]+/", "/path/to/"),
        (r"\"[^\"]+\"", '"..."'),
    ]

    def __init__(self, db: PatternsDatabase) -> None:
        """Initialize the tracker.

        Args:
            db: Patterns database for persistence.
        """
        self.db = db

    def record_failure(
        self,
        error: TaskError,
        context: ExecutionContext,
    ) -> FailurePattern:
        """Record a failure for pattern detection.

        Args:
            error: The error that occurred.
            context: Execution context.

        Returns:
            The recorded or updated FailurePattern.
        """
        # Generalize the error message
        generalized = self._generalize_error(error.message)

        # Create pattern ID from error type and generalized message
        pattern_id = self._generate_pattern_id(error.type, generalized)

        # Find or create pattern
        pattern = self.db.find_failure_pattern(generalized)

        if pattern:
            # Update existing pattern
            pattern.occurrence_count += 1
            pattern.contexts.append(context.summary)
            # Keep only last 10 contexts
            pattern.contexts = pattern.contexts[-10:]
            logger.debug(
                f"Updated failure pattern {pattern.id} (count: {pattern.occurrence_count})"
            )
        else:
            # Create new pattern
            pattern = FailurePattern(
                id=pattern_id,
                error_type=error.type,
                error_message_pattern=generalized,
                occurrence_count=1,
                contexts=[context.summary],
            )
            logger.info(f"Recorded new failure pattern: {error.type}")

        self.db.save_failure_pattern(pattern)
        return pattern

    def _generalize_error(self, message: str) -> str:
        """Convert specific error to pattern.

        Args:
            message: Specific error message.

        Returns:
            Generalized pattern.
        """
        generalized = message
        for pattern, replacement in self.GENERALIZATION_PATTERNS:
            generalized = re.sub(pattern, replacement, generalized)
        return generalized

    def _generate_pattern_id(self, error_type: str, pattern: str) -> str:
        """Generate unique pattern ID.

        Args:
            error_type: Type of error.
            pattern: Generalized pattern.

        Returns:
            Unique pattern ID.
        """
        content = f"{error_type}:{pattern}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"fp-{hash_val}"

    def get_fix_suggestion(self, error: TaskError) -> str | None:
        """Get fix suggestion for known error pattern.

        Only suggests fixes for patterns seen 3+ times.

        Args:
            error: The error to get suggestion for.

        Returns:
            Suggested fix or None.
        """
        generalized = self._generalize_error(error.message)
        pattern = self.db.find_failure_pattern(generalized)

        if pattern and pattern.occurrence_count >= 3:
            return pattern.suggested_fix

        return None

    def set_suggested_fix(self, error: TaskError, fix: str) -> bool:
        """Set the suggested fix for a pattern.

        Args:
            error: The error pattern.
            fix: The suggested fix.

        Returns:
            True if pattern was found and updated.
        """
        generalized = self._generalize_error(error.message)
        pattern = self.db.find_failure_pattern(generalized)

        if pattern:
            pattern.suggested_fix = fix
            self.db.save_failure_pattern(pattern)
            return True

        return False

    def get_common_patterns(self, limit: int = 10) -> list[FailurePattern]:
        """Get most common failure patterns.

        Args:
            limit: Maximum patterns to return.

        Returns:
            List of patterns sorted by occurrence count.
        """
        patterns = self.db.get_all_failure_patterns()
        patterns.sort(key=lambda p: p.occurrence_count, reverse=True)
        return patterns[:limit]


# =============================================================================
# Fix Pattern Tracker
# =============================================================================


class FixPatternTracker:
    """Track what fixes work for what errors.

    Phase 5.2.2: Records fix attempts and their success rates.

    Example:
        ```python
        tracker = FixPatternTracker(db)

        # Record a fix attempt
        tracker.record_fix(error, fix_actions, success=True)

        # Get best fix
        fix = tracker.get_best_fix(error)
        ```
    """

    def __init__(self, db: PatternsDatabase) -> None:
        """Initialize the tracker.

        Args:
            db: Patterns database for persistence.
        """
        self.db = db

    def record_fix(
        self,
        error: TaskError,
        fix_actions: list[FixAction],
        success: bool,
        diff: str = "",
    ) -> FixPattern:
        """Record a fix attempt.

        Args:
            error: The error that was fixed.
            fix_actions: Actions taken to fix.
            success: Whether the fix worked.
            diff: Example diff of the fix.

        Returns:
            The recorded or updated FixPattern.
        """
        # Generalize the error pattern
        error_pattern = self._generalize_error(error.message)

        # Find or create pattern
        pattern = self.db.find_fix_pattern(error_pattern)

        if pattern:
            # Update existing pattern
            if success:
                pattern.success_count += 1
                pattern.fix_approach = self._summarize_fix(fix_actions)
                if diff:
                    pattern.example_diff = diff
            else:
                pattern.failure_count += 1
            logger.debug(
                f"Updated fix pattern {pattern.id} (success rate: {pattern.success_rate:.0%})"
            )
        else:
            # Create new pattern
            pattern = FixPattern(
                id=f"fix-{uuid.uuid4().hex[:12]}",
                error_pattern=error_pattern,
                fix_approach=self._summarize_fix(fix_actions),
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                example_diff=diff,
            )
            logger.info(f"Recorded new fix pattern for {error.type}")

        self.db.save_fix_pattern(pattern)
        return pattern

    def _generalize_error(self, message: str) -> str:
        """Generalize error message."""
        # Use same patterns as FailurePatternTracker
        generalized = message
        for pattern, replacement in FailurePatternTracker.GENERALIZATION_PATTERNS:
            generalized = re.sub(pattern, replacement, generalized)
        return generalized

    def _summarize_fix(self, actions: list[FixAction]) -> str:
        """Summarize fix actions into approach description.

        Args:
            actions: List of fix actions.

        Returns:
            Summary string.
        """
        if not actions:
            return "No specific fix recorded"

        summaries = []
        for action in actions:
            if action.description:
                summaries.append(action.description)
            else:
                summaries.append(f"{action.action_type} {action.file_path}")

        return "; ".join(summaries)

    def get_best_fix(self, error: TaskError) -> FixPattern | None:
        """Get the best fix for an error.

        Args:
            error: The error to fix.

        Returns:
            Best FixPattern or None.
        """
        error_pattern = self._generalize_error(error.message)
        fixes = self.db.get_best_fixes(error_pattern)

        if fixes:
            return fixes[0]
        return None

    def get_all_fixes_for_error(
        self,
        error: TaskError,
        min_success_rate: float = 0.0,
    ) -> list[FixPattern]:
        """Get all fixes for an error type.

        Args:
            error: The error type.
            min_success_rate: Minimum success rate filter.

        Returns:
            List of applicable FixPatterns.
        """
        error_pattern = self._generalize_error(error.message)
        return self.db.get_best_fixes(
            error_pattern,
            min_success_rate=min_success_rate,
            limit=10,
        )


# =============================================================================
# Pattern Suggester
# =============================================================================


class PatternSuggester:
    """Suggest patterns during task execution.

    Phase 5.2.3: Provides suggestions when errors occur based on
    past patterns and fixes.

    Example:
        ```python
        suggester = PatternSuggester(failure_tracker, fix_tracker)

        # Get suggestion for an error
        suggestion = suggester.on_error(error)
        if suggestion:
            print(f"Suggestion: {suggestion}")
        ```
    """

    def __init__(
        self,
        failure_tracker: FailurePatternTracker,
        fix_tracker: FixPatternTracker,
    ) -> None:
        """Initialize the suggester.

        Args:
            failure_tracker: Tracker for failure patterns.
            fix_tracker: Tracker for fix patterns.
        """
        self.failure_tracker = failure_tracker
        self.fix_tracker = fix_tracker

    def on_error(self, error: TaskError) -> str | None:
        """Get suggestion when error occurs.

        Args:
            error: The error that occurred.

        Returns:
            Suggestion string or None.
        """
        # First, check for known fix with high success rate
        fix = self.fix_tracker.get_best_fix(error)
        if fix and fix.success_rate >= 0.7:
            suggestion = f"Known fix (success rate: {fix.success_rate:.0%}):\n{fix.fix_approach}"
            if fix.example_diff:
                suggestion += f"\n\nExample:\n```\n{fix.example_diff[:500]}\n```"
            return suggestion

        # Check for known failure pattern with suggestion
        pattern_suggestion = self.failure_tracker.get_fix_suggestion(error)
        if pattern_suggestion:
            return f"This error has been seen before. Suggested approach:\n{pattern_suggestion}"

        # Check if this is a common error
        generalized = self.failure_tracker._generalize_error(error.message)
        pattern = self.failure_tracker.db.find_failure_pattern(generalized)
        if pattern and pattern.occurrence_count >= 3:
            return (
                f"This error has occurred {pattern.occurrence_count} times. "
                f"Contexts: {', '.join(pattern.contexts[:3])}"
            )

        return None

    def get_common_issues(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get most common issues with suggestions.

        Args:
            limit: Maximum issues to return.

        Returns:
            List of issue dictionaries.
        """
        common = self.failure_tracker.get_common_patterns(limit)
        issues = []

        for pattern in common:
            fix = self.fix_tracker.db.find_fix_pattern(pattern.error_message_pattern)
            issues.append(
                {
                    "error_type": pattern.error_type,
                    "pattern": pattern.error_message_pattern,
                    "count": pattern.occurrence_count,
                    "has_fix": fix is not None,
                    "fix_success_rate": fix.success_rate if fix else 0,
                    "suggested_fix": pattern.suggested_fix or (fix.fix_approach if fix else None),
                }
            )

        return issues


# =============================================================================
# AGENTS.md Updater
# =============================================================================


class AgentsUpdater:
    """Update AGENTS.md with learned patterns.

    Phase 5.2.4: Proposes updates to AGENTS.md based on
    high-confidence patterns learned from execution.

    Example:
        ```python
        from ai_infra.executor.skills import Skill

        updater = AgentsUpdater()
        update = await updater.propose_update(skills)
        if update:
            print(f"Proposed update:\\n{update}")
        ```
    """

    def __init__(
        self,
        min_confidence: float = 0.8,
        min_success_count: int = 5,
    ) -> None:
        """Initialize the updater.

        Args:
            min_confidence: Minimum confidence for inclusion.
            min_success_count: Minimum success count for inclusion.
        """
        self.min_confidence = min_confidence
        self.min_success_count = min_success_count

    def propose_update(
        self,
        patterns: list[Any],
    ) -> str | None:
        """Propose AGENTS.md update with new patterns.

        Args:
            patterns: List of patterns (Skill or FixPattern).

        Returns:
            Markdown update proposal or None.
        """
        if not patterns:
            return None

        # Filter to high-confidence patterns
        worthy = []
        for pattern in patterns:
            # Handle both Skill and FixPattern objects
            if hasattr(pattern, "confidence"):
                if (
                    pattern.confidence >= self.min_confidence
                    and getattr(pattern, "success_count", 0) >= self.min_success_count
                ):
                    worthy.append(pattern)
            elif hasattr(pattern, "success_rate"):
                if (
                    pattern.success_rate >= self.min_confidence
                    and pattern.success_count >= self.min_success_count
                ):
                    worthy.append(pattern)

        if not worthy:
            return None

        # Generate update proposal
        sections = []
        sections.append("## Learned Patterns (Auto-generated)")
        sections.append("")
        sections.append("> The following patterns were learned from execution history.")
        sections.append("> Review before adding to AGENTS.md.")
        sections.append("")

        for i, pattern in enumerate(worthy, 1):
            sections.append(self._format_pattern(pattern, i))

        return "\n".join(sections)

    def _format_pattern(self, pattern: Any, index: int) -> str:
        """Format a pattern for AGENTS.md.

        Args:
            pattern: Pattern to format.
            index: Pattern index.

        Returns:
            Formatted markdown section.
        """
        lines = []

        # Get title
        title = getattr(pattern, "title", None) or getattr(pattern, "error_pattern", "Pattern")
        lines.append(f"### {index}. {title}")
        lines.append("")

        # Get description
        description = getattr(pattern, "description", None)
        if description:
            lines.append(f"**When**: {description}")
            lines.append("")

        # Get pattern/approach
        code_pattern = getattr(pattern, "pattern", None) or getattr(pattern, "fix_approach", None)
        if code_pattern:
            lines.append("**Pattern**:")
            lines.append("```")
            lines.append(code_pattern[:500])
            lines.append("```")
            lines.append("")

        # Get rationale
        rationale = getattr(pattern, "rationale", None)
        if rationale:
            lines.append(f"**Why**: {rationale}")
            lines.append("")

        # Add stats
        success_count = getattr(pattern, "success_count", 0)
        if hasattr(pattern, "confidence"):
            lines.append(f"*Confidence: {pattern.confidence:.0%} ({success_count} successes)*")
        elif hasattr(pattern, "success_rate"):
            lines.append(f"*Success rate: {pattern.success_rate:.0%} ({success_count} successes)*")

        lines.append("")
        return "\n".join(lines)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Models
    "ExecutionContext",
    "FailurePattern",
    "FixAction",
    "FixPattern",
    "TaskError",
    # Database
    "PatternsDatabase",
    # Trackers
    "FailurePatternTracker",
    "FixPatternTracker",
    # Suggester
    "PatternSuggester",
    # Updater
    "AgentsUpdater",
]
