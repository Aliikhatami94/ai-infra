"""Adaptive planning for the Executor module.

Provides infrastructure for analyzing failures and suggesting/applying
plan fixes to the ROADMAP. Supports three modes:
- NO_ADAPT: Never modify ROADMAP, fail on errors
- SUGGEST: Suggest fixes, require human approval (default)
- AUTO_FIX: Automatically insert prerequisite tasks

Example:
    >>> from ai_infra.executor import AdaptiveMode, PlanAnalyzer
    >>>
    >>> analyzer = PlanAnalyzer(
    ...     roadmap=roadmap,
    ...     mode=AdaptiveMode.SUGGEST,
    ... )
    >>>
    >>> # Analyze failure and get suggestions
    >>> suggestions = analyzer.analyze_failure(task, result)
    >>> for suggestion in suggestions:
    ...     print(suggestion.description)
    ...
    >>> # Apply suggestion if approved
    >>> if user_approves:
    ...     analyzer.apply_suggestion(suggestions[0])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ai_infra.executor.failure import FailureCategory, FailureRecord
from ai_infra.executor.loop import ExecutionResult
from ai_infra.executor.roadmap import ParsedTask, Roadmap
from ai_infra.logging import get_logger

logger = get_logger("executor.adaptive")


class AdaptiveMode(str, Enum):
    """Planning modes for error recovery.

    Controls how the executor handles failures.

    Phase 5.7 Update:
        AdaptiveMode now controls WHETHER to retry, not HOW to fix.
        The agent figures out fixes using its existing tools (write_file,
        edit_file, terminal) via enhanced retry context.

    Behavior by mode:
        - NO_ADAPT: Fail immediately, no retry
        - SUGGEST: Show error to user, no automatic retry
        - AUTO_FIX: Retry with enhanced context, agent fixes issues
    """

    NO_ADAPT = "no_adapt"
    """Fail immediately on errors, no retry."""

    SUGGEST = "suggest"
    """Suggest fixes, require human approval (default)."""

    AUTO_FIX = "auto_fix"
    """Retry with enhanced context - agent uses its tools to fix issues."""


class SuggestionType(str, Enum):
    """Types of plan suggestions (for logging/observability).

    .. deprecated:: Phase 5.7
        These suggestion types are no longer automatically applied.
        The agent now handles all fixes using its existing tools.
        This enum is kept for logging, observability, and backwards
        compatibility only.
    """

    CREATE_INIT_FILE = "create_init_file"
    """Create a missing __init__.py file."""

    CREATE_DIRECTORY = "create_directory"
    """Create a missing directory."""

    ADD_IMPORT = "add_import"
    """Add a missing import statement."""

    INSERT_PREREQUISITE_TASK = "insert_prerequisite_task"
    """Insert a new prerequisite task before the failed task."""

    MODIFY_FILE_HINTS = "modify_file_hints"
    """Update the file hints for a task."""

    ADD_DEPENDENCY = "add_dependency"
    """Add a dependency on another task."""


class SuggestionSafety(str, Enum):
    """Safety level of a suggestion for auto-fix mode."""

    SAFE = "safe"
    """Safe to apply automatically (e.g., create __init__.py)."""

    MODERATE = "moderate"
    """Moderately safe, prefer human review."""

    UNSAFE = "unsafe"
    """Not safe to apply automatically, requires human approval."""


@dataclass
class PlanSuggestion:
    """A suggested modification to the execution plan.

    Attributes:
        suggestion_type: Type of suggestion.
        description: Human-readable description of the change.
        safety: Safety level for auto-fix mode.
        target_task_id: ID of the task this relates to.
        new_task: New task to insert, if applicable.
        file_path: File path involved, if applicable.
        file_content: Content to write, if applicable.
        import_statement: Import to add, if applicable.
        line_number: Line number to insert at in ROADMAP.
        metadata: Additional data about the suggestion.
    """

    suggestion_type: SuggestionType
    description: str
    safety: SuggestionSafety
    target_task_id: str
    new_task: ParsedTask | None = None
    file_path: Path | None = None
    file_content: str = ""
    import_statement: str = ""
    line_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "suggestion_type": self.suggestion_type.value,
            "description": self.description,
            "safety": self.safety.value,
            "target_task_id": self.target_task_id,
            "new_task": self.new_task.to_dict() if self.new_task else None,
            "file_path": str(self.file_path) if self.file_path else None,
            "file_content": self.file_content,
            "import_statement": self.import_statement,
            "line_number": self.line_number,
            "metadata": self.metadata,
        }

    def is_safe_for_auto_fix(self) -> bool:
        """Check if this suggestion can be applied in auto-fix mode."""
        return self.safety == SuggestionSafety.SAFE


@dataclass
class SuggestionResult:
    """Result of applying a suggestion.

    Attributes:
        success: Whether the suggestion was applied successfully.
        suggestion: The suggestion that was applied.
        message: Success or error message.
        changes_made: List of changes that were made.
        rollback_info: Information needed to rollback the change.
    """

    success: bool
    suggestion: PlanSuggestion
    message: str = ""
    changes_made: list[str] = field(default_factory=list)
    rollback_info: dict[str, Any] = field(default_factory=dict)


class PlanAnalyzer:
    """Analyze failures and suggest plan fixes.

    The PlanAnalyzer examines task failures and generates suggestions
    for modifying the execution plan to address the issues.

    Example:
        >>> analyzer = PlanAnalyzer(
        ...     roadmap=roadmap,
        ...     mode=AdaptiveMode.SUGGEST,
        ... )
        >>>
        >>> # After a task fails
        >>> suggestions = analyzer.analyze_failure(task, result)
        >>>
        >>> # Display suggestions to user
        >>> for s in suggestions:
        ...     print(f"💡 {s.description}")
        ...
        >>> # Apply if approved
        >>> result = analyzer.apply_suggestion(suggestions[0])
        >>> if result.success:
        ...     print("Applied fix!")
    """

    # Patterns for extracting information from error messages
    MISSING_INIT_PATTERN = re.compile(
        r"(?:Cannot|Unable to|Failed to).*(?:import|resolve).*"
        r"(?:module|package)?\s*['\"]?(\w+(?:\.\w+)*)['\"]?",
        re.IGNORECASE,
    )
    MISSING_FILE_PATTERN = re.compile(
        r"(?:File|Module) ['\"]?([^'\"]+)['\"]? (?:not found|does not exist)",
        re.IGNORECASE,
    )
    MISSING_DIR_PATTERN = re.compile(
        r"(?:Directory|Folder) ['\"]?([^'\"]+)['\"]? (?:not found|does not exist)",
        re.IGNORECASE,
    )
    IMPORT_ERROR_PATTERN = re.compile(
        r"(?:ImportError|ModuleNotFoundError):\s*(?:No module named\s+)?['\"]?(\w+(?:\.\w+)*)['\"]?",
        re.IGNORECASE,
    )

    def __init__(
        self,
        roadmap: Roadmap,
        *,
        mode: AdaptiveMode = AdaptiveMode.SUGGEST,
        project_root: Path | None = None,
        max_suggestions: int = 5,
    ) -> None:
        """Initialize the PlanAnalyzer.

        Args:
            roadmap: The parsed ROADMAP to analyze.
            mode: Adaptive planning mode.
            project_root: Root directory of the project.
            max_suggestions: Maximum suggestions to generate per failure.
        """
        self._roadmap = roadmap
        self._mode = mode
        self._project_root = project_root or Path.cwd()
        self._max_suggestions = max_suggestions
        self._applied_suggestions: list[SuggestionResult] = []

    @property
    def mode(self) -> AdaptiveMode:
        """Get the current adaptive mode."""
        return self._mode

    @mode.setter
    def mode(self, value: AdaptiveMode) -> None:
        """Set the adaptive mode."""
        self._mode = value

    @property
    def roadmap(self) -> Roadmap:
        """Get the roadmap being analyzed."""
        return self._roadmap

    @roadmap.setter
    def roadmap(self, value: Roadmap) -> None:
        """Update the roadmap."""
        self._roadmap = value

    def analyze_failure(
        self,
        task: ParsedTask,
        result: ExecutionResult,
        failure_record: FailureRecord | None = None,
    ) -> list[PlanSuggestion]:
        """Analyze a task failure and generate suggestions.

        Args:
            task: The task that failed.
            result: The execution result with failure details.
            failure_record: Optional failure record with categorization.

        Returns:
            List of suggestions to fix the failure.
        """
        if self._mode == AdaptiveMode.NO_ADAPT:
            logger.debug("no_adapt mode, skipping failure analysis")
            return []

        suggestions: list[PlanSuggestion] = []

        # Determine failure category
        category = failure_record.category if failure_record else self._infer_category(result)

        # Generate suggestions based on category
        if category == FailureCategory.IMPORT_ERROR:
            suggestions.extend(self._analyze_import_error(task, result))
        elif category == FailureCategory.FILE_NOT_FOUND:
            suggestions.extend(self._analyze_file_not_found(task, result))
        elif category == FailureCategory.CONTEXT_MISSING:
            suggestions.extend(self._analyze_context_missing(task, result))
        elif category == FailureCategory.SYNTAX_ERROR:
            suggestions.extend(self._analyze_syntax_error(task, result))
        elif category == FailureCategory.TEST_FAILURE:
            suggestions.extend(self._analyze_test_failure(task, result))

        # Always try generic analysis as fallback
        if not suggestions:
            suggestions.extend(self._analyze_generic(task, result))

        # Limit suggestions
        suggestions = suggestions[: self._max_suggestions]

        logger.info(
            "failure_analyzed",
            task_id=task.id,
            category=category.value if category else "unknown",
            suggestion_count=len(suggestions),
        )

        return suggestions

    def _infer_category(self, result: ExecutionResult) -> FailureCategory:
        """Infer failure category from execution result."""
        error_text = result.error_message.lower() if result.error_message else ""

        if "import" in error_text or "module" in error_text:
            return FailureCategory.IMPORT_ERROR
        elif "file not found" in error_text or "no such file" in error_text:
            return FailureCategory.FILE_NOT_FOUND
        elif "syntax" in error_text:
            return FailureCategory.SYNTAX_ERROR
        elif "test" in error_text and "fail" in error_text:
            return FailureCategory.TEST_FAILURE
        elif "context" in error_text or "information" in error_text:
            return FailureCategory.CONTEXT_MISSING

        return FailureCategory.UNKNOWN

    def _analyze_import_error(
        self,
        task: ParsedTask,
        result: ExecutionResult,
    ) -> list[PlanSuggestion]:
        """Analyze import errors and suggest fixes."""
        suggestions: list[PlanSuggestion] = []
        error_msg = result.error_message or ""

        # Check for missing __init__.py
        match = self.IMPORT_ERROR_PATTERN.search(error_msg)
        if match:
            module_path = match.group(1)
            # Convert module path to directory path
            dir_path = self._project_root / module_path.replace(".", "/")

            # Check if this looks like a package without __init__.py
            if dir_path.exists() and not (dir_path / "__init__.py").exists():
                suggestions.append(
                    PlanSuggestion(
                        suggestion_type=SuggestionType.CREATE_INIT_FILE,
                        description=f"Create `{dir_path / '__init__.py'}` for Python package",
                        safety=SuggestionSafety.SAFE,
                        target_task_id=task.id,
                        file_path=dir_path / "__init__.py",
                        file_content='"""Package initialization."""\n',
                        line_number=task.line_number,
                    )
                )
            elif not dir_path.exists():
                # Suggest creating the directory with __init__.py
                parent_path = self._find_src_directory(dir_path)
                if parent_path:
                    dir_path / "__init__.py"
                    suggestions.append(
                        PlanSuggestion(
                            suggestion_type=SuggestionType.CREATE_DIRECTORY,
                            description=f"Create package directory `{dir_path}`",
                            safety=SuggestionSafety.SAFE,
                            target_task_id=task.id,
                            file_path=dir_path,
                            metadata={"create_init": True},
                        )
                    )

        # Check for missing specific import
        if "from" in error_msg and "import" in error_msg:
            suggestions.append(
                PlanSuggestion(
                    suggestion_type=SuggestionType.INSERT_PREREQUISITE_TASK,
                    description="Insert task to implement missing module/function",
                    safety=SuggestionSafety.MODERATE,
                    target_task_id=task.id,
                    new_task=self._create_prerequisite_task(
                        task,
                        title="Implement missing import target",
                        description=f"Fix import error: {error_msg[:200]}",
                    ),
                )
            )

        return suggestions

    def _analyze_file_not_found(
        self,
        task: ParsedTask,
        result: ExecutionResult,
    ) -> list[PlanSuggestion]:
        """Analyze file not found errors and suggest fixes."""
        suggestions: list[PlanSuggestion] = []
        error_msg = result.error_message or ""

        # Extract file path from error
        match = self.MISSING_FILE_PATTERN.search(error_msg)
        if match:
            file_path = Path(match.group(1))
            if not file_path.is_absolute():
                file_path = self._project_root / file_path

            # Check if parent directory exists
            if not file_path.parent.exists():
                suggestions.append(
                    PlanSuggestion(
                        suggestion_type=SuggestionType.CREATE_DIRECTORY,
                        description=f"Create directory `{file_path.parent}`",
                        safety=SuggestionSafety.SAFE,
                        target_task_id=task.id,
                        file_path=file_path.parent,
                    )
                )

            # If it's a Python file, might need __init__.py
            if file_path.suffix == ".py":
                init_path = file_path.parent / "__init__.py"
                if not init_path.exists():
                    suggestions.append(
                        PlanSuggestion(
                            suggestion_type=SuggestionType.CREATE_INIT_FILE,
                            description=f"Create `{init_path}` for Python package",
                            safety=SuggestionSafety.SAFE,
                            target_task_id=task.id,
                            file_path=init_path,
                            file_content='"""Package initialization."""\n',
                        )
                    )

        # Check for directory not found
        match = self.MISSING_DIR_PATTERN.search(error_msg)
        if match:
            dir_path = Path(match.group(1))
            if not dir_path.is_absolute():
                dir_path = self._project_root / dir_path

            suggestions.append(
                PlanSuggestion(
                    suggestion_type=SuggestionType.CREATE_DIRECTORY,
                    description=f"Create directory `{dir_path}`",
                    safety=SuggestionSafety.SAFE,
                    target_task_id=task.id,
                    file_path=dir_path,
                )
            )

        return suggestions

    def _analyze_context_missing(
        self,
        task: ParsedTask,
        result: ExecutionResult,
    ) -> list[PlanSuggestion]:
        """Analyze context missing errors and suggest fixes."""
        suggestions: list[PlanSuggestion] = []

        # Suggest adding file hints to the task
        suggestions.append(
            PlanSuggestion(
                suggestion_type=SuggestionType.MODIFY_FILE_HINTS,
                description="Add more specific file hints to the task",
                safety=SuggestionSafety.MODERATE,
                target_task_id=task.id,
                metadata={"current_hints": task.file_hints},
            )
        )

        # Suggest breaking down the task
        suggestions.append(
            PlanSuggestion(
                suggestion_type=SuggestionType.INSERT_PREREQUISITE_TASK,
                description="Break down into smaller, more focused tasks",
                safety=SuggestionSafety.MODERATE,
                target_task_id=task.id,
                new_task=self._create_prerequisite_task(
                    task,
                    title="Gather context for implementation",
                    description="Research existing code patterns and gather necessary context",
                ),
            )
        )

        return suggestions

    def _analyze_syntax_error(
        self,
        task: ParsedTask,
        result: ExecutionResult,
    ) -> list[PlanSuggestion]:
        """Analyze syntax errors and suggest fixes."""
        suggestions: list[PlanSuggestion] = []

        # Syntax errors need human review
        suggestions.append(
            PlanSuggestion(
                suggestion_type=SuggestionType.INSERT_PREREQUISITE_TASK,
                description="Add code review task before implementation",
                safety=SuggestionSafety.UNSAFE,
                target_task_id=task.id,
                new_task=self._create_prerequisite_task(
                    task,
                    title="Review and validate code structure",
                    description="Manual review of code structure to prevent syntax errors",
                ),
            )
        )

        return suggestions

    def _analyze_test_failure(
        self,
        task: ParsedTask,
        result: ExecutionResult,
    ) -> list[PlanSuggestion]:
        """Analyze test failures and suggest fixes."""
        suggestions: list[PlanSuggestion] = []

        # Suggest adding test fixture setup
        suggestions.append(
            PlanSuggestion(
                suggestion_type=SuggestionType.INSERT_PREREQUISITE_TASK,
                description="Add task to set up test fixtures",
                safety=SuggestionSafety.MODERATE,
                target_task_id=task.id,
                new_task=self._create_prerequisite_task(
                    task,
                    title="Set up test fixtures and mocks",
                    description="Create necessary test fixtures and mock dependencies",
                ),
            )
        )

        return suggestions

    def _analyze_generic(
        self,
        task: ParsedTask,
        result: ExecutionResult,
    ) -> list[PlanSuggestion]:
        """Generic analysis when no specific category matches."""
        suggestions: list[PlanSuggestion] = []

        # Check for common patterns in error messages
        error_msg = result.error_message or ""

        # Missing __init__.py is very common
        if "__init__" in error_msg.lower() or "package" in error_msg.lower():
            # Find likely package directories from file hints
            for hint in task.file_hints:
                hint_path = self._project_root / hint
                if hint_path.is_dir() or "/" in hint:
                    dir_path = hint_path if hint_path.is_dir() else hint_path.parent
                    init_path = dir_path / "__init__.py"
                    if not init_path.exists():
                        suggestions.append(
                            PlanSuggestion(
                                suggestion_type=SuggestionType.CREATE_INIT_FILE,
                                description=f"Create `{init_path}`",
                                safety=SuggestionSafety.SAFE,
                                target_task_id=task.id,
                                file_path=init_path,
                                file_content='"""Package initialization."""\n',
                            )
                        )

        return suggestions

    def _find_src_directory(self, path: Path) -> Path | None:
        """Find the src directory for a path."""
        current = path
        while current != self._project_root and current.parent != current:
            if current.name == "src":
                return current
            current = current.parent
        return None

    def _create_prerequisite_task(
        self,
        original_task: ParsedTask,
        *,
        title: str,
        description: str,
    ) -> ParsedTask:
        """Create a prerequisite task that should run before the original."""
        # Generate new task ID (insert before original)
        task_id = self._generate_prerequisite_id(original_task.id)

        return ParsedTask(
            id=task_id,
            title=title,
            description=description,
            file_hints=original_task.file_hints.copy(),
            dependencies=[],  # No dependencies for prerequisite
            line_number=original_task.line_number,  # Insert before original
        )

    def _generate_prerequisite_id(self, original_id: str) -> str:
        """Generate an ID for a prerequisite task.

        For task 1.2.3, generates 1.2.2.1 (or 1.2.2.2 if 1.2.2.1 exists).
        """
        parts = original_id.split(".")
        if len(parts) >= 3:
            # Decrement the last part and add .1
            last = int(parts[-1])
            if last > 1:
                return ".".join(parts[:-1] + [str(last - 1), "1"])
        # Fallback: add .pre suffix
        return f"{original_id}.pre"

    def apply_suggestion(
        self,
        suggestion: PlanSuggestion,
        *,
        force: bool = False,
    ) -> SuggestionResult:
        """Apply a suggestion to the project/roadmap.

        .. deprecated:: Phase 5.7
            Hardcoded suggestion application is disabled. The agent now
            handles all fixes using its existing tools. This method logs
            the suggestion but does not execute it unless force=True AND
            legacy_apply=True is set in the suggestion metadata.

        Args:
            suggestion: The suggestion to apply.
            force: Force apply even in NO_ADAPT mode (legacy compatibility).

        Returns:
            Result indicating the suggestion was logged (not applied).
        """
        if self._mode == AdaptiveMode.NO_ADAPT and not force:
            return SuggestionResult(
                success=False,
                suggestion=suggestion,
                message="NO_ADAPT mode: suggestions cannot be applied",
            )

        if self._mode == AdaptiveMode.SUGGEST and not force:
            # In suggest mode, we just return what would be done
            return SuggestionResult(
                success=True,
                suggestion=suggestion,
                message="Suggestion ready for approval (SUGGEST mode)",
                changes_made=[],
            )

        # Phase 5.7: AUTO_FIX mode no longer applies hardcoded fixes
        # The agent figures out fixes using its existing tools via retry context
        # Log for observability but don't execute
        logger.info(
            "suggestion_logged_not_applied",
            suggestion_type=suggestion.suggestion_type.value,
            description=suggestion.description,
            reason="Phase 5.7: Agent handles fixes via retry context",
        )

        # Check for legacy_apply flag (for backwards compatibility if needed)
        if suggestion.metadata.get("legacy_apply") and force:
            logger.warning(
                "legacy_apply_requested",
                suggestion_type=suggestion.suggestion_type.value,
                reason="Using deprecated hardcoded fix - consider agent-driven recovery",
            )
            return self._apply_suggestion_legacy(suggestion)

        return SuggestionResult(
            success=True,
            suggestion=suggestion,
            message="Phase 5.7: Suggestion logged for observability (agent handles fixes)",
            changes_made=[],
        )

    def _apply_suggestion_legacy(
        self,
        suggestion: PlanSuggestion,
    ) -> SuggestionResult:
        """Legacy apply method for backwards compatibility.

        .. deprecated:: Phase 5.7
            Only used when legacy_apply=True is explicitly set.
        """
        try:
            if suggestion.suggestion_type == SuggestionType.CREATE_INIT_FILE:
                return self._apply_create_init_file(suggestion)
            elif suggestion.suggestion_type == SuggestionType.CREATE_DIRECTORY:
                return self._apply_create_directory(suggestion)
            elif suggestion.suggestion_type == SuggestionType.ADD_IMPORT:
                return self._apply_add_import(suggestion)
            elif suggestion.suggestion_type == SuggestionType.INSERT_PREREQUISITE_TASK:
                return self._apply_insert_task(suggestion)
            elif suggestion.suggestion_type == SuggestionType.MODIFY_FILE_HINTS:
                return self._apply_modify_file_hints(suggestion)
            elif suggestion.suggestion_type == SuggestionType.ADD_DEPENDENCY:
                return self._apply_add_dependency(suggestion)
            else:
                return SuggestionResult(
                    success=False,
                    suggestion=suggestion,
                    message=f"Unsupported suggestion type: {suggestion.suggestion_type}",
                )
        except Exception as e:
            logger.exception("suggestion_apply_failed", suggestion=suggestion.to_dict())
            return SuggestionResult(
                success=False,
                suggestion=suggestion,
                message=f"Error applying suggestion: {e}",
            )

    def _apply_create_init_file(self, suggestion: PlanSuggestion) -> SuggestionResult:
        """Apply CREATE_INIT_FILE suggestion."""
        if not suggestion.file_path:
            return SuggestionResult(
                success=False,
                suggestion=suggestion,
                message="No file path specified",
            )

        file_path = Path(suggestion.file_path)

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the __init__.py
        content = suggestion.file_content or '"""Package initialization."""\n'
        file_path.write_text(content)

        logger.info("created_init_file", path=str(file_path))

        return SuggestionResult(
            success=True,
            suggestion=suggestion,
            message=f"Created {file_path}",
            changes_made=[f"Created file: {file_path}"],
            rollback_info={"delete_file": str(file_path)},
        )

    def _apply_create_directory(self, suggestion: PlanSuggestion) -> SuggestionResult:
        """Apply CREATE_DIRECTORY suggestion."""
        if not suggestion.file_path:
            return SuggestionResult(
                success=False,
                suggestion=suggestion,
                message="No directory path specified",
            )

        dir_path = Path(suggestion.file_path)
        changes = []

        # Create the directory
        dir_path.mkdir(parents=True, exist_ok=True)
        changes.append(f"Created directory: {dir_path}")

        # Create __init__.py if requested
        if suggestion.metadata.get("create_init"):
            init_path = dir_path / "__init__.py"
            init_path.write_text('"""Package initialization."""\n')
            changes.append(f"Created file: {init_path}")

        logger.info("created_directory", path=str(dir_path))

        return SuggestionResult(
            success=True,
            suggestion=suggestion,
            message=f"Created {dir_path}",
            changes_made=changes,
            rollback_info={"delete_directory": str(dir_path)},
        )

    def _apply_add_import(self, suggestion: PlanSuggestion) -> SuggestionResult:
        """Apply ADD_IMPORT suggestion."""
        # Adding imports requires modifying source files - moderate risk
        if not suggestion.file_path or not suggestion.import_statement:
            return SuggestionResult(
                success=False,
                suggestion=suggestion,
                message="Missing file path or import statement",
            )

        file_path = Path(suggestion.file_path)
        if not file_path.exists():
            return SuggestionResult(
                success=False,
                suggestion=suggestion,
                message=f"File does not exist: {file_path}",
            )

        # Read current content
        content = file_path.read_text()
        original_content = content

        # Add import at the beginning (after docstring if present)
        import_stmt = suggestion.import_statement
        if not import_stmt.endswith("\n"):
            import_stmt += "\n"

        # Find insertion point (after module docstring)
        lines = content.split("\n")
        insert_idx = 0

        # Skip leading docstring
        if lines and lines[0].startswith('"""'):
            for i, line in enumerate(lines):
                if i > 0 and '"""' in line:
                    insert_idx = i + 1
                    break
        elif lines and lines[0].startswith("'''"):
            for i, line in enumerate(lines):
                if i > 0 and "'''" in line:
                    insert_idx = i + 1
                    break

        # Insert the import
        lines.insert(insert_idx, import_stmt.rstrip())
        new_content = "\n".join(lines)

        file_path.write_text(new_content)

        logger.info("added_import", path=str(file_path), import_stmt=import_stmt.strip())

        return SuggestionResult(
            success=True,
            suggestion=suggestion,
            message=f"Added import to {file_path}",
            changes_made=[f"Added import: {import_stmt.strip()}"],
            rollback_info={
                "restore_file": str(file_path),
                "original_content": original_content,
            },
        )

    def _apply_insert_task(self, suggestion: PlanSuggestion) -> SuggestionResult:
        """Apply INSERT_PREREQUISITE_TASK suggestion."""
        if not suggestion.new_task:
            return SuggestionResult(
                success=False,
                suggestion=suggestion,
                message="No new task specified",
            )

        if not self._roadmap.path:
            return SuggestionResult(
                success=False,
                suggestion=suggestion,
                message="ROADMAP has no path, cannot modify",
            )

        roadmap_path = Path(self._roadmap.path)
        if not roadmap_path.exists():
            return SuggestionResult(
                success=False,
                suggestion=suggestion,
                message=f"ROADMAP file not found: {roadmap_path}",
            )

        # Read current content
        content = roadmap_path.read_text()
        lines = content.split("\n")
        original_content = content

        # Find line number to insert
        insert_line = suggestion.line_number or suggestion.new_task.line_number
        if not insert_line or insert_line > len(lines):
            return SuggestionResult(
                success=False,
                suggestion=suggestion,
                message="Invalid line number for task insertion",
            )

        # Format the new task as Markdown
        new_task_md = self._format_task_markdown(suggestion.new_task)

        # Insert before the target task
        lines.insert(insert_line - 1, new_task_md)

        # Write back
        roadmap_path.write_text("\n".join(lines))

        logger.info(
            "inserted_task",
            roadmap=str(roadmap_path),
            task_id=suggestion.new_task.id,
        )

        return SuggestionResult(
            success=True,
            suggestion=suggestion,
            message=f"Inserted task {suggestion.new_task.id} in ROADMAP",
            changes_made=[f"Inserted task: {suggestion.new_task.title}"],
            rollback_info={
                "restore_file": str(roadmap_path),
                "original_content": original_content,
            },
        )

    def _apply_modify_file_hints(self, suggestion: PlanSuggestion) -> SuggestionResult:
        """Apply MODIFY_FILE_HINTS suggestion."""
        # This requires modifying ROADMAP.md - moderate complexity
        return SuggestionResult(
            success=False,
            suggestion=suggestion,
            message="MODIFY_FILE_HINTS not yet implemented - requires manual update",
        )

    def _apply_add_dependency(self, suggestion: PlanSuggestion) -> SuggestionResult:
        """Apply ADD_DEPENDENCY suggestion."""
        # This requires modifying ROADMAP.md structure
        return SuggestionResult(
            success=False,
            suggestion=suggestion,
            message="ADD_DEPENDENCY not yet implemented - requires manual update",
        )

    def _format_task_markdown(self, task: ParsedTask) -> str:
        """Format a task as Markdown for ROADMAP insertion."""
        lines = [f"- [ ] **{task.title}**"]
        if task.description:
            lines.append(f"  {task.description}")
        if task.file_hints:
            hints = ", ".join(f"`{h}`" for h in task.file_hints)
            lines.append(f"  **Files**: {hints}")
        return "\n".join(lines)

    def can_auto_fix(self, suggestions: list[PlanSuggestion]) -> list[PlanSuggestion]:
        """Filter suggestions that are safe for auto-fix mode.

        Args:
            suggestions: List of suggestions to filter.

        Returns:
            List of suggestions that are safe for automatic application.
        """
        return [s for s in suggestions if s.is_safe_for_auto_fix()]

    def get_applied_suggestions(self) -> list[SuggestionResult]:
        """Get all suggestions that have been applied."""
        return self._applied_suggestions.copy()

    def format_suggestion_prompt(self, suggestion: PlanSuggestion) -> str:
        """Format a suggestion for user display.

        Args:
            suggestion: The suggestion to format.

        Returns:
            Formatted string for terminal display.
        """
        lines = [
            f"Suggested fix: {suggestion.description}",
            f"  Type: {suggestion.suggestion_type.value}",
            f"  Safety: {suggestion.safety.value}",
        ]

        if suggestion.file_path:
            lines.append(f"  File: {suggestion.file_path}")

        if suggestion.new_task:
            lines.append(f"  New task: {suggestion.new_task.title}")

        return "\n".join(lines)


async def analyze_failure_for_plan_fix(
    task: ParsedTask,
    result: ExecutionResult,
    *,
    roadmap: Roadmap | None = None,
    mode: AdaptiveMode = AdaptiveMode.SUGGEST,
    project_root: Path | None = None,
) -> list[PlanSuggestion]:
    """Convenience function to analyze a failure and get suggestions.

    Args:
        task: The task that failed.
        result: The execution result.
        roadmap: Optional roadmap for context.
        mode: Adaptive planning mode.
        project_root: Project root directory.

    Returns:
        List of suggested fixes.
    """
    if mode == AdaptiveMode.NO_ADAPT:
        return []

    # Create a minimal roadmap if not provided
    if roadmap is None:
        roadmap = Roadmap(path="", title="", phases=[])

    analyzer = PlanAnalyzer(
        roadmap=roadmap,
        mode=mode,
        project_root=project_root,
    )

    return analyzer.analyze_failure(task, result)
