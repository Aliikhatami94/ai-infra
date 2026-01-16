"""Subagent output validation.

Phase 16.5.12.3 of EXECUTOR_5.md: Validates subagent output for
completeness and quality, enabling retry logic when validation fails.

This module provides:
- SubagentOutputValidator: Validates output against expectations
- ValidationResult: Contains validation status and issues

Example:
    ```python
    from ai_infra.executor.agents.validator import SubagentOutputValidator
    from ai_infra.executor.agents.base import SubAgentResult
    from ai_infra.executor.todolist import TodoItem
    from pathlib import Path

    validator = SubagentOutputValidator()
    result = SubAgentResult(success=True, files_created=["src/user.py"])
    task = TodoItem(id=1, title="Create src/user.py", description="")

    validation = validator.validate(task, result, Path("/project"))
    if not validation.valid:
        print(f"Issues: {validation.issues}")
    ```
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.agents.base import SubAgentResult
    from ai_infra.executor.todolist import TodoItem

__all__ = [
    "SubagentOutputValidator",
    "ValidationResult",
]

logger = get_logger("executor.agents.validator")


# =============================================================================
# Constants
# =============================================================================

# Minimum file size to be considered non-empty (bytes)
MIN_FILE_SIZE = 10

# Maximum test execution time (seconds)
MAX_TEST_TIMEOUT = 30

# File patterns to validate syntax
PYTHON_EXTENSIONS = {".py", ".pyi"}
JAVASCRIPT_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ValidationResult:
    """Result of subagent output validation.

    Phase 16.5.12.3.2: Contains validation status and any issues found.

    Attributes:
        valid: Whether the output passed validation.
        issues: List of issues found during validation.
        score: Quality score from 0.0 to 1.0.
        files_validated: Number of files validated.
        tests_passed: Whether tests passed (None if not checked).
    """

    valid: bool = True
    issues: list[str] = field(default_factory=list)
    score: float = 1.0
    files_validated: int = 0
    tests_passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "issues": self.issues,
            "score": self.score,
            "files_validated": self.files_validated,
            "tests_passed": self.tests_passed,
        }

    def add_issue(self, issue: str, penalty: float = 0.1) -> None:
        """Add an issue and reduce score.

        Args:
            issue: Description of the issue.
            penalty: Score penalty (0.0 to 1.0).
        """
        self.issues.append(issue)
        self.score = max(0.0, self.score - penalty)
        self.valid = False


# =============================================================================
# Validator Class
# =============================================================================


class SubagentOutputValidator:
    """Validates subagent output for completeness and quality.

    Phase 16.5.12.3: Checks that subagent produced expected output,
    validates syntax, and optionally runs tests.

    Example:
        ```python
        validator = SubagentOutputValidator()
        validation = validator.validate(task, result, workspace)

        if not validation.valid:
            # Retry with feedback
            feedback = "\\n".join(f"- {i}" for i in validation.issues)
            enhanced_prompt = f"{prompt}\\n\\nPREVIOUS ISSUES:\\n{feedback}"
        ```
    """

    def __init__(
        self,
        check_syntax: bool = True,
        run_tests: bool = False,
        min_file_size: int = MIN_FILE_SIZE,
    ) -> None:
        """Initialize validator.

        Args:
            check_syntax: Whether to validate Python syntax.
            run_tests: Whether to run pytest for test files.
            min_file_size: Minimum acceptable file size in bytes.
        """
        self.check_syntax = check_syntax
        self.run_tests = run_tests
        self.min_file_size = min_file_size

    def validate(
        self,
        task: TodoItem,
        result: SubAgentResult,
        workspace: Path,
    ) -> ValidationResult:
        """Validate subagent completed the task correctly.

        Phase 16.5.12.3.2: Main validation entry point.

        Args:
            task: The task that was executed.
            result: SubAgentResult from the subagent.
            workspace: Workspace root path.

        Returns:
            ValidationResult with status and issues.
        """
        validation = ValidationResult()

        # Check if subagent reported success
        if not result.success:
            validation.add_issue(
                f"Subagent reported failure: {result.error or 'unknown error'}",
                penalty=0.5,
            )
            return validation

        # Get all files from result
        all_files = list(set(result.files_created + result.files_modified))

        # Check expected files exist
        expected_files = self._extract_expected_files(task)
        for expected in expected_files:
            found = any(expected in f for f in all_files)
            if not found:
                # Check if file exists in workspace
                if not (workspace / expected).exists():
                    validation.add_issue(
                        f"Expected file not created: {expected}",
                        penalty=0.3,
                    )

        # Validate each file
        for filepath in all_files:
            full_path = workspace / filepath

            # Check file exists
            if not full_path.exists():
                validation.add_issue(
                    f"Claimed file does not exist: {filepath}",
                    penalty=0.2,
                )
                continue

            validation.files_validated += 1

            # Check file is not empty
            if not self._check_file_nonempty(full_path):
                validation.add_issue(
                    f"File is empty or too small: {filepath}",
                    penalty=0.15,
                )
                continue

            # Check syntax for code files
            if self.check_syntax:
                if filepath.endswith(tuple(PYTHON_EXTENSIONS)):
                    syntax_ok, error = self._check_python_syntax(full_path)
                    if not syntax_ok:
                        validation.add_issue(
                            f"Syntax error in {filepath}: {error}",
                            penalty=0.3,
                        )

        # Run tests for test files
        if self.run_tests:
            test_files = [f for f in all_files if "test" in f.lower()]
            if test_files:
                tests_passed = self._run_tests(workspace, test_files)
                validation.tests_passed = tests_passed
                if not tests_passed:
                    validation.add_issue(
                        "Created tests do not pass",
                        penalty=0.4,
                    )

        # Log validation result
        if validation.valid:
            logger.debug(
                f"Validation passed: {validation.files_validated} files, "
                f"score={validation.score:.2f}"
            )
        else:
            logger.warning(
                f"Validation failed: {len(validation.issues)} issues, score={validation.score:.2f}"
            )

        return validation

    def _extract_expected_files(self, task: TodoItem) -> list[str]:
        """Extract expected file paths from task title/description.

        Phase 16.5.12.3.3: Parses task to identify expected outputs.

        Args:
            task: The task to analyze.

        Returns:
            List of expected file paths.
        """
        expected: list[str] = []
        text = f"{task.title} {task.description or ''}"

        # Match file paths in task text
        # Patterns: src/file.py, tests/test_file.py, file.py, etc.
        patterns = [
            r'(?:create|add|write|implement)\s+(?:a\s+)?[`\'"]?([a-zA-Z0-9_/.-]+\.py)',
            r'(?:create|add|write)\s+(?:a\s+)?test\s+file\s+[`\'"]?([a-zA-Z0-9_/.-]+)',
            r'[`\'"]([a-zA-Z0-9_/.-]+\.(?:py|ts|js|tsx|jsx))[`\'"]',
            r"\b((?:src|tests|lib)/[a-zA-Z0-9_/.-]+\.(?:py|ts|js))\b",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and len(match) > 3:
                    # Clean up path
                    clean_path = match.strip("'\"` ")
                    if clean_path and not clean_path.startswith("/"):
                        expected.append(clean_path)

        return list(set(expected))

    def _check_file_nonempty(self, filepath: Path) -> bool:
        """Check that file is not empty.

        Args:
            filepath: Path to the file.

        Returns:
            True if file has content, False otherwise.
        """
        try:
            size = filepath.stat().st_size
            return size >= self.min_file_size
        except Exception:
            return False

    def _check_python_syntax(self, filepath: Path) -> tuple[bool, str]:
        """Check Python file syntax validity.

        Phase 16.5.12.3.4: Uses compile() to validate syntax.

        Args:
            filepath: Path to Python file.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            compile(content, str(filepath), "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def _run_tests(
        self,
        workspace: Path,
        test_files: list[str],
    ) -> bool:
        """Run pytest on test files.

        Phase 16.5.12.3.5: Executes tests to verify they pass.

        Args:
            workspace: Workspace root path.
            test_files: List of test file paths.

        Returns:
            True if tests pass, False otherwise.
        """
        if not test_files:
            return True

        try:
            # Run pytest with minimal output
            result = subprocess.run(
                ["python", "-m", "pytest", "-q", "--tb=no"] + test_files,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=MAX_TEST_TIMEOUT,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning("Test execution timed out")
            return False
        except Exception as e:
            logger.warning(f"Could not run tests: {e}")
            # Don't fail validation if we can't run tests
            return True

    def format_retry_feedback(self, validation: ValidationResult) -> str:
        """Format validation issues as retry feedback.

        Phase 16.5.12.3.6: Creates feedback prompt for retry attempts.

        Args:
            validation: Validation result with issues.

        Returns:
            Formatted feedback string for injection into retry prompt.
        """
        if validation.valid:
            return ""

        feedback_lines = [
            "## Previous Attempt Issues",
            "",
            "Your previous attempt had the following issues that need to be fixed:",
            "",
        ]

        for i, issue in enumerate(validation.issues, 1):
            feedback_lines.append(f"{i}. {issue}")

        feedback_lines.extend(
            [
                "",
                "Please address ALL issues above and ensure:",
                "- All expected files are created",
                "- Files contain valid syntax",
                "- Files are not empty",
                "- Tests pass (if applicable)",
            ]
        )

        return "\n".join(feedback_lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_subagent_output(
    task: TodoItem,
    result: SubAgentResult,
    workspace: Path,
    check_syntax: bool = True,
    run_tests: bool = False,
) -> ValidationResult:
    """Convenience function to validate subagent output.

    Args:
        task: The task that was executed.
        result: SubAgentResult from the subagent.
        workspace: Workspace root path.
        check_syntax: Whether to validate syntax.
        run_tests: Whether to run tests.

    Returns:
        ValidationResult with status and issues.
    """
    validator = SubagentOutputValidator(
        check_syntax=check_syntax,
        run_tests=run_tests,
    )
    return validator.validate(task, result, workspace)


def needs_retry(
    validation: ValidationResult,
    retry_count: int = 0,
    max_retries: int = 2,
) -> bool:
    """Check if validation result warrants a retry.

    Phase 16.5.12.3.6: Determines if retry should be attempted.

    Args:
        validation: Validation result.
        retry_count: Current retry count.
        max_retries: Maximum allowed retries.

    Returns:
        True if retry should be attempted.
    """
    if validation.valid:
        return False

    if retry_count >= max_retries:
        return False

    # Only retry for significant issues (score < 0.7)
    if validation.score >= 0.7:
        return False

    # Don't retry for fundamental failures
    critical_issues = [
        "Subagent reported failure",
        "Expected file not created",
    ]
    has_critical = any(
        any(crit in issue for crit in critical_issues) for issue in validation.issues
    )

    return has_critical
