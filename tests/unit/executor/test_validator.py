"""Tests for Phase 16.5.12.3: SubagentOutputValidator.

Tests the output validation system that checks subagent results
and enables retry logic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ai_infra.executor.agents.base import SubAgentResult
from ai_infra.executor.agents.validator import (
    SubagentOutputValidator,
    ValidationResult,
    needs_retry,
    validate_subagent_output,
)
from ai_infra.executor.todolist import TodoItem

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_workspace() -> Path:
    """Create a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "src").mkdir()
        yield workspace


@pytest.fixture
def validator() -> SubagentOutputValidator:
    """Create a validator instance."""
    return SubagentOutputValidator(
        check_syntax=True,
        run_tests=False,
    )


@pytest.fixture
def sample_task() -> TodoItem:
    """Create a sample task."""
    return TodoItem(
        id=1,
        title="Create src/user.py with User class",
        description="Implement a User class",
    )


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_valid(self) -> None:
        """Test ValidationResult defaults to valid."""
        result = ValidationResult()

        assert result.valid is True
        assert result.score == 1.0
        assert result.issues == []

    def test_add_issue(self) -> None:
        """Test adding issues reduces score."""
        result = ValidationResult()

        result.add_issue("File not found", penalty=0.3)

        assert result.valid is False
        assert result.score == pytest.approx(0.7)
        assert "File not found" in result.issues

    def test_add_multiple_issues(self) -> None:
        """Test multiple issues accumulate."""
        result = ValidationResult()

        result.add_issue("Issue 1", penalty=0.2)
        result.add_issue("Issue 2", penalty=0.3)

        assert len(result.issues) == 2
        assert result.score == pytest.approx(0.5)

    def test_score_doesnt_go_negative(self) -> None:
        """Test score is bounded at 0."""
        result = ValidationResult()

        for i in range(20):
            result.add_issue(f"Issue {i}", penalty=0.1)

        assert result.score >= 0.0

    def test_to_dict(self) -> None:
        """Test serialization."""
        result = ValidationResult(
            valid=False,
            issues=["Error 1"],
            score=0.7,
            files_validated=3,
        )

        data = result.to_dict()

        assert data["valid"] is False
        assert data["score"] == 0.7
        assert data["files_validated"] == 3


# =============================================================================
# SubagentOutputValidator Tests
# =============================================================================


class TestSubagentOutputValidator:
    """Tests for SubagentOutputValidator class."""

    def test_validate_success_with_existing_files(
        self,
        validator: SubagentOutputValidator,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test validation passes when files exist."""
        # Create the expected file
        user_file = temp_workspace / "src" / "user.py"
        user_file.write_text("class User:\n    pass\n")

        result = SubAgentResult(
            success=True,
            files_created=["src/user.py"],
        )

        validation = validator.validate(sample_task, result, temp_workspace)

        assert validation.valid is True
        assert validation.score >= 0.9
        assert validation.files_validated == 1

    def test_validate_fails_when_file_missing(
        self,
        validator: SubagentOutputValidator,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test validation fails when claimed file doesn't exist."""
        result = SubAgentResult(
            success=True,
            files_created=["src/user.py"],  # Doesn't exist
        )

        validation = validator.validate(sample_task, result, temp_workspace)

        assert validation.valid is False
        assert any("does not exist" in issue for issue in validation.issues)

    def test_validate_fails_on_subagent_failure(
        self,
        validator: SubagentOutputValidator,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test validation fails when subagent reports failure."""
        result = SubAgentResult(
            success=False,
            error="Execution failed",
        )

        validation = validator.validate(sample_task, result, temp_workspace)

        assert validation.valid is False
        assert any("failure" in issue.lower() for issue in validation.issues)

    def test_validate_fails_on_empty_file(
        self,
        validator: SubagentOutputValidator,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test validation fails when file is empty."""
        # Create empty file
        user_file = temp_workspace / "src" / "user.py"
        user_file.write_text("")

        result = SubAgentResult(
            success=True,
            files_created=["src/user.py"],
        )

        validation = validator.validate(sample_task, result, temp_workspace)

        assert validation.valid is False
        assert any("empty" in issue.lower() for issue in validation.issues)


class TestSyntaxValidation:
    """Tests for Python syntax validation."""

    def test_valid_python_syntax(
        self,
        validator: SubagentOutputValidator,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test validation passes for valid Python."""
        user_file = temp_workspace / "src" / "user.py"
        user_file.write_text("""
class User:
    def __init__(self, name: str) -> None:
        self.name = name
""")

        result = SubAgentResult(
            success=True,
            files_created=["src/user.py"],
        )

        validation = validator.validate(sample_task, result, temp_workspace)

        assert validation.valid is True

    def test_invalid_python_syntax(
        self,
        validator: SubagentOutputValidator,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test validation fails for invalid Python."""
        user_file = temp_workspace / "src" / "user.py"
        user_file.write_text("""
class User:
    def __init__(self name):  # Missing comma
        self.name = name
""")

        result = SubAgentResult(
            success=True,
            files_created=["src/user.py"],
        )

        validation = validator.validate(sample_task, result, temp_workspace)

        assert validation.valid is False
        assert any("syntax" in issue.lower() for issue in validation.issues)

    def test_syntax_check_can_be_disabled(
        self,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test syntax checking can be disabled."""
        validator = SubagentOutputValidator(check_syntax=False)

        user_file = temp_workspace / "src" / "user.py"
        user_file.write_text("this is not valid python {{{{")

        result = SubAgentResult(
            success=True,
            files_created=["src/user.py"],
        )

        validation = validator.validate(sample_task, result, temp_workspace)

        # Should pass since syntax check is disabled
        assert validation.valid is True


class TestExtractExpectedFiles:
    """Tests for extracting expected files from tasks."""

    def test_extract_explicit_file_path(
        self,
        validator: SubagentOutputValidator,
    ) -> None:
        """Test extraction of explicit file path."""
        task = TodoItem(
            id=1,
            title="Create src/user.py with User class",
            description="",
        )

        expected = validator._extract_expected_files(task)

        assert "src/user.py" in expected

    def test_extract_quoted_file_path(
        self,
        validator: SubagentOutputValidator,
    ) -> None:
        """Test extraction of quoted file path."""
        task = TodoItem(
            id=1,
            title="Add code to `tests/test_auth.py`",
            description="",
        )

        expected = validator._extract_expected_files(task)

        assert "tests/test_auth.py" in expected

    def test_extract_from_description(
        self,
        validator: SubagentOutputValidator,
    ) -> None:
        """Test extraction from description."""
        task = TodoItem(
            id=1,
            title="Create user module",
            description="Create the file src/models/user.py with User class",
        )

        expected = validator._extract_expected_files(task)

        assert any("user.py" in f for f in expected)


class TestRetryFeedback:
    """Tests for retry feedback formatting."""

    def test_format_retry_feedback(
        self,
        validator: SubagentOutputValidator,
    ) -> None:
        """Test formatting validation issues for retry."""
        validation = ValidationResult(
            valid=False,
            issues=[
                "Expected file not created: src/user.py",
                "Syntax error in tests/test_user.py",
            ],
            score=0.4,
        )

        feedback = validator.format_retry_feedback(validation)

        assert "Previous Attempt Issues" in feedback
        assert "src/user.py" in feedback
        assert "Syntax error" in feedback

    def test_no_feedback_when_valid(
        self,
        validator: SubagentOutputValidator,
    ) -> None:
        """Test no feedback for valid results."""
        validation = ValidationResult(valid=True)

        feedback = validator.format_retry_feedback(validation)

        assert feedback == ""


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestValidateSubagentOutput:
    """Tests for validate_subagent_output convenience function."""

    def test_convenience_function(
        self,
        temp_workspace: Path,
        sample_task: TodoItem,
    ) -> None:
        """Test the convenience function works."""
        user_file = temp_workspace / "src" / "user.py"
        user_file.write_text("class User:\n    pass\n")

        result = SubAgentResult(
            success=True,
            files_created=["src/user.py"],
        )

        validation = validate_subagent_output(
            task=sample_task,
            result=result,
            workspace=temp_workspace,
        )

        assert validation.valid is True


class TestNeedsRetry:
    """Tests for needs_retry function."""

    def test_no_retry_when_valid(self) -> None:
        """Test no retry for valid results."""
        validation = ValidationResult(valid=True)

        assert needs_retry(validation) is False

    def test_no_retry_at_max_retries(self) -> None:
        """Test no retry when max retries reached."""
        validation = ValidationResult(valid=False, score=0.3)

        assert needs_retry(validation, retry_count=2, max_retries=2) is False

    def test_no_retry_for_high_score(self) -> None:
        """Test no retry for high score (minor issues)."""
        validation = ValidationResult(valid=False, score=0.8)

        assert needs_retry(validation) is False

    def test_retry_for_critical_issues(self) -> None:
        """Test retry for critical issues."""
        validation = ValidationResult(
            valid=False,
            issues=["Expected file not created: src/user.py"],
            score=0.5,
        )

        assert needs_retry(validation, retry_count=0) is True

    def test_retry_limit_respected(self) -> None:
        """Test retry count is respected."""
        validation = ValidationResult(
            valid=False,
            issues=["Expected file not created: src/user.py"],
            score=0.3,
        )

        assert needs_retry(validation, retry_count=0, max_retries=1) is True
        assert needs_retry(validation, retry_count=1, max_retries=1) is False
