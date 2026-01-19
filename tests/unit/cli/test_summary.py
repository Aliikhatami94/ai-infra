"""Unit tests for ai_infra.cli.summary module.

Phase 16.6.4 of EXECUTOR_6.md: Execution Summary.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from ai_infra.cli.summary import (
    MODEL_PRICING,
    CostEstimator,
    ExecutionResult,
    ExecutionStatus,
    ExecutionSummary,
    FailureInfo,
    FailureSummary,
    FileChange,
    FileChangeSummary,
    FileChangeType,
    GitCheckpoint,
    GitCheckpointSummary,
    NextStep,
    TestResults,
    generate_next_steps,
    render_next_steps,
)

# =============================================================================
# Execution Status Tests
# =============================================================================


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """All expected statuses should exist."""
        assert ExecutionStatus.COMPLETED
        assert ExecutionStatus.FAILED
        assert ExecutionStatus.PARTIAL
        assert ExecutionStatus.CANCELLED

    def test_status_values(self) -> None:
        """Status values should be lowercase strings."""
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"


# =============================================================================
# File Change Tests
# =============================================================================


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_new_file(self) -> None:
        """New file should have correct icon and style."""
        change = FileChange(
            path="src/main.py",
            change_type=FileChangeType.NEW,
            lines_added=45,
        )
        assert change.get_icon() == "+"
        assert "green" in change.get_style()
        assert change.get_stats() == "+45 lines"

    def test_modified_file(self) -> None:
        """Modified file should have correct icon and style."""
        change = FileChange(
            path="tests/test_main.py",
            change_type=FileChangeType.MODIFIED,
            lines_added=23,
            lines_removed=5,
        )
        assert change.get_icon() == "~"
        assert "yellow" in change.get_style()
        assert change.get_stats() == "+23/-5 lines"

    def test_deleted_file(self) -> None:
        """Deleted file should have correct icon and style."""
        change = FileChange(
            path="src/legacy.py",
            change_type=FileChangeType.DELETED,
            lines_removed=89,
        )
        assert change.get_icon() == "-"
        assert "red" in change.get_style()
        assert change.get_stats() == "-89 lines"


class TestFileChangeSummary:
    """Tests for FileChangeSummary."""

    def test_empty_summary(self) -> None:
        """Empty summary should have zero counts."""
        summary = FileChangeSummary()
        assert summary.total_files == 0
        assert summary.files_added == 0
        assert summary.files_modified == 0
        assert summary.files_deleted == 0

    def test_file_counts(self) -> None:
        """Should correctly count file types."""
        changes = [
            FileChange("a.py", FileChangeType.NEW, lines_added=10),
            FileChange("b.py", FileChangeType.NEW, lines_added=20),
            FileChange("c.py", FileChangeType.MODIFIED, lines_added=5, lines_removed=3),
            FileChange("d.py", FileChangeType.DELETED, lines_removed=15),
        ]
        summary = FileChangeSummary(changes=changes)

        assert summary.total_files == 4
        assert summary.files_added == 2
        assert summary.files_modified == 1
        assert summary.files_deleted == 1

    def test_line_totals(self) -> None:
        """Should correctly sum line changes."""
        changes = [
            FileChange("a.py", FileChangeType.NEW, lines_added=10),
            FileChange("b.py", FileChangeType.MODIFIED, lines_added=5, lines_removed=3),
            FileChange("c.py", FileChangeType.DELETED, lines_removed=15),
        ]
        summary = FileChangeSummary(changes=changes)

        assert summary.total_lines_added == 15
        assert summary.total_lines_removed == 18

    def test_render(self) -> None:
        """Should render without error."""
        changes = [
            FileChange("src/main.py", FileChangeType.NEW, lines_added=45),
        ]
        summary = FileChangeSummary(changes=changes)
        table = summary.render()
        assert table is not None


# =============================================================================
# Git Checkpoint Tests
# =============================================================================


class TestGitCheckpoint:
    """Tests for GitCheckpoint dataclass."""

    def test_creation(self) -> None:
        """Should create checkpoint with all fields."""
        checkpoint = GitCheckpoint(
            commit_hash="abc1234",
            message="Complete task 1.1.1",
            task_id="1.1.1",
        )
        assert checkpoint.commit_hash == "abc1234"
        assert checkpoint.message == "Complete task 1.1.1"
        assert checkpoint.task_id == "1.1.1"


class TestGitCheckpointSummary:
    """Tests for GitCheckpointSummary."""

    def test_empty_summary(self) -> None:
        """Empty summary should indicate no checkpoints."""
        summary = GitCheckpointSummary()
        assert summary.total_checkpoints == 0
        text = summary.render()
        assert "No checkpoints" in text.plain

    def test_with_checkpoints(self) -> None:
        """Summary with checkpoints should show count."""
        checkpoints = [
            GitCheckpoint("abc1234", "Task 1", "1.1.1"),
            GitCheckpoint("def5678", "Task 2", "1.1.2"),
            GitCheckpoint("ghi9012", "Task 3", "1.1.3"),
        ]
        summary = GitCheckpointSummary(checkpoints=checkpoints)
        assert summary.total_checkpoints == 3
        text = summary.render()
        assert "3 checkpoint" in text.plain


# =============================================================================
# Cost Estimator Tests
# =============================================================================


class TestCostEstimator:
    """Tests for cost estimation."""

    def test_known_model_pricing(self) -> None:
        """Should use correct pricing for known models."""
        estimator = CostEstimator(
            model_name="claude-sonnet-4",
            tokens_input=1_000_000,
            tokens_output=100_000,
        )
        # Input: 1M * $3/M = $3
        # Output: 100k * $15/M = $1.5
        # Total: $4.5
        cost = estimator.estimate_cost()
        assert 4.4 < cost < 4.6

    def test_prefix_match_pricing(self) -> None:
        """Should match model by prefix."""
        estimator = CostEstimator(
            model_name="claude-sonnet-4-20250514",
            tokens_input=1_000_000,
            tokens_output=0,
        )
        # Should match claude-sonnet-4 pricing
        cost = estimator.estimate_cost()
        assert 2.9 < cost < 3.1  # $3 for 1M input tokens

    def test_unknown_model_uses_default(self) -> None:
        """Unknown models should use default pricing."""
        estimator = CostEstimator(
            model_name="unknown-model-xyz",
            tokens_input=1_000_000,
            tokens_output=0,
        )
        # Default input is $3/M
        cost = estimator.estimate_cost()
        assert 2.9 < cost < 3.1

    def test_format_cost_small(self) -> None:
        """Small costs should show 3 decimal places."""
        estimator = CostEstimator(
            model_name="claude-sonnet-4",
            tokens_input=1000,
            tokens_output=500,
        )
        formatted = estimator.format_cost()
        assert formatted.startswith("~$")
        assert "0.0" in formatted

    def test_format_cost_normal(self) -> None:
        """Normal costs should show 2 decimal places."""
        estimator = CostEstimator(
            model_name="claude-sonnet-4",
            tokens_input=38102,
            tokens_output=7129,
        )
        formatted = estimator.format_cost()
        assert formatted.startswith("~$")

    def test_format_tokens(self) -> None:
        """Should format token breakdown."""
        estimator = CostEstimator(
            model_name="claude-sonnet-4",
            tokens_input=38102,
            tokens_output=7129,
        )
        formatted = estimator.format_tokens()
        assert "45,231" in formatted  # Total
        assert "38,102" in formatted  # Input
        assert "7,129" in formatted  # Output

    def test_total_tokens(self) -> None:
        """Should calculate total tokens."""
        estimator = CostEstimator(
            model_name="test",
            tokens_input=1000,
            tokens_output=500,
        )
        assert estimator.total_tokens == 1500

    def test_model_pricing_coverage(self) -> None:
        """MODEL_PRICING should have entries for major providers."""
        assert "claude-sonnet-4" in MODEL_PRICING
        assert "gpt-4o" in MODEL_PRICING
        assert "gemini-2.0-flash" in MODEL_PRICING
        assert "mistral-large" in MODEL_PRICING


# =============================================================================
# Test Results Tests
# =============================================================================


class TestTestResults:
    """Tests for TestResults dataclass."""

    def test_empty_results(self) -> None:
        """Empty results should have zero counts."""
        results = TestResults()
        assert results.total == 0
        assert results.success_rate == 0.0

    def test_total_calculation(self) -> None:
        """Total should sum all test counts."""
        results = TestResults(passed=10, failed=2, skipped=3)
        assert results.total == 15

    def test_success_rate(self) -> None:
        """Success rate should be percentage of passed."""
        results = TestResults(passed=80, failed=10, skipped=10)
        assert results.success_rate == 80.0

    def test_render_all_pass(self) -> None:
        """Render should show all categories."""
        results = TestResults(passed=12, failed=0, skipped=2, coverage=87.0)
        text = results.render()
        plain = text.plain

        assert "Passed:" in plain
        assert "12" in plain
        assert "Failed:" in plain
        assert "0" in plain
        assert "Skipped:" in plain
        assert "2" in plain
        assert "Coverage:" in plain
        assert "87%" in plain

    def test_render_without_coverage(self) -> None:
        """Render should work without coverage."""
        results = TestResults(passed=10, failed=0, skipped=0)
        text = results.render()
        assert "Coverage:" not in text.plain


# =============================================================================
# Next Steps Tests
# =============================================================================


class TestNextStep:
    """Tests for NextStep dataclass."""

    def test_with_command(self) -> None:
        """NextStep should store command."""
        step = NextStep("Run tests", "pytest -v")
        assert step.description == "Run tests"
        assert step.command == "pytest -v"

    def test_without_command(self) -> None:
        """NextStep should work without command."""
        step = NextStep("Review code")
        assert step.description == "Review code"
        assert step.command is None


class TestGenerateNextSteps:
    """Tests for next step generation."""

    def test_completed_steps(self) -> None:
        """Completed execution should suggest review and testing."""
        steps = generate_next_steps(
            status=ExecutionStatus.COMPLETED,
            has_git_changes=True,
        )
        descriptions = [s.description for s in steps]
        assert any("review" in d.lower() for d in descriptions)
        assert any("test" in d.lower() for d in descriptions)

    def test_failed_steps(self) -> None:
        """Failed execution should suggest retry."""
        steps = generate_next_steps(
            status=ExecutionStatus.FAILED,
            failed_task_id="2.1.3",
        )
        descriptions = [s.description for s in steps]
        assert any("retry" in d.lower() for d in descriptions)

    def test_partial_steps(self) -> None:
        """Partial execution should suggest resume."""
        steps = generate_next_steps(status=ExecutionStatus.PARTIAL)
        descriptions = [s.description for s in steps]
        assert any("resume" in d.lower() for d in descriptions)

    def test_cancelled_steps(self) -> None:
        """Cancelled execution should offer restart options."""
        steps = generate_next_steps(status=ExecutionStatus.CANCELLED)
        descriptions = [s.description for s in steps]
        assert any("resume" in d.lower() or "fresh" in d.lower() for d in descriptions)


class TestRenderNextSteps:
    """Tests for next steps rendering."""

    def test_render_numbered_list(self) -> None:
        """Should render numbered list."""
        steps = [
            NextStep("First step", "command1"),
            NextStep("Second step", "command2"),
        ]
        table = render_next_steps(steps)
        assert table is not None


# =============================================================================
# Failure Summary Tests
# =============================================================================


class TestFailureInfo:
    """Tests for FailureInfo dataclass."""

    def test_creation(self) -> None:
        """Should create with all fields."""
        failure = FailureInfo(
            task_id="2.1.3",
            task_title="Implement OAuth2 flow",
            error_type="Test assertion failed",
            error_message="pytest tests/test_auth.py::test_oauth_callback",
            suggestion="Check the OAUTH_CALLBACK_URL environment variable.",
            retry_command="ai-infra chat --resume --from 2.1.3",
        )
        assert failure.task_id == "2.1.3"
        assert failure.suggestion is not None


class TestFailureSummary:
    """Tests for FailureSummary renderable."""

    def test_renders_without_error(self) -> None:
        """Should render to console without error."""
        failure = FailureInfo(
            task_id="2.1.3",
            task_title="Implement OAuth2 flow",
            error_type="Test assertion failed",
            error_message="Test failed",
            suggestion="Check config",
            retry_command="ai-infra chat --resume --from 2.1.3",
        )
        summary = FailureSummary(failure)

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(summary)
        output = console.file.getvalue()

        assert "EXECUTION FAILED" in output
        assert "2.1.3" in output
        assert "OAuth2" in output
        assert "Check config" in output


# =============================================================================
# Execution Result Tests
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_string_status_conversion(self) -> None:
        """String status should be converted to enum."""
        result = ExecutionResult(status="completed")
        assert result.status == ExecutionStatus.COMPLETED

    def test_enum_status_preserved(self) -> None:
        """Enum status should be preserved."""
        result = ExecutionResult(status=ExecutionStatus.FAILED)
        assert result.status == ExecutionStatus.FAILED

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        result = ExecutionResult(status="completed")
        assert result.completed_tasks == 0
        assert result.total_tasks == 0
        assert result.model_name == "unknown"


# =============================================================================
# Execution Summary Tests
# =============================================================================


class TestExecutionSummary:
    """Tests for ExecutionSummary renderable."""

    def test_completed_summary(self) -> None:
        """Should render completed summary."""
        result = ExecutionResult(
            status="completed",
            completed_tasks=12,
            total_tasks=12,
            duration_seconds=154.0,
            tokens_input=38102,
            tokens_output=7129,
            model_name="claude-sonnet-4",
        )
        summary = ExecutionSummary(result)

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(summary)
        output = console.file.getvalue()

        assert "EXECUTION COMPLETE" in output
        assert "12/12" in output
        assert "claude-sonnet-4" in output
        assert "NEXT STEPS" in output

    def test_failed_summary(self) -> None:
        """Should render failed summary with failure details."""
        failure = FailureInfo(
            task_id="2.1.3",
            task_title="Implement OAuth2",
            error_type="Test failed",
            error_message="Assertion error",
        )
        result = ExecutionResult(
            status="failed",
            completed_tasks=5,
            total_tasks=12,
            duration_seconds=60.0,
            tokens_input=10000,
            tokens_output=2000,
            model_name="gpt-4o",
            failure_info=failure,
        )
        summary = ExecutionSummary(result)

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(summary)
        output = console.file.getvalue()

        assert "EXECUTION FAILED" in output
        assert "5/12" in output
        assert "2.1.3" in output

    def test_with_file_changes(self) -> None:
        """Should render file changes section."""
        changes = FileChangeSummary(
            changes=[
                FileChange("src/main.py", FileChangeType.NEW, lines_added=45),
                FileChange(
                    "tests/test_main.py", FileChangeType.MODIFIED, lines_added=10, lines_removed=2
                ),
            ]
        )
        result = ExecutionResult(
            status="completed",
            completed_tasks=5,
            total_tasks=5,
            file_changes=changes,
        )
        summary = ExecutionSummary(result)

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(summary)
        output = console.file.getvalue()

        assert "FILES MODIFIED" in output
        assert "src/main.py" in output

    def test_with_test_results(self) -> None:
        """Should render test results section."""
        result = ExecutionResult(
            status="completed",
            completed_tasks=5,
            total_tasks=5,
            test_results=TestResults(passed=10, failed=0, skipped=2, coverage=87.0),
        )
        summary = ExecutionSummary(result)

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(summary)
        output = console.file.getvalue()

        assert "TEST RESULTS" in output
        assert "Passed" in output

    def test_with_git_checkpoints(self) -> None:
        """Should render git checkpoints section."""
        checkpoints = GitCheckpointSummary(
            checkpoints=[
                GitCheckpoint("abc1234", "Task 1", "1.1.1"),
            ]
        )
        result = ExecutionResult(
            status="completed",
            completed_tasks=5,
            total_tasks=5,
            git_checkpoints=checkpoints,
        )
        summary = ExecutionSummary(result)

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(summary)
        output = console.file.getvalue()

        assert "GIT CHECKPOINTS" in output
