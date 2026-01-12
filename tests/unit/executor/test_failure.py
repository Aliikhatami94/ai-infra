"""Unit tests for FailureAnalyzer and related classes."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_infra.executor.failure import (
    FailureAnalyzer,
    FailureCategory,
    FailureRecord,
    FailureSeverity,
    FailureStats,
)
from ai_infra.executor.models import Task
from ai_infra.executor.verifier import (
    CheckLevel,
    CheckResult,
    CheckStatus,
    VerificationResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        id="0.1.1",
        title="Implement feature",
        description="Add a new feature to the app",
        file_hints=["main.py", "src/app.py"],
    )


@pytest.fixture
def another_task() -> Task:
    """Create another sample task."""
    return Task(
        id="0.1.2",
        title="Fix bug",
        description="Fix a critical bug",
    )


@pytest.fixture
def analyzer(tmp_path: Path) -> FailureAnalyzer:
    """Create a FailureAnalyzer with temporary storage."""
    return FailureAnalyzer(data_dir=tmp_path, auto_save=False)


@pytest.fixture
def analyzer_with_data(analyzer: FailureAnalyzer, sample_task: Task) -> FailureAnalyzer:
    """Create an analyzer with some failure records."""
    # Add various failures
    analyzer.record_failure(
        sample_task,
        FailureCategory.SYNTAX_ERROR,
        severity=FailureSeverity.HIGH,
        error_message="SyntaxError: unexpected EOF",
        duration_seconds=10.5,
    )
    analyzer.record_failure(
        Task(id="0.1.2", title="Task 2"),
        FailureCategory.SYNTAX_ERROR,
        severity=FailureSeverity.MEDIUM,
        error_message="SyntaxError: invalid syntax",
        duration_seconds=5.0,
    )
    analyzer.record_failure(
        Task(id="0.1.3", title="Task 3"),
        FailureCategory.CONTEXT_MISSING,
        severity=FailureSeverity.HIGH,
        error_message="Not enough context",
        duration_seconds=15.0,
    )
    analyzer.record_failure(
        Task(id="0.1.4", title="Task 4"),
        FailureCategory.TEST_FAILURE,
        severity=FailureSeverity.MEDIUM,
        secondary_categories=[FailureCategory.PARTIAL_CHANGE],
        error_message="2 tests failed",
        duration_seconds=30.0,
    )
    return analyzer


@pytest.fixture
def failed_verification() -> VerificationResult:
    """Create a failed verification result."""
    return VerificationResult(
        task_id="0.1.1",
        checks=[
            CheckResult(
                name="syntax:main.py",
                level=CheckLevel.SYNTAX,
                status=CheckStatus.FAILED,
                error="Line 10: unexpected EOF",
            ),
            CheckResult(
                name="tests",
                level=CheckLevel.TESTS,
                status=CheckStatus.FAILED,
                error="2 tests failed",
            ),
        ],
        levels_run=[CheckLevel.FILES, CheckLevel.SYNTAX, CheckLevel.TESTS],
    )


# =============================================================================
# TestFailureCategory
# =============================================================================


class TestFailureCategory:
    """Tests for FailureCategory enum."""

    def test_all_categories_have_string_values(self):
        """Test that all categories have string values."""
        for category in FailureCategory:
            assert isinstance(category.value, str)
            assert len(category.value) > 0

    def test_category_from_string(self):
        """Test creating category from string value."""
        category = FailureCategory("syntax_error")
        assert category == FailureCategory.SYNTAX_ERROR

    def test_context_categories_exist(self):
        """Test that context-related categories exist."""
        assert FailureCategory.CONTEXT_MISSING
        assert FailureCategory.CONTEXT_STALE
        assert FailureCategory.CONTEXT_OVERWHELMING

    def test_code_quality_categories_exist(self):
        """Test that code quality categories exist."""
        assert FailureCategory.SYNTAX_ERROR
        assert FailureCategory.TYPE_ERROR
        assert FailureCategory.IMPORT_ERROR
        assert FailureCategory.TEST_FAILURE


# =============================================================================
# TestFailureSeverity
# =============================================================================


class TestFailureSeverity:
    """Tests for FailureSeverity enum."""

    def test_severity_levels(self):
        """Test all severity levels exist."""
        assert FailureSeverity.CRITICAL
        assert FailureSeverity.HIGH
        assert FailureSeverity.MEDIUM
        assert FailureSeverity.LOW

    def test_severity_from_string(self):
        """Test creating severity from string value."""
        severity = FailureSeverity("high")
        assert severity == FailureSeverity.HIGH


# =============================================================================
# TestFailureRecord
# =============================================================================


class TestFailureRecord:
    """Tests for FailureRecord dataclass."""

    def test_create_minimal_record(self):
        """Test creating a record with minimal data."""
        record = FailureRecord(
            task_id="0.1.1",
            task_title="Test task",
            category=FailureCategory.SYNTAX_ERROR,
        )
        assert record.task_id == "0.1.1"
        assert record.category == FailureCategory.SYNTAX_ERROR
        assert record.severity == FailureSeverity.MEDIUM  # default

    def test_create_full_record(self):
        """Test creating a record with all fields."""
        record = FailureRecord(
            task_id="0.1.1",
            task_title="Test task",
            category=FailureCategory.SYNTAX_ERROR,
            severity=FailureSeverity.HIGH,
            secondary_categories=[FailureCategory.PARTIAL_CHANGE],
            error_message="SyntaxError: unexpected EOF",
            agent_output="def foo():\n    # incomplete",
            context_summary="Project with 10 files",
            duration_seconds=15.5,
            metadata={"attempt": 1},
        )
        assert record.severity == FailureSeverity.HIGH
        assert len(record.secondary_categories) == 1
        assert record.duration_seconds == 15.5

    def test_to_dict(self):
        """Test serialization to dictionary."""
        record = FailureRecord(
            task_id="0.1.1",
            task_title="Test task",
            category=FailureCategory.SYNTAX_ERROR,
            severity=FailureSeverity.HIGH,
            error_message="Test error",
        )
        data = record.to_dict()

        assert data["task_id"] == "0.1.1"
        assert data["category"] == "syntax_error"
        assert data["severity"] == "high"
        assert "timestamp" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "task_id": "0.1.1",
            "task_title": "Test task",
            "category": "syntax_error",
            "severity": "high",
            "secondary_categories": ["partial_change"],
            "error_message": "Test error",
            "timestamp": "2026-01-06T12:00:00+00:00",
        }
        record = FailureRecord.from_dict(data)

        assert record.task_id == "0.1.1"
        assert record.category == FailureCategory.SYNTAX_ERROR
        assert record.severity == FailureSeverity.HIGH
        assert len(record.secondary_categories) == 1

    def test_agent_output_truncated_in_dict(self):
        """Test that long agent output is truncated."""
        long_output = "x" * 5000
        record = FailureRecord(
            task_id="0.1.1",
            task_title="Test",
            category=FailureCategory.SYNTAX_ERROR,
            agent_output=long_output,
        )
        data = record.to_dict()

        assert len(data["agent_output"]) <= 2000


# =============================================================================
# TestFailureStats
# =============================================================================


class TestFailureStats:
    """Tests for FailureStats dataclass."""

    def test_create_stats(self):
        """Test creating stats."""
        stats = FailureStats(
            total_failures=10,
            by_category={"syntax_error": 5, "test_failure": 3, "timeout": 2},
            by_severity={"high": 4, "medium": 6},
            avg_duration=12.5,
            success_rate=0.7,
        )
        assert stats.total_failures == 10
        assert stats.by_category["syntax_error"] == 5

    def test_to_dict(self):
        """Test serialization."""
        stats = FailureStats(
            total_failures=5,
            by_category={"syntax_error": 5},
            by_severity={"medium": 5},
            avg_duration=10.0,
        )
        data = stats.to_dict()

        assert data["total_failures"] == 5
        assert "by_category" in data

    def test_summary(self):
        """Test human-readable summary."""
        stats = FailureStats(
            total_failures=10,
            by_category={"syntax_error": 5, "test_failure": 5},
            by_severity={"high": 4, "medium": 6},
            avg_duration=12.5,
            success_rate=0.65,
        )
        summary = stats.summary()

        assert "10" in summary
        assert "syntax_error" in summary
        assert "65" in summary  # Could be 65% or 65.0%


# =============================================================================
# TestFailureAnalyzer - Initialization
# =============================================================================


class TestFailureAnalyzerInit:
    """Tests for FailureAnalyzer initialization."""

    def test_init_in_memory(self):
        """Test initialization without data directory."""
        analyzer = FailureAnalyzer()
        assert analyzer.data_dir is None
        assert analyzer.record_count == 0

    def test_init_with_data_dir(self, tmp_path: Path):
        """Test initialization with data directory."""
        analyzer = FailureAnalyzer(data_dir=tmp_path)
        assert analyzer.data_dir == tmp_path
        assert (tmp_path).exists()

    def test_init_creates_directory(self, tmp_path: Path):
        """Test that data directory is created if it doesn't exist."""
        new_dir = tmp_path / "failures"
        FailureAnalyzer(data_dir=new_dir)
        assert new_dir.exists()

    def test_load_existing_data(self, tmp_path: Path):
        """Test loading existing failure data."""
        # Create initial analyzer and save data
        analyzer1 = FailureAnalyzer(data_dir=tmp_path, auto_save=True)
        analyzer1.record_failure(
            Task(id="0.1.1", title="Test"),
            FailureCategory.SYNTAX_ERROR,
        )
        analyzer1.save()

        # Create new analyzer and verify it loads the data
        analyzer2 = FailureAnalyzer(data_dir=tmp_path)
        assert analyzer2.record_count == 1


# =============================================================================
# TestFailureAnalyzer - Recording
# =============================================================================


class TestFailureAnalyzerRecording:
    """Tests for recording failures and successes."""

    def test_record_failure(self, analyzer: FailureAnalyzer, sample_task: Task):
        """Test recording a failure."""
        record = analyzer.record_failure(
            sample_task,
            FailureCategory.SYNTAX_ERROR,
            error_message="Test error",
        )

        assert record.task_id == sample_task.id
        assert record.category == FailureCategory.SYNTAX_ERROR
        assert analyzer.record_count == 1

    def test_record_failure_with_verification(
        self, analyzer: FailureAnalyzer, sample_task: Task, failed_verification: VerificationResult
    ):
        """Test recording a failure with verification result."""
        record = analyzer.record_failure(
            sample_task,
            FailureCategory.SYNTAX_ERROR,
            verification_result=failed_verification,
        )

        assert record.verification_result is not None
        assert "checks" in record.verification_result

    def test_record_success(self, analyzer: FailureAnalyzer, sample_task: Task):
        """Test recording a success."""
        analyzer.record_success(sample_task)

        assert analyzer.total_tasks == 1
        assert analyzer.success_rate == 1.0

    def test_success_rate_calculation(self, analyzer: FailureAnalyzer):
        """Test success rate calculation."""
        # 2 successes, 1 failure = 66.7% success rate
        analyzer.record_success(Task(id="1", title="T1"))
        analyzer.record_success(Task(id="2", title="T2"))
        analyzer.record_failure(
            Task(id="3", title="T3"),
            FailureCategory.SYNTAX_ERROR,
        )

        assert analyzer.total_tasks == 3
        assert abs(analyzer.success_rate - 0.667) < 0.01

    def test_max_records_limit(self, tmp_path: Path):
        """Test that records are trimmed when max is reached."""
        analyzer = FailureAnalyzer(data_dir=tmp_path, max_records=5, auto_save=False)

        for i in range(10):
            analyzer.record_failure(
                Task(id=str(i), title=f"Task {i}"),
                FailureCategory.SYNTAX_ERROR,
            )

        assert analyzer.record_count == 5
        # Should keep the most recent records
        records = analyzer.get_records()
        assert records[0].task_id == "5"


# =============================================================================
# TestFailureAnalyzer - Querying
# =============================================================================


class TestFailureAnalyzerQuerying:
    """Tests for querying failure records."""

    def test_get_records_all(self, analyzer_with_data: FailureAnalyzer):
        """Test getting all records."""
        records = analyzer_with_data.get_records()
        assert len(records) == 4

    def test_get_records_by_category(self, analyzer_with_data: FailureAnalyzer):
        """Test filtering by category."""
        records = analyzer_with_data.get_records(category=FailureCategory.SYNTAX_ERROR)
        assert len(records) == 2
        assert all(r.category == FailureCategory.SYNTAX_ERROR for r in records)

    def test_get_records_by_severity(self, analyzer_with_data: FailureAnalyzer):
        """Test filtering by severity."""
        records = analyzer_with_data.get_records(severity=FailureSeverity.HIGH)
        assert len(records) == 2
        assert all(r.severity == FailureSeverity.HIGH for r in records)

    def test_get_records_since(self, analyzer: FailureAnalyzer, sample_task: Task):
        """Test filtering by timestamp."""
        # Record a failure
        analyzer.record_failure(sample_task, FailureCategory.SYNTAX_ERROR)

        # Query with future timestamp should return nothing
        future = datetime.now(UTC) + timedelta(hours=1)
        records = analyzer.get_records(since=future)
        assert len(records) == 0

        # Query with past timestamp should return the record
        past = datetime.now(UTC) - timedelta(hours=1)
        records = analyzer.get_records(since=past)
        assert len(records) == 1

    def test_get_records_with_limit(self, analyzer_with_data: FailureAnalyzer):
        """Test limiting number of records."""
        records = analyzer_with_data.get_records(limit=2)
        assert len(records) == 2


# =============================================================================
# TestFailureAnalyzer - Statistics
# =============================================================================


class TestFailureAnalyzerStats:
    """Tests for failure statistics."""

    def test_get_stats(self, analyzer_with_data: FailureAnalyzer):
        """Test getting statistics."""
        stats = analyzer_with_data.get_stats()

        assert stats.total_failures == 4
        assert stats.by_category["syntax_error"] == 2
        assert stats.by_category["context_missing"] == 1
        assert stats.by_category["test_failure"] == 1

    def test_get_stats_empty(self, analyzer: FailureAnalyzer):
        """Test stats for empty analyzer."""
        stats = analyzer.get_stats()

        assert stats.total_failures == 0
        assert stats.by_category == {}

    def test_avg_duration(self, analyzer_with_data: FailureAnalyzer):
        """Test average duration calculation."""
        stats = analyzer_with_data.get_stats()

        # (10.5 + 5.0 + 15.0 + 30.0) / 4 = 15.125
        assert abs(stats.avg_duration - 15.125) < 0.01

    def test_top_patterns(self, analyzer_with_data: FailureAnalyzer):
        """Test top patterns extraction."""
        stats = analyzer_with_data.get_stats()

        assert len(stats.top_patterns) > 0
        # syntax_error should be the top pattern with 2 occurrences
        assert stats.top_patterns[0][0] == "syntax_error"
        assert stats.top_patterns[0][1] == 2


# =============================================================================
# TestFailureAnalyzer - Pattern Finding
# =============================================================================


class TestFailureAnalyzerPatterns:
    """Tests for pattern finding."""

    def test_find_patterns(self, analyzer_with_data: FailureAnalyzer):
        """Test finding failure patterns."""
        patterns = analyzer_with_data.find_patterns(min_occurrences=1)

        assert len(patterns) > 0
        # Each pattern should have required fields
        for pattern in patterns:
            assert "pattern" in pattern
            assert "count" in pattern
            assert "category" in pattern

    def test_find_patterns_min_occurrences(self, analyzer_with_data: FailureAnalyzer):
        """Test minimum occurrences filter."""
        patterns = analyzer_with_data.find_patterns(min_occurrences=2)

        # Only syntax_error:high and syntax_error:medium combinations
        # Actually, we need 2 of the same pattern
        for pattern in patterns:
            assert pattern["count"] >= 2


# =============================================================================
# TestFailureAnalyzer - Categorization
# =============================================================================


class TestFailureAnalyzerCategorization:
    """Tests for automatic categorization."""

    def test_categorize_from_verification(
        self, analyzer: FailureAnalyzer, failed_verification: VerificationResult
    ):
        """Test categorizing from verification result."""
        primary, secondary = analyzer.categorize_from_verification(failed_verification)

        assert primary == FailureCategory.SYNTAX_ERROR
        assert FailureCategory.TEST_FAILURE in secondary

    def test_categorize_from_passing_verification(self, analyzer: FailureAnalyzer):
        """Test categorizing from passing verification."""
        passing = VerificationResult(
            task_id="0.1.1",
            checks=[
                CheckResult(
                    name="syntax:main.py",
                    level=CheckLevel.SYNTAX,
                    status=CheckStatus.PASSED,
                ),
            ],
        )

        primary, secondary = analyzer.categorize_from_verification(passing)

        assert primary == FailureCategory.UNKNOWN
        assert len(secondary) == 0


# =============================================================================
# TestFailureAnalyzer - Suggestions
# =============================================================================


class TestFailureAnalyzerSuggestions:
    """Tests for improvement suggestions."""

    def test_suggest_improvements(self, analyzer_with_data: FailureAnalyzer):
        """Test generating improvement suggestions."""
        suggestions = analyzer_with_data.suggest_improvements()

        assert len(suggestions) > 0
        # Should have suggestions for top failure categories
        for suggestion in suggestions:
            assert "category" in suggestion
            assert "issue" in suggestion
            assert "suggestion" in suggestion
            assert "action" in suggestion

    def test_suggest_improvements_for_syntax(self, analyzer: FailureAnalyzer):
        """Test suggestion for syntax errors."""
        for i in range(5):
            analyzer.record_failure(
                Task(id=str(i), title=f"Task {i}"),
                FailureCategory.SYNTAX_ERROR,
            )

        suggestions = analyzer.suggest_improvements(top_n=1)

        assert len(suggestions) == 1
        assert suggestions[0]["category"] == "syntax_error"
        assert "syntax" in suggestions[0]["issue"].lower()


# =============================================================================
# TestFailureAnalyzer - Persistence
# =============================================================================


class TestFailureAnalyzerPersistence:
    """Tests for saving and loading data."""

    def test_save_and_load(self, tmp_path: Path):
        """Test saving and loading failure data."""
        # Create and populate analyzer
        analyzer1 = FailureAnalyzer(data_dir=tmp_path, auto_save=False)
        analyzer1.record_failure(
            Task(id="0.1.1", title="Test"),
            FailureCategory.SYNTAX_ERROR,
            error_message="Test error",
        )
        analyzer1.record_success(Task(id="0.1.2", title="Success"))
        analyzer1.save()

        # Load in new analyzer
        analyzer2 = FailureAnalyzer(data_dir=tmp_path)

        assert analyzer2.record_count == 1
        assert analyzer2.total_tasks == 2

    def test_auto_save(self, tmp_path: Path):
        """Test auto-save functionality."""
        analyzer = FailureAnalyzer(data_dir=tmp_path, auto_save=True)
        analyzer.record_failure(
            Task(id="0.1.1", title="Test"),
            FailureCategory.SYNTAX_ERROR,
        )

        # Check file was created
        data_file = tmp_path / "failures.json"
        assert data_file.exists()

        # Verify content
        with open(data_file) as f:
            data = json.load(f)
        assert len(data["records"]) == 1


# =============================================================================
# TestFailureAnalyzer - Export
# =============================================================================


class TestFailureAnalyzerExport:
    """Tests for exporting failure data."""

    def test_export_json(self, analyzer_with_data: FailureAnalyzer, tmp_path: Path):
        """Test JSON export."""
        output_path = tmp_path / "export.json"
        analyzer_with_data.export_dataset(output_path, format="json")

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert "records" in data
        assert "stats" in data
        assert "patterns" in data
        assert len(data["records"]) == 4

    def test_export_csv(self, analyzer_with_data: FailureAnalyzer, tmp_path: Path):
        """Test CSV export."""
        output_path = tmp_path / "export.csv"
        analyzer_with_data.export_dataset(output_path, format="csv")

        assert output_path.exists()
        content = output_path.read_text()

        # Check header
        assert "task_id" in content
        assert "category" in content
        # Check data rows
        assert "syntax_error" in content


# =============================================================================
# TestFailureAnalyzer - Clear
# =============================================================================


class TestFailureAnalyzerClear:
    """Tests for clearing data."""

    def test_clear(self, analyzer_with_data: FailureAnalyzer):
        """Test clearing all records."""
        assert analyzer_with_data.record_count > 0

        analyzer_with_data.clear()

        assert analyzer_with_data.record_count == 0
        assert analyzer_with_data.total_tasks == 0
        assert analyzer_with_data.success_rate is None
