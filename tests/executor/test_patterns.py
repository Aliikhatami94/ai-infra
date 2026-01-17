"""Tests for executor pattern recognition (Phase 5.2)."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_infra.executor.patterns import (
    AgentsUpdater,
    ExecutionContext,
    FailurePattern,
    FailurePatternTracker,
    FixAction,
    FixPattern,
    FixPatternTracker,
    PatternsDatabase,
    PatternSuggester,
    TaskError,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def patterns_db(temp_dir):
    """Create a patterns database for tests."""
    return PatternsDatabase(path=temp_dir, auto_save=True)


@pytest.fixture
def failure_tracker(patterns_db):
    """Create a failure pattern tracker."""
    return FailurePatternTracker(patterns_db)


@pytest.fixture
def fix_tracker(patterns_db):
    """Create a fix pattern tracker."""
    return FixPatternTracker(patterns_db)


@pytest.fixture
def pattern_suggester(failure_tracker, fix_tracker):
    """Create a pattern suggester."""
    return PatternSuggester(failure_tracker, fix_tracker)


@pytest.fixture
def sample_error():
    """Create a sample error."""
    return TaskError(
        type="ImportError",
        message="No module named 'pandas'",
        file_path="/src/main.py",
        line_number=5,
        context="During data processing task",
    )


@pytest.fixture
def sample_context():
    """Create a sample execution context."""
    return ExecutionContext(
        task_title="Add data processing",
        task_description="Add pandas data processing",
        language="python",
        framework="pandas",
        files_involved=["/src/main.py", "/src/utils.py"],
    )


# =============================================================================
# TaskError Tests
# =============================================================================


class TestTaskError:
    """Tests for TaskError model."""

    def test_create_task_error(self):
        """Test creating a TaskError."""
        error = TaskError(
            type="SyntaxError",
            message="invalid syntax",
            file_path="test.py",
            line_number=10,
            context="parsing",
        )
        assert error.type == "SyntaxError"
        assert error.message == "invalid syntax"
        assert error.file_path == "test.py"
        assert error.line_number == 10
        assert error.context == "parsing"

    def test_task_error_defaults(self):
        """Test TaskError defaults."""
        error = TaskError(type="Error", message="test")
        assert error.file_path is None
        assert error.line_number is None
        assert error.context == ""

    def test_task_error_to_dict(self, sample_error):
        """Test TaskError serialization."""
        data = sample_error.to_dict()
        assert data["type"] == "ImportError"
        assert data["message"] == "No module named 'pandas'"
        assert data["file_path"] == "/src/main.py"
        assert data["line_number"] == 5

    def test_task_error_from_dict(self):
        """Test TaskError deserialization."""
        data = {
            "type": "ValueError",
            "message": "invalid value",
            "file_path": "test.py",
            "line_number": 42,
            "context": "validation",
        }
        error = TaskError.from_dict(data)
        assert error.type == "ValueError"
        assert error.message == "invalid value"

    def test_task_error_from_exception(self):
        """Test creating TaskError from exception."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            error = TaskError.from_exception(e, "test context")
            assert error.type == "ValueError"
            assert error.message == "test error"
            assert error.context == "test context"


# =============================================================================
# ExecutionContext Tests
# =============================================================================


class TestExecutionContext:
    """Tests for ExecutionContext model."""

    def test_create_context(self, sample_context):
        """Test creating ExecutionContext."""
        assert sample_context.task_title == "Add data processing"
        assert sample_context.language == "python"
        assert sample_context.framework == "pandas"

    def test_context_summary(self, sample_context):
        """Test context summary property."""
        summary = sample_context.summary
        assert "Add data processing" in summary
        assert "python" in summary
        assert "pandas" in summary

    def test_context_to_dict(self, sample_context):
        """Test context serialization."""
        data = sample_context.to_dict()
        assert data["task_title"] == "Add data processing"
        assert data["language"] == "python"
        assert len(data["files_involved"]) == 2

    def test_context_from_dict(self):
        """Test context deserialization."""
        data = {
            "task_title": "Test Task",
            "task_description": "Description",
            "language": "typescript",
            "framework": "react",
            "files_involved": ["a.ts", "b.ts"],
        }
        ctx = ExecutionContext.from_dict(data)
        assert ctx.task_title == "Test Task"
        assert ctx.language == "typescript"


# =============================================================================
# FailurePattern Tests
# =============================================================================


class TestFailurePattern:
    """Tests for FailurePattern model."""

    def test_create_failure_pattern(self):
        """Test creating FailurePattern."""
        pattern = FailurePattern(
            id="fp-abc123",
            error_type="ImportError",
            error_message_pattern="No module named '...'",
            occurrence_count=5,
            contexts=["Task A", "Task B"],
            suggested_fix="Install the missing module",
        )
        assert pattern.id == "fp-abc123"
        assert pattern.error_type == "ImportError"
        assert pattern.occurrence_count == 5
        assert len(pattern.contexts) == 2

    def test_failure_pattern_to_dict(self):
        """Test FailurePattern serialization."""
        pattern = FailurePattern(
            id="fp-test",
            error_type="TypeError",
            error_message_pattern="test pattern",
        )
        data = pattern.to_dict()
        assert data["id"] == "fp-test"
        assert "created_at" in data
        assert "updated_at" in data

    def test_failure_pattern_from_dict(self):
        """Test FailurePattern deserialization."""
        data = {
            "id": "fp-123",
            "error_type": "SyntaxError",
            "error_message_pattern": "invalid syntax at line N",
            "occurrence_count": 3,
            "contexts": ["ctx1"],
            "suggested_fix": None,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        pattern = FailurePattern.from_dict(data)
        assert pattern.id == "fp-123"
        assert pattern.occurrence_count == 3

    def test_failure_pattern_contexts_limit(self):
        """Test that contexts are limited in to_dict."""
        pattern = FailurePattern(
            id="fp-test",
            error_type="Error",
            error_message_pattern="test",
            contexts=[f"ctx-{i}" for i in range(20)],
        )
        data = pattern.to_dict()
        assert len(data["contexts"]) == 10  # Only last 10


# =============================================================================
# FixPattern Tests
# =============================================================================


class TestFixPattern:
    """Tests for FixPattern model."""

    def test_create_fix_pattern(self):
        """Test creating FixPattern."""
        pattern = FixPattern(
            id="fix-abc123",
            error_pattern="No module named '...'",
            fix_approach="pip install <module>",
            success_count=8,
            failure_count=2,
        )
        assert pattern.id == "fix-abc123"
        assert pattern.fix_approach == "pip install <module>"

    def test_fix_pattern_success_rate(self):
        """Test success rate calculation."""
        pattern = FixPattern(
            id="fix-test",
            error_pattern="test",
            fix_approach="test fix",
            success_count=8,
            failure_count=2,
        )
        assert pattern.success_rate == 0.8

    def test_fix_pattern_success_rate_zero(self):
        """Test success rate with no attempts."""
        pattern = FixPattern(
            id="fix-test",
            error_pattern="test",
            fix_approach="test fix",
            success_count=0,
            failure_count=0,
        )
        assert pattern.success_rate == 0.0

    def test_fix_pattern_to_dict(self):
        """Test FixPattern serialization."""
        pattern = FixPattern(
            id="fix-test",
            error_pattern="test",
            fix_approach="fix it",
            example_diff="- old\n+ new",
        )
        data = pattern.to_dict()
        assert data["id"] == "fix-test"
        assert data["example_diff"] == "- old\n+ new"

    def test_fix_pattern_from_dict(self):
        """Test FixPattern deserialization."""
        data = {
            "id": "fix-123",
            "error_pattern": "error",
            "fix_approach": "fix",
            "success_count": 5,
            "failure_count": 1,
            "example_diff": "",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        pattern = FixPattern.from_dict(data)
        assert pattern.id == "fix-123"
        assert pattern.success_count == 5


# =============================================================================
# PatternsDatabase Tests
# =============================================================================


class TestPatternsDatabase:
    """Tests for PatternsDatabase."""

    def test_create_database(self, temp_dir):
        """Test creating database."""
        db = PatternsDatabase(path=temp_dir)
        assert db.path == temp_dir
        assert db.path.exists()

    def test_save_and_get_failure_pattern(self, patterns_db):
        """Test saving and retrieving failure patterns."""
        pattern = FailurePattern(
            id="fp-test",
            error_type="Error",
            error_message_pattern="test pattern",
        )
        patterns_db.save_failure_pattern(pattern)

        retrieved = patterns_db.get_failure_pattern("fp-test")
        assert retrieved is not None
        assert retrieved.id == "fp-test"

    def test_find_failure_pattern(self, patterns_db):
        """Test finding failure pattern by pattern string."""
        pattern = FailurePattern(
            id="fp-find-test",
            error_type="ImportError",
            error_message_pattern="No module named '...'",
        )
        patterns_db.save_failure_pattern(pattern)

        found = patterns_db.find_failure_pattern("No module named '...'")
        assert found is not None
        assert found.id == "fp-find-test"

    def test_get_all_failure_patterns(self, patterns_db):
        """Test getting all failure patterns."""
        for i in range(3):
            pattern = FailurePattern(
                id=f"fp-{i}",
                error_type="Error",
                error_message_pattern=f"pattern-{i}",
            )
            patterns_db.save_failure_pattern(pattern)

        all_patterns = patterns_db.get_all_failure_patterns()
        assert len(all_patterns) == 3

    def test_save_and_get_fix_pattern(self, patterns_db):
        """Test saving and retrieving fix patterns."""
        pattern = FixPattern(
            id="fix-test",
            error_pattern="test error",
            fix_approach="test fix",
        )
        patterns_db.save_fix_pattern(pattern)

        retrieved = patterns_db.get_fix_pattern("fix-test")
        assert retrieved is not None
        assert retrieved.id == "fix-test"

    def test_find_fix_pattern(self, patterns_db):
        """Test finding fix pattern by error pattern."""
        pattern = FixPattern(
            id="fix-find-test",
            error_pattern="specific error pattern",
            fix_approach="specific fix",
        )
        patterns_db.save_fix_pattern(pattern)

        found = patterns_db.find_fix_pattern("specific error pattern")
        assert found is not None
        assert found.id == "fix-find-test"

    def test_get_best_fixes(self, patterns_db):
        """Test getting best fixes sorted by success rate."""
        error_pattern = "common error"
        patterns = [
            FixPattern(
                id="fix-1",
                error_pattern=error_pattern,
                fix_approach="fix 1",
                success_count=8,
                failure_count=2,  # 80%
            ),
            FixPattern(
                id="fix-2",
                error_pattern=error_pattern,
                fix_approach="fix 2",
                success_count=9,
                failure_count=1,  # 90%
            ),
            FixPattern(
                id="fix-3",
                error_pattern=error_pattern,
                fix_approach="fix 3",
                success_count=5,
                failure_count=5,  # 50%
            ),
        ]
        for p in patterns:
            patterns_db.save_fix_pattern(p)

        best = patterns_db.get_best_fixes(error_pattern, min_success_rate=0.7)
        assert len(best) == 2
        assert best[0].id == "fix-2"  # 90% first
        assert best[1].id == "fix-1"  # 80% second

    def test_database_persistence(self, temp_dir):
        """Test that database persists across instances."""
        # Create and save
        db1 = PatternsDatabase(path=temp_dir)
        pattern = FailurePattern(
            id="fp-persist",
            error_type="Error",
            error_message_pattern="persisted pattern",
        )
        db1.save_failure_pattern(pattern)

        # Create new instance and verify
        db2 = PatternsDatabase(path=temp_dir)
        retrieved = db2.get_failure_pattern("fp-persist")
        assert retrieved is not None
        assert retrieved.error_message_pattern == "persisted pattern"

    def test_database_auto_save_disabled(self, temp_dir):
        """Test database with auto_save disabled."""
        db = PatternsDatabase(path=temp_dir, auto_save=False)
        pattern = FailurePattern(
            id="fp-no-auto",
            error_type="Error",
            error_message_pattern="no auto save",
        )
        db.save_failure_pattern(pattern)

        # Check file was not written (auto_save=False still saves immediately)
        # The _auto_save method is called, but doesn't save
        # Actually, save_failure_pattern calls _auto_save which checks auto_save flag
        db2 = PatternsDatabase(path=temp_dir, auto_save=False)
        retrieved = db2.get_failure_pattern("fp-no-auto")
        # With auto_save=False, it should NOT persist
        assert retrieved is None


# =============================================================================
# FailurePatternTracker Tests
# =============================================================================


class TestFailurePatternTracker:
    """Tests for FailurePatternTracker."""

    def test_record_failure(self, failure_tracker, sample_error, sample_context):
        """Test recording a failure."""
        pattern = failure_tracker.record_failure(sample_error, sample_context)
        assert pattern is not None
        assert pattern.error_type == "ImportError"
        assert pattern.occurrence_count == 1

    def test_record_same_failure_twice(self, failure_tracker, sample_error, sample_context):
        """Test recording same failure increases count."""
        failure_tracker.record_failure(sample_error, sample_context)
        pattern = failure_tracker.record_failure(sample_error, sample_context)
        assert pattern.occurrence_count == 2

    def test_generalize_error_file_paths(self, failure_tracker):
        """Test error generalization for file paths."""
        message = "Error in '/path/to/file.py': invalid"
        generalized = failure_tracker._generalize_error(message)
        assert "*.py" in generalized

    def test_generalize_error_line_numbers(self, failure_tracker):
        """Test error generalization for line numbers."""
        message = "SyntaxError at line 42, column 10"
        generalized = failure_tracker._generalize_error(message)
        assert "line N" in generalized
        assert "column N" in generalized

    def test_generalize_error_memory_addresses(self, failure_tracker):
        """Test error generalization for memory addresses."""
        message = "Object at 0x7f3abc123def"
        generalized = failure_tracker._generalize_error(message)
        assert "0xXXXX" in generalized

    def test_get_fix_suggestion_requires_count(self, failure_tracker, sample_error, sample_context):
        """Test that suggestions require 3+ occurrences."""
        failure_tracker.record_failure(sample_error, sample_context)
        failure_tracker.set_suggested_fix(sample_error, "Install pandas")

        # Should return None - not enough occurrences
        suggestion = failure_tracker.get_fix_suggestion(sample_error)
        assert suggestion is None

        # Record more failures
        failure_tracker.record_failure(sample_error, sample_context)
        failure_tracker.record_failure(sample_error, sample_context)

        # Now should return suggestion
        suggestion = failure_tracker.get_fix_suggestion(sample_error)
        assert suggestion == "Install pandas"

    def test_set_suggested_fix(self, failure_tracker, sample_error, sample_context):
        """Test setting suggested fix."""
        failure_tracker.record_failure(sample_error, sample_context)
        success = failure_tracker.set_suggested_fix(sample_error, "pip install pandas")
        assert success is True

    def test_set_suggested_fix_unknown_pattern(self, failure_tracker, sample_error):
        """Test setting fix for unknown pattern."""
        success = failure_tracker.set_suggested_fix(sample_error, "fix")
        assert success is False

    def test_get_common_patterns(self, failure_tracker, sample_context):
        """Test getting common patterns."""
        # Create patterns with different counts
        errors = [
            TaskError(type="Error", message="common error"),
            TaskError(type="Error", message="rare error"),
        ]

        # Record common error 5 times
        for _ in range(5):
            failure_tracker.record_failure(errors[0], sample_context)

        # Record rare error once
        failure_tracker.record_failure(errors[1], sample_context)

        common = failure_tracker.get_common_patterns(limit=2)
        assert len(common) == 2
        assert common[0].occurrence_count == 5  # Most common first


# =============================================================================
# FixPatternTracker Tests
# =============================================================================


class TestFixPatternTracker:
    """Tests for FixPatternTracker."""

    def test_record_fix_success(self, fix_tracker, sample_error):
        """Test recording a successful fix."""
        actions = [
            FixAction(
                action_type="edit",
                file_path="requirements.txt",
                description="Add pandas to requirements",
            )
        ]
        pattern = fix_tracker.record_fix(sample_error, actions, success=True)

        assert pattern is not None
        assert pattern.success_count == 1
        assert pattern.failure_count == 0
        assert pattern.success_rate == 1.0

    def test_record_fix_failure(self, fix_tracker, sample_error):
        """Test recording a failed fix."""
        actions = [FixAction(action_type="edit", description="Wrong fix")]
        pattern = fix_tracker.record_fix(sample_error, actions, success=False)

        assert pattern.success_count == 0
        assert pattern.failure_count == 1
        assert pattern.success_rate == 0.0

    def test_record_multiple_fixes(self, fix_tracker, sample_error):
        """Test recording multiple fixes updates counts."""
        actions = [FixAction(action_type="edit", description="Fix")]

        # Record 8 successes and 2 failures
        for _ in range(8):
            fix_tracker.record_fix(sample_error, actions, success=True)
        for _ in range(2):
            fix_tracker.record_fix(sample_error, actions, success=False)

        pattern = fix_tracker.get_best_fix(sample_error)
        assert pattern is not None
        assert pattern.success_count == 8
        assert pattern.failure_count == 2
        assert pattern.success_rate == 0.8

    def test_record_fix_with_diff(self, fix_tracker, sample_error):
        """Test recording fix with diff."""
        actions = [FixAction(action_type="edit", description="Edit")]
        diff = "- old code\n+ new code"
        pattern = fix_tracker.record_fix(sample_error, actions, success=True, diff=diff)

        assert pattern.example_diff == diff

    def test_get_best_fix(self, fix_tracker, sample_error):
        """Test getting best fix."""
        actions = [FixAction(action_type="edit", description="Best fix")]
        fix_tracker.record_fix(sample_error, actions, success=True)

        best = fix_tracker.get_best_fix(sample_error)
        assert best is not None
        assert "Best fix" in best.fix_approach

    def test_get_best_fix_none(self, fix_tracker, sample_error):
        """Test getting best fix when none exists."""
        best = fix_tracker.get_best_fix(sample_error)
        assert best is None

    def test_get_all_fixes_for_error(self, fix_tracker, sample_error):
        """Test getting all fixes for an error."""
        # Record multiple fixes (but same error pattern, so same fix)
        for i in range(3):
            actions = [FixAction(action_type="edit", description=f"Fix {i}")]
            fix_tracker.record_fix(sample_error, actions, success=True)

        fixes = fix_tracker.get_all_fixes_for_error(sample_error)
        # All updates go to same pattern
        assert len(fixes) == 1

    def test_summarize_fix_multiple_actions(self, fix_tracker):
        """Test fix summary with multiple actions."""
        actions = [
            FixAction(action_type="edit", description="Edit file"),
            FixAction(action_type="create", description="Create test"),
        ]
        summary = fix_tracker._summarize_fix(actions)
        assert "Edit file" in summary
        assert "Create test" in summary
        assert ";" in summary

    def test_summarize_fix_no_description(self, fix_tracker):
        """Test fix summary without descriptions."""
        actions = [
            FixAction(action_type="edit", file_path="test.py"),
        ]
        summary = fix_tracker._summarize_fix(actions)
        assert "edit" in summary
        assert "test.py" in summary


# =============================================================================
# PatternSuggester Tests
# =============================================================================


class TestPatternSuggester:
    """Tests for PatternSuggester."""

    def test_on_error_with_high_success_fix(self, pattern_suggester, fix_tracker, sample_error):
        """Test suggestion with high success rate fix."""
        actions = [FixAction(action_type="edit", description="Install module")]

        # Record successful fixes
        for _ in range(7):
            fix_tracker.record_fix(sample_error, actions, success=True)

        suggestion = pattern_suggester.on_error(sample_error)
        assert suggestion is not None
        assert "success rate" in suggestion.lower()
        assert "Install module" in suggestion

    def test_on_error_with_failure_pattern_suggestion(
        self, pattern_suggester, failure_tracker, sample_error, sample_context
    ):
        """Test suggestion from failure pattern."""
        # Record failures and set suggestion
        for _ in range(3):
            failure_tracker.record_failure(sample_error, sample_context)
        failure_tracker.set_suggested_fix(sample_error, "Run pip install")

        suggestion = pattern_suggester.on_error(sample_error)
        assert suggestion is not None
        assert "Run pip install" in suggestion

    def test_on_error_with_common_pattern(
        self, pattern_suggester, failure_tracker, sample_error, sample_context
    ):
        """Test suggestion for common pattern without fix."""
        # Record failures without setting suggestion
        for _ in range(5):
            failure_tracker.record_failure(sample_error, sample_context)

        suggestion = pattern_suggester.on_error(sample_error)
        assert suggestion is not None
        assert "5 times" in suggestion

    def test_on_error_unknown(self, pattern_suggester):
        """Test no suggestion for unknown error."""
        error = TaskError(type="NewError", message="Never seen before")
        suggestion = pattern_suggester.on_error(error)
        assert suggestion is None

    def test_on_error_fix_with_diff(self, pattern_suggester, fix_tracker, sample_error):
        """Test suggestion includes diff."""
        actions = [FixAction(action_type="edit", description="Fix")]
        diff = "- old\n+ new"

        for _ in range(7):
            fix_tracker.record_fix(sample_error, actions, success=True, diff=diff)

        suggestion = pattern_suggester.on_error(sample_error)
        assert suggestion is not None
        assert "Example" in suggestion
        assert "old" in suggestion

    def test_get_common_issues(
        self, pattern_suggester, failure_tracker, fix_tracker, sample_context
    ):
        """Test getting common issues."""
        # Create patterns
        error1 = TaskError(type="Error", message="common error")
        error2 = TaskError(type="Error", message="another error")

        for _ in range(5):
            failure_tracker.record_failure(error1, sample_context)
        for _ in range(3):
            failure_tracker.record_failure(error2, sample_context)

        # Add fix for error1
        actions = [FixAction(action_type="edit", description="Fix")]
        fix_tracker.record_fix(error1, actions, success=True)

        issues = pattern_suggester.get_common_issues(limit=5)
        assert len(issues) == 2
        assert issues[0]["count"] == 5
        assert issues[0]["has_fix"] is True


# =============================================================================
# AgentsUpdater Tests
# =============================================================================


class TestAgentsUpdater:
    """Tests for AgentsUpdater."""

    def test_propose_update_empty(self):
        """Test proposal with no patterns."""
        updater = AgentsUpdater()
        update = updater.propose_update([])
        assert update is None

    def test_propose_update_low_confidence(self):
        """Test proposal filters low confidence patterns."""
        updater = AgentsUpdater(min_confidence=0.8, min_success_count=5)

        # Mock pattern with low confidence
        class MockPattern:
            confidence = 0.5
            success_count = 10

        update = updater.propose_update([MockPattern()])
        assert update is None

    def test_propose_update_low_success_count(self):
        """Test proposal filters low success count."""
        updater = AgentsUpdater(min_confidence=0.8, min_success_count=5)

        class MockPattern:
            confidence = 0.9
            success_count = 2

        update = updater.propose_update([MockPattern()])
        assert update is None

    def test_propose_update_with_fix_pattern(self):
        """Test proposal with FixPattern."""
        updater = AgentsUpdater(min_confidence=0.7, min_success_count=3)

        pattern = FixPattern(
            id="fix-test",
            error_pattern="ImportError: No module named '...'",
            fix_approach="pip install <module>",
            success_count=10,
            failure_count=2,
        )

        update = updater.propose_update([pattern])
        assert update is not None
        assert "Learned Patterns" in update
        assert "pip install" in update

    def test_propose_update_with_skill(self):
        """Test proposal with Skill-like object."""
        updater = AgentsUpdater(min_confidence=0.8, min_success_count=5)

        class MockSkill:
            title = "Use type hints"
            description = "When writing Python functions"
            pattern = "def func(x: int) -> str:"
            rationale = "Type safety"
            confidence = 0.9
            success_count = 10

        update = updater.propose_update([MockSkill()])
        assert update is not None
        assert "Use type hints" in update
        assert "Type safety" in update
        assert "90%" in update

    def test_propose_update_format(self):
        """Test proposal format."""
        updater = AgentsUpdater(min_confidence=0.5, min_success_count=1)

        pattern = FixPattern(
            id="fix-format",
            error_pattern="test error",
            fix_approach="test fix",
            success_count=5,
            failure_count=0,
        )

        update = updater.propose_update([pattern])
        assert update is not None
        assert "## Learned Patterns" in update
        assert "Auto-generated" in update
        assert "### 1." in update


# =============================================================================
# Integration Tests
# =============================================================================


class TestPatternsIntegration:
    """Integration tests for pattern system."""

    def test_full_workflow(self, temp_dir):
        """Test full pattern recording and suggestion workflow."""
        # Setup
        db = PatternsDatabase(path=temp_dir)
        failure_tracker = FailurePatternTracker(db)
        fix_tracker = FixPatternTracker(db)
        suggester = PatternSuggester(failure_tracker, fix_tracker)

        error = TaskError(type="ImportError", message="No module named 'requests'")
        context = ExecutionContext(
            task_title="Add API client",
            language="python",
        )

        # Phase 1: Record failures
        for _ in range(3):
            failure_tracker.record_failure(error, context)

        # Phase 2: Set suggested fix
        failure_tracker.set_suggested_fix(error, "pip install requests")

        # Phase 3: Get suggestion
        suggestion = suggester.on_error(error)
        assert suggestion is not None
        assert "pip install requests" in suggestion

        # Phase 4: Record successful fix
        actions = [FixAction(action_type="install", description="pip install requests")]
        fix_tracker.record_fix(error, actions, success=True)

        # Phase 5: Verify persistence
        db2 = PatternsDatabase(path=temp_dir)
        failure_tracker2 = FailurePatternTracker(db2)
        fix_tracker2 = FixPatternTracker(db2)
        suggester2 = PatternSuggester(failure_tracker2, fix_tracker2)

        suggestion2 = suggester2.on_error(error)
        assert suggestion2 is not None

    def test_pattern_generalization_consistency(self, failure_tracker, sample_context):
        """Test that similar errors match the same pattern."""
        # Two errors that should match same pattern
        error1 = TaskError(
            type="SyntaxError",
            message="invalid syntax at line 10, column 5 in '/path/to/file.py'",
        )
        error2 = TaskError(
            type="SyntaxError",
            message="invalid syntax at line 42, column 12 in '/other/path/test.py'",
        )

        failure_tracker.record_failure(error1, sample_context)
        pattern2 = failure_tracker.record_failure(error2, sample_context)

        # Should be same pattern, count = 2
        assert pattern2.occurrence_count == 2

    def test_agents_updater_with_real_patterns(self, patterns_db, fix_tracker):
        """Test AgentsUpdater with real fix patterns."""
        # Create successful fix patterns
        for i in range(6):
            error = TaskError(type="ImportError", message=f"No module named 'pkg{i}'")
            actions = [FixAction(action_type="install", description=f"Install pkg{i}")]
            fix_tracker.record_fix(error, actions, success=True)

        patterns = patterns_db.get_all_fix_patterns()
        updater = AgentsUpdater(min_confidence=0.5, min_success_count=1)
        update = updater.propose_update(patterns)

        assert update is not None
        # All patterns have 100% success rate
        assert "100%" in update


# =============================================================================
# Edge Cases
# =============================================================================


class TestPatternEdgeCases:
    """Edge case tests."""

    def test_empty_error_message(self, failure_tracker, sample_context):
        """Test handling empty error message."""
        error = TaskError(type="Error", message="")
        pattern = failure_tracker.record_failure(error, sample_context)
        assert pattern is not None

    def test_very_long_error_message(self, failure_tracker, sample_context):
        """Test handling very long error message."""
        error = TaskError(type="Error", message="x" * 10000)
        pattern = failure_tracker.record_failure(error, sample_context)
        assert pattern is not None

    def test_special_characters_in_error(self, failure_tracker, sample_context):
        """Test handling special regex characters in error."""
        error = TaskError(type="Error", message="Error: [.*+?^${}()|[]\\]")
        # Should not raise
        pattern = failure_tracker.record_failure(error, sample_context)
        assert pattern is not None

    def test_unicode_in_error(self, failure_tracker, sample_context):
        """Test handling unicode in error message."""
        error = TaskError(type="Error", message="Error: 你好世界 🎉")
        pattern = failure_tracker.record_failure(error, sample_context)
        assert pattern is not None

    def test_database_corrupted_file(self, temp_dir):
        """Test handling corrupted database file."""
        # Write corrupted JSON
        (temp_dir / "failure_patterns.json").write_text("not valid json")

        # Should not raise, just warn
        db = PatternsDatabase(path=temp_dir)
        patterns = db.get_all_failure_patterns()
        assert patterns == []

    def test_fix_action_minimal(self, fix_tracker, sample_error):
        """Test fix action with minimal data."""
        actions = [FixAction(action_type="edit")]
        pattern = fix_tracker.record_fix(sample_error, actions, success=True)
        assert pattern is not None

    def test_empty_fix_actions(self, fix_tracker, sample_error):
        """Test recording fix with no actions."""
        pattern = fix_tracker.record_fix(sample_error, [], success=True)
        assert pattern is not None
        assert "No specific fix recorded" in pattern.fix_approach
