"""Tests for the learning module (Phase 5.3)."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ai_infra.executor.learning import (
    FailurePattern,
    LearningStats,
    LearningStore,
    PatternType,
    PromptRefiner,
    SuccessPattern,
    TaskType,
)

# =============================================================================
# Test Data Models
# =============================================================================


class TestPatternType:
    """Tests for PatternType enum."""

    def test_pattern_type_values(self):
        """Test all pattern types have correct values."""
        assert PatternType.FAILURE.value == "failure"
        assert PatternType.SUCCESS.value == "success"
        assert PatternType.RETRY.value == "retry"


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_type_values(self):
        """Test all task types have correct values."""
        assert TaskType.CREATE_FILE.value == "create_file"
        assert TaskType.MODIFY_FILE.value == "modify_file"
        assert TaskType.ADD_FEATURE.value == "add_feature"
        assert TaskType.FIX_BUG.value == "fix_bug"
        assert TaskType.REFACTOR.value == "refactor"
        assert TaskType.ADD_TESTS.value == "add_tests"
        assert TaskType.DOCUMENTATION.value == "documentation"
        assert TaskType.CONFIGURATION.value == "configuration"
        assert TaskType.UNKNOWN.value == "unknown"


class TestFailurePattern:
    """Tests for FailurePattern dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        pattern = FailurePattern(
            pattern_id="test_001",
            category="SYNTAX_ERROR",
            error_signature="abc123",
            error_message="SyntaxError: invalid syntax",
        )
        assert pattern.pattern_id == "test_001"
        assert pattern.category == "SYNTAX_ERROR"
        assert pattern.error_signature == "abc123"
        assert pattern.occurrence_count == 1
        assert pattern.resolved is False

    def test_create_full(self):
        """Test creating with all args."""
        now = datetime.now(UTC)
        pattern = FailurePattern(
            pattern_id="test_002",
            category="IMPORT_ERROR",
            error_signature="def456",
            error_message="ImportError: No module named 'foo'",
            task_context="Adding foo feature",
            file_hints=["src/foo.py", "tests/test_foo.py"],
            fix_attempts=[{"attempt": 1, "result": "failed"}],
            occurrence_count=3,
            first_seen=now,
            last_seen=now,
            resolved=True,
            resolution_strategy="Install foo package",
        )
        assert pattern.file_hints == ["src/foo.py", "tests/test_foo.py"]
        assert len(pattern.fix_attempts) == 1
        assert pattern.resolved is True
        assert pattern.resolution_strategy == "Install foo package"

    def test_to_dict(self):
        """Test serialization to dict."""
        pattern = FailurePattern(
            pattern_id="test_003",
            category="TYPE_ERROR",
            error_signature="ghi789",
            error_message="TypeError: expected int",
        )
        data = pattern.to_dict()

        assert data["pattern_id"] == "test_003"
        assert data["category"] == "TYPE_ERROR"
        assert data["error_signature"] == "ghi789"
        assert "first_seen" in data
        assert "last_seen" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "pattern_id": "test_004",
            "category": "ASSERTION_ERROR",
            "error_signature": "jkl012",
            "error_message": "AssertionError",
            "first_seen": "2024-01-01T00:00:00+00:00",
            "last_seen": "2024-01-02T00:00:00+00:00",
        }
        pattern = FailurePattern.from_dict(data)

        assert pattern.pattern_id == "test_004"
        assert pattern.category == "ASSERTION_ERROR"
        assert pattern.first_seen == datetime(2024, 1, 1, tzinfo=UTC)

    def test_round_trip(self):
        """Test serialization round-trip."""
        original = FailurePattern(
            pattern_id="test_005",
            category="RUNTIME_ERROR",
            error_signature="mno345",
            error_message="RuntimeError: bad state",
            file_hints=["src/state.py"],
            occurrence_count=5,
        )
        data = original.to_dict()
        restored = FailurePattern.from_dict(data)

        assert restored.pattern_id == original.pattern_id
        assert restored.category == original.category
        assert restored.occurrence_count == original.occurrence_count
        assert restored.file_hints == original.file_hints


class TestSuccessPattern:
    """Tests for SuccessPattern dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        pattern = SuccessPattern(
            pattern_id="success_001",
            task_type=TaskType.ADD_FEATURE,
            task_signature="abc123",
            task_title="Add login feature",
        )
        assert pattern.pattern_id == "success_001"
        assert pattern.task_type == TaskType.ADD_FEATURE
        assert pattern.success_count == 1

    def test_create_full(self):
        """Test creating with all args."""
        pattern = SuccessPattern(
            pattern_id="success_002",
            task_type=TaskType.FIX_BUG,
            task_signature="def456",
            task_title="Fix login bug",
            task_description="Fix the login timeout issue",
            file_hints=["src/auth.py"],
            prompt_template="Fix the bug by...",
            context_strategy="file_focused",
            verification_level="tests",
            execution_time=45.5,
            files_modified=["src/auth.py", "tests/test_auth.py"],
            success_count=3,
        )
        assert pattern.execution_time == 45.5
        assert len(pattern.files_modified) == 2
        assert pattern.success_count == 3

    def test_to_dict(self):
        """Test serialization to dict."""
        pattern = SuccessPattern(
            pattern_id="success_003",
            task_type=TaskType.REFACTOR,
            task_signature="ghi789",
            task_title="Refactor utils",
        )
        data = pattern.to_dict()

        assert data["pattern_id"] == "success_003"
        assert data["task_type"] == "refactor"  # enum value
        assert data["task_signature"] == "ghi789"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "pattern_id": "success_004",
            "task_type": "add_tests",
            "task_signature": "jkl012",
            "task_title": "Add unit tests",
            "created_at": "2024-01-01T00:00:00+00:00",
            "last_used": "2024-01-02T00:00:00+00:00",
        }
        pattern = SuccessPattern.from_dict(data)

        assert pattern.pattern_id == "success_004"
        assert pattern.task_type == TaskType.ADD_TESTS
        assert pattern.created_at == datetime(2024, 1, 1, tzinfo=UTC)

    def test_round_trip(self):
        """Test serialization round-trip."""
        original = SuccessPattern(
            pattern_id="success_005",
            task_type=TaskType.DOCUMENTATION,
            task_signature="mno345",
            task_title="Update docs",
            execution_time=30.0,
            success_count=7,
        )
        data = original.to_dict()
        restored = SuccessPattern.from_dict(data)

        assert restored.pattern_id == original.pattern_id
        assert restored.task_type == original.task_type
        assert restored.execution_time == original.execution_time


class TestLearningStats:
    """Tests for LearningStats dataclass."""

    def test_default_values(self):
        """Test default values are correct."""
        stats = LearningStats()
        assert stats.total_failures == 0
        assert stats.total_successes == 0
        assert stats.resolved_failures == 0
        assert stats.avg_fix_attempts == 0.0
        assert stats.top_failure_categories == []
        assert stats.top_task_types == []

    def test_to_dict(self):
        """Test serialization to dict."""
        stats = LearningStats(
            total_failures=10,
            total_successes=20,
            resolved_failures=5,
            avg_fix_attempts=2.5,
            top_failure_categories=[("SYNTAX_ERROR", 5), ("TYPE_ERROR", 3)],
        )
        data = stats.to_dict()

        assert data["total_failures"] == 10
        assert data["total_successes"] == 20
        assert len(data["top_failure_categories"]) == 2

    def test_summary(self):
        """Test human-readable summary."""
        stats = LearningStats(
            total_failures=10,
            total_successes=20,
            resolved_failures=5,
            top_failure_categories=[("SYNTAX_ERROR", 5)],
            top_task_types=[("add_feature", 10)],
        )
        summary = stats.summary()

        assert "Learning Statistics" in summary
        assert "Total Failure Patterns: 10" in summary
        assert "Total Success Patterns: 20" in summary
        assert "SYNTAX_ERROR" in summary
        assert "add_feature" in summary


# =============================================================================
# Test LearningStore
# =============================================================================


class TestLearningStoreInMemory:
    """Tests for LearningStore with in-memory storage."""

    def test_create_in_memory(self):
        """Test creating in-memory store."""
        store = LearningStore()
        assert store.failure_count == 0
        assert store.success_count == 0

    def test_generate_error_signature(self):
        """Test error signature generation."""
        sig1 = LearningStore.generate_error_signature("SyntaxError: invalid syntax at line 42")
        sig2 = LearningStore.generate_error_signature("SyntaxError: invalid syntax at line 100")
        # Line numbers should be normalized, so signatures should match
        assert sig1 == sig2

    def test_generate_error_signature_different(self):
        """Test different errors have different signatures."""
        sig1 = LearningStore.generate_error_signature("SyntaxError: invalid syntax")
        sig2 = LearningStore.generate_error_signature("TypeError: expected int")
        assert sig1 != sig2

    def test_generate_task_signature(self):
        """Test task signature generation."""
        sig1 = LearningStore.generate_task_signature(
            "Add login feature",
            ["src/auth.py"],
        )
        sig2 = LearningStore.generate_task_signature(
            "Add login feature",
            ["src/auth.py"],
        )
        assert sig1 == sig2

    def test_infer_task_type_create(self):
        """Test task type inference for create."""
        task_type = LearningStore.infer_task_type("Create a new config file")
        assert task_type == TaskType.CREATE_FILE

    def test_infer_task_type_modify(self):
        """Test task type inference for modify."""
        task_type = LearningStore.infer_task_type("Update the settings module")
        assert task_type == TaskType.MODIFY_FILE

    def test_infer_task_type_bug(self):
        """Test task type inference for bug fix."""
        task_type = LearningStore.infer_task_type("Fix the login bug")
        assert task_type == TaskType.FIX_BUG

    def test_infer_task_type_feature(self):
        """Test task type inference for feature."""
        task_type = LearningStore.infer_task_type("Implement user authentication")
        assert task_type == TaskType.ADD_FEATURE

    def test_infer_task_type_refactor(self):
        """Test task type inference for refactor."""
        task_type = LearningStore.infer_task_type("Refactor the database module")
        assert task_type == TaskType.REFACTOR

    def test_infer_task_type_tests(self):
        """Test task type inference for tests."""
        task_type = LearningStore.infer_task_type("Add tests for auth module")
        assert task_type == TaskType.ADD_TESTS

    def test_infer_task_type_docs(self):
        """Test task type inference for docs."""
        task_type = LearningStore.infer_task_type("Write documentation for module")
        assert task_type == TaskType.DOCUMENTATION

    def test_infer_task_type_config(self):
        """Test task type inference for config."""
        task_type = LearningStore.infer_task_type("Add environment settings")
        assert task_type == TaskType.CONFIGURATION

    def test_infer_task_type_unknown(self):
        """Test task type inference fallback."""
        task_type = LearningStore.infer_task_type("Random task")
        assert task_type == TaskType.UNKNOWN


class TestLearningStoreFailures:
    """Tests for failure pattern operations."""

    def test_record_failure(self):
        """Test recording a failure pattern."""
        store = LearningStore()
        pattern = store.record_failure(
            category="SYNTAX_ERROR",
            error_message="SyntaxError: invalid syntax",
            task_context="Implementing feature X",
            file_hints=["src/feature.py"],
        )

        assert pattern.category == "SYNTAX_ERROR"
        assert pattern.occurrence_count == 1
        assert store.failure_count == 1

    def test_record_failure_updates_existing(self):
        """Test recording same failure increments count."""
        store = LearningStore()

        pattern1 = store.record_failure(
            category="SYNTAX_ERROR",
            error_message="SyntaxError: invalid syntax",
        )
        pattern2 = store.record_failure(
            category="SYNTAX_ERROR",
            error_message="SyntaxError: invalid syntax",
        )

        # Same pattern updated
        assert store.failure_count == 1
        assert pattern2.occurrence_count == 2
        assert pattern1.pattern_id == pattern2.pattern_id

    def test_record_failure_with_fix_attempt(self):
        """Test recording failure with fix attempt."""
        store = LearningStore()
        pattern = store.record_failure(
            category="IMPORT_ERROR",
            error_message="ImportError: No module named 'foo'",
            fix_attempt={"attempt": 1, "strategy": "install package"},
        )

        assert len(pattern.fix_attempts) == 1
        assert pattern.fix_attempts[0]["strategy"] == "install package"

    def test_mark_failure_resolved(self):
        """Test marking a failure as resolved."""
        store = LearningStore()
        pattern = store.record_failure(
            category="TYPE_ERROR",
            error_message="TypeError: expected int",
        )

        result = store.mark_failure_resolved(
            pattern.pattern_id,
            resolution_strategy="Add type conversion",
        )

        assert result is True
        assert pattern.resolved is True
        assert pattern.resolution_strategy == "Add type conversion"

    def test_mark_failure_resolved_not_found(self):
        """Test marking non-existent failure."""
        store = LearningStore()
        result = store.mark_failure_resolved("nonexistent", "strategy")
        assert result is False

    def test_find_similar_failures_exact_match(self):
        """Test finding similar failures by exact signature."""
        store = LearningStore()

        store.record_failure(
            category="SYNTAX_ERROR",
            error_message="SyntaxError: invalid syntax at line 10",
        )

        similar = store.find_similar_failures("SyntaxError: invalid syntax at line 50")

        # Same signature (line numbers normalized)
        assert len(similar) == 1
        assert similar[0].category == "SYNTAX_ERROR"

    def test_find_similar_failures_no_match(self):
        """Test finding similar failures with no match."""
        store = LearningStore()

        store.record_failure(
            category="SYNTAX_ERROR",
            error_message="SyntaxError: invalid syntax",
        )

        similar = store.find_similar_failures("TypeError: expected int")
        assert len(similar) == 0

    def test_get_resolution_strategies(self):
        """Test getting resolution strategies for a category."""
        store = LearningStore()

        pattern = store.record_failure(
            category="IMPORT_ERROR",
            error_message="ImportError: No module named 'foo'",
        )
        store.mark_failure_resolved(pattern.pattern_id, "Install foo package")

        strategies = store.get_resolution_strategies("IMPORT_ERROR")
        assert "Install foo package" in strategies

    def test_get_resolution_strategies_no_resolved(self):
        """Test getting strategies when none resolved."""
        store = LearningStore()

        store.record_failure(
            category="SYNTAX_ERROR",
            error_message="SyntaxError",
        )

        strategies = store.get_resolution_strategies("SYNTAX_ERROR")
        assert strategies == []


class TestLearningStoreSuccesses:
    """Tests for success pattern operations."""

    def test_record_success(self):
        """Test recording a success pattern."""
        store = LearningStore()
        pattern = store.record_success(
            task_title="Add login feature",
            task_description="Implement user login",
            file_hints=["src/auth.py"],
            prompt_template="Implement the feature by...",
            execution_time=30.0,
        )

        assert pattern.task_type == TaskType.ADD_FEATURE
        assert pattern.success_count == 1
        assert store.success_count == 1

    def test_record_success_updates_existing(self):
        """Test recording same success increments count."""
        store = LearningStore()

        store.record_success(
            task_title="Add login feature",
            file_hints=["src/auth.py"],
            execution_time=30.0,
        )
        pattern2 = store.record_success(
            task_title="Add login feature",
            file_hints=["src/auth.py"],
            execution_time=40.0,
        )

        # Same pattern updated
        assert store.success_count == 1
        assert pattern2.success_count == 2
        # Running average: (30 + 40) / 2 = 35
        assert pattern2.execution_time == 35.0

    def test_find_similar_successes_by_signature(self):
        """Test finding similar successes by signature."""
        store = LearningStore()

        store.record_success(
            task_title="Implement login feature",  # "Implement" triggers ADD_FEATURE
            file_hints=["src/auth.py"],
        )

        similar = store.find_similar_successes(
            "Implement login feature",
            file_hints=["src/auth.py"],
        )

        assert len(similar) == 1
        assert similar[0].task_type == TaskType.ADD_FEATURE

    def test_find_similar_successes_by_type(self):
        """Test finding similar successes by task type."""
        store = LearningStore()

        store.record_success(
            task_title="Fix authentication bug",
            file_hints=["src/auth.py"],
        )

        # Different title but same type
        similar = store.find_similar_successes(
            "Fix login issue",
            file_hints=["src/login.py"],
        )

        assert len(similar) == 1
        assert similar[0].task_type == TaskType.FIX_BUG

    def test_get_template_for_task_type(self):
        """Test getting best template for task type."""
        store = LearningStore()

        store.record_success(
            task_title="Fix bug A",
            prompt_template="First fix approach",
        )
        # Record same pattern twice to increase count
        store.record_success(
            task_title="Fix bug A",
            prompt_template="First fix approach",
        )
        store.record_success(
            task_title="Fix bug B",
            prompt_template="Second fix approach",
        )

        template = store.get_template_for_task_type(TaskType.FIX_BUG)
        assert template == "First fix approach"  # Higher count

    def test_get_template_for_task_type_none(self):
        """Test getting template when none exists."""
        store = LearningStore()
        template = store.get_template_for_task_type(TaskType.REFACTOR)
        assert template is None


class TestLearningStoreStats:
    """Tests for statistics."""

    def test_get_stats_empty(self):
        """Test stats on empty store."""
        store = LearningStore()
        stats = store.get_stats()

        assert stats.total_failures == 0
        assert stats.total_successes == 0

    def test_get_stats_with_data(self):
        """Test stats with data."""
        store = LearningStore()

        store.record_failure("SYNTAX_ERROR", "error 1")
        store.record_failure("SYNTAX_ERROR", "error 1")  # Same pattern
        store.record_failure("TYPE_ERROR", "error 2")

        store.record_success("Fix bug", execution_time=30.0)
        store.record_success("Add feature", execution_time=60.0)

        stats = store.get_stats()

        assert stats.total_failures == 3  # 2 occurrences + 1
        assert stats.total_successes == 2
        assert len(stats.top_failure_categories) > 0
        assert len(stats.top_task_types) > 0


class TestLearningStorePersistence:
    """Tests for persistence to disk."""

    def test_save_and_load(self):
        """Test saving and loading patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "learning"

            # Create and populate store
            store1 = LearningStore(data_dir)
            store1.record_failure("SYNTAX_ERROR", "test error", task_context="test")
            store1.record_success("Test task", execution_time=10.0)
            store1.save()

            # Load in new store
            store2 = LearningStore(data_dir)

            assert store2.failure_count == 1
            assert store2.success_count == 1

    def test_auto_save_on_record(self):
        """Test auto-save when recording patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "learning"

            store = LearningStore(data_dir)
            store.record_failure("SYNTAX_ERROR", "test error")

            # Check file exists
            failure_file = data_dir / "failure_patterns.json"
            assert failure_file.exists()

            with open(failure_file) as f:
                data = json.load(f)
            assert len(data["patterns"]) == 1

    def test_clear(self):
        """Test clearing all patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "learning"

            store = LearningStore(data_dir)
            store.record_failure("ERROR", "msg")
            store.record_success("task")

            store.clear()

            assert store.failure_count == 0
            assert store.success_count == 0


class TestLearningStoreTrimming:
    """Tests for pattern trimming."""

    def test_trim_old_patterns(self):
        """Test trimming when max_patterns exceeded."""
        store = LearningStore(max_patterns=5)

        # Record more than max
        for i in range(10):
            store.record_failure("ERROR", f"error message {i}")

        # Should be trimmed to max
        assert store.failure_count <= 5


# =============================================================================
# Test PromptRefiner
# =============================================================================


class TestPromptRefiner:
    """Tests for PromptRefiner."""

    def test_refine_prompt_no_hints(self):
        """Test refinement with no matching patterns."""
        store = LearningStore()
        refiner = PromptRefiner(store)

        refined = refiner.refine_prompt(
            "Original prompt",
            task_title="New task",
        )

        # No hints, original unchanged
        assert refined == "Original prompt"

    def test_refine_prompt_with_success_hint(self):
        """Test refinement adds success hints."""
        store = LearningStore()
        store.record_success(
            task_title="Add login feature",
            prompt_template="Use JWT tokens for auth",
        )

        refiner = PromptRefiner(store)
        refined = refiner.refine_prompt(
            "Implement auth",
            task_title="Add login feature",
        )

        assert "Learning Hints" in refined
        assert "Similar tasks succeeded" in refined

    def test_refine_prompt_with_failure_hint(self):
        """Test refinement adds failure warnings."""
        store = LearningStore()
        store.record_failure(
            "SYNTAX_ERROR",
            "SyntaxError in auth.py",
            file_hints=["src/auth.py"],
        )

        refiner = PromptRefiner(store)
        refined = refiner.refine_prompt(
            "Modify auth module",
            task_title="Update auth",
            file_hints=["src/auth.py"],
        )

        assert "Learning Hints" in refined
        assert "CAUTION" in refined

    def test_suggest_improvements_with_resolution(self):
        """Test suggestions from resolved failures."""
        store = LearningStore()
        pattern = store.record_failure(
            "IMPORT_ERROR",
            "ImportError: No module named 'foo'",
        )
        store.mark_failure_resolved(pattern.pattern_id, "Install foo package")

        refiner = PromptRefiner(store)
        suggestions = refiner.suggest_improvements(
            "Implement feature",
            "ImportError: No module named 'foo'",
        )

        assert len(suggestions) > 0
        assert any("Install foo package" in s for s in suggestions)

    def test_suggest_improvements_syntax_hint(self):
        """Test generic syntax hints."""
        store = LearningStore()
        refiner = PromptRefiner(store)

        suggestions = refiner.suggest_improvements(
            "Bad code",
            "SyntaxError: invalid syntax",
        )

        assert len(suggestions) > 0
        assert any("syntax" in s.lower() for s in suggestions)

    def test_suggest_improvements_import_hint(self):
        """Test generic import hints."""
        store = LearningStore()
        refiner = PromptRefiner(store)

        suggestions = refiner.suggest_improvements(
            "Bad import",
            "ImportError: No module",
        )

        assert len(suggestions) > 0
        assert any("import" in s.lower() for s in suggestions)


# =============================================================================
# Integration Tests
# =============================================================================


class TestLearningStoreIntegration:
    """Integration tests for the learning module."""

    def test_full_workflow(self):
        """Test a complete learning workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "learning"
            store = LearningStore(data_dir)

            # Simulate first attempt - failure
            pattern = store.record_failure(
                category="SYNTAX_ERROR",
                error_message="SyntaxError: unexpected EOF",
                task_context="Adding config parser",
                file_hints=["src/config.py"],
                fix_attempt={"attempt": 1, "action": "check brackets"},
            )

            # Second attempt - still fails
            store.record_failure(
                category="SYNTAX_ERROR",
                error_message="SyntaxError: unexpected EOF",
                fix_attempt={"attempt": 2, "action": "rewrite function"},
            )

            # Third attempt - success
            store.mark_failure_resolved(
                pattern.pattern_id, "Rewrote function with proper indentation"
            )

            store.record_success(
                task_title="Add config parser",
                file_hints=["src/config.py"],
                prompt_template="Use configparser with proper error handling",
                execution_time=45.0,
            )

            # Verify state
            stats = store.get_stats()
            assert stats.total_failures == 2
            assert stats.total_successes == 1
            assert stats.resolved_failures == 1

            # New similar task
            refiner = PromptRefiner(store)
            refined = refiner.refine_prompt(
                "Parse settings from file",
                task_title="Add settings parser",
                file_hints=["src/settings.py"],
            )

            # Should include learned hints
            assert "Learning Hints" in refined

    def test_multiple_task_types(self):
        """Test learning across different task types."""
        store = LearningStore()

        # Record successes for different types
        store.record_success("Create user model", file_hints=["models/user.py"])
        store.record_success("Fix login bug", file_hints=["auth/login.py"])
        store.record_success("Refactor utils", file_hints=["utils.py"])
        store.record_success("Add unit tests", file_hints=["tests/test_user.py"])

        stats = store.get_stats()

        # Should have 4 different task types
        assert len(stats.top_task_types) == 4
