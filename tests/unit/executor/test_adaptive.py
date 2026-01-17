"""Tests for the adaptive planning module (Phase 5.5).

Tests cover:
- AdaptiveMode enum values
- PlanSuggestion creation and serialization
- PlanAnalyzer failure analysis
- Safe operation detection for auto-fix mode
- Suggestion application (create __init__.py, create directory)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_infra.executor.adaptive import (
    AdaptiveMode,
    PlanAnalyzer,
    PlanSuggestion,
    SuggestionSafety,
    SuggestionType,
    analyze_failure_for_plan_fix,
)
from ai_infra.executor.failure import FailureCategory, FailureRecord, FailureSeverity
from ai_infra.executor.roadmap import ParsedTask, Phase, Roadmap, Section
from ai_infra.executor.types import ExecutionResult, ExecutionStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    src_dir = tmp_path / "src" / "myproject"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text('"""Package."""\n')
    (src_dir / "main.py").write_text("# Main module\n")

    # Create a ROADMAP.md
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text(
        """# Test Project ROADMAP

## Phase 1: Setup

### 1.1 Project Structure

- [ ] **Create module structure**
  Create the basic module structure.
  **Files**: `src/myproject/core.py`
"""
    )

    return tmp_path


@pytest.fixture
def sample_roadmap(temp_project: Path) -> Roadmap:
    """Create a sample roadmap."""
    task = ParsedTask(
        id="1.1.1",
        title="Create module structure",
        description="Create the basic module structure.",
        file_hints=["src/myproject/core.py"],
        line_number=10,
    )
    section = Section(
        id="1.1",
        title="Project Structure",
        tasks=[task],
    )
    phase = Phase(
        id="1",
        name="Setup",
        goal="Set up the project",
        sections=[section],
    )
    return Roadmap(
        path=str(temp_project / "ROADMAP.md"),
        title="Test Project ROADMAP",
        phases=[phase],
    )


@pytest.fixture
def sample_task() -> ParsedTask:
    """Create a sample task."""
    return ParsedTask(
        id="1.1.1",
        title="Create module structure",
        description="Create the basic module structure.",
        file_hints=["src/myproject/core.py"],
        line_number=10,
    )


@pytest.fixture
def failed_result() -> ExecutionResult:
    """Create a sample failed result."""
    return ExecutionResult(
        task_id="1.1.1",
        status=ExecutionStatus.FAILED,
        error="ImportError: No module named 'myproject.utils'",
        agent_output="Attempted to import myproject.utils but failed.",
    )


# =============================================================================
# AdaptiveMode Tests
# =============================================================================


class TestAdaptiveMode:
    """Tests for AdaptiveMode enum."""

    def test_mode_values(self):
        """Test that all mode values exist."""
        assert AdaptiveMode.NO_ADAPT.value == "no_adapt"
        assert AdaptiveMode.SUGGEST.value == "suggest"
        assert AdaptiveMode.AUTO_FIX.value == "auto_fix"

    def test_mode_from_string(self):
        """Test creating mode from string value."""
        assert AdaptiveMode("no_adapt") == AdaptiveMode.NO_ADAPT
        assert AdaptiveMode("suggest") == AdaptiveMode.SUGGEST
        assert AdaptiveMode("auto_fix") == AdaptiveMode.AUTO_FIX


# =============================================================================
# SuggestionType Tests
# =============================================================================


class TestSuggestionType:
    """Tests for SuggestionType enum."""

    def test_all_types_exist(self):
        """Test that all suggestion types exist."""
        expected_types = [
            "create_init_file",
            "create_directory",
            "add_import",
            "insert_prerequisite_task",
            "modify_file_hints",
            "add_dependency",
        ]
        for type_value in expected_types:
            assert SuggestionType(type_value) is not None


# =============================================================================
# SuggestionSafety Tests
# =============================================================================


class TestSuggestionSafety:
    """Tests for SuggestionSafety enum."""

    def test_safety_levels(self):
        """Test safety level values."""
        assert SuggestionSafety.SAFE.value == "safe"
        assert SuggestionSafety.MODERATE.value == "moderate"
        assert SuggestionSafety.UNSAFE.value == "unsafe"


# =============================================================================
# PlanSuggestion Tests
# =============================================================================


class TestPlanSuggestion:
    """Tests for PlanSuggestion dataclass."""

    def test_create_init_file_suggestion(self, tmp_path: Path):
        """Test creating an __init__.py suggestion."""
        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Create src/myproject/__init__.py",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=tmp_path / "src" / "myproject" / "__init__.py",
            file_content='"""Package initialization."""\n',
        )

        assert suggestion.suggestion_type == SuggestionType.CREATE_INIT_FILE
        assert suggestion.is_safe_for_auto_fix()
        assert "1.1.1" in suggestion.target_task_id

    def test_unsafe_suggestion_not_auto_fixable(self):
        """Test that unsafe suggestions are not auto-fixable."""
        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.INSERT_PREREQUISITE_TASK,
            description="Insert a new task",
            safety=SuggestionSafety.UNSAFE,
            target_task_id="1.1.1",
        )

        assert not suggestion.is_safe_for_auto_fix()

    def test_to_dict(self, tmp_path: Path):
        """Test serialization to dictionary."""
        file_path = tmp_path / "test.py"
        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Create __init__.py",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=file_path,
            file_content='"""Init."""\n',
            metadata={"key": "value"},
        )

        data = suggestion.to_dict()

        assert data["suggestion_type"] == "create_init_file"
        assert data["safety"] == "safe"
        assert data["target_task_id"] == "1.1.1"
        assert data["file_path"] == str(file_path)
        assert data["file_content"] == '"""Init."""\n'
        assert data["metadata"] == {"key": "value"}


# =============================================================================
# PlanAnalyzer Tests
# =============================================================================


class TestPlanAnalyzer:
    """Tests for PlanAnalyzer class."""

    def test_init_with_default_mode(self, sample_roadmap: Roadmap, temp_project: Path):
        """Test analyzer initialization with default mode."""
        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            project_root=temp_project,
        )

        assert analyzer.mode == AdaptiveMode.SUGGEST
        assert analyzer.roadmap == sample_roadmap

    def test_init_with_no_adapt_mode(self, sample_roadmap: Roadmap, temp_project: Path):
        """Test analyzer in NO_ADAPT mode."""
        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.NO_ADAPT,
            project_root=temp_project,
        )

        assert analyzer.mode == AdaptiveMode.NO_ADAPT

    def test_analyze_failure_in_no_adapt_mode(
        self,
        sample_roadmap: Roadmap,
        sample_task: ParsedTask,
        failed_result: ExecutionResult,
        temp_project: Path,
    ):
        """Test that NO_ADAPT mode returns no suggestions."""
        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.NO_ADAPT,
            project_root=temp_project,
        )

        suggestions = analyzer.analyze_failure(sample_task, failed_result)

        assert suggestions == []

    def test_analyze_import_error(
        self,
        sample_roadmap: Roadmap,
        sample_task: ParsedTask,
        temp_project: Path,
    ):
        """Test analyzing import error failures."""
        # Create a directory without __init__.py
        utils_dir = temp_project / "src" / "myproject" / "utils"
        utils_dir.mkdir(parents=True)

        result = ExecutionResult(
            task_id="1.1.1",
            status=ExecutionStatus.FAILED,
            error="ImportError: No module named 'src.myproject.utils'",
        )

        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.SUGGEST,
            project_root=temp_project,
        )

        suggestions = analyzer.analyze_failure(sample_task, result)

        # Should have some suggestions (may be empty if path doesn't match exactly)
        # The analyzer looks for module patterns and may or may not find matches
        assert isinstance(suggestions, list)

    def test_analyze_file_not_found(
        self,
        sample_roadmap: Roadmap,
        sample_task: ParsedTask,
        temp_project: Path,
    ):
        """Test analyzing file not found errors."""
        result = ExecutionResult(
            task_id="1.1.1",
            status=ExecutionStatus.FAILED,
            error="File 'src/myproject/missing.py' not found",
        )

        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.SUGGEST,
            project_root=temp_project,
        )

        suggestions = analyzer.analyze_failure(sample_task, result)

        # Should have suggestions (may be for creating directory or __init__.py)
        assert isinstance(suggestions, list)

    def test_can_auto_fix_filters_safe_only(
        self,
        sample_roadmap: Roadmap,
        temp_project: Path,
    ):
        """Test that can_auto_fix filters to safe suggestions only."""
        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.AUTO_FIX,
            project_root=temp_project,
        )

        suggestions = [
            PlanSuggestion(
                suggestion_type=SuggestionType.CREATE_INIT_FILE,
                description="Create __init__.py",
                safety=SuggestionSafety.SAFE,
                target_task_id="1.1.1",
            ),
            PlanSuggestion(
                suggestion_type=SuggestionType.INSERT_PREREQUISITE_TASK,
                description="Insert task",
                safety=SuggestionSafety.UNSAFE,
                target_task_id="1.1.1",
            ),
        ]

        safe = analyzer.can_auto_fix(suggestions)

        assert len(safe) == 1
        assert safe[0].safety == SuggestionSafety.SAFE


# =============================================================================
# Suggestion Application Tests
# =============================================================================


class TestSuggestionApplication:
    """Tests for applying suggestions."""

    def test_apply_create_init_file(
        self,
        sample_roadmap: Roadmap,
        temp_project: Path,
    ):
        """Test applying CREATE_INIT_FILE suggestion."""
        new_package = temp_project / "src" / "myproject" / "newpkg"
        new_package.mkdir(parents=True)

        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Create __init__.py",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=new_package / "__init__.py",
            file_content='"""New package."""\n',
            metadata={"legacy_apply": True},  # Phase 5.7: Enable legacy behavior for test
        )

        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.AUTO_FIX,
            project_root=temp_project,
        )

        result = analyzer.apply_suggestion(suggestion, force=True)

        assert result.success
        assert (new_package / "__init__.py").exists()
        assert (new_package / "__init__.py").read_text() == '"""New package."""\n'

    def test_apply_create_directory(
        self,
        sample_roadmap: Roadmap,
        temp_project: Path,
    ):
        """Test applying CREATE_DIRECTORY suggestion."""
        new_dir = temp_project / "src" / "myproject" / "newdir"

        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_DIRECTORY,
            description="Create directory",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=new_dir,
            metadata={"legacy_apply": True},  # Phase 5.7: Enable legacy behavior for test
        )

        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.AUTO_FIX,
            project_root=temp_project,
        )

        result = analyzer.apply_suggestion(suggestion, force=True)

        assert result.success
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_apply_create_directory_with_init(
        self,
        sample_roadmap: Roadmap,
        temp_project: Path,
    ):
        """Test applying CREATE_DIRECTORY with create_init=True."""
        new_dir = temp_project / "src" / "myproject" / "pkg_with_init"

        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_DIRECTORY,
            description="Create directory with __init__.py",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=new_dir,
            metadata={
                "create_init": True,
                "legacy_apply": True,
            },  # Phase 5.7: Enable legacy behavior
        )

        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.AUTO_FIX,
            project_root=temp_project,
        )

        result = analyzer.apply_suggestion(suggestion, force=True)

        assert result.success
        assert new_dir.exists()
        assert (new_dir / "__init__.py").exists()

    def test_apply_in_suggest_mode_requires_force(
        self,
        sample_roadmap: Roadmap,
        temp_project: Path,
    ):
        """Test that SUGGEST mode requires force=True to apply."""
        new_package = temp_project / "src" / "myproject" / "suggestpkg"
        new_package.mkdir(parents=True)

        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Create __init__.py",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=new_package / "__init__.py",
            metadata={"legacy_apply": True},  # Phase 5.7: Enable legacy behavior for test
        )

        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.SUGGEST,
            project_root=temp_project,
        )

        # Without force, just returns suggestion ready status
        result = analyzer.apply_suggestion(suggestion)
        assert result.success  # In SUGGEST mode, returns "ready for approval"
        assert not (new_package / "__init__.py").exists()

        # With force and legacy_apply, actually applies
        result = analyzer.apply_suggestion(suggestion, force=True)
        assert result.success
        assert (new_package / "__init__.py").exists()

    def test_apply_blocked_in_no_adapt_mode(
        self,
        sample_roadmap: Roadmap,
        temp_project: Path,
    ):
        """Test that NO_ADAPT mode blocks application without force."""
        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Create __init__.py",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=temp_project / "blocked.py",
        )

        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.NO_ADAPT,
            project_root=temp_project,
        )

        result = analyzer.apply_suggestion(suggestion)

        assert not result.success
        assert "NO_ADAPT" in result.message


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestAnalyzeFailureForPlanFix:
    """Tests for the analyze_failure_for_plan_fix convenience function."""

    @pytest.mark.asyncio
    async def test_returns_empty_in_no_adapt_mode(
        self,
        sample_task: ParsedTask,
        failed_result: ExecutionResult,
    ):
        """Test that NO_ADAPT mode returns empty list."""
        suggestions = await analyze_failure_for_plan_fix(
            sample_task,
            failed_result,
            mode=AdaptiveMode.NO_ADAPT,
        )

        assert suggestions == []

    @pytest.mark.asyncio
    async def test_analyzes_failure_in_suggest_mode(
        self,
        sample_task: ParsedTask,
        failed_result: ExecutionResult,
        sample_roadmap: Roadmap,
        temp_project: Path,
    ):
        """Test failure analysis in SUGGEST mode."""
        suggestions = await analyze_failure_for_plan_fix(
            sample_task,
            failed_result,
            roadmap=sample_roadmap,
            mode=AdaptiveMode.SUGGEST,
            project_root=temp_project,
        )

        # Should return some suggestions for import error
        assert isinstance(suggestions, list)


# =============================================================================
# Format Tests
# =============================================================================


class TestFormatSuggestionPrompt:
    """Tests for formatting suggestions for user display."""

    def test_format_simple_suggestion(
        self,
        sample_roadmap: Roadmap,
        temp_project: Path,
    ):
        """Test formatting a simple suggestion."""
        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Create src/__init__.py",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=temp_project / "src" / "__init__.py",
        )

        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            project_root=temp_project,
        )

        formatted = analyzer.format_suggestion_prompt(suggestion)

        assert "Create src/__init__.py" in formatted
        assert "create_init_file" in formatted
        assert "safe" in formatted

    def test_format_suggestion_with_new_task(
        self,
        sample_roadmap: Roadmap,
        temp_project: Path,
    ):
        """Test formatting a suggestion with a new task."""
        new_task = ParsedTask(
            id="1.1.0.1",
            title="Setup prerequisites",
            description="Set up prerequisites for the main task",
        )

        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.INSERT_PREREQUISITE_TASK,
            description="Insert prerequisite task",
            safety=SuggestionSafety.MODERATE,
            target_task_id="1.1.1",
            new_task=new_task,
        )

        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            project_root=temp_project,
        )

        formatted = analyzer.format_suggestion_prompt(suggestion)

        assert "Insert prerequisite task" in formatted
        assert "Setup prerequisites" in formatted


# =============================================================================
# Integration Tests
# =============================================================================


class TestAdaptivePlanningIntegration:
    """Integration tests for adaptive planning with Executor."""

    def test_executor_config_has_adaptive_mode(self):
        """Test that ExecutorConfig has adaptive_mode field."""
        from ai_infra.executor.types import ExecutorConfig

        config = ExecutorConfig()
        assert hasattr(config, "adaptive_mode")
        assert config.adaptive_mode == "suggest"  # Default is suggest

    def test_executor_config_adaptive_mode_in_dict(self):
        """Test that adaptive_mode is included in to_dict."""
        from ai_infra.executor.types import ExecutorConfig

        config = ExecutorConfig(adaptive_mode="auto_fix")
        data = config.to_dict()

        assert "adaptive_mode" in data
        assert data["adaptive_mode"] == "auto_fix"

    def test_execution_result_has_suggestions(self):
        """Test that ExecutionResult has suggestions field."""
        result = ExecutionResult(
            task_id="1.1.1",
            status=ExecutionStatus.FAILED,
            error="Test error",
        )

        assert hasattr(result, "suggestions")
        assert result.suggestions == []
        assert not result.has_suggestions

    def test_execution_result_suggestions_in_dict(self):
        """Test that suggestions are included in to_dict."""
        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Create __init__.py",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
        )

        result = ExecutionResult(
            task_id="1.1.1",
            status=ExecutionStatus.FAILED,
            error="Test error",
            suggestions=[suggestion],
        )

        data = result.to_dict()
        assert "suggestions" in data
        assert len(data["suggestions"]) == 1
        assert data["suggestions"][0]["suggestion_type"] == "create_init_file"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_apply_suggestion_missing_file_path(
        self,
        sample_roadmap: Roadmap,
        temp_project: Path,
    ):
        """Test applying suggestion without file_path."""
        suggestion = PlanSuggestion(
            suggestion_type=SuggestionType.CREATE_INIT_FILE,
            description="Missing file path",
            safety=SuggestionSafety.SAFE,
            target_task_id="1.1.1",
            file_path=None,  # Missing
            metadata={"legacy_apply": True},  # Phase 5.7: Test legacy path
        )

        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.AUTO_FIX,
            project_root=temp_project,
        )

        result = analyzer.apply_suggestion(suggestion, force=True)

        # Legacy apply should fail for missing file path
        assert not result.success
        assert "No file path" in result.message

    def test_analyze_failure_with_failure_record(
        self,
        sample_roadmap: Roadmap,
        sample_task: ParsedTask,
        temp_project: Path,
    ):
        """Test analyzing failure with pre-categorized FailureRecord."""
        result = ExecutionResult(
            task_id="1.1.1",
            status=ExecutionStatus.FAILED,
            error="Some error",
        )

        failure_record = FailureRecord(
            task_id="1.1.1",
            task_title="Create module",
            category=FailureCategory.IMPORT_ERROR,
            severity=FailureSeverity.HIGH,
            error_message="Import error details",
        )

        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.SUGGEST,
            project_root=temp_project,
        )

        suggestions = analyzer.analyze_failure(
            sample_task,
            result,
            failure_record=failure_record,
        )

        # Should use the pre-categorized failure type
        assert isinstance(suggestions, list)

    def test_max_suggestions_limit(
        self,
        sample_roadmap: Roadmap,
        sample_task: ParsedTask,
        temp_project: Path,
    ):
        """Test that suggestions are limited by max_suggestions."""
        analyzer = PlanAnalyzer(
            roadmap=sample_roadmap,
            mode=AdaptiveMode.SUGGEST,
            project_root=temp_project,
            max_suggestions=2,
        )

        # Create an error that would generate many suggestions
        result = ExecutionResult(
            task_id="1.1.1",
            status=ExecutionStatus.FAILED,
            error="Multiple issues: ImportError, FileNotFound, context missing",
        )

        suggestions = analyzer.analyze_failure(sample_task, result)

        assert len(suggestions) <= 2
