"""Tests for file size linter.

Tests Phase 13.1 of EXECUTOR_4.md - File Size Enforcement.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts directory to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from lint_file_size import (
    DEFAULT_EXCEPTIONS,
    DEFAULT_MAX_LINES,
    FileInfo,
    LintResult,
    check_file_sizes,
    count_lines,
    format_result,
    get_refactoring_recommendations,
    should_skip_directory,
)

# =============================================================================
# count_lines Tests
# =============================================================================


class TestCountLines:
    """Tests for count_lines function."""

    def test_count_lines_simple(self, tmp_path: Path) -> None:
        """Test counting lines in a simple file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        assert count_lines(test_file) == 3

    def test_count_lines_empty(self, tmp_path: Path) -> None:
        """Test counting lines in an empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        assert count_lines(test_file) == 0

    def test_count_lines_single_line(self, tmp_path: Path) -> None:
        """Test counting lines in a single line file."""
        test_file = tmp_path / "single.py"
        test_file.write_text("single line")

        assert count_lines(test_file) == 1

    def test_count_lines_nonexistent(self, tmp_path: Path) -> None:
        """Test counting lines in a nonexistent file."""
        test_file = tmp_path / "nonexistent.py"

        assert count_lines(test_file) == 0


# =============================================================================
# should_skip_directory Tests
# =============================================================================


class TestShouldSkipDirectory:
    """Tests for should_skip_directory function."""

    def test_skip_pycache(self) -> None:
        """Test skipping __pycache__ directory."""
        assert should_skip_directory(Path("__pycache__")) is True

    def test_skip_venv(self) -> None:
        """Test skipping venv directory."""
        assert should_skip_directory(Path(".venv")) is True
        assert should_skip_directory(Path("venv")) is True

    def test_skip_git(self) -> None:
        """Test skipping .git directory."""
        assert should_skip_directory(Path(".git")) is True

    def test_not_skip_src(self) -> None:
        """Test not skipping src directory."""
        assert should_skip_directory(Path("src")) is False

    def test_not_skip_tests(self) -> None:
        """Test not skipping tests directory."""
        assert should_skip_directory(Path("tests")) is False


# =============================================================================
# FileInfo Tests
# =============================================================================


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_file_info_creation(self, tmp_path: Path) -> None:
        """Test FileInfo creation."""
        test_file = tmp_path / "test.py"
        info = FileInfo(path=test_file, line_count=100)

        assert info.path == test_file
        assert info.line_count == 100
        assert info.is_exception is False

    def test_file_info_to_dict(self, tmp_path: Path) -> None:
        """Test FileInfo to_dict."""
        test_file = tmp_path / "test.py"
        info = FileInfo(path=test_file, line_count=100, is_exception=True)

        d = info.to_dict()

        assert d["line_count"] == 100
        assert d["is_exception"] is True


# =============================================================================
# LintResult Tests
# =============================================================================


class TestLintResult:
    """Tests for LintResult dataclass."""

    def test_result_creation(self) -> None:
        """Test LintResult creation."""
        result = LintResult()

        assert result.violations == []
        assert result.warnings == []
        assert result.passed == []
        assert result.max_lines == DEFAULT_MAX_LINES

    def test_has_violations_false(self) -> None:
        """Test has_violations when no violations."""
        result = LintResult()

        assert result.has_violations is False

    def test_has_violations_true(self, tmp_path: Path) -> None:
        """Test has_violations when violations exist."""
        result = LintResult()
        result.violations.append(FileInfo(path=tmp_path / "test.py", line_count=600))

        assert result.has_violations is True

    def test_total_files(self, tmp_path: Path) -> None:
        """Test total_files calculation."""
        result = LintResult()
        result.violations.append(FileInfo(path=tmp_path / "v1.py", line_count=600))
        result.warnings.append(FileInfo(path=tmp_path / "w1.py", line_count=450))
        result.passed.append(FileInfo(path=tmp_path / "p1.py", line_count=100))

        assert result.total_files == 3

    def test_to_dict(self, tmp_path: Path) -> None:
        """Test to_dict serialization."""
        result = LintResult(max_lines=500)
        result.violations.append(FileInfo(path=tmp_path / "v1.py", line_count=600))

        d = result.to_dict()

        assert d["max_lines"] == 500
        assert d["violations_count"] == 1
        assert d["total_files"] == 1


# =============================================================================
# check_file_sizes Tests
# =============================================================================


class TestCheckFileSizes:
    """Tests for check_file_sizes function."""

    def test_check_empty_directory(self, tmp_path: Path) -> None:
        """Test checking empty directory."""
        result = check_file_sizes(tmp_path)

        assert result.total_files == 0
        assert result.has_violations is False

    def test_check_finds_violations(self, tmp_path: Path) -> None:
        """Test finding files that exceed limit."""
        # Create a file with 600 lines
        large_file = tmp_path / "large.py"
        large_file.write_text("\n".join([f"line{i}" for i in range(600)]))

        result = check_file_sizes(tmp_path, max_lines=500)

        assert len(result.violations) == 1
        assert result.violations[0].line_count == 600

    def test_check_passes_small_files(self, tmp_path: Path) -> None:
        """Test passing files under limit."""
        small_file = tmp_path / "small.py"
        small_file.write_text("\n".join([f"line{i}" for i in range(100)]))

        result = check_file_sizes(tmp_path, max_lines=500)

        assert len(result.passed) == 1
        assert result.has_violations is False

    def test_check_respects_exceptions(self, tmp_path: Path) -> None:
        """Test exceptions are not violations."""
        # Create a large state.py file (exception)
        state_file = tmp_path / "state.py"
        state_file.write_text("\n".join([f"line{i}" for i in range(800)]))

        result = check_file_sizes(tmp_path, max_lines=500)

        # Should be in warnings, not violations
        assert len(result.violations) == 0
        assert len(result.warnings) == 1
        assert result.warnings[0].is_exception is True

    def test_check_warns_approaching_limit(self, tmp_path: Path) -> None:
        """Test warning when approaching limit."""
        warning_file = tmp_path / "warning.py"
        warning_file.write_text("\n".join([f"line{i}" for i in range(450)]))  # 90% of 500

        result = check_file_sizes(tmp_path, max_lines=500, warning_threshold=0.8)

        assert len(result.violations) == 0
        assert len(result.warnings) == 1

    def test_check_skips_pycache(self, tmp_path: Path) -> None:
        """Test skipping __pycache__ directory."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        cached_file = pycache / "module.cpython-311.pyc"
        cached_file.write_text("\n".join([f"line{i}" for i in range(1000)]))

        result = check_file_sizes(tmp_path, max_lines=500)

        assert result.total_files == 0

    def test_check_recursive(self, tmp_path: Path) -> None:
        """Test recursive file checking."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        large_file = subdir / "large.py"
        large_file.write_text("\n".join([f"line{i}" for i in range(600)]))

        result = check_file_sizes(tmp_path, max_lines=500)

        assert len(result.violations) == 1

    def test_check_custom_exceptions(self, tmp_path: Path) -> None:
        """Test custom exceptions."""
        custom_file = tmp_path / "custom.py"
        custom_file.write_text("\n".join([f"line{i}" for i in range(600)]))

        # Without exception - violation
        result1 = check_file_sizes(tmp_path, max_lines=500)
        assert len(result1.violations) == 1

        # With exception - warning
        result2 = check_file_sizes(tmp_path, max_lines=500, exceptions={"custom.py"})
        assert len(result2.violations) == 0
        assert len(result2.warnings) == 1


# =============================================================================
# format_result Tests
# =============================================================================


class TestFormatResult:
    """Tests for format_result function."""

    def test_format_empty_result(self) -> None:
        """Test formatting empty result."""
        result = LintResult()

        output = format_result(result)

        assert "Summary:" in output
        assert "Total files: 0" in output
        assert "PASSED" in output

    def test_format_with_violations(self, tmp_path: Path) -> None:
        """Test formatting result with violations."""
        result = LintResult(max_lines=500)
        result.violations.append(FileInfo(path=tmp_path / "large.py", line_count=600))

        output = format_result(result)

        assert "exceeding" in output
        assert "600 lines" in output
        assert "FAILED" in output

    def test_format_verbose(self, tmp_path: Path) -> None:
        """Test verbose formatting includes passed files."""
        result = LintResult()
        result.passed.append(FileInfo(path=tmp_path / "small.py", line_count=100))

        output = format_result(result, verbose=True)

        assert "Passed files:" in output


# =============================================================================
# get_refactoring_recommendations Tests
# =============================================================================


class TestGetRefactoringRecommendations:
    """Tests for get_refactoring_recommendations function."""

    def test_recommendations_for_loop(self, tmp_path: Path) -> None:
        """Test recommendations for loop.py."""
        result = LintResult(max_lines=500)
        result.violations.append(FileInfo(path=tmp_path / "loop.py", line_count=3000))

        recommendations = get_refactoring_recommendations(result)

        assert len(recommendations) == 1
        assert recommendations[0]["priority"] == "high"
        assert "Split" in recommendations[0]["suggestion"]

    def test_recommendations_for_dependencies(self, tmp_path: Path) -> None:
        """Test recommendations for dependencies.py."""
        result = LintResult(max_lines=500)
        result.violations.append(FileInfo(path=tmp_path / "dependencies.py", line_count=1500))

        recommendations = get_refactoring_recommendations(result)

        assert len(recommendations) == 1
        assert recommendations[0]["priority"] == "medium"

    def test_recommendations_for_graph(self, tmp_path: Path) -> None:
        """Test recommendations for graph.py."""
        result = LintResult(max_lines=500)
        result.violations.append(FileInfo(path=tmp_path / "graph.py", line_count=1500))

        recommendations = get_refactoring_recommendations(result)

        assert len(recommendations) == 1
        assert recommendations[0]["priority"] == "high"
        assert "node" in recommendations[0]["suggestion"].lower()

    def test_recommendations_reduction_needed(self, tmp_path: Path) -> None:
        """Test reduction_needed calculation."""
        result = LintResult(max_lines=500)
        result.violations.append(FileInfo(path=tmp_path / "large.py", line_count=800))

        recommendations = get_refactoring_recommendations(result)

        assert recommendations[0]["reduction_needed"] == 300


# =============================================================================
# Default Configuration Tests
# =============================================================================


class TestDefaultConfiguration:
    """Tests for default configuration values."""

    def test_default_max_lines(self) -> None:
        """Test default max lines is 500."""
        assert DEFAULT_MAX_LINES == 500

    def test_default_exceptions_contains_state(self) -> None:
        """Test default exceptions contains state.py."""
        assert "state.py" in DEFAULT_EXCEPTIONS

    def test_default_exceptions_contains_types(self) -> None:
        """Test default exceptions contains types.py."""
        assert "types.py" in DEFAULT_EXCEPTIONS

    def test_default_exceptions_contains_init(self) -> None:
        """Test default exceptions contains __init__.py."""
        assert "__init__.py" in DEFAULT_EXCEPTIONS


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for file size linter."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete linting workflow."""
        # Create directory structure
        src = tmp_path / "src"
        src.mkdir()

        # Create files of various sizes
        small = src / "small.py"
        small.write_text("\n".join([f"line{i}" for i in range(100)]))

        medium = src / "medium.py"
        medium.write_text("\n".join([f"line{i}" for i in range(450)]))

        large = src / "large.py"
        large.write_text("\n".join([f"line{i}" for i in range(600)]))

        exception = src / "state.py"
        exception.write_text("\n".join([f"line{i}" for i in range(800)]))

        # Run check
        result = check_file_sizes(src, max_lines=500)

        # Verify
        assert len(result.violations) == 1
        assert result.violations[0].line_count == 600

        assert len(result.warnings) == 2  # medium approaching + state.py exception

        assert len(result.passed) == 1
        assert result.passed[0].line_count == 100

    def test_json_output(self, tmp_path: Path) -> None:
        """Test JSON output serialization."""
        import json

        src = tmp_path / "src"
        src.mkdir()

        large = src / "large.py"
        large.write_text("\n".join([f"line{i}" for i in range(600)]))

        result = check_file_sizes(src, max_lines=500)

        # Should serialize without errors
        json_output = json.dumps(result.to_dict())
        parsed = json.loads(json_output)

        assert parsed["violations_count"] == 1
        assert parsed["max_lines"] == 500
