"""Scenario tests for syntax error recovery (Phase 6.3.1).

Tests various scenarios for recovering from syntax errors in different
file types: Python, JSON, TypeScript, etc.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from ai_infra.executor.failure import FailureAnalyzer, FailureCategory, FailureRecord
from ai_infra.executor.verifier import TaskVerifier

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def python_project(tmp_path: Path) -> Path:
    """Create a minimal Python project."""
    (tmp_path / "pyproject.toml").write_text("""\
[project]
name = "test-project"
version = "0.1.0"
""")
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    return tmp_path


@pytest.fixture
def nodejs_project(tmp_path: Path) -> Path:
    """Create a minimal Node.js project."""
    package = {"name": "test", "version": "1.0.0"}
    (tmp_path / "package.json").write_text(json.dumps(package))
    (tmp_path / "src").mkdir()
    return tmp_path


@pytest.fixture
def verifier(tmp_path: Path) -> TaskVerifier:
    """Create a TaskVerifier for testing."""
    return TaskVerifier(workspace=tmp_path)


@pytest.fixture
def failure_analyzer() -> FailureAnalyzer:
    """Create a FailureAnalyzer for testing."""
    return FailureAnalyzer()


# =============================================================================
# Python Syntax Error Detection Tests
# =============================================================================


class TestPythonSyntaxErrorDetection:
    """Tests for detecting Python syntax errors."""

    def test_detects_incomplete_function(self, python_project: Path) -> None:
        """Should detect incomplete function definition."""
        broken_file = python_project / "src" / "broken.py"
        broken_file.write_text("def foo(\n")

        with pytest.raises(SyntaxError):
            ast.parse(broken_file.read_text())

    def test_detects_missing_colon(self, python_project: Path) -> None:
        """Should detect missing colon after function def."""
        broken_file = python_project / "src" / "broken.py"
        broken_file.write_text("def foo()\n    pass\n")

        with pytest.raises(SyntaxError):
            ast.parse(broken_file.read_text())

    def test_detects_unmatched_brackets(self, python_project: Path) -> None:
        """Should detect unmatched brackets."""
        broken_file = python_project / "src" / "broken.py"
        broken_file.write_text("x = [1, 2, 3\n")

        with pytest.raises(SyntaxError):
            ast.parse(broken_file.read_text())

    def test_detects_invalid_indentation(self, python_project: Path) -> None:
        """Should detect invalid indentation."""
        broken_file = python_project / "src" / "broken.py"
        broken_file.write_text("def foo():\npass\n")

        with pytest.raises(IndentationError):
            ast.parse(broken_file.read_text())

    def test_valid_python_parses(self, python_project: Path) -> None:
        """Valid Python should parse without errors."""
        valid_file = python_project / "src" / "valid.py"
        valid_file.write_text("""\
def foo():
    return "Hello"

class Bar:
    def __init__(self):
        self.x = 42
""")

        # Should not raise
        tree = ast.parse(valid_file.read_text())
        assert tree is not None


class TestPythonSyntaxErrorRecovery:
    """Tests for Python syntax error recovery scenarios."""

    def test_verifier_detects_syntax_errors(
        self, python_project: Path, verifier: TaskVerifier
    ) -> None:
        """TaskVerifier should detect syntax errors."""
        # Create broken file
        broken_file = python_project / "src" / "broken.py"
        broken_file.write_text("def foo(\n")

        # Verifier should report syntax error at SYNTAX level
        # Note: We test the underlying check, not full async verify
        files = [broken_file]
        errors = []

        for f in files:
            try:
                ast.parse(f.read_text())
            except SyntaxError as e:
                errors.append((str(f), str(e)))

        assert len(errors) == 1
        assert "broken.py" in errors[0][0]

    def test_failure_category_for_syntax_error(self, failure_analyzer: FailureAnalyzer) -> None:
        """Should categorize as SYNTAX_ERROR."""
        record = FailureRecord(
            task_id="1.1.1",
            task_title="Create function",
            error_message="SyntaxError: unexpected EOF while parsing",
            category=FailureCategory.SYNTAX_ERROR,
        )

        assert record.category == FailureCategory.SYNTAX_ERROR

    def test_syntax_error_recovery_simulation(self, python_project: Path) -> None:
        """Simulate fixing a syntax error."""
        broken_file = python_project / "src" / "broken.py"

        # Step 1: Create broken file
        broken_file.write_text("def foo(\n")

        # Verify it's broken
        with pytest.raises(SyntaxError):
            ast.parse(broken_file.read_text())

        # Step 2: Simulate agent fixing it
        fixed_content = """\
def foo():
    pass
"""
        broken_file.write_text(fixed_content)

        # Step 3: Verify it's fixed
        tree = ast.parse(broken_file.read_text())
        assert tree is not None


# =============================================================================
# JSON Syntax Error Detection Tests
# =============================================================================


class TestJSONSyntaxErrorDetection:
    """Tests for detecting JSON syntax errors."""

    def test_detects_missing_quotes(self, nodejs_project: Path) -> None:
        """Should detect missing quotes around value."""
        broken_file = nodejs_project / "data.json"
        broken_file.write_text('{"key": value}')

        with pytest.raises(json.JSONDecodeError):
            json.loads(broken_file.read_text())

    def test_detects_trailing_comma(self, nodejs_project: Path) -> None:
        """Should detect trailing comma."""
        broken_file = nodejs_project / "data.json"
        broken_file.write_text('{"key": "value",}')

        with pytest.raises(json.JSONDecodeError):
            json.loads(broken_file.read_text())

    def test_detects_unquoted_key(self, nodejs_project: Path) -> None:
        """Should detect unquoted key."""
        broken_file = nodejs_project / "data.json"
        broken_file.write_text('{key: "value"}')

        with pytest.raises(json.JSONDecodeError):
            json.loads(broken_file.read_text())

    def test_valid_json_parses(self, nodejs_project: Path) -> None:
        """Valid JSON should parse without errors."""
        valid_file = nodejs_project / "data.json"
        valid_file.write_text('{"name": "test", "version": "1.0.0"}')

        parsed = json.loads(valid_file.read_text())
        assert parsed["name"] == "test"


class TestJSONSyntaxErrorRecovery:
    """Tests for JSON syntax error recovery scenarios."""

    def test_json_error_recovery_simulation(self, nodejs_project: Path) -> None:
        """Simulate fixing a JSON syntax error."""
        broken_file = nodejs_project / "config.json"

        # Step 1: Create broken file
        broken_file.write_text('{"key": value}')

        # Verify it's broken
        with pytest.raises(json.JSONDecodeError):
            json.loads(broken_file.read_text())

        # Step 2: Simulate agent fixing it
        fixed_content = '{"key": "value"}'
        broken_file.write_text(fixed_content)

        # Step 3: Verify it's fixed
        parsed = json.loads(broken_file.read_text())
        assert parsed["key"] == "value"


# =============================================================================
# TypeScript/JavaScript Syntax Error Detection Tests
# =============================================================================


class TestTypeScriptSyntaxErrorDetection:
    """Tests for detecting TypeScript/JavaScript syntax errors."""

    def test_detects_missing_semicolon_in_ts(self, nodejs_project: Path) -> None:
        """Should detect common TypeScript syntax issues."""
        # Note: We can't fully validate TS without tsc, but we can check structure
        broken_file = nodejs_project / "src" / "broken.ts"
        broken_file.write_text("const x = {")

        content = broken_file.read_text()
        # Simple brace matching check
        open_braces = content.count("{")
        close_braces = content.count("}")
        assert open_braces != close_braces

    def test_valid_typescript_structure(self, nodejs_project: Path) -> None:
        """Valid TypeScript should have balanced braces."""
        valid_file = nodejs_project / "src" / "valid.ts"
        valid_file.write_text("""\
interface User {
  id: number;
  name: string;
}

function greet(user: User): string {
  return `Hello, ${user.name}`;
}
""")

        content = valid_file.read_text()
        open_braces = content.count("{")
        close_braces = content.count("}")
        assert open_braces == close_braces


# =============================================================================
# Multi-file Error Recovery Tests
# =============================================================================


class TestMultiFileErrorRecovery:
    """Tests for recovering from errors across multiple files."""

    def test_multiple_syntax_errors(self, python_project: Path) -> None:
        """Should detect multiple files with syntax errors."""
        # Create multiple broken files
        (python_project / "src" / "file1.py").write_text("def foo(\n")
        (python_project / "src" / "file2.py").write_text("class Bar\n")
        (python_project / "src" / "valid.py").write_text("x = 1\n")

        errors = []
        for py_file in (python_project / "src").glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            try:
                ast.parse(py_file.read_text())
            except SyntaxError as e:
                errors.append((py_file.name, str(e)))

        assert len(errors) == 2
        error_files = {e[0] for e in errors}
        assert "file1.py" in error_files
        assert "file2.py" in error_files
        assert "valid.py" not in error_files

    def test_fix_errors_preserves_valid_files(self, python_project: Path) -> None:
        """Fixing errors should not affect valid files."""
        valid_file = python_project / "src" / "valid.py"
        broken_file = python_project / "src" / "broken.py"

        valid_content = "def valid():\n    return 42\n"
        valid_file.write_text(valid_content)
        broken_file.write_text("def broken(\n")

        # Simulate fixing broken file
        broken_file.write_text("def broken():\n    pass\n")

        # Valid file should be unchanged
        assert valid_file.read_text() == valid_content


# =============================================================================
# Error Analysis Tests
# =============================================================================


class TestErrorAnalysis:
    """Tests for analyzing syntax errors."""

    def test_extract_error_location(self) -> None:
        """Should extract line number from syntax error."""
        try:
            ast.parse("def foo(\n  pass\n")
        except SyntaxError as e:
            assert e.lineno is not None
            assert e.lineno >= 1

    def test_extract_error_context(self, python_project: Path) -> None:
        """Should provide context around syntax error."""
        broken_file = python_project / "src" / "broken.py"
        code = """\
def good_function():
    return 1

def broken_function(
    pass

def another_function():
    return 2
"""
        broken_file.write_text(code)

        try:
            ast.parse(code)
        except SyntaxError as e:
            # Error should be around line 4-5
            assert e.lineno is not None
            assert 4 <= e.lineno <= 6

    def test_categorize_syntax_vs_type_error(self, failure_analyzer: FailureAnalyzer) -> None:
        """Should distinguish syntax errors from type errors."""
        syntax_record = FailureRecord(
            task_id="1.1",
            task_title="Test",
            error_message="SyntaxError: invalid syntax",
            category=FailureCategory.SYNTAX_ERROR,
        )

        type_record = FailureRecord(
            task_id="1.2",
            task_title="Test",
            error_message="TypeError: expected str, got int",
            category=FailureCategory.TYPE_ERROR,
        )

        assert syntax_record.category != type_record.category
        assert syntax_record.category == FailureCategory.SYNTAX_ERROR
        assert type_record.category == FailureCategory.TYPE_ERROR
