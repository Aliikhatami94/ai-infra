"""Scenario tests for test failure recovery (Phase 6.3.2).

Tests scenarios for recovering from test failures, including:
- Identifying failing tests
- Fixing implementation vs test
- Handling multiple test failures
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ai_infra.executor.failure import FailureCategory, FailureRecord
from ai_infra.executor.verifier import CheckLevel, TaskVerifier

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def python_project(tmp_path: Path) -> Path:
    """Create a minimal Python project with test structure."""
    # pyproject.toml
    (tmp_path / "pyproject.toml").write_text("""\
[project]
name = "test-project"
version = "0.1.0"

[tool.pytest.ini_options]
testpaths = ["tests"]
""")

    # Source directory
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")

    # Tests directory
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "conftest.py").write_text("""\
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
""")

    return tmp_path


@pytest.fixture
def project_with_failing_test(python_project: Path) -> Path:
    """Create a project with a failing test."""
    # Implementation with bug
    (python_project / "src" / "math_ops.py").write_text("""\
def add(a: int, b: int) -> int:
    return a - b  # Bug: should be a + b

def multiply(a: int, b: int) -> int:
    return a * b
""")

    # Test that will fail
    (python_project / "tests" / "test_math.py").write_text("""\
from math_ops import add, multiply

def test_add():
    assert add(2, 3) == 5  # Will fail because of bug

def test_multiply():
    assert multiply(2, 3) == 6  # Will pass
""")

    return python_project


@pytest.fixture
def project_with_passing_tests(python_project: Path) -> Path:
    """Create a project with passing tests."""
    # Correct implementation
    (python_project / "src" / "math_ops.py").write_text("""\
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b
""")

    # Tests that pass
    (python_project / "tests" / "test_math.py").write_text("""\
from math_ops import add, multiply

def test_add():
    assert add(2, 3) == 5

def test_multiply():
    assert multiply(2, 3) == 6
""")

    return python_project


# =============================================================================
# Test Failure Detection Tests
# =============================================================================


class TestTestFailureDetection:
    """Tests for detecting test failures."""

    def test_detects_assertion_failure(self, project_with_failing_test: Path) -> None:
        """Should detect assertion failures in tests."""
        result = subprocess.run(
            ["python", "-m", "pytest", "-v", "--tb=short"],
            cwd=project_with_failing_test,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "FAILED" in result.stdout or "FAILED" in result.stderr

    def test_identifies_which_test_failed(self, project_with_failing_test: Path) -> None:
        """Should identify which specific test failed."""
        result = subprocess.run(
            ["python", "-m", "pytest", "-v", "--tb=short"],
            cwd=project_with_failing_test,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr
        # Should mention the failing test
        assert "test_add" in output or "test_math" in output

    def test_passing_tests_succeed(self, project_with_passing_tests: Path) -> None:
        """Passing tests should succeed."""
        result = subprocess.run(
            ["python", "-m", "pytest", "-v"],
            cwd=project_with_passing_tests,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0


class TestTestFailureAnalysis:
    """Tests for analyzing test failures."""

    def test_categorize_as_test_failure(self) -> None:
        """Should categorize as TEST_FAILURE."""
        record = FailureRecord(
            task_id="1.1.1",
            task_title="Fix tests",
            error_message="AssertionError: assert 5 == -1",
            category=FailureCategory.TEST_FAILURE,
        )

        assert record.category == FailureCategory.TEST_FAILURE

    def test_distinguish_from_syntax_error(self) -> None:
        """Should distinguish test failures from syntax errors."""
        test_failure = FailureRecord(
            task_id="1.1",
            task_title="Test",
            error_message="AssertionError",
            category=FailureCategory.TEST_FAILURE,
        )

        syntax_error = FailureRecord(
            task_id="1.2",
            task_title="Test",
            error_message="SyntaxError",
            category=FailureCategory.SYNTAX_ERROR,
        )

        assert test_failure.category != syntax_error.category


# =============================================================================
# Test Failure Recovery Tests
# =============================================================================


class TestTestFailureRecovery:
    """Tests for test failure recovery scenarios."""

    def test_fix_implementation_not_test(self, project_with_failing_test: Path) -> None:
        """Should fix implementation, not the test."""
        # Initial state: test fails
        result = subprocess.run(
            ["python", "-m", "pytest", "-v"],
            cwd=project_with_failing_test,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

        # Simulate agent fixing implementation (not test)
        fixed_impl = """\
def add(a: int, b: int) -> int:
    return a + b  # Fixed!

def multiply(a: int, b: int) -> int:
    return a * b
"""
        (project_with_failing_test / "src" / "math_ops.py").write_text(fixed_impl)

        # Now test should pass
        result = subprocess.run(
            ["python", "-m", "pytest", "-v"],
            cwd=project_with_failing_test,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_test_file_unchanged_after_fix(self, project_with_failing_test: Path) -> None:
        """Test file should remain unchanged when fixing implementation."""
        test_content_before = (project_with_failing_test / "tests" / "test_math.py").read_text()

        # Fix implementation
        fixed_impl = """\
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b
"""
        (project_with_failing_test / "src" / "math_ops.py").write_text(fixed_impl)

        test_content_after = (project_with_failing_test / "tests" / "test_math.py").read_text()

        assert test_content_before == test_content_after


class TestMultipleTestFailures:
    """Tests for handling multiple test failures."""

    @pytest.fixture
    def project_with_multiple_failures(self, python_project: Path) -> Path:
        """Create project with multiple failing tests."""
        # Implementation with multiple bugs
        (python_project / "src" / "math_ops.py").write_text("""\
def add(a: int, b: int) -> int:
    return a - b  # Bug 1

def subtract(a: int, b: int) -> int:
    return a + b  # Bug 2

def multiply(a: int, b: int) -> int:
    return a * b  # Correct
""")

        (python_project / "tests" / "test_math.py").write_text("""\
from math_ops import add, subtract, multiply

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 3) == 2

def test_multiply():
    assert multiply(2, 3) == 6
""")

        return python_project

    def test_detects_multiple_failures(self, project_with_multiple_failures: Path) -> None:
        """Should detect multiple test failures."""
        result = subprocess.run(
            ["python", "-m", "pytest", "-v"],
            cwd=project_with_multiple_failures,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        # Should have 2 failures
        assert result.returncode != 0
        # Count FAILED occurrences
        failed_count = output.count("FAILED")
        assert failed_count >= 2

    def test_fix_all_failures(self, project_with_multiple_failures: Path) -> None:
        """Should be able to fix all failures."""
        # Fix all bugs
        fixed_impl = """\
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b

def multiply(a: int, b: int) -> int:
    return a * b
"""
        (project_with_multiple_failures / "src" / "math_ops.py").write_text(fixed_impl)

        result = subprocess.run(
            ["python", "-m", "pytest", "-v"],
            cwd=project_with_multiple_failures,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0


# =============================================================================
# Test Output Parsing Tests
# =============================================================================


class TestTestOutputParsing:
    """Tests for parsing test output."""

    def test_parse_pytest_failure_output(self) -> None:
        """Should parse pytest failure output format."""
        # Sample pytest output
        output = """\
============================= test session starts ==============================
collected 2 items

tests/test_math.py .F                                                    [100%]

=================================== FAILURES ===================================
______________________________ test_add ________________________________________

    def test_add():
>       assert add(2, 3) == 5
E       assert -1 == 5
E        +  where -1 = add(2, 3)

tests/test_math.py:4: AssertionError
=========================== short test summary info ============================
FAILED tests/test_math.py::test_add - assert -1 == 5
========================= 1 failed, 1 passed in 0.12s =========================
"""

        # Parse key information
        assert "FAILED" in output
        assert "test_add" in output
        assert "AssertionError" in output
        assert "assert -1 == 5" in output

    def test_extract_failing_test_names(self) -> None:
        """Should extract failing test names from output."""
        output = """\
FAILED tests/test_math.py::test_add - assert -1 == 5
FAILED tests/test_math.py::test_subtract - assert 8 == 2
"""

        import re

        pattern = r"FAILED ([\w/.:]+)"
        matches = re.findall(pattern, output)

        assert len(matches) == 2
        assert "test_math.py::test_add" in matches[0]
        assert "test_math.py::test_subtract" in matches[1]


# =============================================================================
# Verifier Integration Tests
# =============================================================================


class TestVerifierTestDetection:
    """Tests for TaskVerifier test failure detection."""

    @pytest.mark.asyncio
    async def test_verifier_reports_test_failures(self, project_with_failing_test: Path) -> None:
        """TaskVerifier should report test failures."""
        from ai_infra.executor.models import Task

        verifier = TaskVerifier(workspace=project_with_failing_test)

        task = Task(
            id="test-1",
            title="Fix math operations",
            file_hints=["src/math_ops.py"],
        )

        # Run verification at TESTS level
        result = await verifier.verify(task, levels=[CheckLevel.TESTS])

        # Should fail due to test failure
        assert not result.overall

    @pytest.mark.asyncio
    async def test_verifier_passes_with_good_tests(self, project_with_passing_tests: Path) -> None:
        """TaskVerifier should pass with passing tests."""
        from ai_infra.executor.models import Task

        verifier = TaskVerifier(workspace=project_with_passing_tests)

        task = Task(
            id="test-2",
            title="Math operations",
            file_hints=["src/math_ops.py"],
        )

        result = await verifier.verify(task, levels=[CheckLevel.TESTS])

        # Should pass
        assert result.overall
