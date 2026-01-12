"""Unit tests for TaskVerifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_infra.executor.models import Task
from ai_infra.executor.verifier import (
    CheckLevel,
    CheckResult,
    CheckStatus,
    TaskVerifier,
    VerificationResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a basic workspace structure."""
    # Create a valid Python file
    (tmp_path / "main.py").write_text("def hello():\n    return 'world'\n")

    # Create a src directory with a module
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("")
    (src_dir / "app.py").write_text("import os\nimport sys\n\ndef run(): pass\n")

    return tmp_path


@pytest.fixture
def workspace_with_tests(workspace: Path) -> Path:
    """Workspace with a tests directory and Python project indicator."""
    # Add pyproject.toml so the project is detected as Python
    (workspace / "pyproject.toml").write_text(
        '[project]\nname = "test-project"\nversion = "0.1.0"\n'
    )
    tests_dir = workspace / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("")
    (tests_dir / "test_main.py").write_text(
        "def test_hello():\n    from main import hello\n    assert hello() == 'world'\n"
    )
    return workspace


@pytest.fixture
def workspace_with_pyproject(workspace: Path) -> Path:
    """Workspace with pyproject.toml."""
    (workspace / "pyproject.toml").write_text(
        '[project]\nname = "test-project"\nversion = "0.1.0"\n'
    )
    return workspace


@pytest.fixture
def workspace_with_syntax_error(tmp_path: Path) -> Path:
    """Workspace with a Python syntax error."""
    (tmp_path / "broken.py").write_text("def broken(\n    # missing closing paren\n")
    return tmp_path


@pytest.fixture
def workspace_with_bad_import(tmp_path: Path) -> Path:
    """Workspace with an unresolvable import."""
    (tmp_path / "bad_import.py").write_text("import nonexistent_module_xyz\n")
    return tmp_path


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
def task_with_missing_file() -> Task:
    """Task that expects a file that doesn't exist."""
    return Task(
        id="0.1.2",
        title="Create new file",
        file_hints=["nonexistent.py"],
    )


# =============================================================================
# TestCheckResult
# =============================================================================


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_passed_property_true(self):
        """Test passed property returns True for PASSED status."""
        result = CheckResult(
            name="test",
            level=CheckLevel.FILES,
            status=CheckStatus.PASSED,
        )
        assert result.passed is True

    def test_passed_property_false_for_failed(self):
        """Test passed property returns False for FAILED status."""
        result = CheckResult(
            name="test",
            level=CheckLevel.FILES,
            status=CheckStatus.FAILED,
        )
        assert result.passed is False

    def test_passed_property_false_for_skipped(self):
        """Test passed property returns False for SKIPPED status."""
        result = CheckResult(
            name="test",
            level=CheckLevel.FILES,
            status=CheckStatus.SKIPPED,
        )
        assert result.passed is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = CheckResult(
            name="file_exists:main.py",
            level=CheckLevel.FILES,
            status=CheckStatus.PASSED,
            message="File exists",
            metadata={"path": "main.py"},
        )
        data = result.to_dict()

        assert data["name"] == "file_exists:main.py"
        assert data["level"] == "files"
        assert data["status"] == "passed"
        assert data["message"] == "File exists"
        assert data["metadata"]["path"] == "main.py"


# =============================================================================
# TestVerificationResult
# =============================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_overall_true_when_all_passed(self):
        """Test overall is True when all checks pass."""
        result = VerificationResult(
            task_id="0.1.1",
            checks=[
                CheckResult(name="a", level=CheckLevel.FILES, status=CheckStatus.PASSED),
                CheckResult(name="b", level=CheckLevel.SYNTAX, status=CheckStatus.PASSED),
            ],
        )
        assert result.overall is True

    def test_overall_true_with_skipped(self):
        """Test overall is True when some checks are skipped."""
        result = VerificationResult(
            task_id="0.1.1",
            checks=[
                CheckResult(name="a", level=CheckLevel.FILES, status=CheckStatus.PASSED),
                CheckResult(name="b", level=CheckLevel.TESTS, status=CheckStatus.SKIPPED),
            ],
        )
        assert result.overall is True

    def test_overall_false_when_any_failed(self):
        """Test overall is False when any check fails."""
        result = VerificationResult(
            task_id="0.1.1",
            checks=[
                CheckResult(name="a", level=CheckLevel.FILES, status=CheckStatus.PASSED),
                CheckResult(name="b", level=CheckLevel.SYNTAX, status=CheckStatus.FAILED),
            ],
        )
        assert result.overall is False

    def test_passed_count(self):
        """Test counting passed checks."""
        result = VerificationResult(
            task_id="0.1.1",
            checks=[
                CheckResult(name="a", level=CheckLevel.FILES, status=CheckStatus.PASSED),
                CheckResult(name="b", level=CheckLevel.SYNTAX, status=CheckStatus.PASSED),
                CheckResult(name="c", level=CheckLevel.TESTS, status=CheckStatus.FAILED),
            ],
        )
        assert result.passed_count == 2

    def test_failed_count(self):
        """Test counting failed checks."""
        result = VerificationResult(
            task_id="0.1.1",
            checks=[
                CheckResult(name="a", level=CheckLevel.FILES, status=CheckStatus.PASSED),
                CheckResult(name="b", level=CheckLevel.SYNTAX, status=CheckStatus.FAILED),
                CheckResult(name="c", level=CheckLevel.TESTS, status=CheckStatus.FAILED),
            ],
        )
        assert result.failed_count == 2

    def test_get_failures(self):
        """Test getting all failed checks."""
        failed_check = CheckResult(name="b", level=CheckLevel.SYNTAX, status=CheckStatus.FAILED)
        result = VerificationResult(
            task_id="0.1.1",
            checks=[
                CheckResult(name="a", level=CheckLevel.FILES, status=CheckStatus.PASSED),
                failed_check,
            ],
        )
        failures = result.get_failures()

        assert len(failures) == 1
        assert failures[0] == failed_check

    def test_summary(self):
        """Test human-readable summary."""
        result = VerificationResult(
            task_id="0.1.1",
            checks=[
                CheckResult(name="a", level=CheckLevel.FILES, status=CheckStatus.PASSED),
                CheckResult(name="b", level=CheckLevel.SYNTAX, status=CheckStatus.FAILED),
            ],
            total_duration_ms=123.45,
        )
        summary = result.summary()

        assert "FAILED" in summary
        assert "1 passed" in summary
        assert "1 failed" in summary
        assert "123ms" in summary

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = VerificationResult(
            task_id="0.1.1",
            checks=[
                CheckResult(name="a", level=CheckLevel.FILES, status=CheckStatus.PASSED),
            ],
            levels_run=[CheckLevel.FILES],
            total_duration_ms=100.0,
        )
        data = result.to_dict()

        assert data["task_id"] == "0.1.1"
        assert len(data["checks"]) == 1
        assert data["overall"] is True
        assert data["levels_run"] == ["files"]


# =============================================================================
# TestTaskVerifier - Initialization
# =============================================================================


class TestTaskVerifierInit:
    """Tests for TaskVerifier initialization."""

    def test_init_with_path(self, workspace: Path):
        """Test initialization with a Path."""
        verifier = TaskVerifier(workspace)
        assert verifier.workspace == workspace.resolve()

    def test_init_with_string(self, workspace: Path):
        """Test initialization with a string path."""
        verifier = TaskVerifier(str(workspace))
        assert verifier.workspace == workspace.resolve()

    def test_default_exclude_dirs(self, workspace: Path):
        """Test default exclude directories are set."""
        verifier = TaskVerifier(workspace)
        assert "__pycache__" in verifier.exclude_dirs
        assert ".git" in verifier.exclude_dirs
        assert "node_modules" in verifier.exclude_dirs

    def test_custom_exclude_dirs(self, workspace: Path):
        """Test custom exclude directories."""
        custom = frozenset({"custom_exclude"})
        verifier = TaskVerifier(workspace, exclude_dirs=custom)
        assert verifier.exclude_dirs == custom

    def test_custom_pytest_args(self, workspace: Path):
        """Test custom pytest arguments."""
        verifier = TaskVerifier(workspace, pytest_args=["-v", "--no-header"])
        assert "-v" in verifier.pytest_args

    def test_custom_mypy_args(self, workspace: Path):
        """Test custom mypy arguments."""
        verifier = TaskVerifier(workspace, mypy_args=["--strict"])
        assert "--strict" in verifier.mypy_args


# =============================================================================
# TestTaskVerifier - File Checks (Level 1)
# =============================================================================


class TestTaskVerifierFileChecks:
    """Tests for Level 1: File existence checks."""

    @pytest.mark.asyncio
    async def test_file_exists_passes(self, workspace: Path, sample_task: Task):
        """Test that existing files pass the check."""
        verifier = TaskVerifier(workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.FILES])

        assert result.overall is True
        file_checks = [c for c in result.checks if c.level == CheckLevel.FILES]
        assert len(file_checks) == 2
        assert all(c.passed for c in file_checks)

    @pytest.mark.asyncio
    async def test_missing_file_fails(self, workspace: Path, task_with_missing_file: Task):
        """Test that missing files fail the check."""
        verifier = TaskVerifier(workspace)
        result = await verifier.verify(task_with_missing_file, levels=[CheckLevel.FILES])

        assert result.overall is False
        assert result.failed_count == 1

    @pytest.mark.asyncio
    async def test_no_file_hints_skipped(self, workspace: Path):
        """Test that tasks with no file hints skip the check."""
        task = Task(id="0.1.1", title="No hints")
        verifier = TaskVerifier(workspace)
        result = await verifier.verify(task, levels=[CheckLevel.FILES])

        assert result.overall is True
        assert result.checks[0].status == CheckStatus.SKIPPED


# =============================================================================
# TestTaskVerifier - Syntax Checks (Level 2)
# =============================================================================


class TestTaskVerifierSyntaxChecks:
    """Tests for Level 2: Python syntax validation."""

    @pytest.mark.asyncio
    async def test_valid_syntax_passes(self, workspace: Path, sample_task: Task):
        """Test that valid Python files pass syntax check."""
        verifier = TaskVerifier(workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.SYNTAX])

        assert result.overall is True
        syntax_checks = [c for c in result.checks if c.level == CheckLevel.SYNTAX]
        assert len(syntax_checks) > 0
        assert all(c.passed for c in syntax_checks)

    @pytest.mark.asyncio
    async def test_syntax_error_fails(self, workspace_with_syntax_error: Path, sample_task: Task):
        """Test that syntax errors fail the check."""
        verifier = TaskVerifier(workspace_with_syntax_error)
        result = await verifier.verify(sample_task, levels=[CheckLevel.SYNTAX])

        assert result.overall is False
        failures = result.get_failures()
        assert len(failures) == 1
        assert "syntax" in failures[0].name.lower()

    @pytest.mark.asyncio
    async def test_excludes_pycache(self, workspace: Path, sample_task: Task):
        """Test that __pycache__ directories are excluded."""
        # Create a file in __pycache__
        pycache = workspace / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.py").write_text("invalid syntax {{{")

        verifier = TaskVerifier(workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.SYNTAX])

        # Should still pass because __pycache__ is excluded
        assert result.overall is True

    @pytest.mark.asyncio
    async def test_no_python_files_skipped(self, tmp_path: Path, sample_task: Task):
        """Test that empty workspace skips syntax check."""
        (tmp_path / "readme.txt").write_text("No Python here")

        verifier = TaskVerifier(tmp_path)
        result = await verifier.verify(sample_task, levels=[CheckLevel.SYNTAX])

        assert result.overall is True
        assert result.checks[0].status == CheckStatus.SKIPPED


# =============================================================================
# TestTaskVerifier - Import Checks (Level 3)
# =============================================================================


class TestTaskVerifierImportChecks:
    """Tests for Level 3: Import resolution checks."""

    @pytest.mark.asyncio
    async def test_stdlib_imports_pass(self, workspace: Path, sample_task: Task):
        """Test that standard library imports pass."""
        verifier = TaskVerifier(workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.IMPORTS])

        assert result.overall is True
        import_checks = [c for c in result.checks if c.level == CheckLevel.IMPORTS]
        # Should find 'os' and 'sys' imports from src/app.py
        import_names = [c.name for c in import_checks]
        assert any("os" in name for name in import_names)
        assert any("sys" in name for name in import_names)

    @pytest.mark.asyncio
    async def test_missing_import_fails(self, workspace_with_bad_import: Path, sample_task: Task):
        """Test that unresolvable imports fail."""
        verifier = TaskVerifier(workspace_with_bad_import)
        result = await verifier.verify(sample_task, levels=[CheckLevel.IMPORTS])

        assert result.overall is False
        failures = result.get_failures()
        assert len(failures) == 1
        assert "nonexistent_module_xyz" in failures[0].name

    @pytest.mark.asyncio
    async def test_local_module_import_passes(self, workspace: Path, sample_task: Task):
        """Test that local module imports are detected."""
        # Add a file that imports from src
        (workspace / "use_app.py").write_text("from src import app\n")

        verifier = TaskVerifier(workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.IMPORTS])

        # 'src' should be found as a local module
        import_checks = [c for c in result.checks if c.level == CheckLevel.IMPORTS]
        src_check = next((c for c in import_checks if "src" in c.name), None)
        assert src_check is not None
        assert src_check.passed


# =============================================================================
# TestTaskVerifier - Test Execution (Level 4)
# =============================================================================


class TestTaskVerifierTestChecks:
    """Tests for Level 4: Test execution."""

    @pytest.mark.asyncio
    async def test_no_tests_dir_skipped(self, workspace: Path, sample_task: Task):
        """Test that missing tests directory skips the check."""
        verifier = TaskVerifier(workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.TESTS])

        assert result.overall is True
        assert result.checks[0].status == CheckStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_passing_tests_pass(self, workspace_with_tests: Path, sample_task: Task):
        """Test that passing tests result in passed check."""
        verifier = TaskVerifier(workspace_with_tests)
        result = await verifier.verify(sample_task, levels=[CheckLevel.TESTS])

        assert result.overall is True
        test_check = result.checks[0]
        assert test_check.status == CheckStatus.PASSED

    @pytest.mark.asyncio
    async def test_failing_tests_fail(self, workspace_with_tests: Path, sample_task: Task):
        """Test that failing tests result in failed check."""
        # Modify the test to fail
        test_file = workspace_with_tests / "tests" / "test_main.py"
        test_file.write_text("def test_always_fails():\n    assert False\n")

        verifier = TaskVerifier(workspace_with_tests)
        result = await verifier.verify(sample_task, levels=[CheckLevel.TESTS])

        assert result.overall is False
        test_check = result.checks[0]
        assert test_check.status == CheckStatus.FAILED


# =============================================================================
# TestTaskVerifier - Type Checking (Level 5)
# =============================================================================


class TestTaskVerifierTypeChecks:
    """Tests for Level 5: Type checking with mypy."""

    @pytest.mark.asyncio
    async def test_no_pyproject_skipped(self, workspace: Path, sample_task: Task):
        """Test that missing pyproject.toml skips the check."""
        verifier = TaskVerifier(workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.TYPES])

        assert result.overall is True
        assert result.checks[0].status == CheckStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_with_pyproject_runs_mypy(
        self, workspace_with_pyproject: Path, sample_task: Task
    ):
        """Test that mypy runs when pyproject.toml exists."""
        verifier = TaskVerifier(workspace_with_pyproject)
        result = await verifier.verify(sample_task, levels=[CheckLevel.TYPES])

        # Check should run (pass or fail, not skipped)
        type_check = result.checks[0]
        assert type_check.status in (CheckStatus.PASSED, CheckStatus.FAILED, CheckStatus.ERROR)


# =============================================================================
# TestTaskVerifier - Full Verification
# =============================================================================


class TestTaskVerifierFullVerification:
    """Tests for complete verification workflow."""

    @pytest.mark.asyncio
    async def test_verify_all_levels(self, workspace: Path, sample_task: Task):
        """Test running all verification levels."""
        verifier = TaskVerifier(workspace)
        result = await verifier.verify(sample_task)

        assert len(result.levels_run) == 6
        assert CheckLevel.FILES in result.levels_run
        assert CheckLevel.SYNTAX in result.levels_run
        assert CheckLevel.IMPORTS in result.levels_run
        assert CheckLevel.RUNTIME in result.levels_run
        assert CheckLevel.TESTS in result.levels_run
        assert CheckLevel.TYPES in result.levels_run

    @pytest.mark.asyncio
    async def test_stop_on_failure(self, workspace_with_syntax_error: Path):
        """Test stopping at first failure."""
        # Use task without file hints so FILES check is skipped
        task = Task(id="0.1.1", title="Test task")

        verifier = TaskVerifier(workspace_with_syntax_error)
        result = await verifier.verify(
            task,
            levels=[CheckLevel.FILES, CheckLevel.SYNTAX, CheckLevel.TESTS],
            stop_on_failure=True,
        )

        # FILES should be skipped, SYNTAX should fail
        levels_with_checks = {c.level for c in result.checks}
        assert CheckLevel.SYNTAX in levels_with_checks
        # TESTS level should not have been reached due to SYNTAX failure
        assert CheckLevel.TESTS not in levels_with_checks

    @pytest.mark.asyncio
    async def test_quick_verify(self, workspace: Path, sample_task: Task):
        """Test quick verification (files + syntax only)."""
        verifier = TaskVerifier(workspace)
        result = await verifier.quick_verify(sample_task)

        levels_run = set(result.levels_run)
        assert levels_run == {CheckLevel.FILES, CheckLevel.SYNTAX}

    @pytest.mark.asyncio
    async def test_full_verify(self, workspace: Path, sample_task: Task):
        """Test full verification (all levels)."""
        verifier = TaskVerifier(workspace)
        result = await verifier.full_verify(sample_task)

        assert len(result.levels_run) == 6

    @pytest.mark.asyncio
    async def test_duration_tracking(self, workspace: Path, sample_task: Task):
        """Test that duration is tracked."""
        verifier = TaskVerifier(workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.SYNTAX])

        assert result.total_duration_ms > 0


# =============================================================================
# TestTaskVerifier - Edge Cases
# =============================================================================


class TestTaskVerifierEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_workspace(self, tmp_path: Path, sample_task: Task):
        """Test verification on empty workspace."""
        verifier = TaskVerifier(tmp_path)
        result = await verifier.verify(sample_task)

        # Should complete without errors
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_binary_file_ignored(self, workspace: Path, sample_task: Task):
        """Test that binary files don't cause errors."""
        # Create a binary file
        (workspace / "binary.py").write_bytes(b"\x00\x01\x02\x03")

        verifier = TaskVerifier(workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.SYNTAX])

        # Should handle gracefully (error status, not exception)
        binary_check = next((c for c in result.checks if "binary.py" in c.name), None)
        if binary_check:
            assert binary_check.status in (CheckStatus.ERROR, CheckStatus.FAILED)

    @pytest.mark.asyncio
    async def test_nested_exclude_dirs(self, workspace: Path, sample_task: Task):
        """Test that nested excluded directories are skipped."""
        # Create a nested __pycache__
        nested = workspace / "src" / "__pycache__"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "cached.py").write_text("invalid {{{")

        verifier = TaskVerifier(workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.SYNTAX])

        # Should pass because __pycache__ is excluded
        assert result.overall is True


# =============================================================================
# TestRuntimeCheck - Runtime Import Verification
# =============================================================================


class TestRuntimeCheck:
    """Tests for runtime import verification."""

    @pytest.fixture
    def python_workspace(self, tmp_path: Path) -> Path:
        """Create a Python workspace with pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_path / "src").mkdir()
        return tmp_path

    @pytest.fixture
    def sample_task(self) -> Task:
        """Create a sample task."""
        return Task(id="0.1.1", title="Test", file_hints=["src/main.py"])

    @pytest.mark.asyncio
    async def test_runtime_check_success(self, python_workspace: Path, sample_task: Task):
        """Test runtime check with valid Python module."""
        # Create a valid Python module
        (python_workspace / "src" / "valid_module.py").write_text(
            "def hello():\n    return 'world'\n"
        )

        verifier = TaskVerifier(python_workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.RUNTIME])

        assert CheckLevel.RUNTIME in result.levels_run
        runtime_checks = [c for c in result.checks if c.level == CheckLevel.RUNTIME]
        assert len(runtime_checks) >= 1
        # Should pass because module imports successfully
        passed = [c for c in runtime_checks if c.status == CheckStatus.PASSED]
        assert len(passed) >= 1

    @pytest.mark.asyncio
    async def test_runtime_check_circular_import(self, python_workspace: Path, sample_task: Task):
        """Test runtime check detects circular imports."""
        src = python_workspace / "src"

        # Create module A that imports from B
        (src / "module_a.py").write_text(
            "from src.module_b import func_b\n\ndef func_a():\n    return func_b()\n"
        )

        # Create module B that imports from A (circular!)
        (src / "module_b.py").write_text(
            "from src.module_a import func_a\n\ndef func_b():\n    return 'b'\n"
        )

        verifier = TaskVerifier(python_workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.RUNTIME])

        runtime_checks = [c for c in result.checks if c.level == CheckLevel.RUNTIME]
        assert len(runtime_checks) >= 1

        # At least one should fail due to circular import
        failed = [c for c in runtime_checks if c.status == CheckStatus.FAILED]
        assert len(failed) >= 1

        # Error should mention circular import
        circular_errors = [c for c in failed if c.error and "circular" in c.error.lower()]
        assert len(circular_errors) >= 1

    @pytest.mark.asyncio
    async def test_runtime_check_import_error(self, python_workspace: Path, sample_task: Task):
        """Test runtime check detects missing module imports."""
        src = python_workspace / "src"

        # Create module that imports non-existent module
        (src / "bad_import.py").write_text("from nonexistent_module import something\n")

        verifier = TaskVerifier(python_workspace)
        result = await verifier.verify(sample_task, levels=[CheckLevel.RUNTIME])

        runtime_checks = [c for c in result.checks if c.level == CheckLevel.RUNTIME]
        failed = [c for c in runtime_checks if c.status == CheckStatus.FAILED]
        assert len(failed) >= 1

    @pytest.mark.asyncio
    async def test_runtime_check_no_python_files(self, tmp_path: Path, sample_task: Task):
        """Test runtime check with no Python files."""
        verifier = TaskVerifier(tmp_path)
        result = await verifier.verify(sample_task, levels=[CheckLevel.RUNTIME])

        runtime_checks = [c for c in result.checks if c.level == CheckLevel.RUNTIME]
        assert len(runtime_checks) >= 1
        # Should be skipped since no project type detected
        skipped = [c for c in runtime_checks if c.status == CheckStatus.SKIPPED]
        assert len(skipped) >= 1
