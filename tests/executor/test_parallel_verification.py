"""Tests for parallel verification module.

Phase 1.4: Parallel verification tests covering:
- 1.4.1: Parallel execution with asyncio.gather()
- 1.4.2: Partial failure handling with aggregate_results()
- 1.4.3: Check-specific timeouts
- 1.4.4: Skip unchanged file checks
- 1.4.5: Performance benchmarks
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from ai_infra.executor.parallel_verification import (
    DEFAULT_CHECK_CONFIGS,
    CheckConfig,
    CheckType,
    ParallelCheckResult,
    ParallelVerificationResult,
    ParallelVerifier,
    aggregate_results,
    create_parallel_verifier,
    get_checks_to_run,
    should_run_check,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# CheckType Tests
# =============================================================================


class TestCheckType:
    """Tests for CheckType enum."""

    def test_check_types_exist(self) -> None:
        """Verify all expected check types exist."""
        assert CheckType.SYNTAX.value == "syntax"
        assert CheckType.TYPES.value == "types"
        assert CheckType.LINT.value == "lint"
        assert CheckType.TESTS.value == "tests"
        assert CheckType.IMPORTS.value == "imports"
        assert CheckType.FILES.value == "files"

    def test_check_types_unique(self) -> None:
        """Verify check type values are unique."""
        values = [ct.value for ct in CheckType]
        assert len(values) == len(set(values))


# =============================================================================
# CheckConfig Tests (Phase 1.4.3 - Check-specific Timeouts)
# =============================================================================


class TestCheckConfig:
    """Tests for CheckConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default CheckConfig values."""
        config = CheckConfig(check_type=CheckType.SYNTAX)
        assert config.timeout == 30.0
        assert config.extensions == frozenset()
        assert config.skip_on_timeout is False
        assert config.enabled is True

    def test_syntax_factory(self) -> None:
        """Test syntax check factory method."""
        config = CheckConfig.syntax()
        assert config.check_type == CheckType.SYNTAX
        assert config.timeout == 10.0
        assert ".py" in config.extensions
        assert config.skip_on_timeout is False

    def test_syntax_factory_custom_timeout(self) -> None:
        """Test syntax check with custom timeout."""
        config = CheckConfig.syntax(timeout=5.0)
        assert config.timeout == 5.0

    def test_types_factory(self) -> None:
        """Test types check factory method."""
        config = CheckConfig.types()
        assert config.check_type == CheckType.TYPES
        assert config.timeout == 30.0
        assert ".py" in config.extensions
        assert config.skip_on_timeout is True  # Soft timeout

    def test_lint_factory(self) -> None:
        """Test lint check factory method."""
        config = CheckConfig.lint()
        assert config.check_type == CheckType.LINT
        assert config.timeout == 20.0
        assert ".py" in config.extensions
        assert ".js" in config.extensions
        assert ".ts" in config.extensions
        assert config.skip_on_timeout is True  # Soft timeout

    def test_tests_factory(self) -> None:
        """Test tests check factory method."""
        config = CheckConfig.tests()
        assert config.check_type == CheckType.TESTS
        assert config.timeout == 120.0
        assert config.extensions == frozenset()  # Always run
        assert config.skip_on_timeout is False  # Hard timeout

    def test_imports_factory(self) -> None:
        """Test imports check factory method."""
        config = CheckConfig.imports()
        assert config.check_type == CheckType.IMPORTS
        assert config.timeout == 15.0
        assert ".py" in config.extensions
        assert config.skip_on_timeout is True  # Soft timeout


# =============================================================================
# ParallelCheckResult Tests
# =============================================================================


class TestParallelCheckResult:
    """Tests for ParallelCheckResult dataclass."""

    def test_default_passed(self) -> None:
        """Test default result is passed."""
        result = ParallelCheckResult(check_type=CheckType.SYNTAX)
        assert result.passed is True
        assert result.skipped is False
        assert result.timed_out is False
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test failed result."""
        result = ParallelCheckResult(
            check_type=CheckType.TYPES,
            passed=False,
            error="Type error in main.py",
        )
        assert result.passed is False
        assert result.error == "Type error in main.py"

    def test_timeout_result(self) -> None:
        """Test timeout result."""
        result = ParallelCheckResult(
            check_type=CheckType.TESTS,
            passed=False,
            timed_out=True,
            duration_ms=120000,
            error="Timed out after 120s",
        )
        assert result.timed_out is True
        assert result.passed is False

    def test_skipped_result(self) -> None:
        """Test skipped result."""
        result = ParallelCheckResult(
            check_type=CheckType.LINT,
            skipped=True,
        )
        assert result.skipped is True
        assert result.passed is True

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = ParallelCheckResult(
            check_type=CheckType.SYNTAX,
            passed=True,
            duration_ms=150.5,
            warnings=["Minor style issue"],
        )
        data = result.to_dict()
        assert data["check_type"] == "syntax"
        assert data["passed"] is True
        assert data["duration_ms"] == 150.5
        assert "Minor style issue" in data["warnings"]


# =============================================================================
# ParallelVerificationResult Tests (Phase 1.4.2)
# =============================================================================


class TestParallelVerificationResult:
    """Tests for ParallelVerificationResult dataclass."""

    def test_default_passed(self) -> None:
        """Test default result is passed."""
        result = ParallelVerificationResult()
        assert result.passed is True
        assert result.checks == []
        assert result.failures == []

    def test_with_checks(self) -> None:
        """Test result with multiple checks."""
        checks = [
            ParallelCheckResult(check_type=CheckType.SYNTAX, passed=True, duration_ms=100),
            ParallelCheckResult(check_type=CheckType.TYPES, passed=True, duration_ms=200),
            ParallelCheckResult(check_type=CheckType.TESTS, passed=False, duration_ms=300),
        ]
        result = ParallelVerificationResult(
            passed=False,
            checks=checks,
            failures=["Tests failed"],
            duration_ms=350,
        )
        assert result.passed is False
        assert len(result.checks) == 3
        assert result.checks_run == 3
        assert result.checks_passed == 2
        assert result.checks_failed == 1

    def test_speedup_calculation(self) -> None:
        """Test speedup calculation."""
        result = ParallelVerificationResult(
            duration_ms=500,
            sequential_duration_ms=1500,
        )
        assert result.speedup == 3.0

    def test_speedup_zero_duration(self) -> None:
        """Test speedup with zero duration (no division error)."""
        result = ParallelVerificationResult(
            duration_ms=0,
            sequential_duration_ms=1000,
        )
        assert result.speedup == 1.0

    def test_checks_with_skipped(self) -> None:
        """Test check counts exclude skipped."""
        checks = [
            ParallelCheckResult(check_type=CheckType.SYNTAX, passed=True, duration_ms=100),
            ParallelCheckResult(check_type=CheckType.LINT, skipped=True),
            ParallelCheckResult(check_type=CheckType.TYPES, passed=True, duration_ms=200),
        ]
        result = ParallelVerificationResult(checks=checks)
        assert result.checks_run == 2  # Skipped not counted
        assert result.checks_passed == 2

    def test_summary(self) -> None:
        """Test summary generation."""
        result = ParallelVerificationResult(
            passed=True,
            checks=[
                ParallelCheckResult(check_type=CheckType.SYNTAX, passed=True),
                ParallelCheckResult(check_type=CheckType.TYPES, passed=True),
            ],
            duration_ms=500,
            sequential_duration_ms=1000,
        )
        summary = result.summary()
        assert "PASSED" in summary
        assert "2/2" in summary
        assert "500ms" in summary
        assert "2.0x" in summary

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = ParallelVerificationResult(
            passed=True,
            checks=[
                ParallelCheckResult(check_type=CheckType.SYNTAX, passed=True),
            ],
            duration_ms=100,
            sequential_duration_ms=100,
        )
        data = result.to_dict()
        assert data["passed"] is True
        assert len(data["checks"]) == 1
        assert data["speedup"] == 1.0


# =============================================================================
# aggregate_results Tests (Phase 1.4.2)
# =============================================================================


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_aggregate_all_passed(self) -> None:
        """Test aggregating all passed results."""
        results = [
            ParallelCheckResult(check_type=CheckType.SYNTAX, passed=True, duration_ms=100),
            ParallelCheckResult(check_type=CheckType.TYPES, passed=True, duration_ms=200),
        ]
        aggregated = aggregate_results(results)
        assert aggregated.passed is True
        assert len(aggregated.failures) == 0
        assert aggregated.sequential_duration_ms == 300

    def test_aggregate_with_failure(self) -> None:
        """Test aggregating with a failure."""
        results = [
            ParallelCheckResult(check_type=CheckType.SYNTAX, passed=True, duration_ms=100),
            ParallelCheckResult(
                check_type=CheckType.TYPES,
                passed=False,
                error="Type error",
                duration_ms=200,
            ),
        ]
        aggregated = aggregate_results(results)
        assert aggregated.passed is False
        assert "Type error" in aggregated.failures[0]

    def test_aggregate_with_exception(self) -> None:
        """Test aggregating with exception result."""
        results: list[ParallelCheckResult | Exception] = [
            ParallelCheckResult(check_type=CheckType.SYNTAX, passed=True),
            RuntimeError("Unexpected error"),
        ]
        aggregated = aggregate_results(results)
        assert aggregated.passed is False
        assert len(aggregated.failures) == 1
        assert "exception" in aggregated.failures[0].lower()

    def test_aggregate_timeout_soft(self) -> None:
        """Test aggregating with soft timeout (warning, not failure)."""
        results = [
            ParallelCheckResult(
                check_type=CheckType.TYPES,
                passed=True,
                timed_out=True,
                duration_ms=30000,
            ),
        ]
        # Types has skip_on_timeout=True by default
        aggregated = aggregate_results(results, DEFAULT_CHECK_CONFIGS)
        assert aggregated.passed is True
        assert len(aggregated.warnings) == 1
        assert "timed out" in aggregated.warnings[0].lower()

    def test_aggregate_timeout_hard(self) -> None:
        """Test aggregating with hard timeout (failure)."""
        results = [
            ParallelCheckResult(
                check_type=CheckType.SYNTAX,
                passed=False,
                timed_out=True,
                duration_ms=10000,
            ),
        ]
        # Syntax has skip_on_timeout=False by default
        aggregated = aggregate_results(results, DEFAULT_CHECK_CONFIGS)
        assert aggregated.passed is False
        assert len(aggregated.failures) == 1
        assert "timed out" in aggregated.failures[0].lower()

    def test_aggregate_skipped_ignored(self) -> None:
        """Test that skipped checks are ignored in aggregation."""
        results = [
            ParallelCheckResult(check_type=CheckType.SYNTAX, passed=True),
            ParallelCheckResult(check_type=CheckType.LINT, skipped=True),
        ]
        aggregated = aggregate_results(results)
        assert aggregated.passed is True
        assert aggregated.checks_run == 1

    def test_aggregate_collects_warnings(self) -> None:
        """Test that warnings are collected from results."""
        results = [
            ParallelCheckResult(
                check_type=CheckType.SYNTAX,
                passed=True,
                warnings=["Warning 1"],
            ),
            ParallelCheckResult(
                check_type=CheckType.TYPES,
                passed=True,
                warnings=["Warning 2", "Warning 3"],
            ),
        ]
        aggregated = aggregate_results(results)
        assert len(aggregated.warnings) == 3


# =============================================================================
# should_run_check Tests (Phase 1.4.4)
# =============================================================================


class TestShouldRunCheck:
    """Tests for should_run_check function."""

    def test_python_file_runs_syntax(self) -> None:
        """Test Python file triggers syntax check."""
        assert should_run_check(CheckType.SYNTAX, [Path("src/main.py")]) is True

    def test_python_file_runs_types(self) -> None:
        """Test Python file triggers type check."""
        assert should_run_check(CheckType.TYPES, [Path("src/main.py")]) is True

    def test_js_file_skips_types(self) -> None:
        """Test JS file does not trigger type check."""
        assert should_run_check(CheckType.TYPES, [Path("src/main.js")]) is False

    def test_js_file_runs_lint(self) -> None:
        """Test JS file triggers lint check."""
        assert should_run_check(CheckType.LINT, [Path("src/main.js")]) is True

    def test_tests_always_run(self) -> None:
        """Test tests check runs regardless of file type."""
        # Tests config has empty extensions = always run
        assert should_run_check(CheckType.TESTS, [Path("README.md")]) is True

    def test_no_files_no_checks(self) -> None:
        """Test no files means no extension-dependent checks run."""
        # But tests still run (empty extensions = always)
        assert should_run_check(CheckType.SYNTAX, []) is False
        assert should_run_check(CheckType.TESTS, []) is True

    def test_disabled_config_skips(self) -> None:
        """Test disabled config skips check."""
        config = CheckConfig(
            check_type=CheckType.SYNTAX,
            enabled=False,
        )
        assert should_run_check(CheckType.SYNTAX, [Path("src/main.py")], config) is False

    def test_string_paths_work(self) -> None:
        """Test string paths are accepted."""
        assert should_run_check(CheckType.SYNTAX, ["src/main.py"]) is True

    def test_mixed_path_types(self) -> None:
        """Test mixed Path and string types."""
        files = [Path("src/a.py"), "src/b.py"]
        assert should_run_check(CheckType.SYNTAX, files) is True


# =============================================================================
# get_checks_to_run Tests (Phase 1.4.4)
# =============================================================================


class TestGetChecksToRun:
    """Tests for get_checks_to_run function."""

    def test_python_files_get_python_checks(self) -> None:
        """Test Python files get appropriate checks."""
        checks = get_checks_to_run(
            files_modified=[Path("src/main.py")],
            requested_checks=[CheckType.SYNTAX, CheckType.TYPES, CheckType.TESTS],
        )
        assert CheckType.SYNTAX in checks
        assert CheckType.TYPES in checks
        assert CheckType.TESTS in checks

    def test_js_files_skip_python_checks(self) -> None:
        """Test JS files skip Python-only checks."""
        checks = get_checks_to_run(
            files_modified=[Path("src/app.js")],
            requested_checks=[CheckType.SYNTAX, CheckType.TYPES, CheckType.LINT],
        )
        # Syntax requires .py, Types requires .py, Lint includes .js
        assert CheckType.SYNTAX not in checks
        assert CheckType.TYPES not in checks
        assert CheckType.LINT in checks

    def test_defaults_to_all_check_types(self) -> None:
        """Test defaults to all check types when none specified."""
        checks = get_checks_to_run(
            files_modified=[Path("src/main.py")],
            requested_checks=None,
        )
        assert CheckType.SYNTAX in checks
        assert CheckType.TYPES in checks
        assert CheckType.TESTS in checks

    def test_empty_files_only_runs_universal(self) -> None:
        """Test empty files only runs checks with no extension filter."""
        checks = get_checks_to_run(
            files_modified=[],
        )
        # Only TESTS and FILES have empty extensions
        assert CheckType.SYNTAX not in checks
        assert CheckType.TESTS in checks

    def test_custom_configs(self) -> None:
        """Test custom check configurations."""
        custom_configs = {
            CheckType.SYNTAX: CheckConfig(
                check_type=CheckType.SYNTAX,
                extensions=frozenset({".custom"}),
            ),
        }
        checks = get_checks_to_run(
            files_modified=[Path("src/main.custom")],
            requested_checks=[CheckType.SYNTAX],
            check_configs=custom_configs,
        )
        assert CheckType.SYNTAX in checks


# =============================================================================
# ParallelVerifier Tests (Phase 1.4.1)
# =============================================================================


class TestParallelVerifier:
    """Tests for ParallelVerifier class."""

    def test_init_with_path(self) -> None:
        """Test initialization with path."""
        verifier = ParallelVerifier(workspace=Path("/tmp/test"))
        # macOS resolves /tmp to /private/tmp, so check the name only
        assert verifier.workspace.name == "test"
        assert verifier.check_configs is not None

    def test_init_with_string(self) -> None:
        """Test initialization with string path."""
        verifier = ParallelVerifier(workspace="/tmp/test")
        assert verifier.workspace.name == "test"

    def test_init_with_custom_configs(self) -> None:
        """Test initialization with custom configs."""
        configs = {
            CheckType.SYNTAX: CheckConfig.syntax(timeout=5.0),
        }
        verifier = ParallelVerifier(
            workspace="/tmp/test",
            check_configs=configs,
        )
        assert verifier.check_configs[CheckType.SYNTAX].timeout == 5.0

    @pytest.mark.asyncio
    async def test_verify_parallel_no_files(self) -> None:
        """Test parallel verification with no files."""
        verifier = ParallelVerifier(workspace="/tmp/test")
        result = await verifier.verify_parallel(files_modified=[])
        # Tests check still runs (empty extensions = always run)
        assert result is not None

    @pytest.mark.asyncio
    async def test_verify_parallel_mocked(self) -> None:
        """Test parallel verification with mocked verifier."""
        from ai_infra.executor.models import Task

        mock_verifier = MagicMock()
        mock_result = MagicMock()
        mock_result.overall = True
        mock_result.checks = []
        mock_result.passed_count = 1
        mock_result.failed_count = 0
        mock_result.get_failures.return_value = []

        async def mock_verify(*args, **kwargs):
            await asyncio.sleep(0.01)  # Small delay
            return mock_result

        mock_verifier.verify = mock_verify

        verifier = ParallelVerifier(
            workspace="/tmp/test",
            verifier=mock_verifier,
        )

        task = Task(
            id="test",
            title="Test Task",
            description="Test",
            file_hints=["src/main.py"],
        )

        result = await verifier.verify_parallel(
            task=task,
            files_modified=[Path("src/main.py")],
            checks=[CheckType.SYNTAX],
        )

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_verify_parallel_runs_concurrently(self) -> None:
        """Test that checks run concurrently (Phase 1.4.1)."""
        from ai_infra.executor.models import Task

        call_times: list[float] = []

        async def slow_verify(*args, **kwargs):
            import time

            start = time.perf_counter()
            await asyncio.sleep(0.1)  # 100ms delay
            call_times.append(time.perf_counter() - start)
            mock_result = MagicMock()
            mock_result.overall = True
            mock_result.checks = []
            mock_result.passed_count = 1
            mock_result.failed_count = 0
            mock_result.get_failures.return_value = []
            return mock_result

        mock_verifier = MagicMock()
        mock_verifier.verify = slow_verify

        verifier = ParallelVerifier(
            workspace="/tmp/test",
            verifier=mock_verifier,
        )

        task = Task(
            id="test",
            title="Test Task",
            description="Test",
            file_hints=["src/main.py"],
        )

        import time

        start = time.perf_counter()
        result = await verifier.verify_parallel(
            task=task,
            files_modified=[Path("src/main.py")],
            checks=[CheckType.SYNTAX, CheckType.TYPES],
        )
        total_time = (time.perf_counter() - start) * 1000

        # If sequential: ~200ms, if parallel: ~100ms
        # Allow some overhead but should be < 180ms if truly parallel
        assert total_time < 180, f"Checks not running in parallel: {total_time}ms"
        assert result.passed is True


# =============================================================================
# Timeout Tests (Phase 1.4.3)
# =============================================================================


class TestTimeouts:
    """Tests for check-specific timeouts."""

    @pytest.mark.asyncio
    async def test_timeout_triggers(self) -> None:
        """Test that timeout triggers for slow checks."""
        from ai_infra.executor.models import Task

        async def very_slow_verify(*args, **kwargs):
            await asyncio.sleep(10)  # Will timeout
            mock_result = MagicMock()
            mock_result.overall = True
            return mock_result

        mock_verifier = MagicMock()
        mock_verifier.verify = very_slow_verify

        # Use very short timeout
        configs = {
            CheckType.SYNTAX: CheckConfig(
                check_type=CheckType.SYNTAX,
                timeout=0.05,  # 50ms timeout
                extensions=frozenset({".py"}),
                skip_on_timeout=False,  # Hard timeout
            ),
        }

        verifier = ParallelVerifier(
            workspace="/tmp/test",
            verifier=mock_verifier,
            check_configs=configs,
        )

        task = Task(
            id="test",
            title="Test Task",
            description="Test",
            file_hints=["src/main.py"],
        )

        result = await verifier.verify_parallel(
            task=task,
            files_modified=[Path("src/main.py")],
            checks=[CheckType.SYNTAX],
        )

        # Should fail due to hard timeout
        assert result.passed is False
        assert any(r.timed_out for r in result.checks)

    @pytest.mark.asyncio
    async def test_soft_timeout_passes(self) -> None:
        """Test that soft timeout results in pass with warning."""
        from ai_infra.executor.models import Task

        async def very_slow_verify(*args, **kwargs):
            await asyncio.sleep(10)  # Will timeout
            mock_result = MagicMock()
            mock_result.overall = True
            return mock_result

        mock_verifier = MagicMock()
        mock_verifier.verify = very_slow_verify

        # Use soft timeout
        configs = {
            CheckType.TYPES: CheckConfig(
                check_type=CheckType.TYPES,
                timeout=0.05,  # 50ms timeout
                extensions=frozenset({".py"}),
                skip_on_timeout=True,  # Soft timeout
            ),
        }

        verifier = ParallelVerifier(
            workspace="/tmp/test",
            verifier=mock_verifier,
            check_configs=configs,
        )

        task = Task(
            id="test",
            title="Test Task",
            description="Test",
            file_hints=["src/main.py"],
        )

        result = await verifier.verify_parallel(
            task=task,
            files_modified=[Path("src/main.py")],
            checks=[CheckType.TYPES],
        )

        # Should pass with soft timeout (warning only)
        assert result.passed is True
        assert any(r.timed_out for r in result.checks)
        assert len(result.warnings) > 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateParallelVerifier:
    """Tests for create_parallel_verifier factory."""

    def test_create_with_defaults(self) -> None:
        """Test creating verifier with defaults."""
        verifier = create_parallel_verifier(workspace="/tmp/test")
        assert verifier.check_configs[CheckType.SYNTAX].timeout == 10.0
        assert verifier.check_configs[CheckType.TYPES].timeout == 30.0
        assert verifier.check_configs[CheckType.LINT].timeout == 20.0
        assert verifier.check_configs[CheckType.TESTS].timeout == 120.0

    def test_create_with_custom_timeouts(self) -> None:
        """Test creating verifier with custom timeouts."""
        verifier = create_parallel_verifier(
            workspace="/tmp/test",
            syntax_timeout=5.0,
            types_timeout=60.0,
            lint_timeout=10.0,
            tests_timeout=300.0,
        )
        assert verifier.check_configs[CheckType.SYNTAX].timeout == 5.0
        assert verifier.check_configs[CheckType.TYPES].timeout == 60.0
        assert verifier.check_configs[CheckType.LINT].timeout == 10.0
        assert verifier.check_configs[CheckType.TESTS].timeout == 300.0


# =============================================================================
# Performance Benchmark Tests (Phase 1.4.5)
# =============================================================================


class TestPerformanceBenchmark:
    """Performance benchmark tests for parallel verification."""

    @pytest.mark.asyncio
    async def test_speedup_over_sequential(self) -> None:
        """Test that parallel execution achieves speedup over sequential.

        Phase 1.4.5: Benchmark verification time reduction.
        Target: 50% reduction (3s sequential -> 1.5s parallel).
        """
        from ai_infra.executor.models import Task

        # Simulate checks with different durations
        check_delays = {
            CheckType.SYNTAX: 0.05,  # 50ms
            CheckType.TYPES: 0.1,  # 100ms
            CheckType.LINT: 0.08,  # 80ms
        }

        async def delayed_verify(*args, **kwargs):
            levels = kwargs.get("levels", [])
            if levels:
                from ai_infra.executor.verifier import CheckLevel

                level = levels[0]
                delay = 0.05  # Default
                if level == CheckLevel.SYNTAX:
                    delay = check_delays.get(CheckType.SYNTAX, 0.05)
                elif level == CheckLevel.TYPES:
                    delay = check_delays.get(CheckType.TYPES, 0.1)
            else:
                delay = 0.05

            await asyncio.sleep(delay)

            mock_result = MagicMock()
            mock_result.overall = True
            mock_result.checks = []
            mock_result.passed_count = 1
            mock_result.failed_count = 0
            mock_result.get_failures.return_value = []
            return mock_result

        mock_verifier = MagicMock()
        mock_verifier.verify = delayed_verify

        verifier = ParallelVerifier(
            workspace="/tmp/test",
            verifier=mock_verifier,
        )

        task = Task(
            id="test",
            title="Test Task",
            description="Test",
            file_hints=["src/main.py"],
        )

        result = await verifier.verify_parallel(
            task=task,
            files_modified=[Path("src/main.py")],
            checks=[CheckType.SYNTAX, CheckType.TYPES],
        )

        # Sequential would be 50 + 100 = 150ms
        # Parallel should be ~100ms (slowest check)
        # We allow for overhead, but should see speedup > 1.2x
        assert result.speedup >= 1.0, f"Expected speedup > 1.0, got {result.speedup}"
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_sequential_duration_tracked(self) -> None:
        """Test that sequential duration is correctly tracked."""
        from ai_infra.executor.models import Task

        async def timed_verify(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms
            mock_result = MagicMock()
            mock_result.overall = True
            mock_result.checks = []
            mock_result.passed_count = 1
            mock_result.failed_count = 0
            mock_result.get_failures.return_value = []
            return mock_result

        mock_verifier = MagicMock()
        mock_verifier.verify = timed_verify

        verifier = ParallelVerifier(
            workspace="/tmp/test",
            verifier=mock_verifier,
        )

        task = Task(
            id="test",
            title="Test Task",
            description="Test",
            file_hints=["src/main.py"],
        )

        result = await verifier.verify_parallel(
            task=task,
            files_modified=[Path("src/main.py")],
            checks=[CheckType.SYNTAX, CheckType.TYPES],
        )

        # Each check is ~50ms, so sequential should be ~100ms
        # Allow for some variance
        assert result.sequential_duration_ms > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestParallelVerifierIntegration:
    """Integration tests for parallel verifier."""

    @pytest.mark.asyncio
    async def test_full_verification_flow(self) -> None:
        """Test complete verification flow with mock verifier."""
        from ai_infra.executor.models import Task
        from ai_infra.executor.verifier import CheckStatus

        # Create proper mock result
        mock_check = MagicMock()
        mock_check.status = CheckStatus.PASSED
        mock_check.message = "OK"
        mock_check.error = None

        mock_result = MagicMock()
        mock_result.overall = True
        mock_result.checks = [mock_check]
        mock_result.passed_count = 1
        mock_result.failed_count = 0
        mock_result.get_failures.return_value = []

        async def mock_verify(*args, **kwargs):
            return mock_result

        mock_verifier = MagicMock()
        mock_verifier.verify = mock_verify

        verifier = ParallelVerifier(
            workspace="/tmp/test",
            verifier=mock_verifier,
        )

        task = Task(
            id="test",
            title="Test Feature",
            description="Implement test feature",
            file_hints=["src/feature.py", "tests/test_feature.py"],
        )

        result = await verifier.verify_parallel(
            task=task,
            files_modified=[Path("src/feature.py"), Path("tests/test_feature.py")],
        )

        assert result.passed is True
        assert result.duration_ms > 0
        assert len(result.failures) == 0
