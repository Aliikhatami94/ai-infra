"""Parallel verification for the Executor.

Phase 1.4: Parallel verification to reduce task verification latency.

This module provides:
- ParallelVerifier: Runs multiple verification checks concurrently
- CheckConfig: Configuration for individual checks (timeouts, skip conditions)
- aggregate_results: Combines results from parallel checks with partial failure handling

The problem: Verification checks run sequentially, adding latency (3s total).
The solution: Run checks in parallel, limited only by the slowest check (~2s).

Example:
    >>> verifier = ParallelVerifier(workspace=Path("./project"))
    >>> result = await verifier.verify_parallel(
    ...     task=task,
    ...     files_modified=files,
    ...     checks=["syntax", "types", "lint", "tests"],
    ... )
    >>> print(f"Verified in {result.duration_ms}ms")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.models import Task
    from ai_infra.executor.verifier import TaskVerifier

logger = get_logger("executor.parallel_verification")


# =============================================================================
# Check Types and Configuration (Phase 1.4.4)
# =============================================================================


class CheckType(Enum):
    """Types of verification checks available."""

    SYNTAX = "syntax"
    TYPES = "types"
    LINT = "lint"
    TESTS = "tests"
    IMPORTS = "imports"
    FILES = "files"


@dataclass
class CheckConfig:
    """Configuration for a verification check.

    Phase 1.4.3: Check-specific timeouts.
    Phase 1.4.4: Skip conditions based on file types.

    Attributes:
        check_type: The type of check.
        timeout: Maximum time for this check in seconds.
        extensions: File extensions that trigger this check (empty = always run).
        skip_on_timeout: If True, timeout is a warning not a failure.
        enabled: Whether this check is enabled.
    """

    check_type: CheckType
    timeout: float = 30.0
    extensions: frozenset[str] = field(default_factory=frozenset)
    skip_on_timeout: bool = False
    enabled: bool = True

    @classmethod
    def syntax(cls, timeout: float = 10.0) -> CheckConfig:
        """Create syntax check config (always runs on Python files)."""
        return cls(
            check_type=CheckType.SYNTAX,
            timeout=timeout,
            extensions=frozenset({".py"}),
            skip_on_timeout=False,
        )

    @classmethod
    def types(cls, timeout: float = 30.0) -> CheckConfig:
        """Create type check config (runs on Python files, soft timeout)."""
        return cls(
            check_type=CheckType.TYPES,
            timeout=timeout,
            extensions=frozenset({".py"}),
            skip_on_timeout=True,  # Don't block on slow type check
        )

    @classmethod
    def lint(cls, timeout: float = 20.0) -> CheckConfig:
        """Create lint check config (runs on code files)."""
        return cls(
            check_type=CheckType.LINT,
            timeout=timeout,
            extensions=frozenset({".py", ".js", ".ts", ".jsx", ".tsx", ".rs"}),
            skip_on_timeout=True,
        )

    @classmethod
    def tests(cls, timeout: float = 120.0) -> CheckConfig:
        """Create test check config (always runs, hardest timeout)."""
        return cls(
            check_type=CheckType.TESTS,
            timeout=timeout,
            extensions=frozenset(),  # Always run if tests exist
            skip_on_timeout=False,
        )

    @classmethod
    def imports(cls, timeout: float = 15.0) -> CheckConfig:
        """Create import check config (Python files only)."""
        return cls(
            check_type=CheckType.IMPORTS,
            timeout=timeout,
            extensions=frozenset({".py"}),
            skip_on_timeout=True,
        )


# Default check configurations
DEFAULT_CHECK_CONFIGS: dict[CheckType, CheckConfig] = {
    CheckType.SYNTAX: CheckConfig.syntax(),
    CheckType.TYPES: CheckConfig.types(),
    CheckType.LINT: CheckConfig.lint(),
    CheckType.TESTS: CheckConfig.tests(),
    CheckType.IMPORTS: CheckConfig.imports(),
    CheckType.FILES: CheckConfig(check_type=CheckType.FILES, timeout=5.0),
}


# =============================================================================
# Parallel Check Result (Phase 1.4.2)
# =============================================================================


@dataclass
class ParallelCheckResult:
    """Result from a single parallel check.

    Attributes:
        check_type: Type of check that was run.
        passed: Whether the check passed.
        skipped: Whether the check was skipped.
        timed_out: Whether the check timed out.
        duration_ms: How long the check took.
        error: Error message if failed.
        warnings: Any warnings generated.
        details: Additional check-specific details.
    """

    check_type: CheckType
    passed: bool = True
    skipped: bool = False
    timed_out: bool = False
    duration_ms: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type.value,
            "passed": self.passed,
            "skipped": self.skipped,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "warnings": self.warnings,
            "details": self.details,
        }


@dataclass
class ParallelVerificationResult:
    """Aggregated result from parallel verification.

    Phase 1.4.2: Handles partial failures gracefully.

    Attributes:
        passed: Whether all critical checks passed.
        checks: Individual check results.
        failures: List of failure messages.
        warnings: List of warning messages.
        duration_ms: Total wall-clock time for parallel execution.
        sequential_duration_ms: What sequential execution would have taken.
        speedup: How much faster parallel was (ratio).
    """

    passed: bool = True
    checks: list[ParallelCheckResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    sequential_duration_ms: float = 0.0

    @property
    def speedup(self) -> float:
        """Calculate speedup from parallelization."""
        if self.duration_ms <= 0:
            return 1.0
        return self.sequential_duration_ms / self.duration_ms

    @property
    def checks_run(self) -> int:
        """Number of checks that were run (not skipped)."""
        return sum(1 for c in self.checks if not c.skipped)

    @property
    def checks_passed(self) -> int:
        """Number of checks that passed."""
        return sum(1 for c in self.checks if c.passed and not c.skipped)

    @property
    def checks_failed(self) -> int:
        """Number of checks that failed."""
        return sum(1 for c in self.checks if not c.passed and not c.skipped)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "failures": self.failures,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
            "sequential_duration_ms": self.sequential_duration_ms,
            "speedup": self.speedup,
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Verification {status}: {self.checks_passed}/{self.checks_run} passed "
            f"({self.duration_ms:.0f}ms, {self.speedup:.1f}x speedup)"
        )


# =============================================================================
# Result Aggregation (Phase 1.4.2)
# =============================================================================


def aggregate_results(
    results: list[ParallelCheckResult | Exception],
    check_configs: dict[CheckType, CheckConfig] | None = None,
) -> ParallelVerificationResult:
    """Aggregate results from parallel checks.

    Phase 1.4.2: Handles partial failures gracefully.

    Args:
        results: List of check results or exceptions.
        check_configs: Check configurations for timeout behavior.

    Returns:
        Aggregated ParallelVerificationResult.
    """
    if check_configs is None:
        check_configs = DEFAULT_CHECK_CONFIGS

    failures: list[str] = []
    warnings: list[str] = []
    checks: list[ParallelCheckResult] = []
    sequential_duration = 0.0

    for result in results:
        if isinstance(result, Exception):
            # Handle exception as failure
            error_msg = f"Check failed with exception: {result}"
            failures.append(error_msg)
            checks.append(
                ParallelCheckResult(
                    check_type=CheckType.SYNTAX,  # Unknown
                    passed=False,
                    error=error_msg,
                )
            )
        elif isinstance(result, ParallelCheckResult):
            checks.append(result)
            sequential_duration += result.duration_ms

            if result.skipped:
                continue

            if result.timed_out:
                # Check if timeout is a warning or failure
                config = check_configs.get(result.check_type)
                if config and config.skip_on_timeout:
                    warnings.append(
                        f"{result.check_type.value} timed out after {result.duration_ms:.0f}ms"
                    )
                else:
                    failures.append(
                        f"{result.check_type.value} timed out after {result.duration_ms:.0f}ms"
                    )
            elif not result.passed:
                failures.append(result.error or f"{result.check_type.value} failed")

            warnings.extend(result.warnings)

    # Calculate overall pass (no failures)
    passed = len(failures) == 0

    return ParallelVerificationResult(
        passed=passed,
        checks=checks,
        failures=failures,
        warnings=warnings,
        sequential_duration_ms=sequential_duration,
    )


# =============================================================================
# Check Skipping Logic (Phase 1.4.4)
# =============================================================================


def should_run_check(
    check_type: CheckType,
    files_modified: list[Path] | list[str],
    config: CheckConfig | None = None,
) -> bool:
    """Determine if a check should run based on modified files.

    Phase 1.4.4: Skip unchanged file checks.

    Args:
        check_type: Type of check to consider.
        files_modified: List of modified file paths.
        config: Optional check configuration.

    Returns:
        True if the check should run.
    """
    if config is None:
        config = DEFAULT_CHECK_CONFIGS.get(check_type)

    if config is None or not config.enabled:
        return False

    # No extensions filter = always run
    if not config.extensions:
        return True

    # Check if any modified file matches the extension filter
    for file_path in files_modified:
        path = Path(file_path) if isinstance(file_path, str) else file_path
        if path.suffix in config.extensions:
            return True

    return False


def get_checks_to_run(
    files_modified: list[Path] | list[str],
    requested_checks: list[CheckType] | None = None,
    check_configs: dict[CheckType, CheckConfig] | None = None,
) -> list[CheckType]:
    """Get list of checks to run based on modified files.

    Phase 1.4.4: Smart check filtering.

    Args:
        files_modified: List of modified file paths.
        requested_checks: Optional list of specific checks to run.
        check_configs: Check configurations.

    Returns:
        List of check types to run.
    """
    if check_configs is None:
        check_configs = DEFAULT_CHECK_CONFIGS

    if requested_checks is None:
        requested_checks = list(CheckType)

    checks_to_run = []
    for check_type in requested_checks:
        config = check_configs.get(check_type)
        if should_run_check(check_type, files_modified, config):
            checks_to_run.append(check_type)

    return checks_to_run


# =============================================================================
# Parallel Verifier (Phase 1.4.1)
# =============================================================================


class ParallelVerifier:
    """Run verification checks in parallel.

    Phase 1.4.1: Refactors verification to parallel execution.

    This class wraps TaskVerifier methods and runs them concurrently,
    reducing total verification time from the sum of all checks to
    approximately the duration of the slowest check.

    Attributes:
        workspace: Project workspace directory.
        verifier: The underlying TaskVerifier instance.
        check_configs: Configuration for each check type.

    Example:
        >>> verifier = ParallelVerifier(workspace=Path("./project"))
        >>> result = await verifier.verify_parallel(
        ...     task=task,
        ...     files_modified=["src/main.py"],
        ... )
        >>> if result.passed:
        ...     print("All checks passed!")
    """

    def __init__(
        self,
        workspace: Path | str,
        *,
        verifier: TaskVerifier | None = None,
        check_configs: dict[CheckType, CheckConfig] | None = None,
    ):
        """Initialize parallel verifier.

        Args:
            workspace: Project workspace directory.
            verifier: Optional existing TaskVerifier to wrap.
            check_configs: Optional custom check configurations.
        """
        self.workspace = Path(workspace).resolve()
        self._verifier = verifier
        self.check_configs = check_configs or DEFAULT_CHECK_CONFIGS.copy()

    @property
    def verifier(self) -> TaskVerifier:
        """Get or create the underlying TaskVerifier."""
        if self._verifier is None:
            from ai_infra.executor.verifier import TaskVerifier

            self._verifier = TaskVerifier(workspace=self.workspace)
        return self._verifier

    async def verify_parallel(
        self,
        task: Task | None = None,
        files_modified: list[Path] | list[str] | None = None,
        checks: list[CheckType] | None = None,
    ) -> ParallelVerificationResult:
        """Run verification checks in parallel.

        Phase 1.4.1: Parallel execution of verification checks.

        Args:
            task: Optional task being verified (for file hints).
            files_modified: List of modified files to check.
            checks: Optional specific checks to run.

        Returns:
            ParallelVerificationResult with aggregated results.
        """
        files_modified = files_modified or []
        start_time = time.perf_counter()

        # Determine which checks to run (Phase 1.4.4)
        checks_to_run = get_checks_to_run(
            files_modified=files_modified,
            requested_checks=checks,
            check_configs=self.check_configs,
        )

        if not checks_to_run:
            logger.info("No checks to run (no relevant files modified)")
            return ParallelVerificationResult(
                passed=True,
                checks=[],
                duration_ms=0.0,
            )

        logger.info(
            f"Running {len(checks_to_run)} checks in parallel: {[c.value for c in checks_to_run]}"
        )

        # Create coroutines for each check
        check_coroutines = []
        for check_type in checks_to_run:
            config = self.check_configs.get(check_type, DEFAULT_CHECK_CONFIGS.get(check_type))
            coro = self._run_check_with_timeout(
                check_type=check_type,
                task=task,
                files_modified=files_modified,
                config=config,
            )
            check_coroutines.append(coro)

        # Run all checks in parallel (Phase 1.4.1)
        results = await asyncio.gather(*check_coroutines, return_exceptions=True)

        # Aggregate results (Phase 1.4.2)
        aggregated = aggregate_results(
            results=list(results),
            check_configs=self.check_configs,
        )

        # Set actual parallel duration
        aggregated.duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(aggregated.summary())
        return aggregated

    async def _run_check_with_timeout(
        self,
        check_type: CheckType,
        task: Task | None,
        files_modified: list[Path] | list[str],
        config: CheckConfig | None,
    ) -> ParallelCheckResult:
        """Run a single check with timeout.

        Phase 1.4.3: Check-specific timeouts.

        Args:
            check_type: Type of check to run.
            task: Optional task being verified.
            files_modified: List of modified files.
            config: Check configuration.

        Returns:
            ParallelCheckResult from the check.
        """
        if config is None:
            config = CheckConfig(check_type=check_type)

        start_time = time.perf_counter()
        timeout = config.timeout

        try:
            result = await asyncio.wait_for(
                self._run_check(check_type, task, files_modified),
                timeout=timeout,
            )
            result.duration_ms = (time.perf_counter() - start_time) * 1000
            return result

        except TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"{check_type.value} timed out after {timeout}s")

            return ParallelCheckResult(
                check_type=check_type,
                passed=config.skip_on_timeout,  # Pass if soft timeout
                timed_out=True,
                duration_ms=duration_ms,
                warnings=[f"Timed out after {timeout}s"] if config.skip_on_timeout else [],
                error=None if config.skip_on_timeout else f"Timed out after {timeout}s",
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"{check_type.value} failed with exception: {e}")

            return ParallelCheckResult(
                check_type=check_type,
                passed=False,
                duration_ms=duration_ms,
                error=str(e),
            )

    async def _run_check(
        self,
        check_type: CheckType,
        task: Task | None,
        files_modified: list[Path] | list[str],
    ) -> ParallelCheckResult:
        """Run a specific check type.

        Args:
            check_type: Type of check to run.
            task: Optional task being verified.
            files_modified: List of modified files.

        Returns:
            ParallelCheckResult from the check.
        """
        from ai_infra.executor.verifier import CheckLevel, CheckStatus

        # Map CheckType to CheckLevel
        level_map = {
            CheckType.SYNTAX: CheckLevel.SYNTAX,
            CheckType.TYPES: CheckLevel.TYPES,
            CheckType.LINT: CheckLevel.SYNTAX,  # Use syntax level for lint (placeholder)
            CheckType.TESTS: CheckLevel.TESTS,
            CheckType.IMPORTS: CheckLevel.IMPORTS,
            CheckType.FILES: CheckLevel.FILES,
        }

        level = level_map.get(check_type)
        if level is None:
            return ParallelCheckResult(
                check_type=check_type,
                skipped=True,
                warnings=[f"Unknown check type: {check_type}"],
            )

        # Run the check via TaskVerifier
        try:
            # Create a minimal task if none provided
            if task is None:
                from ai_infra.executor.models import Task

                task = Task(
                    id="verification",
                    title="Verification",
                    description="Parallel verification check",
                    file_hints=[str(f) for f in files_modified],
                )

            result = await self.verifier.verify(
                task=task,
                levels=[level],
                stop_on_failure=False,
            )

            # Convert VerificationResult to ParallelCheckResult
            passed = result.overall
            errors = [c.error for c in result.get_failures() if c.error]
            warnings = [c.message for c in result.checks if c.status == CheckStatus.SKIPPED]

            return ParallelCheckResult(
                check_type=check_type,
                passed=passed,
                error=errors[0] if errors else None,
                warnings=warnings,
                details={
                    "checks_run": len(result.checks),
                    "passed": result.passed_count,
                    "failed": result.failed_count,
                },
            )

        except Exception as e:
            logger.exception(f"Error running {check_type.value} check: {e}")
            return ParallelCheckResult(
                check_type=check_type,
                passed=False,
                error=str(e),
            )


# =============================================================================
# Factory Functions
# =============================================================================


def create_parallel_verifier(
    workspace: Path | str,
    *,
    syntax_timeout: float = 10.0,
    types_timeout: float = 30.0,
    lint_timeout: float = 20.0,
    tests_timeout: float = 120.0,
) -> ParallelVerifier:
    """Create a ParallelVerifier with custom timeouts.

    Args:
        workspace: Project workspace directory.
        syntax_timeout: Timeout for syntax check.
        types_timeout: Timeout for type check.
        lint_timeout: Timeout for lint check.
        tests_timeout: Timeout for tests.

    Returns:
        Configured ParallelVerifier.
    """
    configs = {
        CheckType.SYNTAX: CheckConfig.syntax(timeout=syntax_timeout),
        CheckType.TYPES: CheckConfig.types(timeout=types_timeout),
        CheckType.LINT: CheckConfig.lint(timeout=lint_timeout),
        CheckType.TESTS: CheckConfig.tests(timeout=tests_timeout),
        CheckType.IMPORTS: CheckConfig.imports(),
        CheckType.FILES: CheckConfig(check_type=CheckType.FILES, timeout=5.0),
    }

    return ParallelVerifier(
        workspace=workspace,
        check_configs=configs,
    )
