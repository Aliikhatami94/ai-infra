"""Phase 1 verification metrics for the Executor.

Phase 1.5: Phase 1 Verification - Metrics collection and validation.

This module provides:
- Phase1Metrics: Track and validate Phase 1 success criteria
- MetricCollector: Collect timing and cost metrics during execution
- BenchmarkRunner: Run benchmarks to validate targets

Success Criteria (from EXECUTOR_1.md):
| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Time to first token | 3-5s | <500ms | <500ms |
| Simple task latency | 3-5s | ~500ms | <1s |
| Context build time (blocking) | 1-2s | 0s | 0s |
| Verification time | 3s | 1.5s | <2s |
| Cost per task (average) | $0.05 | $0.015 | <$0.02 |

Example:
    >>> collector = MetricCollector()
    >>> async with collector.measure("time_to_first_token"):
    ...     token = await get_first_token()
    >>> metrics = collector.get_metrics()
    >>> validator = Phase1Validator(metrics)
    >>> report = validator.validate_all()
    >>> print(report.summary())
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ai_infra.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("executor.phase1_metrics")


# =============================================================================
# Metric Types and Targets
# =============================================================================


class MetricType(Enum):
    """Types of Phase 1 metrics."""

    TIME_TO_FIRST_TOKEN = "time_to_first_token"
    SIMPLE_TASK_LATENCY = "simple_task_latency"
    CONTEXT_BUILD_TIME = "context_build_time"
    VERIFICATION_TIME = "verification_time"
    COST_PER_TASK = "cost_per_task"


@dataclass(frozen=True)
class MetricTarget:
    """Target for a Phase 1 metric.

    Attributes:
        metric_type: Type of metric.
        target_value: Target value to achieve.
        unit: Unit of measurement (ms, s, $).
        before_value: Value before Phase 1 optimizations.
        description: Human-readable description.
    """

    metric_type: MetricType
    target_value: float
    unit: str
    before_value: float
    description: str

    def is_met(self, actual: float) -> bool:
        """Check if the target is met."""
        return actual <= self.target_value


# Phase 1 target definitions
PHASE1_TARGETS: dict[MetricType, MetricTarget] = {
    MetricType.TIME_TO_FIRST_TOKEN: MetricTarget(
        metric_type=MetricType.TIME_TO_FIRST_TOKEN,
        target_value=500.0,  # 500ms
        unit="ms",
        before_value=4000.0,  # 3-5s, use 4s midpoint
        description="Time from request to first token streamed",
    ),
    MetricType.SIMPLE_TASK_LATENCY: MetricTarget(
        metric_type=MetricType.SIMPLE_TASK_LATENCY,
        target_value=1000.0,  # ~500ms target, 1s acceptable
        unit="ms",
        before_value=4000.0,  # 3-5s, use 4s midpoint
        description="Total latency for simple tasks (comments, docstrings)",
    ),
    MetricType.CONTEXT_BUILD_TIME: MetricTarget(
        metric_type=MetricType.CONTEXT_BUILD_TIME,
        target_value=0.0,  # 0s (non-blocking)
        unit="ms",
        before_value=1500.0,  # 1-2s, use 1.5s midpoint
        description="Blocking time for context building (should be 0 with pre-warming)",
    ),
    MetricType.VERIFICATION_TIME: MetricTarget(
        metric_type=MetricType.VERIFICATION_TIME,
        target_value=2000.0,  # 1.5s target, 2s acceptable
        unit="ms",
        before_value=3000.0,  # 3s
        description="Time for parallel verification checks",
    ),
    MetricType.COST_PER_TASK: MetricTarget(
        metric_type=MetricType.COST_PER_TASK,
        target_value=0.02,  # $0.015 target, $0.02 acceptable
        unit="$",
        before_value=0.05,  # $0.05
        description="Average cost per task with model routing",
    ),
}


# =============================================================================
# Metric Sample and Collection
# =============================================================================


@dataclass
class MetricSample:
    """A single metric sample.

    Attributes:
        metric_type: Type of metric.
        value: Measured value.
        timestamp: When the sample was taken.
        metadata: Additional context about the sample.
    """

    metric_type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class MetricStats:
    """Statistics for a metric.

    Attributes:
        metric_type: Type of metric.
        count: Number of samples.
        min_value: Minimum value.
        max_value: Maximum value.
        avg_value: Average value.
        p50_value: 50th percentile (median).
        p95_value: 95th percentile.
        p99_value: 99th percentile.
    """

    metric_type: MetricType
    count: int = 0
    min_value: float = float("inf")
    max_value: float = 0.0
    avg_value: float = 0.0
    p50_value: float = 0.0
    p95_value: float = 0.0
    p99_value: float = 0.0
    samples: list[float] = field(default_factory=list)

    def add_sample(self, value: float) -> None:
        """Add a sample and update stats."""
        self.samples.append(value)
        self.count = len(self.samples)
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.avg_value = sum(self.samples) / self.count
        self._update_percentiles()

    def _update_percentiles(self) -> None:
        """Update percentile calculations."""
        if not self.samples:
            return

        sorted_samples = sorted(self.samples)
        n = len(sorted_samples)

        def percentile(p: float) -> float:
            idx = int(p * n)
            return sorted_samples[min(idx, n - 1)]

        self.p50_value = percentile(0.50)
        self.p95_value = percentile(0.95)
        self.p99_value = percentile(0.99)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "count": self.count,
            "min": self.min_value if self.count > 0 else None,
            "max": self.max_value if self.count > 0 else None,
            "avg": self.avg_value if self.count > 0 else None,
            "p50": self.p50_value if self.count > 0 else None,
            "p95": self.p95_value if self.count > 0 else None,
            "p99": self.p99_value if self.count > 0 else None,
        }


class MetricCollector:
    """Collect metrics during execution.

    Thread-safe collector for Phase 1 metrics with timing context managers.

    Example:
        >>> collector = MetricCollector()
        >>> with collector.measure_sync(MetricType.VERIFICATION_TIME):
        ...     run_verification()
        >>> async with collector.measure(MetricType.TIME_TO_FIRST_TOKEN):
        ...     await get_first_token()
        >>> stats = collector.get_stats()
    """

    def __init__(self) -> None:
        """Initialize collector."""
        self._samples: list[MetricSample] = []
        self._stats: dict[MetricType, MetricStats] = {
            mt: MetricStats(metric_type=mt) for mt in MetricType
        }
        self._lock = asyncio.Lock()

    def record(
        self,
        metric_type: MetricType,
        value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a metric sample.

        Args:
            metric_type: Type of metric.
            value: Measured value.
            metadata: Optional additional context.
        """
        sample = MetricSample(
            metric_type=metric_type,
            value=value,
            metadata=metadata or {},
        )
        self._samples.append(sample)
        self._stats[metric_type].add_sample(value)
        logger.debug(f"Recorded {metric_type.value}: {value}")

    @asynccontextmanager
    async def measure(
        self,
        metric_type: MetricType,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[None]:
        """Async context manager for timing metrics.

        Args:
            metric_type: Type of metric to measure.
            metadata: Optional additional context.

        Yields:
            None - the timing is recorded automatically.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(metric_type, elapsed_ms, metadata)

    @contextmanager
    def measure_sync(
        self,
        metric_type: MetricType,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Sync context manager for timing metrics.

        Args:
            metric_type: Type of metric to measure.
            metadata: Optional additional context.

        Yields:
            None - the timing is recorded automatically.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(metric_type, elapsed_ms, metadata)

    def record_cost(
        self,
        cost: float,
        model: str | None = None,
        tokens: int | None = None,
    ) -> None:
        """Record a cost metric.

        Args:
            cost: Cost in dollars.
            model: Optional model name.
            tokens: Optional token count.
        """
        metadata = {}
        if model:
            metadata["model"] = model
        if tokens:
            metadata["tokens"] = tokens
        self.record(MetricType.COST_PER_TASK, cost, metadata)

    def get_samples(
        self,
        metric_type: MetricType | None = None,
    ) -> list[MetricSample]:
        """Get recorded samples.

        Args:
            metric_type: Optional filter by type.

        Returns:
            List of metric samples.
        """
        if metric_type is None:
            return list(self._samples)
        return [s for s in self._samples if s.metric_type == metric_type]

    def get_stats(
        self,
        metric_type: MetricType | None = None,
    ) -> dict[MetricType, MetricStats] | MetricStats:
        """Get metric statistics.

        Args:
            metric_type: Optional specific metric type.

        Returns:
            Stats for the metric(s).
        """
        if metric_type is not None:
            return self._stats[metric_type]
        return dict(self._stats)

    def clear(self) -> None:
        """Clear all collected samples."""
        self._samples.clear()
        self._stats = {mt: MetricStats(metric_type=mt) for mt in MetricType}

    def summary(self) -> str:
        """Get human-readable summary of metrics."""
        lines = ["Phase 1 Metrics Summary:", "-" * 40]

        for metric_type, stats in self._stats.items():
            if stats.count == 0:
                continue

            target = PHASE1_TARGETS.get(metric_type)
            unit = target.unit if target else "ms"

            status = ""
            if target:
                met = target.is_met(stats.avg_value)
                status = " [TARGET MET]" if met else " [TARGET MISSED]"

            lines.append(
                f"{metric_type.value}:{status}\n"
                f"  Count: {stats.count}\n"
                f"  Avg: {stats.avg_value:.2f}{unit}\n"
                f"  P50: {stats.p50_value:.2f}{unit}\n"
                f"  P95: {stats.p95_value:.2f}{unit}"
            )

        return "\n".join(lines)


# =============================================================================
# Metric Validation
# =============================================================================


@dataclass
class ValidationResult:
    """Result of validating a single metric.

    Attributes:
        metric_type: Type of metric validated.
        target: The target value.
        actual: The actual measured value.
        passed: Whether the target was met.
        improvement: Improvement ratio from before value.
    """

    metric_type: MetricType
    target: float
    actual: float
    passed: bool
    improvement: float
    unit: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "target": self.target,
            "actual": self.actual,
            "passed": self.passed,
            "improvement": self.improvement,
            "unit": self.unit,
        }


@dataclass
class Phase1ValidationReport:
    """Full Phase 1 validation report.

    Attributes:
        results: Individual metric validation results.
        overall_passed: Whether all targets are met.
        timestamp: When validation was performed.
    """

    results: list[ValidationResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def overall_passed(self) -> bool:
        """Check if all targets are met."""
        return all(r.passed for r in self.results)

    @property
    def passed_count(self) -> int:
        """Count of passed validations."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        """Count of failed validations."""
        return sum(1 for r in self.results if not r.passed)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "overall_passed": self.overall_passed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Phase 1 Validation Report",
            "=" * 50,
            "",
            f"Overall: {'PASSED' if self.overall_passed else 'FAILED'}",
            f"Results: {self.passed_count}/{len(self.results)} targets met",
            "",
            "Details:",
            "-" * 50,
        ]

        for result in self.results:
            status = "[PASS]" if result.passed else "[FAIL]"
            lines.append(
                f"{status} {result.metric_type.value}:\n"
                f"       Target: {result.target}{result.unit}\n"
                f"       Actual: {result.actual:.2f}{result.unit}\n"
                f"       Improvement: {result.improvement:.1f}x"
            )

        return "\n".join(lines)

    def markdown_table(self) -> str:
        """Generate markdown table for documentation."""
        lines = [
            "| Metric | Before | After | Target | Status |",
            "|--------|--------|-------|--------|--------|",
        ]

        for result in self.results:
            target_info = PHASE1_TARGETS.get(result.metric_type)
            before = target_info.before_value if target_info else "N/A"
            status = "[x]" if result.passed else "[ ]"

            # Format values with units
            if result.unit == "$":
                before_str = f"${before}"
                actual_str = f"${result.actual:.3f}"
                target_str = f"${result.target}"
            else:
                before_str = f"{before}ms"
                actual_str = f"{result.actual:.0f}ms"
                target_str = f"{result.target}ms"

            lines.append(
                f"| {result.metric_type.value} | {before_str} | "
                f"{actual_str} | {target_str} | {status} |"
            )

        return "\n".join(lines)


class Phase1Validator:
    """Validate Phase 1 metrics against targets.

    Example:
        >>> collector = MetricCollector()
        >>> # ... run tasks and collect metrics ...
        >>> validator = Phase1Validator(collector)
        >>> report = validator.validate_all()
        >>> if report.overall_passed:
        ...     print("Phase 1 complete!")
    """

    def __init__(
        self,
        collector: MetricCollector | None = None,
        targets: dict[MetricType, MetricTarget] | None = None,
    ) -> None:
        """Initialize validator.

        Args:
            collector: Optional metric collector with samples.
            targets: Optional custom targets (defaults to PHASE1_TARGETS).
        """
        self._collector = collector
        self._targets = targets or PHASE1_TARGETS
        self._overrides: dict[MetricType, float] = {}

    def set_override(self, metric_type: MetricType, value: float) -> None:
        """Set an override value for validation.

        Useful when the metric is calculated differently or from external sources.

        Args:
            metric_type: Type of metric.
            value: Value to use for validation.
        """
        self._overrides[metric_type] = value

    def validate(self, metric_type: MetricType) -> ValidationResult:
        """Validate a single metric.

        Args:
            metric_type: Type of metric to validate.

        Returns:
            ValidationResult for the metric.
        """
        target = self._targets.get(metric_type)
        if target is None:
            raise ValueError(f"No target defined for {metric_type}")

        # Get actual value from override or collector
        if metric_type in self._overrides:
            actual = self._overrides[metric_type]
        elif self._collector:
            stats = self._collector.get_stats(metric_type)
            if isinstance(stats, MetricStats) and stats.count > 0:
                actual = stats.avg_value
            else:
                # No samples, use a large value to fail validation
                actual = float("inf")
        else:
            actual = float("inf")

        passed = target.is_met(actual)
        improvement = target.before_value / actual if actual > 0 else float("inf")

        return ValidationResult(
            metric_type=metric_type,
            target=target.target_value,
            actual=actual,
            passed=passed,
            improvement=improvement,
            unit=target.unit,
        )

    def validate_all(self) -> Phase1ValidationReport:
        """Validate all Phase 1 metrics.

        Returns:
            Complete validation report.
        """
        results = []
        for metric_type in self._targets:
            try:
                result = self.validate(metric_type)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to validate {metric_type}: {e}")

        return Phase1ValidationReport(results=results)


# =============================================================================
# Benchmark Runner
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result from a benchmark run.

    Attributes:
        name: Benchmark name.
        iterations: Number of iterations run.
        total_ms: Total time in milliseconds.
        avg_ms: Average time per iteration.
        min_ms: Minimum iteration time.
        max_ms: Maximum iteration time.
    """

    name: str
    iterations: int
    total_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_ms": self.total_ms,
            "avg_ms": self.avg_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


class BenchmarkRunner:
    """Run Phase 1 benchmarks.

    Example:
        >>> runner = BenchmarkRunner()
        >>> result = await runner.benchmark_time_to_first_token(
        ...     llm=llm,
        ...     prompt="Hello",
        ...     iterations=10,
        ... )
        >>> print(f"Avg: {result.avg_ms}ms")
    """

    def __init__(self, collector: MetricCollector | None = None) -> None:
        """Initialize benchmark runner.

        Args:
            collector: Optional collector to record metrics.
        """
        self._collector = collector or MetricCollector()

    @property
    def collector(self) -> MetricCollector:
        """Get the metric collector."""
        return self._collector

    async def benchmark_streaming(
        self,
        stream_func: Any,
        iterations: int = 10,
        warmup: int = 2,
    ) -> BenchmarkResult:
        """Benchmark time to first token for streaming.

        Args:
            stream_func: Async function that yields tokens.
            iterations: Number of benchmark iterations.
            warmup: Number of warmup iterations (not counted).

        Returns:
            BenchmarkResult with timing statistics.
        """
        times: list[float] = []

        # Warmup
        for _ in range(warmup):
            async for _ in stream_func():
                break

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            async for _ in stream_func():
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                self._collector.record(MetricType.TIME_TO_FIRST_TOKEN, elapsed)
                break

        return BenchmarkResult(
            name="time_to_first_token",
            iterations=len(times),
            total_ms=sum(times),
            avg_ms=sum(times) / len(times) if times else 0,
            min_ms=min(times) if times else 0,
            max_ms=max(times) if times else 0,
        )

    async def benchmark_task_latency(
        self,
        task_func: Any,
        iterations: int = 10,
        warmup: int = 2,
    ) -> BenchmarkResult:
        """Benchmark simple task latency.

        Args:
            task_func: Async function that runs a simple task.
            iterations: Number of benchmark iterations.
            warmup: Number of warmup iterations.

        Returns:
            BenchmarkResult with timing statistics.
        """
        times: list[float] = []

        # Warmup
        for _ in range(warmup):
            await task_func()

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            await task_func()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            self._collector.record(MetricType.SIMPLE_TASK_LATENCY, elapsed)

        return BenchmarkResult(
            name="simple_task_latency",
            iterations=len(times),
            total_ms=sum(times),
            avg_ms=sum(times) / len(times) if times else 0,
            min_ms=min(times) if times else 0,
            max_ms=max(times) if times else 0,
        )

    async def benchmark_verification(
        self,
        verify_func: Any,
        iterations: int = 5,
        warmup: int = 1,
    ) -> BenchmarkResult:
        """Benchmark verification time.

        Args:
            verify_func: Async function that runs verification.
            iterations: Number of benchmark iterations.
            warmup: Number of warmup iterations.

        Returns:
            BenchmarkResult with timing statistics.
        """
        times: list[float] = []

        # Warmup
        for _ in range(warmup):
            await verify_func()

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            await verify_func()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            self._collector.record(MetricType.VERIFICATION_TIME, elapsed)

        return BenchmarkResult(
            name="verification_time",
            iterations=len(times),
            total_ms=sum(times),
            avg_ms=sum(times) / len(times) if times else 0,
            min_ms=min(times) if times else 0,
            max_ms=max(times) if times else 0,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_collector() -> MetricCollector:
    """Create a new metric collector."""
    return MetricCollector()


def create_validator(
    collector: MetricCollector | None = None,
) -> Phase1Validator:
    """Create a Phase 1 validator.

    Args:
        collector: Optional collector with metrics.

    Returns:
        Configured Phase1Validator.
    """
    return Phase1Validator(collector=collector)


def create_benchmark_runner(
    collector: MetricCollector | None = None,
) -> BenchmarkRunner:
    """Create a benchmark runner.

    Args:
        collector: Optional collector to use.

    Returns:
        Configured BenchmarkRunner.
    """
    return BenchmarkRunner(collector=collector)


def validate_phase1_targets(
    time_to_first_token_ms: float | None = None,
    simple_task_latency_ms: float | None = None,
    context_build_time_ms: float | None = None,
    verification_time_ms: float | None = None,
    cost_per_task: float | None = None,
) -> Phase1ValidationReport:
    """Quick validation of Phase 1 targets with explicit values.

    Args:
        time_to_first_token_ms: Time to first token in ms.
        simple_task_latency_ms: Simple task latency in ms.
        context_build_time_ms: Context build blocking time in ms.
        verification_time_ms: Verification time in ms.
        cost_per_task: Cost per task in dollars.

    Returns:
        Validation report.
    """
    validator = Phase1Validator()

    if time_to_first_token_ms is not None:
        validator.set_override(MetricType.TIME_TO_FIRST_TOKEN, time_to_first_token_ms)
    if simple_task_latency_ms is not None:
        validator.set_override(MetricType.SIMPLE_TASK_LATENCY, simple_task_latency_ms)
    if context_build_time_ms is not None:
        validator.set_override(MetricType.CONTEXT_BUILD_TIME, context_build_time_ms)
    if verification_time_ms is not None:
        validator.set_override(MetricType.VERIFICATION_TIME, verification_time_ms)
    if cost_per_task is not None:
        validator.set_override(MetricType.COST_PER_TASK, cost_per_task)

    return validator.validate_all()
