"""Core benchmark harness for performance testing.

This module provides the base benchmark infrastructure:
- BenchmarkResult: Data class for benchmark results
- benchmark: Async benchmark function
- benchmark_sync: Synchronous benchmark function
- BenchmarkSuite: Collection of benchmarks
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "benchmark",
    "benchmark_sync",
    "compare_results",
    "run_benchmark_suite",
]


# =============================================================================
# Benchmark Result Data Class
# =============================================================================


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Result of a benchmark run with statistical analysis.

    Attributes:
        name: Name of the benchmark.
        iterations: Number of iterations run (excluding warmup).
        mean_ms: Mean execution time in milliseconds.
        median_ms: Median execution time in milliseconds.
        p95_ms: 95th percentile execution time.
        p99_ms: 99th percentile execution time.
        min_ms: Minimum execution time.
        max_ms: Maximum execution time.
        std_dev_ms: Standard deviation of execution times.
        warmup_iterations: Number of warmup iterations run.
        timestamp: ISO timestamp when benchmark was run.
        metadata: Additional metadata about the benchmark.
    """

    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float = 0.0
    warmup_iterations: int = 0
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format benchmark result as human-readable string."""
        return (
            f"{self.name}:\n"
            f"  Mean: {self.mean_ms:.1f}ms, Median: {self.median_ms:.1f}ms\n"
            f"  P95: {self.p95_ms:.1f}ms, P99: {self.p99_ms:.1f}ms\n"
            f"  Min: {self.min_ms:.1f}ms, Max: {self.max_ms:.1f}ms\n"
            f"  StdDev: {self.std_dev_ms:.1f}ms ({self.iterations} iterations)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def passes_target(self, target_ms: float, percentile: str = "p95") -> bool:
        """Check if benchmark passes a target threshold.

        Args:
            target_ms: Target time in milliseconds.
            percentile: Which percentile to check ("mean", "median", "p95", "p99").

        Returns:
            True if the specified percentile is under target.
        """
        value = getattr(self, f"{percentile}_ms", self.p95_ms)
        return value <= target_ms

    def summary(self) -> str:
        """Return a one-line summary."""
        return (
            f"{self.name}: mean={self.mean_ms:.1f}ms, "
            f"p95={self.p95_ms:.1f}ms, p99={self.p99_ms:.1f}ms"
        )


# =============================================================================
# Benchmark Functions
# =============================================================================


async def benchmark(
    name: str,
    fn: Callable[[], Coroutine[Any, Any, Any]],
    iterations: int = 10,
    warmup: int = 2,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run an async benchmark and collect statistics.

    Executes the provided async function multiple times, measuring execution
    time for each iteration and computing statistical metrics.

    Args:
        name: Name for the benchmark (used in output).
        fn: Async function to benchmark (no arguments, returns awaitable).
        iterations: Number of timed iterations to run.
        warmup: Number of warmup iterations (not timed).
        metadata: Optional metadata to include in result.

    Returns:
        BenchmarkResult with statistical analysis of execution times.

    Example:
        >>> async def fetch_data():
        ...     async with httpx.AsyncClient() as client:
        ...         await client.get("https://api.example.com")
        >>>
        >>> result = await benchmark("fetch_data", fetch_data, iterations=20)
        >>> print(f"P95 latency: {result.p95_ms:.1f}ms")
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    # Warmup runs (not timed)
    for _ in range(warmup):
        await fn()

    # Timed runs
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        await fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return _compute_result(
        name=name,
        times=times,
        warmup=warmup,
        metadata=metadata,
    )


def benchmark_sync(
    name: str,
    fn: Callable[[], Any],
    iterations: int = 10,
    warmup: int = 2,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run a synchronous benchmark and collect statistics.

    Executes the provided function multiple times, measuring execution
    time for each iteration and computing statistical metrics.

    Args:
        name: Name for the benchmark (used in output).
        fn: Synchronous function to benchmark (no arguments).
        iterations: Number of timed iterations to run.
        warmup: Number of warmup iterations (not timed).
        metadata: Optional metadata to include in result.

    Returns:
        BenchmarkResult with statistical analysis of execution times.

    Example:
        >>> def compute_heavy():
        ...     return sum(i ** 2 for i in range(10000))
        >>>
        >>> result = benchmark_sync("compute", compute_heavy, iterations=100)
        >>> print(f"Mean time: {result.mean_ms:.1f}ms")
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    # Warmup runs (not timed)
    for _ in range(warmup):
        fn()

    # Timed runs
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return _compute_result(
        name=name,
        times=times,
        warmup=warmup,
        metadata=metadata,
    )


def _compute_result(
    name: str,
    times: list[float],
    warmup: int,
    metadata: dict[str, Any] | None,
) -> BenchmarkResult:
    """Compute BenchmarkResult from raw timing data."""
    times_sorted = sorted(times)
    n = len(times_sorted)

    # Calculate percentile indices
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1) if n >= 100 else n - 1

    # Calculate standard deviation
    std_dev = statistics.stdev(times) if n >= 2 else 0.0

    return BenchmarkResult(
        name=name,
        iterations=n,
        mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        p95_ms=times_sorted[p95_idx],
        p99_ms=times_sorted[p99_idx],
        min_ms=min(times),
        max_ms=max(times),
        std_dev_ms=std_dev,
        warmup_iterations=warmup,
        timestamp=datetime.now(UTC).isoformat(),
        metadata=metadata or {},
    )


# =============================================================================
# Benchmark Suite
# =============================================================================


@dataclass
class BenchmarkSuite:
    """Collection of benchmarks to run together.

    Example:
        >>> suite = BenchmarkSuite(name="executor")
        >>> suite.add("init", async_init_fn)
        >>> suite.add("execute", async_execute_fn)
        >>> results = await suite.run(iterations=10)
        >>> for r in results:
        ...     print(r.summary())
    """

    name: str
    benchmarks: list[tuple[str, Callable[[], Coroutine[Any, Any, Any]]]] = field(
        default_factory=list
    )
    results: list[BenchmarkResult] = field(default_factory=list)

    def add(
        self,
        name: str,
        fn: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        """Add a benchmark to the suite.

        Args:
            name: Name for the benchmark.
            fn: Async function to benchmark.
        """
        self.benchmarks.append((name, fn))

    async def run(
        self,
        iterations: int = 10,
        warmup: int = 2,
    ) -> list[BenchmarkResult]:
        """Run all benchmarks in the suite.

        Args:
            iterations: Number of timed iterations per benchmark.
            warmup: Number of warmup iterations per benchmark.

        Returns:
            List of BenchmarkResult objects.
        """
        self.results = []
        for bench_name, fn in self.benchmarks:
            full_name = f"{self.name}/{bench_name}"
            result = await benchmark(
                name=full_name,
                fn=fn,
                iterations=iterations,
                warmup=warmup,
                metadata={"suite": self.name},
            )
            self.results.append(result)
        return self.results

    def summary(self) -> str:
        """Return a summary of all benchmark results."""
        if not self.results:
            return f"Suite '{self.name}': No results yet"

        lines = [f"Suite '{self.name}' ({len(self.results)} benchmarks):"]
        for result in self.results:
            lines.append(f"  {result.summary()}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert suite results to dictionary."""
        return {
            "name": self.name,
            "benchmarks": [r.to_dict() for r in self.results],
            "count": len(self.results),
        }


async def run_benchmark_suite(
    name: str,
    benchmarks: dict[str, Callable[[], Coroutine[Any, Any, Any]]],
    iterations: int = 10,
    warmup: int = 2,
) -> list[BenchmarkResult]:
    """Run a suite of benchmarks.

    Convenience function for running multiple benchmarks at once.

    Args:
        name: Name for the benchmark suite.
        benchmarks: Mapping of benchmark names to async functions.
        iterations: Number of iterations per benchmark.
        warmup: Number of warmup iterations.

    Returns:
        List of BenchmarkResult objects.

    Example:
        >>> results = await run_benchmark_suite(
        ...     "executor",
        ...     {
        ...         "init": async_init,
        ...         "execute": async_execute,
        ...     },
        ...     iterations=20,
        ... )
    """
    suite = BenchmarkSuite(name=name)
    for bench_name, fn in benchmarks.items():
        suite.add(bench_name, fn)
    return await suite.run(iterations=iterations, warmup=warmup)


# =============================================================================
# Benchmark Comparison
# =============================================================================


def compare_results(
    baseline: BenchmarkResult,
    current: BenchmarkResult,
    threshold_pct: float = 10.0,
) -> dict[str, Any]:
    """Compare two benchmark results.

    Args:
        baseline: Previous benchmark result.
        current: Current benchmark result.
        threshold_pct: Percentage threshold for regression detection.

    Returns:
        Dictionary with comparison metrics.
    """

    def pct_change(old: float, new: float) -> float:
        if old == 0:
            return 0.0
        return ((new - old) / old) * 100

    mean_change = pct_change(baseline.mean_ms, current.mean_ms)
    p95_change = pct_change(baseline.p95_ms, current.p95_ms)
    p99_change = pct_change(baseline.p99_ms, current.p99_ms)

    return {
        "baseline_name": baseline.name,
        "current_name": current.name,
        "mean_change_pct": round(mean_change, 2),
        "p95_change_pct": round(p95_change, 2),
        "p99_change_pct": round(p99_change, 2),
        "is_regression": max(mean_change, p95_change) > threshold_pct,
        "is_improvement": min(mean_change, p95_change) < -threshold_pct,
    }
