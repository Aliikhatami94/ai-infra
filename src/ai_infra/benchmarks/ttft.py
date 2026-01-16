"""Time-to-First-Token (TTFT) benchmark module.

Phase 12.2 of EXECUTOR_4.md - Time-to-First-Token Benchmark.

This module provides comprehensive TTFT benchmarking for the executor:
- TTFTBenchmark class for measuring time to first token
- TTFTProfile for breaking down TTFT into components
- CLI integration for running TTFT benchmarks

Target: <500ms from prompt submission to first token.

Example:
    >>> from ai_infra.benchmarks.ttft import TTFTBenchmark, TTFTProfile
    >>>
    >>> # Measure TTFT
    >>> benchmark = TTFTBenchmark()
    >>> result = await benchmark.measure()
    >>> print(f"TTFT: {result.ttft_ms:.1f}ms")
    >>>
    >>> # Profile TTFT breakdown
    >>> profile = TTFTProfile()
    >>> breakdown = await profile.profile()
    >>> print(f"LLM latency: {breakdown['llm_request_ms']:.1f}ms")
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ai_infra.benchmarks import BenchmarkResult, benchmark

if TYPE_CHECKING:
    pass


__all__ = [
    "TTFTBenchmark",
    "TTFTMeasurement",
    "TTFTProfile",
    "TTFTProfileResult",
    "TTFTTarget",
    "measure_ttft",
]


# =============================================================================
# TTFT Targets
# =============================================================================


@dataclass(frozen=True, slots=True)
class TTFTTarget:
    """Target TTFT values for different scenarios.

    Attributes:
        name: Target name/description.
        p50_ms: Target for 50th percentile (median).
        p95_ms: Target for 95th percentile.
        p99_ms: Target for 99th percentile.
    """

    name: str
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def passes(self, result: BenchmarkResult) -> bool:
        """Check if a benchmark result meets the target."""
        return (
            result.median_ms <= self.p50_ms
            and result.p95_ms <= self.p95_ms
            and result.p99_ms <= self.p99_ms
        )

    def summary(self, result: BenchmarkResult) -> str:
        """Get summary of target vs actual."""
        p50_status = "✓" if result.median_ms <= self.p50_ms else "✗"
        p95_status = "✓" if result.p95_ms <= self.p95_ms else "✗"
        p99_status = "✓" if result.p99_ms <= self.p99_ms else "✗"

        return (
            f"{self.name}:\n"
            f"  P50: {result.median_ms:.1f}ms / {self.p50_ms:.0f}ms {p50_status}\n"
            f"  P95: {result.p95_ms:.1f}ms / {self.p95_ms:.0f}ms {p95_status}\n"
            f"  P99: {result.p99_ms:.1f}ms / {self.p99_ms:.0f}ms {p99_status}"
        )


# Standard TTFT targets per EXECUTOR_4.md
TTFT_TARGET_DEFAULT = TTFTTarget(
    name="Default (<500ms P95)",
    p50_ms=300.0,
    p95_ms=500.0,
    p99_ms=1000.0,
)

TTFT_TARGET_FAST = TTFTTarget(
    name="Fast (<200ms P95)",
    p50_ms=100.0,
    p95_ms=200.0,
    p99_ms=400.0,
)

TTFT_TARGET_STREAMING = TTFTTarget(
    name="Streaming (<100ms P95)",
    p50_ms=50.0,
    p95_ms=100.0,
    p99_ms=200.0,
)


# =============================================================================
# TTFT Measurement
# =============================================================================


@dataclass(frozen=True, slots=True)
class TTFTMeasurement:
    """Single TTFT measurement.

    Attributes:
        ttft_ms: Time to first token in milliseconds.
        total_ms: Total time to completion (optional).
        first_token: Content of the first token received.
        timestamp: When measurement was taken.
        metadata: Additional measurement context.
    """

    ttft_ms: float
    total_ms: float = 0.0
    first_token: str = ""
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ttft_ms": round(self.ttft_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "first_token": self.first_token[:50] if self.first_token else "",
            "timestamp": self.timestamp,
            **self.metadata,
        }


async def measure_ttft(
    stream_fn: Callable[[], AsyncIterator[str | tuple[str, Any]]],
    timeout: float = 30.0,
) -> TTFTMeasurement:
    """Measure time to first token from an async stream.

    Args:
        stream_fn: Async generator function that yields tokens.
        timeout: Maximum time to wait for first token.

    Returns:
        TTFTMeasurement with timing data.

    Example:
        >>> async def my_stream():
        ...     yield "Hello"
        ...     yield "World"
        >>>
        >>> measurement = await measure_ttft(my_stream)
        >>> print(f"TTFT: {measurement.ttft_ms:.1f}ms")
    """
    start = time.perf_counter()
    first_token = ""
    ttft_ms = 0.0

    try:
        async with asyncio.timeout(timeout):
            async for item in stream_fn():
                ttft_ms = (time.perf_counter() - start) * 1000
                if isinstance(item, tuple):
                    first_token = str(item[0])
                else:
                    first_token = str(item)
                break
    except TimeoutError:
        ttft_ms = timeout * 1000

    total_ms = (time.perf_counter() - start) * 1000

    return TTFTMeasurement(
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        first_token=first_token,
        timestamp=datetime.now(UTC).isoformat(),
    )


# =============================================================================
# TTFT Benchmark Class
# =============================================================================


class TTFTBenchmark:
    """Time-to-First-Token benchmark runner.

    Provides comprehensive TTFT benchmarking including:
    - Multiple iterations with warmup
    - Statistical analysis (mean, median, percentiles)
    - Target comparison
    - Component profiling

    Example:
        >>> benchmark = TTFTBenchmark(target=TTFT_TARGET_DEFAULT)
        >>>
        >>> # With simulated stream
        >>> async def simulated_stream():
        ...     await asyncio.sleep(0.1)  # 100ms TTFT
        ...     yield "First token"
        >>>
        >>> result = await benchmark.run(simulated_stream, iterations=10)
        >>> print(result)
        >>> print(f"Passes target: {benchmark.target.passes(result)}")
    """

    def __init__(
        self,
        target: TTFTTarget | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize TTFT benchmark.

        Args:
            target: TTFT target to validate against.
            timeout: Maximum time to wait for first token.
        """
        self.target = target or TTFT_TARGET_DEFAULT
        self.timeout = timeout
        self._measurements: list[TTFTMeasurement] = []

    async def run(
        self,
        stream_fn: Callable[[], AsyncIterator[str | tuple[str, Any]]],
        iterations: int = 10,
        warmup: int = 2,
    ) -> BenchmarkResult:
        """Run TTFT benchmark.

        Args:
            stream_fn: Async generator function that yields tokens.
            iterations: Number of timed iterations.
            warmup: Number of warmup iterations (not counted).

        Returns:
            BenchmarkResult with TTFT statistics.
        """
        self._measurements = []

        async def measure() -> float:
            measurement = await measure_ttft(stream_fn, self.timeout)
            self._measurements.append(measurement)
            return measurement.ttft_ms

        # Use existing benchmark harness
        async def benchmark_fn():
            await measure()

        result = await benchmark(
            name="TTFT",
            fn=benchmark_fn,
            iterations=iterations,
            warmup=warmup,
            metadata={
                "target": self.target.name,
                "timeout": self.timeout,
            },
        )

        return result

    def passes_target(self, result: BenchmarkResult) -> bool:
        """Check if benchmark result meets target."""
        return self.target.passes(result)

    def get_measurements(self) -> list[TTFTMeasurement]:
        """Get all individual measurements."""
        return self._measurements.copy()

    def target_summary(self, result: BenchmarkResult) -> str:
        """Get summary comparing result to target."""
        return self.target.summary(result)


# =============================================================================
# TTFT Profile
# =============================================================================


@dataclass
class TTFTProfileResult:
    """Result of TTFT profiling with component breakdown.

    Attributes:
        total_ttft_ms: Total time to first token.
        components: Breakdown of time by component.
        timestamp: When profile was taken.
        metadata: Additional context.
    """

    total_ttft_ms: float
    components: dict[str, float]
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_ttft_ms": round(self.total_ttft_ms, 3),
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "timestamp": self.timestamp,
            **self.metadata,
        }

    def __str__(self) -> str:
        """Format as string."""
        lines = [f"TTFT Profile (Total: {self.total_ttft_ms:.1f}ms):"]
        for name, ms in sorted(self.components.items(), key=lambda x: -x[1]):
            pct = (ms / self.total_ttft_ms * 100) if self.total_ttft_ms > 0 else 0
            lines.append(f"  {name}: {ms:.1f}ms ({pct:.1f}%)")
        return "\n".join(lines)

    def bottleneck(self) -> tuple[str, float]:
        """Return the largest component (bottleneck)."""
        if not self.components:
            return ("unknown", 0.0)
        return max(self.components.items(), key=lambda x: x[1])


class TTFTProfile:
    """Profile TTFT to identify bottlenecks.

    Breaks down TTFT into measurable components:
    - initialization: Time to set up executor/graph
    - context_build: Time to build context/prompt
    - llm_request: Time for LLM API call to return first token
    - network: Estimated network latency
    - other: Unattributed time

    Example:
        >>> profile = TTFTProfile()
        >>>
        >>> # Profile with custom components
        >>> result = await profile.profile_components({
        ...     "init": init_fn,
        ...     "build": build_fn,
        ...     "llm": llm_fn,
        ... })
        >>> print(result)
        >>> print(f"Bottleneck: {result.bottleneck()}")
    """

    def __init__(self) -> None:
        """Initialize profiler."""
        self._results: list[TTFTProfileResult] = []

    async def profile_components(
        self,
        components: dict[str, Callable[[], Coroutine[Any, Any, Any]]],
    ) -> TTFTProfileResult:
        """Profile individual components.

        Args:
            components: Map of component name to async function.

        Returns:
            TTFTProfileResult with timing breakdown.
        """
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        for name, fn in components.items():
            start = time.perf_counter()
            await fn()
            timings[name] = (time.perf_counter() - start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        # Calculate unattributed time
        component_total = sum(timings.values())
        if total_ms > component_total:
            timings["overhead"] = total_ms - component_total

        result = TTFTProfileResult(
            total_ttft_ms=total_ms,
            components=timings,
            timestamp=datetime.now(UTC).isoformat(),
        )

        self._results.append(result)
        return result

    async def profile_streaming(
        self,
        stream_fn: Callable[[], AsyncIterator[str | tuple[str, Any]]],
        setup_fn: Callable[[], Coroutine[Any, Any, Any]] | None = None,
    ) -> TTFTProfileResult:
        """Profile a streaming operation.

        Measures:
        - setup: Time for setup_fn (if provided)
        - stream_start: Time to start streaming
        - first_token: Time to receive first token

        Args:
            stream_fn: Async generator that yields tokens.
            setup_fn: Optional setup function to measure.

        Returns:
            TTFTProfileResult with breakdown.
        """
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # Setup phase
        if setup_fn:
            setup_start = time.perf_counter()
            await setup_fn()
            timings["setup"] = (time.perf_counter() - setup_start) * 1000

        # Stream start phase
        stream_start = time.perf_counter()
        stream = stream_fn()

        # First token phase
        token_start = time.perf_counter()
        timings["stream_init"] = (token_start - stream_start) * 1000

        async for _ in stream:
            timings["first_token"] = (time.perf_counter() - token_start) * 1000
            break

        total_ms = (time.perf_counter() - total_start) * 1000

        result = TTFTProfileResult(
            total_ttft_ms=total_ms,
            components=timings,
            timestamp=datetime.now(UTC).isoformat(),
        )

        self._results.append(result)
        return result

    def get_results(self) -> list[TTFTProfileResult]:
        """Get all profile results."""
        return self._results.copy()

    def average_components(self) -> dict[str, float]:
        """Get average timing for each component across all profiles."""
        if not self._results:
            return {}

        all_components: dict[str, list[float]] = {}
        for result in self._results:
            for name, ms in result.components.items():
                if name not in all_components:
                    all_components[name] = []
                all_components[name].append(ms)

        return {name: sum(values) / len(values) for name, values in all_components.items()}


# =============================================================================
# Simulated TTFT for Testing
# =============================================================================


async def simulated_stream(
    ttft_ms: float = 100.0,
    token_count: int = 10,
    token_delay_ms: float = 10.0,
) -> AsyncIterator[str]:
    """Create a simulated token stream for testing.

    Args:
        ttft_ms: Time to first token in milliseconds.
        token_count: Number of tokens to generate.
        token_delay_ms: Delay between tokens.

    Yields:
        Simulated tokens.
    """
    # Simulate TTFT
    await asyncio.sleep(ttft_ms / 1000)
    yield "First"

    # Stream remaining tokens
    for i in range(token_count - 1):
        await asyncio.sleep(token_delay_ms / 1000)
        yield f"token_{i}"


def create_simulated_stream(
    ttft_ms: float = 100.0,
    token_count: int = 10,
    token_delay_ms: float = 10.0,
) -> Callable[[], AsyncIterator[str]]:
    """Create a factory for simulated streams.

    Useful for passing to benchmark functions that expect a callable.

    Args:
        ttft_ms: Time to first token.
        token_count: Number of tokens.
        token_delay_ms: Delay between tokens.

    Returns:
        Callable that creates a simulated stream.
    """

    def factory() -> AsyncIterator[str]:
        return simulated_stream(ttft_ms, token_count, token_delay_ms)

    return factory
