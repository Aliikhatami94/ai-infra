"""Tests for benchmark harness module.

Phase 12.1 of EXECUTOR_4.md - Benchmark Framework Tests.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from ai_infra.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    benchmark,
    benchmark_sync,
    compare_results,
    run_benchmark_suite,
)

# =============================================================================
# Test BenchmarkResult
# =============================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_basic_result(self) -> None:
        """Basic BenchmarkResult should store all fields."""
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=100.0,
            median_ms=95.0,
            p95_ms=150.0,
            p99_ms=180.0,
            min_ms=80.0,
            max_ms=200.0,
        )
        assert result.name == "test"
        assert result.iterations == 10
        assert result.mean_ms == 100.0
        assert result.median_ms == 95.0
        assert result.p95_ms == 150.0
        assert result.p99_ms == 180.0
        assert result.min_ms == 80.0
        assert result.max_ms == 200.0

    def test_str_format(self) -> None:
        """__str__ should format result nicely."""
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=100.0,
            median_ms=95.0,
            p95_ms=150.0,
            p99_ms=180.0,
            min_ms=80.0,
            max_ms=200.0,
            std_dev_ms=25.0,
        )
        s = str(result)
        assert "test:" in s
        assert "Mean: 100.0ms" in s
        assert "P95: 150.0ms" in s
        assert "StdDev: 25.0ms" in s

    def test_to_dict(self) -> None:
        """to_dict should include all fields."""
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=100.0,
            median_ms=95.0,
            p95_ms=150.0,
            p99_ms=180.0,
            min_ms=80.0,
            max_ms=200.0,
            timestamp="2024-01-15T10:00:00Z",
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert d["name"] == "test"
        assert d["mean_ms"] == 100.0
        assert d["timestamp"] == "2024-01-15T10:00:00Z"
        assert d["metadata"] == {"key": "value"}

    def test_passes_target_p95(self) -> None:
        """passes_target should check p95 by default."""
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=100.0,
            median_ms=95.0,
            p95_ms=150.0,
            p99_ms=180.0,
            min_ms=80.0,
            max_ms=200.0,
        )
        assert result.passes_target(200.0) is True  # 150 < 200
        assert result.passes_target(100.0) is False  # 150 > 100

    def test_passes_target_mean(self) -> None:
        """passes_target should check specified percentile."""
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=100.0,
            median_ms=95.0,
            p95_ms=150.0,
            p99_ms=180.0,
            min_ms=80.0,
            max_ms=200.0,
        )
        assert result.passes_target(120.0, percentile="mean") is True
        assert result.passes_target(80.0, percentile="mean") is False

    def test_summary(self) -> None:
        """summary should return one-line string."""
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=100.0,
            median_ms=95.0,
            p95_ms=150.0,
            p99_ms=180.0,
            min_ms=80.0,
            max_ms=200.0,
        )
        summary = result.summary()
        assert "test:" in summary
        assert "mean=100.0ms" in summary
        assert "p95=150.0ms" in summary


# =============================================================================
# Test benchmark() Function
# =============================================================================


class TestBenchmarkAsync:
    """Tests for async benchmark() function."""

    @pytest.mark.asyncio
    async def test_benchmark_simple_function(self) -> None:
        """benchmark should measure async function execution."""

        async def simple_async():
            await asyncio.sleep(0.001)  # 1ms

        result = await benchmark("simple", simple_async, iterations=5, warmup=1)

        assert result.name == "simple"
        assert result.iterations == 5
        assert result.warmup_iterations == 1
        assert result.mean_ms >= 1.0  # At least 1ms
        assert result.min_ms >= 1.0
        assert result.timestamp  # Should have timestamp

    @pytest.mark.asyncio
    async def test_benchmark_with_metadata(self) -> None:
        """benchmark should include metadata in result."""

        async def fast_fn():
            pass

        result = await benchmark(
            "meta_test",
            fast_fn,
            iterations=3,
            metadata={"version": "1.0", "type": "unit"},
        )

        assert result.metadata["version"] == "1.0"
        assert result.metadata["type"] == "unit"

    @pytest.mark.asyncio
    async def test_benchmark_invalid_iterations(self) -> None:
        """benchmark should raise on invalid iterations."""

        async def fn():
            pass

        with pytest.raises(ValueError, match="iterations must be >= 1"):
            await benchmark("test", fn, iterations=0)

    @pytest.mark.asyncio
    async def test_benchmark_invalid_warmup(self) -> None:
        """benchmark should raise on negative warmup."""

        async def fn():
            pass

        with pytest.raises(ValueError, match="warmup must be >= 0"):
            await benchmark("test", fn, warmup=-1)

    @pytest.mark.asyncio
    async def test_benchmark_percentiles(self) -> None:
        """benchmark should compute correct percentiles."""

        call_count = 0

        async def varying_fn():
            nonlocal call_count
            call_count += 1
            # Variable sleep to create spread
            await asyncio.sleep(0.001 * (call_count % 5 + 1))

        result = await benchmark("varying", varying_fn, iterations=10, warmup=0)

        # P95 should be <= P99 <= max
        assert result.p95_ms <= result.p99_ms <= result.max_ms
        # Min should be <= mean <= max
        assert result.min_ms <= result.mean_ms <= result.max_ms


# =============================================================================
# Test benchmark_sync() Function
# =============================================================================


class TestBenchmarkSync:
    """Tests for sync benchmark_sync() function."""

    def test_benchmark_sync_simple(self) -> None:
        """benchmark_sync should measure sync function execution."""

        def simple_fn():
            time.sleep(0.001)

        result = benchmark_sync("simple_sync", simple_fn, iterations=5, warmup=1)

        assert result.name == "simple_sync"
        assert result.iterations == 5
        assert result.mean_ms >= 1.0

    def test_benchmark_sync_fast(self) -> None:
        """benchmark_sync should handle very fast functions."""

        def fast_fn():
            return 1 + 1

        result = benchmark_sync("fast", fast_fn, iterations=100, warmup=10)

        assert result.iterations == 100
        assert result.min_ms >= 0  # Should be very fast
        assert result.std_dev_ms is not None

    def test_benchmark_sync_invalid_iterations(self) -> None:
        """benchmark_sync should raise on invalid iterations."""

        def fn():
            pass

        with pytest.raises(ValueError, match="iterations must be >= 1"):
            benchmark_sync("test", fn, iterations=0)


# =============================================================================
# Test BenchmarkSuite
# =============================================================================


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite class."""

    def test_suite_creation(self) -> None:
        """BenchmarkSuite should initialize correctly."""
        suite = BenchmarkSuite(name="test_suite")
        assert suite.name == "test_suite"
        assert len(suite.benchmarks) == 0
        assert len(suite.results) == 0

    def test_suite_add(self) -> None:
        """add() should add benchmarks to suite."""

        async def fn1():
            pass

        async def fn2():
            pass

        suite = BenchmarkSuite(name="test")
        suite.add("fn1", fn1)
        suite.add("fn2", fn2)

        assert len(suite.benchmarks) == 2

    @pytest.mark.asyncio
    async def test_suite_run(self) -> None:
        """run() should execute all benchmarks."""

        async def fn1():
            await asyncio.sleep(0.001)

        async def fn2():
            await asyncio.sleep(0.002)

        suite = BenchmarkSuite(name="test_suite")
        suite.add("fn1", fn1)
        suite.add("fn2", fn2)

        results = await suite.run(iterations=3, warmup=1)

        assert len(results) == 2
        assert results[0].name == "test_suite/fn1"
        assert results[1].name == "test_suite/fn2"
        assert suite.results == results

    def test_suite_summary_no_results(self) -> None:
        """summary() should handle no results."""
        suite = BenchmarkSuite(name="empty")
        summary = suite.summary()
        assert "No results yet" in summary

    @pytest.mark.asyncio
    async def test_suite_summary_with_results(self) -> None:
        """summary() should format results."""

        async def fn():
            pass

        suite = BenchmarkSuite(name="test")
        suite.add("fast", fn)
        await suite.run(iterations=2)

        summary = suite.summary()
        assert "test" in summary
        assert "1 benchmarks" in summary

    @pytest.mark.asyncio
    async def test_suite_to_dict(self) -> None:
        """to_dict() should serialize suite."""

        async def fn():
            pass

        suite = BenchmarkSuite(name="test")
        suite.add("fn", fn)
        await suite.run(iterations=2)

        d = suite.to_dict()
        assert d["name"] == "test"
        assert d["count"] == 1
        assert len(d["benchmarks"]) == 1


# =============================================================================
# Test run_benchmark_suite() Function
# =============================================================================


class TestRunBenchmarkSuite:
    """Tests for run_benchmark_suite() convenience function."""

    @pytest.mark.asyncio
    async def test_run_benchmark_suite(self) -> None:
        """run_benchmark_suite should run all benchmarks."""

        async def fn1():
            pass

        async def fn2():
            pass

        results = await run_benchmark_suite(
            "test",
            {"fn1": fn1, "fn2": fn2},
            iterations=3,
            warmup=1,
        )

        assert len(results) == 2
        names = [r.name for r in results]
        assert "test/fn1" in names
        assert "test/fn2" in names


# =============================================================================
# Test compare_results() Function
# =============================================================================


class TestCompareResults:
    """Tests for compare_results() function."""

    def test_compare_no_change(self) -> None:
        """compare_results should detect no significant change."""
        baseline = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=100.0,
            median_ms=100.0,
            p95_ms=100.0,
            p99_ms=100.0,
            min_ms=100.0,
            max_ms=100.0,
        )
        current = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=102.0,  # 2% increase
            median_ms=102.0,
            p95_ms=102.0,
            p99_ms=102.0,
            min_ms=102.0,
            max_ms=102.0,
        )

        comparison = compare_results(baseline, current, threshold_pct=10.0)

        assert comparison["is_regression"] is False
        assert comparison["is_improvement"] is False
        assert comparison["mean_change_pct"] == 2.0

    def test_compare_regression(self) -> None:
        """compare_results should detect regression."""
        baseline = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=100.0,
            median_ms=100.0,
            p95_ms=100.0,
            p99_ms=100.0,
            min_ms=100.0,
            max_ms=100.0,
        )
        current = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=120.0,  # 20% increase
            median_ms=120.0,
            p95_ms=120.0,
            p99_ms=120.0,
            min_ms=120.0,
            max_ms=120.0,
        )

        comparison = compare_results(baseline, current, threshold_pct=10.0)

        assert comparison["is_regression"] is True
        assert comparison["is_improvement"] is False
        assert comparison["mean_change_pct"] == 20.0

    def test_compare_improvement(self) -> None:
        """compare_results should detect improvement."""
        baseline = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=100.0,
            median_ms=100.0,
            p95_ms=100.0,
            p99_ms=100.0,
            min_ms=100.0,
            max_ms=100.0,
        )
        current = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=80.0,  # 20% decrease (improvement)
            median_ms=80.0,
            p95_ms=80.0,
            p99_ms=80.0,
            min_ms=80.0,
            max_ms=80.0,
        )

        comparison = compare_results(baseline, current, threshold_pct=10.0)

        assert comparison["is_regression"] is False
        assert comparison["is_improvement"] is True
        assert comparison["mean_change_pct"] == -20.0

    def test_compare_custom_threshold(self) -> None:
        """compare_results should respect custom threshold."""
        baseline = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=100.0,
            median_ms=100.0,
            p95_ms=100.0,
            p99_ms=100.0,
            min_ms=100.0,
            max_ms=100.0,
        )
        current = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=107.0,  # 7% increase
            median_ms=107.0,
            p95_ms=107.0,
            p99_ms=107.0,
            min_ms=107.0,
            max_ms=107.0,
        )

        # With 10% threshold, not a regression
        comparison_10 = compare_results(baseline, current, threshold_pct=10.0)
        assert comparison_10["is_regression"] is False

        # With 5% threshold, is a regression
        comparison_5 = compare_results(baseline, current, threshold_pct=5.0)
        assert comparison_5["is_regression"] is True
