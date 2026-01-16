"""Tests for TTFT (Time-to-First-Token) benchmark module.

Tests Phase 12.2 of EXECUTOR_4.md - Time-to-First-Token Benchmark.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from ai_infra.benchmarks import BenchmarkResult
from ai_infra.benchmarks.ttft import (
    TTFT_TARGET_DEFAULT,
    TTFT_TARGET_FAST,
    TTFT_TARGET_STREAMING,
    TTFTBenchmark,
    TTFTMeasurement,
    TTFTProfile,
    TTFTProfileResult,
    TTFTTarget,
    create_simulated_stream,
    measure_ttft,
    simulated_stream,
)

# =============================================================================
# TTFTTarget Tests
# =============================================================================


class TestTTFTTarget:
    """Tests for TTFTTarget dataclass."""

    def test_target_creation(self) -> None:
        """Test target creation with valid values."""
        target = TTFTTarget(
            name="Test Target",
            p50_ms=100.0,
            p95_ms=200.0,
            p99_ms=300.0,
        )

        assert target.name == "Test Target"
        assert target.p50_ms == 100.0
        assert target.p95_ms == 200.0
        assert target.p99_ms == 300.0

    def test_target_passes_when_under_threshold(self) -> None:
        """Test passes() returns True when result is under threshold."""
        target = TTFTTarget("Test", p50_ms=200.0, p95_ms=400.0, p99_ms=600.0)
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=150.0,
            median_ms=140.0,  # Under 200
            p95_ms=350.0,  # Under 400
            p99_ms=500.0,  # Under 600
            min_ms=100.0,
            max_ms=600.0,
        )

        assert target.passes(result) is True

    def test_target_fails_when_p50_exceeded(self) -> None:
        """Test passes() returns False when p50 is exceeded."""
        target = TTFTTarget("Test", p50_ms=100.0, p95_ms=200.0, p99_ms=300.0)
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=150.0,
            median_ms=150.0,  # Exceeds 100
            p95_ms=180.0,
            p99_ms=200.0,
            min_ms=100.0,
            max_ms=220.0,
        )

        assert target.passes(result) is False

    def test_target_fails_when_p95_exceeded(self) -> None:
        """Test passes() returns False when p95 is exceeded."""
        target = TTFTTarget("Test", p50_ms=100.0, p95_ms=200.0, p99_ms=300.0)
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=150.0,
            median_ms=90.0,
            p95_ms=250.0,  # Exceeds 200
            p99_ms=280.0,
            min_ms=80.0,
            max_ms=300.0,
        )

        assert target.passes(result) is False

    def test_target_fails_when_p99_exceeded(self) -> None:
        """Test passes() returns False when p99 is exceeded."""
        target = TTFTTarget("Test", p50_ms=100.0, p95_ms=200.0, p99_ms=300.0)
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=150.0,
            median_ms=90.0,
            p95_ms=180.0,
            p99_ms=350.0,  # Exceeds 300
            min_ms=80.0,
            max_ms=400.0,
        )

        assert target.passes(result) is False

    def test_target_summary_format(self) -> None:
        """Test summary() produces readable output."""
        target = TTFTTarget("Test", p50_ms=100.0, p95_ms=200.0, p99_ms=300.0)
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=90.0,
            median_ms=90.0,
            p95_ms=180.0,
            p99_ms=250.0,
            min_ms=80.0,
            max_ms=300.0,
        )

        summary = target.summary(result)

        assert "Test:" in summary
        assert "P50:" in summary
        assert "P95:" in summary
        assert "P99:" in summary
        assert "✓" in summary  # Should pass all


class TestDefaultTargets:
    """Tests for predefined TTFT targets."""

    def test_default_target_values(self) -> None:
        """Test TTFT_TARGET_DEFAULT has correct values."""
        assert TTFT_TARGET_DEFAULT.p50_ms == 300.0
        assert TTFT_TARGET_DEFAULT.p95_ms == 500.0
        assert TTFT_TARGET_DEFAULT.p99_ms == 1000.0

    def test_fast_target_values(self) -> None:
        """Test TTFT_TARGET_FAST has correct values."""
        assert TTFT_TARGET_FAST.p50_ms == 100.0
        assert TTFT_TARGET_FAST.p95_ms == 200.0
        assert TTFT_TARGET_FAST.p99_ms == 400.0

    def test_streaming_target_values(self) -> None:
        """Test TTFT_TARGET_STREAMING has correct values."""
        assert TTFT_TARGET_STREAMING.p50_ms == 50.0
        assert TTFT_TARGET_STREAMING.p95_ms == 100.0
        assert TTFT_TARGET_STREAMING.p99_ms == 200.0


# =============================================================================
# TTFTMeasurement Tests
# =============================================================================


class TestTTFTMeasurement:
    """Tests for TTFTMeasurement dataclass."""

    def test_measurement_creation(self) -> None:
        """Test measurement creation with required fields."""
        measurement = TTFTMeasurement(
            ttft_ms=150.0,
            total_ms=500.0,
            first_token="Hello",
        )

        assert measurement.ttft_ms == 150.0
        assert measurement.total_ms == 500.0
        assert measurement.first_token == "Hello"

    def test_measurement_to_dict(self) -> None:
        """Test to_dict() serializes correctly."""
        measurement = TTFTMeasurement(
            ttft_ms=150.123,
            total_ms=500.456,
            first_token="Hello World",
            timestamp="2024-01-01T00:00:00Z",
        )

        d = measurement.to_dict()

        assert d["ttft_ms"] == 150.123
        assert d["total_ms"] == 500.456
        assert d["first_token"] == "Hello World"
        assert d["timestamp"] == "2024-01-01T00:00:00Z"

    def test_measurement_truncates_long_token(self) -> None:
        """Test first_token is truncated in to_dict()."""
        long_token = "x" * 100
        measurement = TTFTMeasurement(
            ttft_ms=100.0,
            first_token=long_token,
        )

        d = measurement.to_dict()

        assert len(d["first_token"]) == 50


# =============================================================================
# measure_ttft Tests
# =============================================================================


class TestMeasureTTFT:
    """Tests for measure_ttft function."""

    @pytest.mark.asyncio
    async def test_measures_time_to_first_token(self) -> None:
        """Test measure_ttft captures TTFT correctly."""

        # Create stream with ~50ms TTFT
        async def test_stream() -> AsyncIterator[str]:
            await asyncio.sleep(0.05)
            yield "First"
            yield "Second"

        measurement = await measure_ttft(test_stream)

        # Should be around 50ms (with some tolerance)
        assert 40 < measurement.ttft_ms < 150
        assert measurement.first_token == "First"

    @pytest.mark.asyncio
    async def test_handles_tuple_stream(self) -> None:
        """Test measure_ttft handles tuple-yielding streams."""

        async def tuple_stream() -> AsyncIterator[tuple[str, int]]:
            await asyncio.sleep(0.01)
            yield ("node_1", {"state": "data"})
            yield ("node_2", {"state": "more"})

        measurement = await measure_ttft(tuple_stream)

        assert measurement.first_token == "node_1"

    @pytest.mark.asyncio
    async def test_handles_timeout(self) -> None:
        """Test measure_ttft handles timeout correctly."""

        async def slow_stream() -> AsyncIterator[str]:
            await asyncio.sleep(10.0)  # Would take 10s
            yield "Finally"

        measurement = await measure_ttft(slow_stream, timeout=0.1)

        # Should timeout after ~100ms
        assert measurement.ttft_ms >= 100.0

    @pytest.mark.asyncio
    async def test_captures_total_time(self) -> None:
        """Test total_ms is captured correctly."""

        async def test_stream() -> AsyncIterator[str]:
            await asyncio.sleep(0.02)  # TTFT
            yield "First"

        measurement = await measure_ttft(test_stream)

        assert measurement.total_ms >= measurement.ttft_ms

    @pytest.mark.asyncio
    async def test_sets_timestamp(self) -> None:
        """Test timestamp is set on measurement."""

        async def test_stream() -> AsyncIterator[str]:
            yield "Token"

        measurement = await measure_ttft(test_stream)

        assert measurement.timestamp != ""
        assert "T" in measurement.timestamp  # ISO format


# =============================================================================
# TTFTBenchmark Tests
# =============================================================================


class TestTTFTBenchmark:
    """Tests for TTFTBenchmark class."""

    def test_init_with_default_target(self) -> None:
        """Test initialization with default target."""
        benchmark = TTFTBenchmark()

        assert benchmark.target == TTFT_TARGET_DEFAULT
        assert benchmark.timeout == 30.0

    def test_init_with_custom_target(self) -> None:
        """Test initialization with custom target."""
        custom_target = TTFTTarget("Custom", 50.0, 100.0, 200.0)
        benchmark = TTFTBenchmark(target=custom_target)

        assert benchmark.target == custom_target

    @pytest.mark.asyncio
    async def test_run_benchmark(self) -> None:
        """Test run() executes benchmark iterations."""
        benchmark = TTFTBenchmark()

        stream_factory = create_simulated_stream(ttft_ms=50.0)
        result = await benchmark.run(stream_factory, iterations=5, warmup=1)

        assert result.name == "TTFT"
        assert result.iterations == 5
        assert result.mean_ms > 0

    @pytest.mark.asyncio
    async def test_stores_measurements(self) -> None:
        """Test measurements are stored during run."""
        benchmark = TTFTBenchmark()

        stream_factory = create_simulated_stream(ttft_ms=30.0)
        await benchmark.run(stream_factory, iterations=3, warmup=0)

        measurements = benchmark.get_measurements()
        assert len(measurements) == 3

    def test_passes_target_when_under(self) -> None:
        """Test passes_target() returns True when under threshold."""
        benchmark = TTFTBenchmark(target=TTFTTarget("Test", 200.0, 400.0, 600.0))
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=100.0,
            median_ms=100.0,
            p95_ms=200.0,
            p99_ms=300.0,
            min_ms=80.0,
            max_ms=350.0,
        )

        assert benchmark.passes_target(result) is True

    def test_passes_target_when_over(self) -> None:
        """Test passes_target() returns False when over threshold."""
        benchmark = TTFTBenchmark(target=TTFTTarget("Test", 100.0, 200.0, 300.0))
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_ms=150.0,
            median_ms=150.0,
            p95_ms=250.0,
            p99_ms=350.0,
            min_ms=100.0,
            max_ms=400.0,
        )

        assert benchmark.passes_target(result) is False


# =============================================================================
# TTFTProfileResult Tests
# =============================================================================


class TestTTFTProfileResult:
    """Tests for TTFTProfileResult dataclass."""

    def test_result_creation(self) -> None:
        """Test profile result creation."""
        result = TTFTProfileResult(
            total_ttft_ms=200.0,
            components={
                "init": 50.0,
                "build": 100.0,
                "llm": 50.0,
            },
        )

        assert result.total_ttft_ms == 200.0
        assert len(result.components) == 3

    def test_to_dict(self) -> None:
        """Test to_dict() serializes correctly."""
        result = TTFTProfileResult(
            total_ttft_ms=200.123,
            components={"init": 50.456, "llm": 100.789},
            timestamp="2024-01-01T00:00:00Z",
        )

        d = result.to_dict()

        assert d["total_ttft_ms"] == 200.123
        assert d["components"]["init"] == 50.456

    def test_str_format(self) -> None:
        """Test __str__() produces readable output."""
        result = TTFTProfileResult(
            total_ttft_ms=200.0,
            components={
                "init": 50.0,
                "llm": 100.0,
                "build": 50.0,
            },
        )

        output = str(result)

        assert "TTFT Profile" in output
        assert "200.0ms" in output
        assert "init" in output
        assert "llm" in output

    def test_bottleneck_returns_largest(self) -> None:
        """Test bottleneck() returns the largest component."""
        result = TTFTProfileResult(
            total_ttft_ms=200.0,
            components={
                "init": 30.0,
                "llm": 150.0,  # Largest
                "build": 20.0,
            },
        )

        name, ms = result.bottleneck()

        assert name == "llm"
        assert ms == 150.0

    def test_bottleneck_empty_components(self) -> None:
        """Test bottleneck() handles empty components."""
        result = TTFTProfileResult(
            total_ttft_ms=0.0,
            components={},
        )

        name, ms = result.bottleneck()

        assert name == "unknown"
        assert ms == 0.0


# =============================================================================
# TTFTProfile Tests
# =============================================================================


class TestTTFTProfile:
    """Tests for TTFTProfile class."""

    @pytest.mark.asyncio
    async def test_profile_components(self) -> None:
        """Test profile_components measures each component."""
        profile = TTFTProfile()

        async def init_fn():
            await asyncio.sleep(0.02)

        async def llm_fn():
            await asyncio.sleep(0.05)

        result = await profile.profile_components(
            {
                "init": init_fn,
                "llm": llm_fn,
            }
        )

        assert result.total_ttft_ms > 0
        assert "init" in result.components
        assert "llm" in result.components
        # LLM should be larger
        assert result.components["llm"] > result.components["init"]

    @pytest.mark.asyncio
    async def test_profile_streaming(self) -> None:
        """Test profile_streaming measures stream phases."""
        profile = TTFTProfile()

        async def setup():
            await asyncio.sleep(0.01)

        stream_factory = create_simulated_stream(ttft_ms=30.0)

        result = await profile.profile_streaming(
            stream_fn=stream_factory,
            setup_fn=setup,
        )

        assert "setup" in result.components
        assert "first_token" in result.components

    @pytest.mark.asyncio
    async def test_stores_results(self) -> None:
        """Test profile results are stored."""
        profile = TTFTProfile()

        async def dummy():
            pass

        await profile.profile_components({"test": dummy})
        await profile.profile_components({"test": dummy})

        results = profile.get_results()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_average_components(self) -> None:
        """Test average_components calculates averages."""
        profile = TTFTProfile()

        async def fn():
            await asyncio.sleep(0.01)

        # Run multiple profiles
        await profile.profile_components({"test": fn})
        await profile.profile_components({"test": fn})

        averages = profile.average_components()

        assert "test" in averages
        assert averages["test"] > 0


# =============================================================================
# Simulated Stream Tests
# =============================================================================


class TestSimulatedStream:
    """Tests for simulated stream utilities."""

    @pytest.mark.asyncio
    async def test_simulated_stream_yields_tokens(self) -> None:
        """Test simulated_stream yields expected tokens."""
        tokens = []
        async for token in simulated_stream(ttft_ms=10, token_count=3):
            tokens.append(token)

        assert len(tokens) == 3
        assert tokens[0] == "First"

    @pytest.mark.asyncio
    async def test_simulated_stream_ttft_timing(self) -> None:
        """Test simulated_stream TTFT is approximately correct."""
        import time

        start = time.perf_counter()
        async for _ in simulated_stream(ttft_ms=50, token_count=1):
            ttft = (time.perf_counter() - start) * 1000
            break

        # Should be around 50ms (with tolerance)
        assert 30 < ttft < 150

    def test_create_simulated_stream_factory(self) -> None:
        """Test create_simulated_stream returns callable."""
        factory = create_simulated_stream(ttft_ms=100)

        assert callable(factory)
        # Calling factory returns async iterator
        stream = factory()
        assert hasattr(stream, "__anext__")


# =============================================================================
# Integration Tests
# =============================================================================


class TestTTFTIntegration:
    """Integration tests for TTFT benchmarking."""

    @pytest.mark.asyncio
    async def test_full_benchmark_workflow(self) -> None:
        """Test complete benchmark -> profile -> analyze workflow."""
        # Step 1: Run benchmark
        benchmark = TTFTBenchmark(target=TTFT_TARGET_DEFAULT)
        stream_factory = create_simulated_stream(ttft_ms=100)

        result = await benchmark.run(stream_factory, iterations=5, warmup=1)

        # Step 2: Check target
        assert result.iterations == 5
        passes = benchmark.passes_target(result)
        assert isinstance(passes, bool)

        # Step 3: Get summary
        summary = benchmark.target_summary(result)
        assert "P50" in summary
        assert "P95" in summary

    @pytest.mark.asyncio
    async def test_profile_and_identify_bottleneck(self) -> None:
        """Test profiling identifies the bottleneck component."""
        profile = TTFTProfile()

        async def fast_init():
            await asyncio.sleep(0.01)

        async def slow_llm():
            await asyncio.sleep(0.05)

        result = await profile.profile_components(
            {
                "init": fast_init,
                "llm": slow_llm,
            }
        )

        bottleneck_name, bottleneck_ms = result.bottleneck()

        # LLM should be the bottleneck
        assert bottleneck_name == "llm"
        assert bottleneck_ms > result.components["init"]

    @pytest.mark.asyncio
    async def test_compare_against_targets(self) -> None:
        """Test comparing results against different targets."""
        stream_factory = create_simulated_stream(ttft_ms=80)

        benchmark = TTFTBenchmark()
        result = await benchmark.run(stream_factory, iterations=3, warmup=0)

        # Check against different targets
        assert TTFT_TARGET_DEFAULT.passes(result)  # 500ms p95 - should pass
        # Streaming target at 100ms p95 - may or may not pass depending on variance
