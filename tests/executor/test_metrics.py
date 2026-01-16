"""Tests for executor metrics module.

Tests for performance metrics, benchmarking, and validation.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from ai_infra.executor.phase1_metrics import (
    PHASE1_TARGETS,
    BenchmarkResult,
    BenchmarkRunner,
    MetricCollector,
    MetricSample,
    MetricStats,
    MetricTarget,
    MetricType,
    Phase1ValidationReport,
    Phase1Validator,
    ValidationResult,
    create_benchmark_runner,
    create_collector,
    create_validator,
    validate_phase1_targets,
)

# =============================================================================
# MetricType Tests
# =============================================================================


class TestMetricType:
    """Tests for MetricType enum."""

    def test_all_metric_types_exist(self) -> None:
        """Verify all expected metric types exist."""
        assert MetricType.TIME_TO_FIRST_TOKEN.value == "time_to_first_token"
        assert MetricType.SIMPLE_TASK_LATENCY.value == "simple_task_latency"
        assert MetricType.CONTEXT_BUILD_TIME.value == "context_build_time"
        assert MetricType.VERIFICATION_TIME.value == "verification_time"
        assert MetricType.COST_PER_TASK.value == "cost_per_task"

    def test_metric_types_unique(self) -> None:
        """Verify metric type values are unique."""
        values = [mt.value for mt in MetricType]
        assert len(values) == len(set(values))


# =============================================================================
# MetricTarget Tests
# =============================================================================


class TestMetricTarget:
    """Tests for MetricTarget dataclass."""

    def test_target_creation(self) -> None:
        """Test creating a metric target."""
        target = MetricTarget(
            metric_type=MetricType.TIME_TO_FIRST_TOKEN,
            target_value=500.0,
            unit="ms",
            before_value=4000.0,
            description="Time to first token",
        )
        assert target.metric_type == MetricType.TIME_TO_FIRST_TOKEN
        assert target.target_value == 500.0
        assert target.unit == "ms"

    def test_is_met_when_below_target(self) -> None:
        """Test target is met when actual is below target."""
        target = MetricTarget(
            metric_type=MetricType.TIME_TO_FIRST_TOKEN,
            target_value=500.0,
            unit="ms",
            before_value=4000.0,
            description="Test",
        )
        assert target.is_met(400.0) is True
        assert target.is_met(500.0) is True

    def test_is_met_when_above_target(self) -> None:
        """Test target is not met when actual is above target."""
        target = MetricTarget(
            metric_type=MetricType.TIME_TO_FIRST_TOKEN,
            target_value=500.0,
            unit="ms",
            before_value=4000.0,
            description="Test",
        )
        assert target.is_met(501.0) is False
        assert target.is_met(1000.0) is False


# =============================================================================
# Phase 1 Targets Tests
# =============================================================================


class TestPhase1Targets:
    """Tests for predefined Phase 1 targets."""

    def test_all_targets_defined(self) -> None:
        """Verify all metric types have targets."""
        for metric_type in MetricType:
            assert metric_type in PHASE1_TARGETS

    def test_time_to_first_token_target(self) -> None:
        """Test time to first token target."""
        target = PHASE1_TARGETS[MetricType.TIME_TO_FIRST_TOKEN]
        assert target.target_value == 500.0
        assert target.unit == "ms"
        assert target.before_value == 4000.0

    def test_verification_time_target(self) -> None:
        """Test verification time target."""
        target = PHASE1_TARGETS[MetricType.VERIFICATION_TIME]
        assert target.target_value == 2000.0
        assert target.unit == "ms"
        assert target.before_value == 3000.0

    def test_cost_per_task_target(self) -> None:
        """Test cost per task target."""
        target = PHASE1_TARGETS[MetricType.COST_PER_TASK]
        assert target.target_value == 0.02
        assert target.unit == "$"
        assert target.before_value == 0.05


# =============================================================================
# MetricSample Tests
# =============================================================================


class TestMetricSample:
    """Tests for MetricSample dataclass."""

    def test_sample_creation(self) -> None:
        """Test creating a metric sample."""
        sample = MetricSample(
            metric_type=MetricType.VERIFICATION_TIME,
            value=1500.0,
        )
        assert sample.metric_type == MetricType.VERIFICATION_TIME
        assert sample.value == 1500.0
        assert sample.timestamp > 0

    def test_sample_with_metadata(self) -> None:
        """Test sample with metadata."""
        sample = MetricSample(
            metric_type=MetricType.COST_PER_TASK,
            value=0.015,
            metadata={"model": "gpt-4o-mini", "tokens": 1000},
        )
        assert sample.metadata["model"] == "gpt-4o-mini"
        assert sample.metadata["tokens"] == 1000

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        sample = MetricSample(
            metric_type=MetricType.TIME_TO_FIRST_TOKEN,
            value=250.0,
        )
        data = sample.to_dict()
        assert data["metric_type"] == "time_to_first_token"
        assert data["value"] == 250.0


# =============================================================================
# MetricStats Tests
# =============================================================================


class TestMetricStats:
    """Tests for MetricStats dataclass."""

    def test_empty_stats(self) -> None:
        """Test stats with no samples."""
        stats = MetricStats(metric_type=MetricType.VERIFICATION_TIME)
        assert stats.count == 0
        assert stats.min_value == float("inf")
        assert stats.max_value == 0.0

    def test_add_single_sample(self) -> None:
        """Test adding a single sample."""
        stats = MetricStats(metric_type=MetricType.VERIFICATION_TIME)
        stats.add_sample(100.0)
        assert stats.count == 1
        assert stats.min_value == 100.0
        assert stats.max_value == 100.0
        assert stats.avg_value == 100.0

    def test_add_multiple_samples(self) -> None:
        """Test adding multiple samples."""
        stats = MetricStats(metric_type=MetricType.VERIFICATION_TIME)
        stats.add_sample(100.0)
        stats.add_sample(200.0)
        stats.add_sample(300.0)
        assert stats.count == 3
        assert stats.min_value == 100.0
        assert stats.max_value == 300.0
        assert stats.avg_value == 200.0

    def test_percentiles(self) -> None:
        """Test percentile calculations."""
        stats = MetricStats(metric_type=MetricType.VERIFICATION_TIME)
        for i in range(1, 101):
            stats.add_sample(float(i))
        # With 100 samples (1-100), p50 index is 50, value is 51
        assert stats.p50_value >= 50.0
        assert stats.p50_value <= 52.0
        assert stats.p95_value >= 95.0
        assert stats.p99_value >= 99.0

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        stats = MetricStats(metric_type=MetricType.VERIFICATION_TIME)
        stats.add_sample(100.0)
        data = stats.to_dict()
        assert data["metric_type"] == "verification_time"
        assert data["count"] == 1
        assert data["avg"] == 100.0


# =============================================================================
# MetricCollector Tests
# =============================================================================


class TestMetricCollector:
    """Tests for MetricCollector class."""

    def test_create_collector(self) -> None:
        """Test creating a collector."""
        collector = MetricCollector()
        assert collector is not None

    def test_record_metric(self) -> None:
        """Test recording a metric."""
        collector = MetricCollector()
        collector.record(MetricType.VERIFICATION_TIME, 1500.0)
        samples = collector.get_samples()
        assert len(samples) == 1
        assert samples[0].value == 1500.0

    def test_record_with_metadata(self) -> None:
        """Test recording with metadata."""
        collector = MetricCollector()
        collector.record(
            MetricType.COST_PER_TASK,
            0.015,
            metadata={"model": "gpt-4o-mini"},
        )
        samples = collector.get_samples()
        assert samples[0].metadata["model"] == "gpt-4o-mini"

    def test_record_cost(self) -> None:
        """Test recording a cost metric."""
        collector = MetricCollector()
        collector.record_cost(0.015, model="gpt-4o-mini", tokens=1000)
        samples = collector.get_samples(MetricType.COST_PER_TASK)
        assert len(samples) == 1
        assert samples[0].value == 0.015

    def test_get_samples_filtered(self) -> None:
        """Test getting samples filtered by type."""
        collector = MetricCollector()
        collector.record(MetricType.VERIFICATION_TIME, 1000.0)
        collector.record(MetricType.COST_PER_TASK, 0.01)
        collector.record(MetricType.VERIFICATION_TIME, 1500.0)

        samples = collector.get_samples(MetricType.VERIFICATION_TIME)
        assert len(samples) == 2

    def test_get_stats(self) -> None:
        """Test getting stats."""
        collector = MetricCollector()
        collector.record(MetricType.VERIFICATION_TIME, 1000.0)
        collector.record(MetricType.VERIFICATION_TIME, 2000.0)

        stats = collector.get_stats(MetricType.VERIFICATION_TIME)
        assert isinstance(stats, MetricStats)
        assert stats.count == 2
        assert stats.avg_value == 1500.0

    def test_clear(self) -> None:
        """Test clearing samples."""
        collector = MetricCollector()
        collector.record(MetricType.VERIFICATION_TIME, 1000.0)
        collector.clear()
        assert len(collector.get_samples()) == 0

    @pytest.mark.asyncio
    async def test_measure_async(self) -> None:
        """Test async timing context manager."""
        collector = MetricCollector()

        async with collector.measure(MetricType.VERIFICATION_TIME):
            await asyncio.sleep(0.01)  # 10ms

        stats = collector.get_stats(MetricType.VERIFICATION_TIME)
        assert isinstance(stats, MetricStats)
        assert stats.count == 1
        assert stats.avg_value >= 10  # At least 10ms

    def test_measure_sync(self) -> None:
        """Test sync timing context manager."""
        collector = MetricCollector()

        with collector.measure_sync(MetricType.VERIFICATION_TIME):
            time.sleep(0.01)  # 10ms

        stats = collector.get_stats(MetricType.VERIFICATION_TIME)
        assert isinstance(stats, MetricStats)
        assert stats.count == 1
        assert stats.avg_value >= 10  # At least 10ms

    def test_summary(self) -> None:
        """Test summary generation."""
        collector = MetricCollector()
        collector.record(MetricType.VERIFICATION_TIME, 1500.0)
        collector.record(MetricType.COST_PER_TASK, 0.015)

        summary = collector.summary()
        assert "Phase 1 Metrics Summary" in summary
        assert "verification_time" in summary
        assert "cost_per_task" in summary


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_passed_result(self) -> None:
        """Test a passed validation result."""
        result = ValidationResult(
            metric_type=MetricType.VERIFICATION_TIME,
            target=2000.0,
            actual=1500.0,
            passed=True,
            improvement=2.0,
            unit="ms",
        )
        assert result.passed is True
        assert result.improvement == 2.0

    def test_failed_result(self) -> None:
        """Test a failed validation result."""
        result = ValidationResult(
            metric_type=MetricType.VERIFICATION_TIME,
            target=2000.0,
            actual=2500.0,
            passed=False,
            improvement=1.2,
            unit="ms",
        )
        assert result.passed is False

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = ValidationResult(
            metric_type=MetricType.VERIFICATION_TIME,
            target=2000.0,
            actual=1500.0,
            passed=True,
            improvement=2.0,
            unit="ms",
        )
        data = result.to_dict()
        assert data["metric_type"] == "verification_time"
        assert data["passed"] is True


# =============================================================================
# Phase1ValidationReport Tests
# =============================================================================


class TestPhase1ValidationReport:
    """Tests for Phase1ValidationReport dataclass."""

    def test_empty_report(self) -> None:
        """Test empty report."""
        report = Phase1ValidationReport()
        assert report.overall_passed is True  # Vacuously true
        assert report.passed_count == 0
        assert report.failed_count == 0

    def test_all_passed_report(self) -> None:
        """Test report with all passed."""
        results = [
            ValidationResult(
                metric_type=MetricType.VERIFICATION_TIME,
                target=2000.0,
                actual=1500.0,
                passed=True,
                improvement=2.0,
                unit="ms",
            ),
            ValidationResult(
                metric_type=MetricType.COST_PER_TASK,
                target=0.02,
                actual=0.015,
                passed=True,
                improvement=3.3,
                unit="$",
            ),
        ]
        report = Phase1ValidationReport(results=results)
        assert report.overall_passed is True
        assert report.passed_count == 2
        assert report.failed_count == 0

    def test_some_failed_report(self) -> None:
        """Test report with some failures."""
        results = [
            ValidationResult(
                metric_type=MetricType.VERIFICATION_TIME,
                target=2000.0,
                actual=1500.0,
                passed=True,
                improvement=2.0,
                unit="ms",
            ),
            ValidationResult(
                metric_type=MetricType.TIME_TO_FIRST_TOKEN,
                target=500.0,
                actual=600.0,
                passed=False,
                improvement=6.7,
                unit="ms",
            ),
        ]
        report = Phase1ValidationReport(results=results)
        assert report.overall_passed is False
        assert report.passed_count == 1
        assert report.failed_count == 1

    def test_summary(self) -> None:
        """Test summary generation."""
        results = [
            ValidationResult(
                metric_type=MetricType.VERIFICATION_TIME,
                target=2000.0,
                actual=1500.0,
                passed=True,
                improvement=2.0,
                unit="ms",
            ),
        ]
        report = Phase1ValidationReport(results=results)
        summary = report.summary()
        assert "Phase 1 Validation Report" in summary
        assert "PASSED" in summary
        assert "verification_time" in summary

    def test_markdown_table(self) -> None:
        """Test markdown table generation."""
        results = [
            ValidationResult(
                metric_type=MetricType.VERIFICATION_TIME,
                target=2000.0,
                actual=1500.0,
                passed=True,
                improvement=2.0,
                unit="ms",
            ),
        ]
        report = Phase1ValidationReport(results=results)
        table = report.markdown_table()
        assert "| Metric |" in table
        assert "[x]" in table

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        results = [
            ValidationResult(
                metric_type=MetricType.VERIFICATION_TIME,
                target=2000.0,
                actual=1500.0,
                passed=True,
                improvement=2.0,
                unit="ms",
            ),
        ]
        report = Phase1ValidationReport(results=results)
        data = report.to_dict()
        assert data["overall_passed"] is True
        assert len(data["results"]) == 1


# =============================================================================
# Phase1Validator Tests
# =============================================================================


class TestPhase1Validator:
    """Tests for Phase1Validator class."""

    def test_create_validator(self) -> None:
        """Test creating a validator."""
        validator = Phase1Validator()
        assert validator is not None

    def test_validate_with_override(self) -> None:
        """Test validation with override value."""
        validator = Phase1Validator()
        validator.set_override(MetricType.VERIFICATION_TIME, 1500.0)
        result = validator.validate(MetricType.VERIFICATION_TIME)
        assert result.passed is True
        assert result.actual == 1500.0

    def test_validate_failing_metric(self) -> None:
        """Test validation of failing metric."""
        validator = Phase1Validator()
        validator.set_override(MetricType.VERIFICATION_TIME, 3000.0)  # Above target
        result = validator.validate(MetricType.VERIFICATION_TIME)
        assert result.passed is False

    def test_validate_with_collector(self) -> None:
        """Test validation with collector data."""
        collector = MetricCollector()
        collector.record(MetricType.VERIFICATION_TIME, 1500.0)
        collector.record(MetricType.VERIFICATION_TIME, 1700.0)

        validator = Phase1Validator(collector=collector)
        result = validator.validate(MetricType.VERIFICATION_TIME)
        assert result.actual == 1600.0  # Average
        assert result.passed is True

    def test_validate_all(self) -> None:
        """Test validating all metrics."""
        validator = Phase1Validator()
        validator.set_override(MetricType.TIME_TO_FIRST_TOKEN, 400.0)
        validator.set_override(MetricType.SIMPLE_TASK_LATENCY, 500.0)
        validator.set_override(MetricType.CONTEXT_BUILD_TIME, 0.0)
        validator.set_override(MetricType.VERIFICATION_TIME, 1500.0)
        validator.set_override(MetricType.COST_PER_TASK, 0.015)

        report = validator.validate_all()
        assert report.overall_passed is True
        assert len(report.results) == 5

    def test_validate_with_custom_targets(self) -> None:
        """Test validation with custom targets."""
        custom_targets = {
            MetricType.VERIFICATION_TIME: MetricTarget(
                metric_type=MetricType.VERIFICATION_TIME,
                target_value=3000.0,  # Looser target
                unit="ms",
                before_value=5000.0,
                description="Custom target",
            ),
        }
        validator = Phase1Validator(targets=custom_targets)
        validator.set_override(MetricType.VERIFICATION_TIME, 2500.0)
        report = validator.validate_all()
        # Only one target defined
        assert len(report.results) == 1
        assert report.results[0].passed is True


# =============================================================================
# BenchmarkResult Tests
# =============================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            name="test",
            iterations=10,
            total_ms=1000.0,
            avg_ms=100.0,
            min_ms=90.0,
            max_ms=110.0,
        )
        assert result.name == "test"
        assert result.avg_ms == 100.0

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = BenchmarkResult(
            name="test",
            iterations=10,
            total_ms=1000.0,
            avg_ms=100.0,
            min_ms=90.0,
            max_ms=110.0,
        )
        data = result.to_dict()
        assert data["name"] == "test"
        assert data["avg_ms"] == 100.0


# =============================================================================
# BenchmarkRunner Tests
# =============================================================================


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_create_runner(self) -> None:
        """Test creating a benchmark runner."""
        runner = BenchmarkRunner()
        assert runner.collector is not None

    def test_create_with_collector(self) -> None:
        """Test creating with existing collector."""
        collector = MetricCollector()
        runner = BenchmarkRunner(collector=collector)
        assert runner.collector is collector

    @pytest.mark.asyncio
    async def test_benchmark_streaming(self) -> None:
        """Test streaming benchmark."""

        async def mock_stream() -> AsyncIterator[str]:
            await asyncio.sleep(0.01)  # 10ms delay
            yield "token1"
            yield "token2"

        runner = BenchmarkRunner()
        result = await runner.benchmark_streaming(mock_stream, iterations=3, warmup=1)

        assert result.name == "time_to_first_token"
        assert result.iterations == 3
        assert result.avg_ms >= 10  # At least 10ms

    @pytest.mark.asyncio
    async def test_benchmark_task_latency(self) -> None:
        """Test task latency benchmark."""

        async def mock_task() -> None:
            await asyncio.sleep(0.01)  # 10ms

        runner = BenchmarkRunner()
        result = await runner.benchmark_task_latency(mock_task, iterations=3, warmup=1)

        assert result.name == "simple_task_latency"
        assert result.iterations == 3
        assert result.avg_ms >= 10

    @pytest.mark.asyncio
    async def test_benchmark_verification(self) -> None:
        """Test verification benchmark."""

        async def mock_verify() -> None:
            await asyncio.sleep(0.01)  # 10ms

        runner = BenchmarkRunner()
        result = await runner.benchmark_verification(mock_verify, iterations=3, warmup=1)

        assert result.name == "verification_time"
        assert result.iterations == 3
        assert result.avg_ms >= 10


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_collector(self) -> None:
        """Test create_collector factory."""
        collector = create_collector()
        assert isinstance(collector, MetricCollector)

    def test_create_validator(self) -> None:
        """Test create_validator factory."""
        validator = create_validator()
        assert isinstance(validator, Phase1Validator)

    def test_create_validator_with_collector(self) -> None:
        """Test create_validator with collector."""
        collector = MetricCollector()
        validator = create_validator(collector=collector)
        assert validator is not None

    def test_create_benchmark_runner(self) -> None:
        """Test create_benchmark_runner factory."""
        runner = create_benchmark_runner()
        assert isinstance(runner, BenchmarkRunner)

    def test_validate_phase1_targets(self) -> None:
        """Test quick validation function."""
        report = validate_phase1_targets(
            time_to_first_token_ms=400.0,
            simple_task_latency_ms=500.0,
            context_build_time_ms=0.0,
            verification_time_ms=1500.0,
            cost_per_task=0.015,
        )
        assert report.overall_passed is True
        assert report.passed_count == 5

    def test_validate_phase1_targets_partial(self) -> None:
        """Test partial validation."""
        report = validate_phase1_targets(
            verification_time_ms=1500.0,
        )
        # Only verification should pass, others will fail (no value = inf)
        assert report.passed_count >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for Phase 1 metrics."""

    @pytest.mark.asyncio
    async def test_full_workflow(self) -> None:
        """Test complete metrics collection and validation workflow."""
        # Create collector
        collector = create_collector()

        # Simulate collecting metrics
        async with collector.measure(MetricType.TIME_TO_FIRST_TOKEN):
            await asyncio.sleep(0.01)

        async with collector.measure(MetricType.VERIFICATION_TIME):
            await asyncio.sleep(0.01)

        collector.record_cost(0.015, model="gpt-4o-mini")

        # Validate
        validator = create_validator(collector)
        # Override unmeasured metrics
        validator.set_override(MetricType.SIMPLE_TASK_LATENCY, 500.0)
        validator.set_override(MetricType.CONTEXT_BUILD_TIME, 0.0)

        report = validator.validate_all()

        # Should have results for all metrics
        assert len(report.results) == 5
        # Summary should be generated
        assert "Phase 1 Validation Report" in report.summary()

    def test_phase1_complete_scenario(self) -> None:
        """Test Phase 1 complete validation scenario.

        This simulates what the final Phase 1 validation would look like
        with all targets met.
        """
        report = validate_phase1_targets(
            time_to_first_token_ms=300.0,  # Target: <500ms
            simple_task_latency_ms=450.0,  # Target: <1000ms
            context_build_time_ms=0.0,  # Target: 0ms (non-blocking)
            verification_time_ms=1200.0,  # Target: <2000ms
            cost_per_task=0.012,  # Target: <$0.02
        )

        assert report.overall_passed is True
        assert report.passed_count == 5
        assert report.failed_count == 0

        # Verify markdown output for documentation
        table = report.markdown_table()
        assert "[x]" in table
        assert "[ ]" not in table
