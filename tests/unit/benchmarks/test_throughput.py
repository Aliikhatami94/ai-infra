"""Tests for throughput benchmark module.

Tests Phase 12.4 - Throughput Benchmark.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from ai_infra.benchmarks.throughput import (
    STANDARD_WORKLOADS,
    THROUGHPUT_TARGET_BASELINE,
    THROUGHPUT_TARGET_BATCH,
    THROUGHPUT_TARGET_FAST,
    WORKLOAD_LARGE,
    WORKLOAD_MEDIUM,
    WORKLOAD_SMALL,
    ThroughputBenchmark,
    ThroughputResult,
    ThroughputTarget,
    WorkloadConfig,
    compare_throughput_results,
    generate_multi_task_roadmap,
    generate_varied_task_roadmap,
    summarize_throughput_results,
)

# =============================================================================
# ThroughputTarget Tests
# =============================================================================


class TestThroughputTarget:
    """Tests for ThroughputTarget dataclass."""

    def test_target_creation(self) -> None:
        """Test target creation with valid values."""
        target = ThroughputTarget(
            name="Test Target",
            min_tasks_per_minute=5.0,
            target_tasks_per_minute=10.0,
        )

        assert target.name == "Test Target"
        assert target.min_tasks_per_minute == 5.0
        assert target.target_tasks_per_minute == 10.0

    def test_passes_when_above_minimum(self) -> None:
        """Test passes() returns True when above minimum."""
        target = ThroughputTarget("Test", 5.0, 10.0)

        assert target.passes(5.0) is True
        assert target.passes(7.0) is True
        assert target.passes(15.0) is True

    def test_passes_when_below_minimum(self) -> None:
        """Test passes() returns False when below minimum."""
        target = ThroughputTarget("Test", 5.0, 10.0)

        assert target.passes(4.9) is False
        assert target.passes(2.0) is False

    def test_exceeds_target(self) -> None:
        """Test exceeds_target() check."""
        target = ThroughputTarget("Test", 5.0, 10.0)

        assert target.exceeds_target(10.0) is True
        assert target.exceeds_target(15.0) is True
        assert target.exceeds_target(9.0) is False


class TestDefaultTargets:
    """Tests for predefined throughput targets."""

    def test_baseline_target(self) -> None:
        """Test THROUGHPUT_TARGET_BASELINE values."""
        assert THROUGHPUT_TARGET_BASELINE.min_tasks_per_minute == 2.0
        assert THROUGHPUT_TARGET_BASELINE.target_tasks_per_minute == 5.0

    def test_fast_target(self) -> None:
        """Test THROUGHPUT_TARGET_FAST values."""
        assert THROUGHPUT_TARGET_FAST.min_tasks_per_minute == 5.0
        assert THROUGHPUT_TARGET_FAST.target_tasks_per_minute == 10.0

    def test_batch_target(self) -> None:
        """Test THROUGHPUT_TARGET_BATCH values."""
        assert THROUGHPUT_TARGET_BATCH.min_tasks_per_minute == 10.0
        assert THROUGHPUT_TARGET_BATCH.target_tasks_per_minute == 20.0


# =============================================================================
# WorkloadConfig Tests
# =============================================================================


class TestWorkloadConfig:
    """Tests for WorkloadConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test workload config creation."""
        config = WorkloadConfig(
            name="Test Workload",
            task_count=10,
            task_template="Create test{n}.py",
        )

        assert config.name == "Test Workload"
        assert config.task_count == 10

    def test_generate_roadmap(self) -> None:
        """Test generate_roadmap produces valid roadmap."""
        config = WorkloadConfig(
            name="Test",
            task_count=3,
            task_template="Create file{n}.py",
        )

        roadmap = config.generate_roadmap()

        assert "# Test" in roadmap
        assert "Create file1.py" in roadmap
        assert "Create file2.py" in roadmap
        assert "Create file3.py" in roadmap


class TestStandardWorkloads:
    """Tests for predefined workloads."""

    def test_small_workload(self) -> None:
        """Test WORKLOAD_SMALL configuration."""
        assert WORKLOAD_SMALL.task_count == 5
        assert WORKLOAD_SMALL.name == "Small Workload"

    def test_medium_workload(self) -> None:
        """Test WORKLOAD_MEDIUM configuration."""
        assert WORKLOAD_MEDIUM.task_count == 10
        assert WORKLOAD_MEDIUM.name == "Medium Workload"

    def test_large_workload(self) -> None:
        """Test WORKLOAD_LARGE configuration."""
        assert WORKLOAD_LARGE.task_count == 20
        assert WORKLOAD_LARGE.name == "Large Workload"

    def test_standard_workloads_contains_all(self) -> None:
        """Test STANDARD_WORKLOADS contains all workloads."""
        assert len(STANDARD_WORKLOADS) == 3
        assert WORKLOAD_SMALL in STANDARD_WORKLOADS
        assert WORKLOAD_MEDIUM in STANDARD_WORKLOADS
        assert WORKLOAD_LARGE in STANDARD_WORKLOADS


# =============================================================================
# Roadmap Generation Tests
# =============================================================================


class TestGenerateMultiTaskRoadmap:
    """Tests for generate_multi_task_roadmap function."""

    def test_generates_correct_task_count(self) -> None:
        """Test generates correct number of tasks."""
        roadmap = generate_multi_task_roadmap(5)

        assert roadmap.count("- [ ]") == 5

    def test_generates_numbered_tasks(self) -> None:
        """Test tasks are numbered correctly."""
        roadmap = generate_multi_task_roadmap(3)

        assert "file1.py" in roadmap
        assert "file2.py" in roadmap
        assert "file3.py" in roadmap

    def test_custom_template(self) -> None:
        """Test custom task template."""
        roadmap = generate_multi_task_roadmap(
            task_count=2,
            task_template="Write module{n}.py",
        )

        assert "Write module1.py" in roadmap
        assert "Write module2.py" in roadmap

    def test_custom_title(self) -> None:
        """Test custom roadmap title."""
        roadmap = generate_multi_task_roadmap(
            task_count=1,
            title="My Custom Test",
        )

        assert "# My Custom Test" in roadmap


class TestGenerateVariedTaskRoadmap:
    """Tests for generate_varied_task_roadmap function."""

    def test_generates_varied_tasks(self) -> None:
        """Test generates different task types."""
        roadmap = generate_varied_task_roadmap(5)

        assert "module" in roadmap or "test" in roadmap or "config" in roadmap

    def test_generates_correct_count(self) -> None:
        """Test generates correct number of tasks."""
        roadmap = generate_varied_task_roadmap(10)

        assert roadmap.count("- [ ]") == 10


# =============================================================================
# ThroughputResult Tests
# =============================================================================


class TestThroughputResult:
    """Tests for ThroughputResult dataclass."""

    def test_result_creation(self) -> None:
        """Test result creation with required fields."""
        result = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=10,
            elapsed_seconds=60.0,
            tasks_per_minute=10.0,
            tasks_per_second=0.167,
            avg_task_seconds=6.0,
        )

        assert result.workload_name == "Test"
        assert result.task_count == 10
        assert result.tasks_per_minute == 10.0

    def test_completion_rate_full(self) -> None:
        """Test completion_rate when all tasks complete."""
        result = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=10,
            elapsed_seconds=60.0,
            tasks_per_minute=10.0,
            tasks_per_second=0.167,
            avg_task_seconds=6.0,
        )

        assert result.completion_rate == 100.0

    def test_completion_rate_partial(self) -> None:
        """Test completion_rate with partial completion."""
        result = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=7,
            elapsed_seconds=60.0,
            tasks_per_minute=7.0,
            tasks_per_second=0.117,
            avg_task_seconds=8.57,
        )

        assert result.completion_rate == 70.0

    def test_is_complete(self) -> None:
        """Test is_complete property."""
        complete = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=10,
            elapsed_seconds=60.0,
            tasks_per_minute=10.0,
            tasks_per_second=0.167,
            avg_task_seconds=6.0,
        )

        partial = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=5,
            elapsed_seconds=60.0,
            tasks_per_minute=5.0,
            tasks_per_second=0.083,
            avg_task_seconds=12.0,
        )

        assert complete.is_complete is True
        assert partial.is_complete is False

    def test_passes_target(self) -> None:
        """Test passes_target method."""
        result = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=10,
            elapsed_seconds=60.0,
            tasks_per_minute=5.0,
            tasks_per_second=0.083,
            avg_task_seconds=6.0,
        )

        assert result.passes_target(THROUGHPUT_TARGET_BASELINE) is True
        assert result.passes_target(THROUGHPUT_TARGET_FAST) is True
        assert result.passes_target(THROUGHPUT_TARGET_BATCH) is False

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        result = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=10,
            elapsed_seconds=60.0,
            tasks_per_minute=10.0,
            tasks_per_second=0.167,
            avg_task_seconds=6.0,
        )

        d = result.to_dict()

        assert d["workload_name"] == "Test"
        assert d["task_count"] == 10
        assert d["tasks_per_minute"] == 10.0
        assert d["completion_rate"] == 100.0

    def test_str_format(self) -> None:
        """Test __str__ produces readable output."""
        result = ThroughputResult(
            workload_name="Test Workload",
            task_count=10,
            tasks_completed=10,
            elapsed_seconds=60.0,
            tasks_per_minute=10.0,
            tasks_per_second=0.167,
            avg_task_seconds=6.0,
        )

        output = str(result)

        assert "Test Workload" in output
        assert "10/10" in output
        assert "tasks/min" in output

    def test_timestamp_auto_set(self) -> None:
        """Test timestamp is automatically set."""
        result = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=10,
            elapsed_seconds=60.0,
            tasks_per_minute=10.0,
            tasks_per_second=0.167,
            avg_task_seconds=6.0,
        )

        assert result.timestamp != ""
        assert "T" in result.timestamp


# =============================================================================
# ThroughputBenchmark Tests
# =============================================================================


class TestThroughputBenchmark:
    """Tests for ThroughputBenchmark class."""

    def test_init_default(self) -> None:
        """Test initialization with defaults."""
        benchmark = ThroughputBenchmark()

        assert benchmark.target == THROUGHPUT_TARGET_BASELINE
        assert benchmark.timeout == 600.0

    def test_init_custom_target(self) -> None:
        """Test initialization with custom target."""
        benchmark = ThroughputBenchmark(target=THROUGHPUT_TARGET_FAST)

        assert benchmark.target == THROUGHPUT_TARGET_FAST

    @pytest.mark.asyncio
    async def test_run_simulated(self) -> None:
        """Test run_simulated executes and returns result."""
        benchmark = ThroughputBenchmark()

        result = await benchmark.run_simulated(task_count=10)

        assert result.task_count == 10
        assert result.tasks_completed == 10
        assert result.tasks_per_minute > 0

    @pytest.mark.asyncio
    async def test_run_simulated_custom_time(self) -> None:
        """Test run_simulated with custom time per task."""
        benchmark = ThroughputBenchmark()

        result = await benchmark.run_simulated(
            task_count=10,
            sim_seconds_per_task=6.0,  # 6s per task = 10 tasks/min
        )

        assert result.avg_task_seconds == 6.0
        assert result.tasks_per_minute == 10.0

    @pytest.mark.asyncio
    async def test_run_simulated_partial_completion(self) -> None:
        """Test run_simulated with partial completion."""
        benchmark = ThroughputBenchmark()

        result = await benchmark.run_simulated(
            task_count=10,
            sim_completion_rate=0.5,
        )

        assert result.tasks_completed == 5
        assert result.status == "partial"

    @pytest.mark.asyncio
    async def test_run_simulated_marks_metadata(self) -> None:
        """Test run_simulated adds simulated metadata."""
        benchmark = ThroughputBenchmark()

        result = await benchmark.run_simulated(task_count=5)

        assert result.metadata.get("simulated") is True

    @pytest.mark.asyncio
    async def test_run_with_executor(self) -> None:
        """Test run() with mock executor function."""
        benchmark = ThroughputBenchmark()

        async def mock_executor(roadmap_path: str) -> dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"status": "completed", "tasks_completed": 10}

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await benchmark.run(
                task_count=10,
                executor_fn=mock_executor,
                work_dir=Path(tmp_dir),
            )

        assert result.task_count == 10
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_run_handles_timeout(self) -> None:
        """Test run() handles executor timeout."""
        benchmark = ThroughputBenchmark(timeout=0.1)

        async def slow_executor(roadmap_path: str) -> dict[str, Any]:
            await asyncio.sleep(10.0)
            return {"status": "completed"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await benchmark.run(
                task_count=10,
                executor_fn=slow_executor,
                work_dir=Path(tmp_dir),
            )

        assert result.status == "timeout"

    @pytest.mark.asyncio
    async def test_run_handles_exception(self) -> None:
        """Test run() handles executor exception."""
        benchmark = ThroughputBenchmark()

        async def failing_executor(roadmap_path: str) -> dict[str, Any]:
            raise RuntimeError("Test error")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await benchmark.run(
                task_count=10,
                executor_fn=failing_executor,
                work_dir=Path(tmp_dir),
            )

        assert "failed" in result.status

    @pytest.mark.asyncio
    async def test_run_workload_simulated(self) -> None:
        """Test run_workload with simulated execution."""
        benchmark = ThroughputBenchmark()

        result = await benchmark.run_workload(WORKLOAD_SMALL, simulated=True)

        assert result.task_count == 5
        assert result.workload_name == "Small Workload"

    @pytest.mark.asyncio
    async def test_run_workload_requires_executor(self) -> None:
        """Test run_workload requires executor when not simulated."""
        benchmark = ThroughputBenchmark()

        with pytest.raises(ValueError, match="executor_fn required"):
            await benchmark.run_workload(WORKLOAD_SMALL, simulated=False)

    @pytest.mark.asyncio
    async def test_run_all_workloads_simulated(self) -> None:
        """Test run_all_workloads with simulated execution."""
        benchmark = ThroughputBenchmark()

        results = await benchmark.run_all_workloads(simulated=True)

        assert len(results) == 3
        assert results[0].task_count == 5  # Small
        assert results[1].task_count == 10  # Medium
        assert results[2].task_count == 20  # Large

    @pytest.mark.asyncio
    async def test_run_scaling_test(self) -> None:
        """Test run_scaling_test with multiple task counts."""
        benchmark = ThroughputBenchmark()

        results = await benchmark.run_scaling_test(
            task_counts=[5, 10, 15],
            simulated=True,
        )

        assert len(results) == 3
        assert results[0].task_count == 5
        assert results[1].task_count == 10
        assert results[2].task_count == 15

    @pytest.mark.asyncio
    async def test_run_scaling_test_default_counts(self) -> None:
        """Test run_scaling_test with default task counts."""
        benchmark = ThroughputBenchmark()

        results = await benchmark.run_scaling_test(simulated=True)

        assert len(results) == 4  # Default: [5, 10, 20, 50]

    @pytest.mark.asyncio
    async def test_get_results(self) -> None:
        """Test get_results returns all results."""
        benchmark = ThroughputBenchmark()

        await benchmark.run_simulated(task_count=5)
        await benchmark.run_simulated(task_count=10)

        results = benchmark.get_results()

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_passes_target(self) -> None:
        """Test passes_target method."""
        benchmark = ThroughputBenchmark(target=THROUGHPUT_TARGET_BASELINE)

        result = await benchmark.run_simulated(
            task_count=10,
            sim_seconds_per_task=10.0,  # 6 tasks/min > 2 min
        )

        assert benchmark.passes_target(result) is True

    @pytest.mark.asyncio
    async def test_summary(self) -> None:
        """Test summary produces readable output."""
        benchmark = ThroughputBenchmark()

        await benchmark.run_simulated(task_count=5)
        await benchmark.run_simulated(task_count=10)

        summary = benchmark.summary()

        assert "Throughput Benchmark Results" in summary
        assert "tasks/min" in summary

    def test_summary_no_results(self) -> None:
        """Test summary handles no results."""
        benchmark = ThroughputBenchmark()

        summary = benchmark.summary()

        assert summary == "No results"


# =============================================================================
# Comparison Utility Tests
# =============================================================================


class TestCompareThroughputResults:
    """Tests for compare_throughput_results function."""

    def test_compare_improvement(self) -> None:
        """Test comparison detects improvement."""
        baseline = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=10,
            elapsed_seconds=120.0,
            tasks_per_minute=5.0,
            tasks_per_second=0.083,
            avg_task_seconds=12.0,
        )
        current = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=10,
            elapsed_seconds=60.0,
            tasks_per_minute=10.0,  # Faster
            tasks_per_second=0.167,
            avg_task_seconds=6.0,
        )

        comparison = compare_throughput_results(baseline, current)

        assert comparison["is_improvement"] is True
        assert comparison["throughput_diff"] == 5.0

    def test_compare_regression(self) -> None:
        """Test comparison detects regression."""
        baseline = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=10,
            elapsed_seconds=60.0,
            tasks_per_minute=10.0,
            tasks_per_second=0.167,
            avg_task_seconds=6.0,
        )
        current = ThroughputResult(
            workload_name="Test",
            task_count=10,
            tasks_completed=10,
            elapsed_seconds=300.0,
            tasks_per_minute=2.0,  # Slower by more than 10%
            tasks_per_second=0.033,
            avg_task_seconds=30.0,
        )

        comparison = compare_throughput_results(baseline, current)

        assert comparison["is_regression"] is True


class TestSummarizeThroughputResults:
    """Tests for summarize_throughput_results function."""

    def test_summarize_empty(self) -> None:
        """Test summarize handles empty list."""
        summary = summarize_throughput_results([])

        assert summary["total"] == 0

    def test_summarize_multiple(self) -> None:
        """Test summarize with multiple results."""
        results = [
            ThroughputResult(
                workload_name="Test1",
                task_count=10,
                tasks_completed=10,
                elapsed_seconds=60.0,
                tasks_per_minute=10.0,
                tasks_per_second=0.167,
                avg_task_seconds=6.0,
            ),
            ThroughputResult(
                workload_name="Test2",
                task_count=20,
                tasks_completed=20,
                elapsed_seconds=120.0,
                tasks_per_minute=10.0,
                tasks_per_second=0.167,
                avg_task_seconds=6.0,
            ),
        ]

        summary = summarize_throughput_results(results)

        assert summary["total"] == 2
        assert summary["total_tasks"] == 30
        assert summary["total_completed"] == 30
        assert summary["avg_throughput"] == 10.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestThroughputIntegration:
    """Integration tests for throughput benchmarking."""

    @pytest.mark.asyncio
    async def test_full_benchmark_workflow(self) -> None:
        """Test complete benchmark -> summarize workflow."""
        benchmark = ThroughputBenchmark(target=THROUGHPUT_TARGET_BASELINE)

        # Run all standard workloads
        results = await benchmark.run_all_workloads(simulated=True)

        # Verify results
        assert len(results) == 3
        for result in results:
            assert result.is_complete

        # Get summary
        summary = benchmark.summary()
        assert "Throughput Benchmark Results" in summary

    @pytest.mark.asyncio
    async def test_10_task_benchmark_per_spec(self) -> None:
        """Test 10-task benchmark as specified in EXECUTOR_4.md."""
        benchmark = ThroughputBenchmark()

        # Simulate 10-task workload per spec
        result = await benchmark.run_simulated(
            task_count=10,
            sim_seconds_per_task=20.0,  # 3 tasks/min
            workload_name="10-task throughput test",
        )

        # Should pass baseline target (>2 tasks/min)
        assert result.tasks_per_minute == 3.0
        assert result.passes_target(THROUGHPUT_TARGET_BASELINE)

    @pytest.mark.asyncio
    async def test_scaling_analysis(self) -> None:
        """Test throughput scaling analysis."""
        benchmark = ThroughputBenchmark()

        results = await benchmark.run_scaling_test(
            task_counts=[5, 10, 20],
            simulated=True,
        )

        summary = summarize_throughput_results(results)

        assert summary["total"] == 3
        assert summary["total_tasks"] == 35  # 5 + 10 + 20
