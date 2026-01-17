"""Tests for task completion benchmark module.

Tests Phase 12.3 of EXECUTOR_4.md - Task Completion Benchmark.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from ai_infra.benchmarks.task_completion import (
    COMPLEX_TASK,
    MEDIUM_TASK,
    SIMPLE_TASK,
    STANDARD_TASKS,
    StandardTask,
    TaskCompletionBenchmark,
    TaskCompletionResult,
    TaskCompletionTarget,
    TaskComplexity,
    compare_task_results,
    get_target_for_complexity,
    summarize_all_results,
)

# =============================================================================
# TaskComplexity Tests
# =============================================================================


class TestTaskComplexity:
    """Tests for TaskComplexity enum."""

    def test_enum_values(self) -> None:
        """Test enum has correct values."""
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MEDIUM.value == "medium"
        assert TaskComplexity.COMPLEX.value == "complex"

    def test_target_seconds_simple(self) -> None:
        """Test simple task target is 30 seconds."""
        assert TaskComplexity.SIMPLE.target_seconds == 30.0

    def test_target_seconds_medium(self) -> None:
        """Test medium task target is 60 seconds."""
        assert TaskComplexity.MEDIUM.target_seconds == 60.0

    def test_target_seconds_complex(self) -> None:
        """Test complex task target is 180 seconds."""
        assert TaskComplexity.COMPLEX.target_seconds == 180.0

    def test_description_exists(self) -> None:
        """Test all complexity levels have descriptions."""
        for complexity in TaskComplexity:
            assert complexity.description != ""
            assert isinstance(complexity.description, str)


# =============================================================================
# TaskCompletionTarget Tests
# =============================================================================


class TestTaskCompletionTarget:
    """Tests for TaskCompletionTarget dataclass."""

    def test_target_creation(self) -> None:
        """Test target creation with valid values."""
        target = TaskCompletionTarget(
            complexity=TaskComplexity.SIMPLE,
            target_seconds=30.0,
            warning_seconds=20.0,
        )

        assert target.complexity == TaskComplexity.SIMPLE
        assert target.target_seconds == 30.0
        assert target.warning_seconds == 20.0

    def test_passes_when_under_target(self) -> None:
        """Test passes() returns True when under target."""
        target = TaskCompletionTarget(
            complexity=TaskComplexity.SIMPLE,
            target_seconds=30.0,
        )

        assert target.passes(25.0) is True
        assert target.passes(30.0) is True

    def test_passes_when_over_target(self) -> None:
        """Test passes() returns False when over target."""
        target = TaskCompletionTarget(
            complexity=TaskComplexity.SIMPLE,
            target_seconds=30.0,
        )

        assert target.passes(31.0) is False
        assert target.passes(100.0) is False

    def test_is_warning_in_warning_zone(self) -> None:
        """Test is_warning() returns True in warning zone."""
        target = TaskCompletionTarget(
            complexity=TaskComplexity.SIMPLE,
            target_seconds=30.0,
            warning_seconds=20.0,
        )

        assert target.is_warning(25.0) is True
        assert target.is_warning(20.0) is True

    def test_is_warning_outside_warning_zone(self) -> None:
        """Test is_warning() returns False outside warning zone."""
        target = TaskCompletionTarget(
            complexity=TaskComplexity.SIMPLE,
            target_seconds=30.0,
            warning_seconds=20.0,
        )

        assert target.is_warning(15.0) is False
        assert target.is_warning(35.0) is False

    def test_is_warning_no_warning_set(self) -> None:
        """Test is_warning() returns False when no warning set."""
        target = TaskCompletionTarget(
            complexity=TaskComplexity.SIMPLE,
            target_seconds=30.0,
        )

        assert target.is_warning(25.0) is False


class TestGetTargetForComplexity:
    """Tests for get_target_for_complexity function."""

    def test_returns_correct_target_simple(self) -> None:
        """Test returns correct target for simple complexity."""
        target = get_target_for_complexity(TaskComplexity.SIMPLE)

        assert target.complexity == TaskComplexity.SIMPLE
        assert target.target_seconds == 30.0

    def test_returns_correct_target_medium(self) -> None:
        """Test returns correct target for medium complexity."""
        target = get_target_for_complexity(TaskComplexity.MEDIUM)

        assert target.complexity == TaskComplexity.MEDIUM
        assert target.target_seconds == 60.0

    def test_returns_correct_target_complex(self) -> None:
        """Test returns correct target for complex complexity."""
        target = get_target_for_complexity(TaskComplexity.COMPLEX)

        assert target.complexity == TaskComplexity.COMPLEX
        assert target.target_seconds == 180.0


# =============================================================================
# StandardTask Tests
# =============================================================================


class TestStandardTask:
    """Tests for StandardTask dataclass."""

    def test_task_creation(self) -> None:
        """Test task creation with required fields."""
        task = StandardTask(
            name="test_task",
            complexity=TaskComplexity.SIMPLE,
            roadmap_content="# Test\n- [ ] Do something",
        )

        assert task.name == "test_task"
        assert task.complexity == TaskComplexity.SIMPLE
        assert "# Test" in task.roadmap_content

    def test_task_target_property(self) -> None:
        """Test target property returns correct target."""
        task = StandardTask(
            name="test",
            complexity=TaskComplexity.MEDIUM,
            roadmap_content="# Test",
        )

        assert task.target.complexity == TaskComplexity.MEDIUM
        assert task.target.target_seconds == 60.0

    def test_task_to_dict(self) -> None:
        """Test to_dict() serializes correctly."""
        task = StandardTask(
            name="test_task",
            complexity=TaskComplexity.SIMPLE,
            roadmap_content="# Test",
            expected_files=("file1.py", "file2.py"),
            description="Test description",
        )

        d = task.to_dict()

        assert d["name"] == "test_task"
        assert d["complexity"] == "simple"
        assert d["target_seconds"] == 30.0
        assert d["expected_files"] == ["file1.py", "file2.py"]
        assert d["description"] == "Test description"


class TestStandardTasks:
    """Tests for predefined standard tasks."""

    def test_simple_task_definition(self) -> None:
        """Test SIMPLE_TASK is correctly defined."""
        assert SIMPLE_TASK.name == "simple_hello_world"
        assert SIMPLE_TASK.complexity == TaskComplexity.SIMPLE
        assert "hello.py" in SIMPLE_TASK.expected_files
        assert "Hello, World!" in SIMPLE_TASK.roadmap_content

    def test_medium_task_definition(self) -> None:
        """Test MEDIUM_TASK is correctly defined."""
        assert MEDIUM_TASK.name == "medium_calculator"
        assert MEDIUM_TASK.complexity == TaskComplexity.MEDIUM
        assert "calculator.py" in MEDIUM_TASK.expected_files
        assert "test_calculator.py" in MEDIUM_TASK.expected_files

    def test_complex_task_definition(self) -> None:
        """Test COMPLEX_TASK is correctly defined."""
        assert COMPLEX_TASK.name == "complex_fastapi_app"
        assert COMPLEX_TASK.complexity == TaskComplexity.COMPLEX
        assert "FastAPI" in COMPLEX_TASK.roadmap_content
        assert "JWT" in COMPLEX_TASK.roadmap_content

    def test_standard_tasks_contains_all(self) -> None:
        """Test STANDARD_TASKS contains all three tasks."""
        assert len(STANDARD_TASKS) == 3
        assert SIMPLE_TASK in STANDARD_TASKS
        assert MEDIUM_TASK in STANDARD_TASKS
        assert COMPLEX_TASK in STANDARD_TASKS


# =============================================================================
# TaskCompletionResult Tests
# =============================================================================


class TestTaskCompletionResult:
    """Tests for TaskCompletionResult dataclass."""

    def test_result_creation(self) -> None:
        """Test result creation with required fields."""
        result = TaskCompletionResult(
            task_name="test_task",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=25.0,
            target_seconds=30.0,
        )

        assert result.task_name == "test_task"
        assert result.elapsed_seconds == 25.0
        assert result.status == "completed"

    def test_passes_target_when_under(self) -> None:
        """Test passes_target is True when under target."""
        result = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=25.0,
            target_seconds=30.0,
            status="completed",
        )

        assert result.passes_target is True

    def test_passes_target_when_over(self) -> None:
        """Test passes_target is False when over target."""
        result = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=35.0,
            target_seconds=30.0,
            status="completed",
        )

        assert result.passes_target is False

    def test_passes_target_when_failed(self) -> None:
        """Test passes_target is False when status is not completed."""
        result = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=25.0,
            target_seconds=30.0,
            status="failed",
        )

        assert result.passes_target is False

    def test_margin_seconds_positive(self) -> None:
        """Test margin_seconds is positive when under target."""
        result = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=20.0,
            target_seconds=30.0,
        )

        assert result.margin_seconds == 10.0

    def test_margin_seconds_negative(self) -> None:
        """Test margin_seconds is negative when over target."""
        result = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=40.0,
            target_seconds=30.0,
        )

        assert result.margin_seconds == -10.0

    def test_margin_percent(self) -> None:
        """Test margin_percent calculation."""
        result = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=15.0,
            target_seconds=30.0,
        )

        assert result.margin_percent == 50.0  # 15s under 30s = 50%

    def test_to_dict(self) -> None:
        """Test to_dict() serializes correctly."""
        result = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=25.0,
            target_seconds=30.0,
            files_created=["file1.py"],
        )

        d = result.to_dict()

        assert d["task_name"] == "test"
        assert d["complexity"] == "simple"
        assert d["elapsed_seconds"] == 25.0
        assert d["passes_target"] is True
        assert d["files_created"] == ["file1.py"]

    def test_str_format(self) -> None:
        """Test __str__() produces readable output."""
        result = TaskCompletionResult(
            task_name="test_task",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=25.0,
            target_seconds=30.0,
        )

        output = str(result)

        assert "test_task" in output
        assert "simple" in output
        assert "25.0s" in output
        assert "30s" in output

    def test_timestamp_auto_set(self) -> None:
        """Test timestamp is automatically set."""
        result = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=25.0,
            target_seconds=30.0,
        )

        assert result.timestamp != ""
        assert "T" in result.timestamp  # ISO format


# =============================================================================
# TaskCompletionBenchmark Tests
# =============================================================================


class TestTaskCompletionBenchmark:
    """Tests for TaskCompletionBenchmark class."""

    def test_init_default(self) -> None:
        """Test initialization with defaults."""
        benchmark = TaskCompletionBenchmark()

        assert benchmark.timeout == 300.0

    def test_init_custom_timeout(self) -> None:
        """Test initialization with custom timeout."""
        benchmark = TaskCompletionBenchmark(timeout=60.0)

        assert benchmark.timeout == 60.0

    @pytest.mark.asyncio
    async def test_run_simulated(self) -> None:
        """Test run_simulated executes and returns result."""
        benchmark = TaskCompletionBenchmark()

        result = await benchmark.run_simulated(SIMPLE_TASK, sim_time=20.0)

        assert result.task_name == "simple_hello_world"
        assert result.elapsed_seconds == 20.0
        assert result.passes_target is True

    @pytest.mark.asyncio
    async def test_run_simulated_default_time(self) -> None:
        """Test run_simulated uses 80% of target by default."""
        benchmark = TaskCompletionBenchmark()

        result = await benchmark.run_simulated(SIMPLE_TASK)

        # Default is 80% of 30s = 24s
        assert result.elapsed_seconds == 24.0

    @pytest.mark.asyncio
    async def test_run_simulated_marks_as_simulated(self) -> None:
        """Test run_simulated adds simulated metadata."""
        benchmark = TaskCompletionBenchmark()

        result = await benchmark.run_simulated(SIMPLE_TASK)

        assert result.metadata.get("simulated") is True

    @pytest.mark.asyncio
    async def test_run_with_executor(self) -> None:
        """Test run() with mock executor function."""
        benchmark = TaskCompletionBenchmark()

        async def mock_executor(roadmap_path: str) -> dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"status": "completed"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await benchmark.run(
                SIMPLE_TASK,
                executor_fn=mock_executor,
                work_dir=Path(tmp_dir),
            )

        assert result.task_name == "simple_hello_world"
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_run_handles_timeout(self) -> None:
        """Test run() handles executor timeout."""
        benchmark = TaskCompletionBenchmark(timeout=0.1)

        async def slow_executor(roadmap_path: str) -> dict[str, Any]:
            await asyncio.sleep(10.0)
            return {"status": "completed"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await benchmark.run(
                SIMPLE_TASK,
                executor_fn=slow_executor,
                work_dir=Path(tmp_dir),
            )

        assert result.status == "timeout"

    @pytest.mark.asyncio
    async def test_run_handles_exception(self) -> None:
        """Test run() handles executor exception."""
        benchmark = TaskCompletionBenchmark()

        async def failing_executor(roadmap_path: str) -> dict[str, Any]:
            raise RuntimeError("Test error")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await benchmark.run(
                SIMPLE_TASK,
                executor_fn=failing_executor,
                work_dir=Path(tmp_dir),
            )

        assert "failed" in result.status
        assert "Test error" in result.status

    @pytest.mark.asyncio
    async def test_run_all_standard_simulated(self) -> None:
        """Test run_all_standard with simulated execution."""
        benchmark = TaskCompletionBenchmark()

        results = await benchmark.run_all_standard(simulated=True)

        assert len(results) == 3
        assert results[0].task_name == "simple_hello_world"
        assert results[1].task_name == "medium_calculator"
        assert results[2].task_name == "complex_fastapi_app"

    @pytest.mark.asyncio
    async def test_run_all_standard_requires_executor(self) -> None:
        """Test run_all_standard requires executor_fn when not simulated."""
        benchmark = TaskCompletionBenchmark()

        with pytest.raises(ValueError, match="executor_fn required"):
            await benchmark.run_all_standard(simulated=False)

    @pytest.mark.asyncio
    async def test_get_results(self) -> None:
        """Test get_results returns all results."""
        benchmark = TaskCompletionBenchmark()

        await benchmark.run_simulated(SIMPLE_TASK)
        await benchmark.run_simulated(MEDIUM_TASK)

        results = benchmark.get_results()

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_summary(self) -> None:
        """Test summary produces readable output."""
        benchmark = TaskCompletionBenchmark()

        await benchmark.run_simulated(SIMPLE_TASK, sim_time=20.0)
        await benchmark.run_simulated(MEDIUM_TASK, sim_time=50.0)

        summary = benchmark.summary()

        assert "Task Completion Benchmark Results" in summary
        assert "simple_hello_world" in summary
        assert "medium_calculator" in summary

    def test_summary_no_results(self) -> None:
        """Test summary handles no results."""
        benchmark = TaskCompletionBenchmark()

        summary = benchmark.summary()

        assert summary == "No results"


# =============================================================================
# Comparison Utility Tests
# =============================================================================


class TestCompareTaskResults:
    """Tests for compare_task_results function."""

    def test_compare_improvement(self) -> None:
        """Test comparison detects improvement."""
        baseline = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=25.0,
            target_seconds=30.0,
        )
        current = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=20.0,
            target_seconds=30.0,
        )

        comparison = compare_task_results(baseline, current)

        assert comparison["is_improvement"] is True
        assert comparison["diff_seconds"] == -5.0

    def test_compare_regression(self) -> None:
        """Test comparison detects regression."""
        baseline = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=25.0,
            target_seconds=30.0,
        )
        current = TaskCompletionResult(
            task_name="test",
            complexity=TaskComplexity.SIMPLE,
            elapsed_seconds=35.0,  # Over target
            target_seconds=30.0,
        )

        comparison = compare_task_results(baseline, current)

        assert comparison["is_regression"] is True
        assert comparison["diff_seconds"] == 10.0


class TestSummarizeAllResults:
    """Tests for summarize_all_results function."""

    def test_summarize_empty(self) -> None:
        """Test summarize handles empty list."""
        summary = summarize_all_results([])

        assert summary["total"] == 0
        assert summary["passed"] == 0

    def test_summarize_all_passed(self) -> None:
        """Test summarize with all passing results."""
        results = [
            TaskCompletionResult(
                task_name="test1",
                complexity=TaskComplexity.SIMPLE,
                elapsed_seconds=20.0,
                target_seconds=30.0,
            ),
            TaskCompletionResult(
                task_name="test2",
                complexity=TaskComplexity.MEDIUM,
                elapsed_seconds=50.0,
                target_seconds=60.0,
            ),
        ]

        summary = summarize_all_results(results)

        assert summary["total"] == 2
        assert summary["passed"] == 2
        assert summary["failed"] == 0
        assert summary["pass_rate"] == 100.0

    def test_summarize_by_complexity(self) -> None:
        """Test summarize groups by complexity."""
        results = [
            TaskCompletionResult(
                task_name="s1",
                complexity=TaskComplexity.SIMPLE,
                elapsed_seconds=20.0,
                target_seconds=30.0,
            ),
            TaskCompletionResult(
                task_name="s2",
                complexity=TaskComplexity.SIMPLE,
                elapsed_seconds=25.0,
                target_seconds=30.0,
            ),
            TaskCompletionResult(
                task_name="m1",
                complexity=TaskComplexity.MEDIUM,
                elapsed_seconds=50.0,
                target_seconds=60.0,
            ),
        ]

        summary = summarize_all_results(results)

        assert "simple" in summary["by_complexity"]
        assert summary["by_complexity"]["simple"]["total"] == 2
        assert summary["by_complexity"]["simple"]["avg_seconds"] == 22.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestTaskCompletionIntegration:
    """Integration tests for task completion benchmarking."""

    @pytest.mark.asyncio
    async def test_full_benchmark_workflow(self) -> None:
        """Test complete benchmark -> summarize workflow."""
        benchmark = TaskCompletionBenchmark()

        # Run all standard tasks simulated
        results = await benchmark.run_all_standard(simulated=True)

        # Verify results
        assert len(results) == 3
        for result in results:
            assert result.passes_target is True

        # Get summary
        summary = benchmark.summary()
        assert "3/3 passed" in summary

    @pytest.mark.asyncio
    async def test_parametrized_task_benchmark(self) -> None:
        """Test parametrized benchmarking like pytest.mark.parametrize."""
        benchmark = TaskCompletionBenchmark()

        test_cases = [
            (SIMPLE_TASK, 30),
            (MEDIUM_TASK, 60),
            (COMPLEX_TASK, 180),
        ]

        for task, target_seconds in test_cases:
            result = await benchmark.run_simulated(task)

            assert result.target_seconds == target_seconds
            assert result.passes_target is True

    @pytest.mark.asyncio
    async def test_comparison_workflow(self) -> None:
        """Test baseline vs current comparison workflow."""
        # Simulate baseline run
        baseline_results = [
            TaskCompletionResult(
                task_name="test",
                complexity=TaskComplexity.SIMPLE,
                elapsed_seconds=25.0,
                target_seconds=30.0,
            ),
        ]

        # Simulate current run (faster)
        current_results = [
            TaskCompletionResult(
                task_name="test",
                complexity=TaskComplexity.SIMPLE,
                elapsed_seconds=20.0,
                target_seconds=30.0,
            ),
        ]

        # Compare
        comparison = compare_task_results(baseline_results[0], current_results[0])

        assert comparison["is_improvement"] is True
        assert comparison["diff_percent"] == -20.0  # 20% faster
