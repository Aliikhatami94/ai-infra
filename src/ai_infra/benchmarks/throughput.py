"""Throughput benchmark module.

Phase 12.4 of EXECUTOR_4.md - Throughput Benchmark.

This module provides benchmarking for multi-task throughput:
- ThroughputBenchmark for measuring tasks/minute
- ThroughputResult for storing throughput metrics
- Workload generators for different task counts

Target: Track tasks/minute for different workloads.

Example:
    >>> from ai_infra.benchmarks.throughput import (
    ...     ThroughputBenchmark,
    ...     generate_multi_task_roadmap,
    ... )
    >>>
    >>> benchmark = ThroughputBenchmark()
    >>> result = await benchmark.run(task_count=10, executor_fn=my_executor)
    >>> print(f"Throughput: {result.tasks_per_minute:.1f} tasks/minute")
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "ThroughputBenchmark",
    "ThroughputResult",
    "ThroughputTarget",
    "WorkloadConfig",
    "compare_throughput_results",
    "generate_multi_task_roadmap",
]


# =============================================================================
# Throughput Target
# =============================================================================


@dataclass(frozen=True, slots=True)
class ThroughputTarget:
    """Target throughput for a workload.

    Attributes:
        name: Target name/description.
        min_tasks_per_minute: Minimum acceptable throughput.
        target_tasks_per_minute: Target throughput.
    """

    name: str
    min_tasks_per_minute: float
    target_tasks_per_minute: float

    def passes(self, tasks_per_minute: float) -> bool:
        """Check if throughput meets minimum."""
        return tasks_per_minute >= self.min_tasks_per_minute

    def exceeds_target(self, tasks_per_minute: float) -> bool:
        """Check if throughput exceeds target."""
        return tasks_per_minute >= self.target_tasks_per_minute


# Default throughput targets
THROUGHPUT_TARGET_BASELINE = ThroughputTarget(
    name="Baseline (2+ tasks/min)",
    min_tasks_per_minute=2.0,
    target_tasks_per_minute=5.0,
)

THROUGHPUT_TARGET_FAST = ThroughputTarget(
    name="Fast (5+ tasks/min)",
    min_tasks_per_minute=5.0,
    target_tasks_per_minute=10.0,
)

THROUGHPUT_TARGET_BATCH = ThroughputTarget(
    name="Batch (10+ tasks/min)",
    min_tasks_per_minute=10.0,
    target_tasks_per_minute=20.0,
)


# =============================================================================
# Workload Configuration
# =============================================================================


@dataclass(frozen=True)
class WorkloadConfig:
    """Configuration for a throughput workload.

    Attributes:
        name: Workload name for identification.
        task_count: Number of tasks in the workload.
        task_template: Template for generating task content.
        description: Human-readable description.
    """

    name: str
    task_count: int
    task_template: str = "Create file{n}.py"
    description: str = ""

    def generate_roadmap(self) -> str:
        """Generate roadmap content for this workload."""
        return generate_multi_task_roadmap(
            task_count=self.task_count,
            task_template=self.task_template,
            title=self.name,
        )


# Standard workload configurations
WORKLOAD_SMALL = WorkloadConfig(
    name="Small Workload",
    task_count=5,
    description="5 simple file creation tasks",
)

WORKLOAD_MEDIUM = WorkloadConfig(
    name="Medium Workload",
    task_count=10,
    description="10 simple file creation tasks",
)

WORKLOAD_LARGE = WorkloadConfig(
    name="Large Workload",
    task_count=20,
    description="20 simple file creation tasks",
)

STANDARD_WORKLOADS = (WORKLOAD_SMALL, WORKLOAD_MEDIUM, WORKLOAD_LARGE)


# =============================================================================
# Roadmap Generation
# =============================================================================


def generate_multi_task_roadmap(
    task_count: int,
    task_template: str = "Create file{n}.py",
    title: str = "Throughput Test",
) -> str:
    """Generate a roadmap with multiple tasks.

    Args:
        task_count: Number of tasks to generate.
        task_template: Template for task content with {n} placeholder.
        title: Roadmap title.

    Returns:
        Roadmap content as string.

    Example:
        >>> roadmap = generate_multi_task_roadmap(3)
        >>> print(roadmap)
        # Throughput Test
        - [ ] Create file1.py
        - [ ] Create file2.py
        - [ ] Create file3.py
    """
    lines = [f"# {title}", ""]
    for i in range(1, task_count + 1):
        task_content = task_template.format(n=i)
        lines.append(f"- [ ] {task_content}")

    return "\n".join(lines)


def generate_varied_task_roadmap(task_count: int) -> str:
    """Generate a roadmap with varied task types.

    Creates a mix of task types for more realistic benchmarking.

    Args:
        task_count: Number of tasks to generate.

    Returns:
        Roadmap content with varied tasks.
    """
    task_types = [
        "Create module{n}.py with a simple function",
        "Create test_module{n}.py with basic tests",
        "Create config{n}.json with default settings",
        "Create README{n}.md with documentation",
        "Create script{n}.py that prints 'Hello {n}'",
    ]

    lines = ["# Varied Throughput Test", ""]
    for i in range(1, task_count + 1):
        template = task_types[(i - 1) % len(task_types)]
        task_content = template.format(n=i)
        lines.append(f"- [ ] {task_content}")

    return "\n".join(lines)


# =============================================================================
# Throughput Result
# =============================================================================


@dataclass
class ThroughputResult:
    """Result of a throughput benchmark.

    Attributes:
        workload_name: Name of the workload.
        task_count: Total number of tasks.
        tasks_completed: Number of tasks completed.
        elapsed_seconds: Total time taken.
        tasks_per_minute: Throughput in tasks per minute.
        tasks_per_second: Throughput in tasks per second.
        avg_task_seconds: Average time per task.
        status: Overall status (completed, partial, failed).
        timestamp: When the benchmark was run.
        metadata: Additional context.
    """

    workload_name: str
    task_count: int
    tasks_completed: int
    elapsed_seconds: float
    tasks_per_minute: float
    tasks_per_second: float
    avg_task_seconds: float
    status: str = "completed"
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    @property
    def completion_rate(self) -> float:
        """Get task completion rate as percentage."""
        if self.task_count == 0:
            return 0.0
        return (self.tasks_completed / self.task_count) * 100

    @property
    def is_complete(self) -> bool:
        """Check if all tasks were completed."""
        return self.tasks_completed >= self.task_count

    def passes_target(self, target: ThroughputTarget) -> bool:
        """Check if result meets target."""
        return target.passes(self.tasks_per_minute)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workload_name": self.workload_name,
            "task_count": self.task_count,
            "tasks_completed": self.tasks_completed,
            "completion_rate": round(self.completion_rate, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "tasks_per_minute": round(self.tasks_per_minute, 2),
            "tasks_per_second": round(self.tasks_per_second, 3),
            "avg_task_seconds": round(self.avg_task_seconds, 3),
            "status": self.status,
            "timestamp": self.timestamp,
            **self.metadata,
        }

    def __str__(self) -> str:
        """Format as human-readable string."""
        status_icon = "✓" if self.is_complete else "⚠"
        return (
            f"{self.workload_name} {status_icon}:\n"
            f"  Tasks: {self.tasks_completed}/{self.task_count} "
            f"({self.completion_rate:.0f}%)\n"
            f"  Time: {self.elapsed_seconds:.1f}s\n"
            f"  Throughput: {self.tasks_per_minute:.1f} tasks/min "
            f"({self.tasks_per_second:.2f} tasks/sec)\n"
            f"  Avg: {self.avg_task_seconds:.2f}s per task"
        )


# =============================================================================
# Throughput Benchmark
# =============================================================================


class ThroughputBenchmark:
    """Benchmark for measuring multi-task throughput.

    Executes roadmaps with multiple tasks and measures throughput
    in tasks per minute.

    Example:
        >>> benchmark = ThroughputBenchmark()
        >>>
        >>> # Run with simulated executor
        >>> result = await benchmark.run_simulated(task_count=10)
        >>> print(f"Throughput: {result.tasks_per_minute:.1f} tasks/min")
        >>>
        >>> # Run with real executor
        >>> result = await benchmark.run(
        ...     task_count=10,
        ...     executor_fn=my_executor,
        ... )
        >>> assert result.passes_target(THROUGHPUT_TARGET_BASELINE)
    """

    def __init__(
        self,
        target: ThroughputTarget | None = None,
        timeout: float = 600.0,
    ) -> None:
        """Initialize benchmark.

        Args:
            target: Throughput target to validate against.
            timeout: Maximum time to wait for completion.
        """
        self.target = target or THROUGHPUT_TARGET_BASELINE
        self.timeout = timeout
        self._results: list[ThroughputResult] = []

    async def run(
        self,
        task_count: int,
        executor_fn: Callable[[str], Coroutine[Any, Any, dict[str, Any]]],
        work_dir: Path | None = None,
        task_template: str = "Create file{n}.py",
        workload_name: str | None = None,
    ) -> ThroughputResult:
        """Run a throughput benchmark.

        Args:
            task_count: Number of tasks in the workload.
            executor_fn: Async function that executes the roadmap.
            work_dir: Working directory for the tasks.
            task_template: Template for generating tasks.
            workload_name: Name for this workload.

        Returns:
            ThroughputResult with throughput metrics.
        """
        import tempfile
        from pathlib import Path

        # Create working directory if needed
        if work_dir is None:
            temp_dir = tempfile.mkdtemp(prefix=f"throughput_{task_count}_")
            work_dir = Path(temp_dir)

        # Generate and write roadmap
        roadmap_content = generate_multi_task_roadmap(
            task_count=task_count,
            task_template=task_template,
        )
        roadmap_path = work_dir / "ROADMAP.md"
        roadmap_path.write_text(roadmap_content)

        # Execute and time
        start = time.perf_counter()
        status = "completed"
        tasks_completed = 0

        try:
            async with asyncio.timeout(self.timeout):
                exec_result = await executor_fn(str(roadmap_path))
                status = exec_result.get("status", "completed")
                tasks_completed = exec_result.get("tasks_completed", task_count)
        except TimeoutError:
            status = "timeout"
        except Exception as e:
            status = f"failed: {e!s}"

        elapsed = time.perf_counter() - start

        # Calculate metrics
        elapsed_minutes = elapsed / 60.0
        tasks_per_minute = tasks_completed / elapsed_minutes if elapsed_minutes > 0 else 0
        tasks_per_second = tasks_completed / elapsed if elapsed > 0 else 0
        avg_task_seconds = elapsed / tasks_completed if tasks_completed > 0 else 0

        throughput_result = ThroughputResult(
            workload_name=workload_name or f"{task_count}-task workload",
            task_count=task_count,
            tasks_completed=tasks_completed,
            elapsed_seconds=elapsed,
            tasks_per_minute=tasks_per_minute,
            tasks_per_second=tasks_per_second,
            avg_task_seconds=avg_task_seconds,
            status=status,
        )

        self._results.append(throughput_result)
        return throughput_result

    async def run_simulated(
        self,
        task_count: int,
        sim_seconds_per_task: float = 3.0,
        sim_completion_rate: float = 1.0,
        workload_name: str | None = None,
    ) -> ThroughputResult:
        """Run a simulated throughput benchmark.

        Useful for testing the benchmark infrastructure.

        Args:
            task_count: Number of tasks in the workload.
            sim_seconds_per_task: Simulated time per task.
            sim_completion_rate: Fraction of tasks completed (0.0-1.0).
            workload_name: Name for this workload.

        Returns:
            ThroughputResult with simulated metrics.
        """
        # Simulate minimal delay
        await asyncio.sleep(0.01)

        # Calculate simulated metrics
        tasks_completed = int(task_count * sim_completion_rate)
        elapsed_seconds = tasks_completed * sim_seconds_per_task
        elapsed_minutes = elapsed_seconds / 60.0

        tasks_per_minute = tasks_completed / elapsed_minutes if elapsed_minutes > 0 else 0
        tasks_per_second = tasks_completed / elapsed_seconds if elapsed_seconds > 0 else 0
        avg_task_seconds = sim_seconds_per_task

        status = "completed" if sim_completion_rate >= 1.0 else "partial"

        result = ThroughputResult(
            workload_name=workload_name or f"{task_count}-task simulated",
            task_count=task_count,
            tasks_completed=tasks_completed,
            elapsed_seconds=elapsed_seconds,
            tasks_per_minute=tasks_per_minute,
            tasks_per_second=tasks_per_second,
            avg_task_seconds=avg_task_seconds,
            status=status,
            metadata={"simulated": True},
        )

        self._results.append(result)
        return result

    async def run_workload(
        self,
        workload: WorkloadConfig,
        executor_fn: Callable[[str], Coroutine[Any, Any, dict[str, Any]]] | None = None,
        simulated: bool = True,
    ) -> ThroughputResult:
        """Run a predefined workload configuration.

        Args:
            workload: WorkloadConfig to run.
            executor_fn: Executor function (required if not simulated).
            simulated: Whether to use simulated execution.

        Returns:
            ThroughputResult for the workload.
        """
        if simulated:
            return await self.run_simulated(
                task_count=workload.task_count,
                workload_name=workload.name,
            )
        else:
            if executor_fn is None:
                msg = "executor_fn required when simulated=False"
                raise ValueError(msg)
            return await self.run(
                task_count=workload.task_count,
                executor_fn=executor_fn,
                task_template=workload.task_template,
                workload_name=workload.name,
            )

    async def run_all_workloads(
        self,
        executor_fn: Callable[[str], Coroutine[Any, Any, dict[str, Any]]] | None = None,
        simulated: bool = True,
    ) -> list[ThroughputResult]:
        """Run all standard workloads.

        Args:
            executor_fn: Executor function (required if not simulated).
            simulated: Whether to use simulated execution.

        Returns:
            List of results for all workloads.
        """
        results = []
        for workload in STANDARD_WORKLOADS:
            result = await self.run_workload(
                workload=workload,
                executor_fn=executor_fn,
                simulated=simulated,
            )
            results.append(result)
        return results

    async def run_scaling_test(
        self,
        task_counts: list[int] | None = None,
        executor_fn: Callable[[str], Coroutine[Any, Any, dict[str, Any]]] | None = None,
        simulated: bool = True,
    ) -> list[ThroughputResult]:
        """Run throughput tests at different scales.

        Useful for understanding how throughput scales with workload size.

        Args:
            task_counts: List of task counts to test.
            executor_fn: Executor function (required if not simulated).
            simulated: Whether to use simulated execution.

        Returns:
            List of results at each scale.
        """
        if task_counts is None:
            task_counts = [5, 10, 20, 50]

        results = []
        for count in task_counts:
            if simulated:
                result = await self.run_simulated(
                    task_count=count,
                    workload_name=f"Scale test ({count} tasks)",
                )
            else:
                if executor_fn is None:
                    msg = "executor_fn required when simulated=False"
                    raise ValueError(msg)
                result = await self.run(
                    task_count=count,
                    executor_fn=executor_fn,
                    workload_name=f"Scale test ({count} tasks)",
                )
            results.append(result)
        return results

    def get_results(self) -> list[ThroughputResult]:
        """Get all benchmark results."""
        return self._results.copy()

    def passes_target(self, result: ThroughputResult) -> bool:
        """Check if result meets target."""
        return self.target.passes(result.tasks_per_minute)

    def summary(self) -> str:
        """Get summary of all results."""
        if not self._results:
            return "No results"

        lines = ["Throughput Benchmark Results:", "=" * 40]

        passed = sum(1 for r in self._results if self.target.passes(r.tasks_per_minute))
        total = len(self._results)
        lines.append(f"Target: {self.target.name}")
        lines.append(f"Overall: {passed}/{total} passed\n")

        for result in self._results:
            passes = "✓" if self.target.passes(result.tasks_per_minute) else "✗"
            lines.append(f"{result.workload_name} {passes}")
            lines.append(f"  {result.tasks_per_minute:.1f} tasks/min")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Comparison Utilities
# =============================================================================


def compare_throughput_results(
    baseline: ThroughputResult,
    current: ThroughputResult,
) -> dict[str, Any]:
    """Compare two throughput results.

    Args:
        baseline: Previous result.
        current: Current result.

    Returns:
        Dictionary with comparison metrics.
    """
    throughput_diff = current.tasks_per_minute - baseline.tasks_per_minute
    throughput_diff_pct = (
        (throughput_diff / baseline.tasks_per_minute * 100) if baseline.tasks_per_minute > 0 else 0
    )

    avg_task_diff = current.avg_task_seconds - baseline.avg_task_seconds
    avg_task_diff_pct = (
        (avg_task_diff / baseline.avg_task_seconds * 100) if baseline.avg_task_seconds > 0 else 0
    )

    return {
        "workload_name": current.workload_name,
        "baseline_throughput": round(baseline.tasks_per_minute, 2),
        "current_throughput": round(current.tasks_per_minute, 2),
        "throughput_diff": round(throughput_diff, 2),
        "throughput_diff_pct": round(throughput_diff_pct, 1),
        "baseline_avg_task_s": round(baseline.avg_task_seconds, 3),
        "current_avg_task_s": round(current.avg_task_seconds, 3),
        "avg_task_diff_pct": round(avg_task_diff_pct, 1),
        "is_improvement": throughput_diff > 0,
        "is_regression": throughput_diff < 0 and throughput_diff_pct < -10,
    }


def summarize_throughput_results(results: list[ThroughputResult]) -> dict[str, Any]:
    """Summarize multiple throughput results.

    Args:
        results: List of throughput results.

    Returns:
        Summary dictionary with aggregate metrics.
    """
    if not results:
        return {"total": 0}

    throughputs = [r.tasks_per_minute for r in results]
    avg_tasks = [r.avg_task_seconds for r in results]

    return {
        "total": len(results),
        "total_tasks": sum(r.task_count for r in results),
        "total_completed": sum(r.tasks_completed for r in results),
        "total_time_seconds": sum(r.elapsed_seconds for r in results),
        "avg_throughput": sum(throughputs) / len(throughputs),
        "max_throughput": max(throughputs),
        "min_throughput": min(throughputs),
        "avg_task_time": sum(avg_tasks) / len(avg_tasks),
        "completion_rate": (
            sum(r.tasks_completed for r in results) / sum(r.task_count for r in results) * 100
        ),
    }
