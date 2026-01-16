"""Task completion benchmark module.

Phase 12.3 of EXECUTOR_4.md - Task Completion Benchmark.

This module provides benchmarking for task completion times:
- TaskComplexity enum for categorizing tasks
- StandardTask definitions (SIMPLE, MEDIUM, COMPLEX)
- TaskCompletionBenchmark for measuring completion times
- TaskCompletionResult for storing results

Targets:
- Simple task: <30s
- Medium task: <60s
- Complex task: <180s

Example:
    >>> from ai_infra.benchmarks.task_completion import (
    ...     TaskCompletionBenchmark,
    ...     SIMPLE_TASK,
    ...     TaskComplexity,
    ... )
    >>>
    >>> benchmark = TaskCompletionBenchmark()
    >>> result = await benchmark.run(SIMPLE_TASK)
    >>> print(f"Completed in {result.elapsed_seconds:.1f}s")
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "COMPLEX_TASK",
    "MEDIUM_TASK",
    "SIMPLE_TASK",
    "StandardTask",
    "TaskComplexity",
    "TaskCompletionBenchmark",
    "TaskCompletionResult",
    "TaskCompletionTarget",
    "get_target_for_complexity",
]


# =============================================================================
# Task Complexity Enum
# =============================================================================


class TaskComplexity(str, Enum):
    """Task complexity levels for benchmarking.

    Defines standardized complexity categories with associated
    time targets for completion benchmarks.
    """

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

    @property
    def target_seconds(self) -> float:
        """Get target completion time in seconds."""
        targets = {
            TaskComplexity.SIMPLE: 30.0,
            TaskComplexity.MEDIUM: 60.0,
            TaskComplexity.COMPLEX: 180.0,
        }
        return targets[self]

    @property
    def description(self) -> str:
        """Get human-readable description."""
        descriptions = {
            TaskComplexity.SIMPLE: "Single file, basic functionality",
            TaskComplexity.MEDIUM: "Multiple files, moderate complexity",
            TaskComplexity.COMPLEX: "Full application with multiple components",
        }
        return descriptions[self]


# =============================================================================
# Task Completion Target
# =============================================================================


@dataclass(frozen=True, slots=True)
class TaskCompletionTarget:
    """Target completion time for a task complexity level.

    Attributes:
        complexity: The task complexity level.
        target_seconds: Maximum allowed completion time.
        warning_seconds: Time at which to warn (optional).
    """

    complexity: TaskComplexity
    target_seconds: float
    warning_seconds: float = 0.0

    def passes(self, elapsed_seconds: float) -> bool:
        """Check if elapsed time meets target."""
        return elapsed_seconds <= self.target_seconds

    def is_warning(self, elapsed_seconds: float) -> bool:
        """Check if elapsed time is in warning zone."""
        if self.warning_seconds <= 0:
            return False
        return self.warning_seconds <= elapsed_seconds <= self.target_seconds


# Default targets per EXECUTOR_4.md
TARGET_SIMPLE = TaskCompletionTarget(
    complexity=TaskComplexity.SIMPLE,
    target_seconds=30.0,
    warning_seconds=20.0,
)

TARGET_MEDIUM = TaskCompletionTarget(
    complexity=TaskComplexity.MEDIUM,
    target_seconds=60.0,
    warning_seconds=45.0,
)

TARGET_COMPLEX = TaskCompletionTarget(
    complexity=TaskComplexity.COMPLEX,
    target_seconds=180.0,
    warning_seconds=120.0,
)


def get_target_for_complexity(complexity: TaskComplexity) -> TaskCompletionTarget:
    """Get the target for a given complexity level."""
    targets = {
        TaskComplexity.SIMPLE: TARGET_SIMPLE,
        TaskComplexity.MEDIUM: TARGET_MEDIUM,
        TaskComplexity.COMPLEX: TARGET_COMPLEX,
    }
    return targets[complexity]


# =============================================================================
# Standard Task Definitions
# =============================================================================


@dataclass(frozen=True)
class StandardTask:
    """A standardized benchmark task definition.

    Attributes:
        name: Task name for identification.
        complexity: Complexity level.
        roadmap_content: The ROADMAP.md content to execute.
        expected_files: Files that should be created.
        description: Human-readable description.
    """

    name: str
    complexity: TaskComplexity
    roadmap_content: str
    expected_files: tuple[str, ...] = ()
    description: str = ""

    @property
    def target(self) -> TaskCompletionTarget:
        """Get the completion target for this task."""
        return get_target_for_complexity(self.complexity)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "complexity": self.complexity.value,
            "target_seconds": self.target.target_seconds,
            "expected_files": list(self.expected_files),
            "description": self.description,
        }


# Standard task definitions per EXECUTOR_4.md
SIMPLE_TASK = StandardTask(
    name="simple_hello_world",
    complexity=TaskComplexity.SIMPLE,
    roadmap_content="""# Simple Task

- [ ] Create hello.py that prints "Hello, World!"
""",
    expected_files=("hello.py",),
    description="Create a single Python file that prints Hello World",
)

MEDIUM_TASK = StandardTask(
    name="medium_calculator",
    complexity=TaskComplexity.MEDIUM,
    roadmap_content="""# Medium Task

- [ ] Create calculator.py with add, subtract, multiply, divide functions
- [ ] Create test_calculator.py with tests for all functions
""",
    expected_files=("calculator.py", "test_calculator.py"),
    description="Create a calculator module with comprehensive tests",
)

COMPLEX_TASK = StandardTask(
    name="complex_fastapi_app",
    complexity=TaskComplexity.COMPLEX,
    roadmap_content="""# Complex Task

- [ ] Create a FastAPI app with:
    - User model with id, email, password_hash
    - POST /register endpoint
    - POST /login endpoint with JWT
    - GET /me endpoint (authenticated)
- [ ] Create tests for all endpoints
""",
    expected_files=("main.py", "models.py", "test_endpoints.py"),
    description="Create a FastAPI application with authentication",
)

# All standard tasks for iteration
STANDARD_TASKS = (SIMPLE_TASK, MEDIUM_TASK, COMPLEX_TASK)


# =============================================================================
# Task Completion Result
# =============================================================================


@dataclass
class TaskCompletionResult:
    """Result of a task completion benchmark.

    Attributes:
        task_name: Name of the task that was run.
        complexity: Task complexity level.
        elapsed_seconds: Time taken to complete.
        target_seconds: Target time for this complexity.
        status: Completion status (completed, failed, timeout).
        files_created: Files that were created during execution.
        timestamp: When the benchmark was run.
        metadata: Additional context.
    """

    task_name: str
    complexity: TaskComplexity
    elapsed_seconds: float
    target_seconds: float
    status: str = "completed"
    files_created: list[str] = field(default_factory=list)
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    @property
    def passes_target(self) -> bool:
        """Check if result meets target."""
        return self.status == "completed" and self.elapsed_seconds <= self.target_seconds

    @property
    def margin_seconds(self) -> float:
        """Get margin vs target (positive = under, negative = over)."""
        return self.target_seconds - self.elapsed_seconds

    @property
    def margin_percent(self) -> float:
        """Get margin as percentage of target."""
        if self.target_seconds == 0:
            return 0.0
        return (self.margin_seconds / self.target_seconds) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_name": self.task_name,
            "complexity": self.complexity.value,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "target_seconds": self.target_seconds,
            "passes_target": self.passes_target,
            "margin_seconds": round(self.margin_seconds, 3),
            "status": self.status,
            "files_created": self.files_created,
            "timestamp": self.timestamp,
            **self.metadata,
        }

    def __str__(self) -> str:
        """Format as human-readable string."""
        status_icon = "✓" if self.passes_target else "✗"
        return (
            f"{self.task_name} ({self.complexity.value}):\n"
            f"  Status: {self.status} {status_icon}\n"
            f"  Time: {self.elapsed_seconds:.1f}s / {self.target_seconds:.0f}s\n"
            f"  Margin: {self.margin_seconds:+.1f}s ({self.margin_percent:+.1f}%)"
        )


# =============================================================================
# Task Completion Benchmark
# =============================================================================


class TaskCompletionBenchmark:
    """Benchmark for measuring task completion times.

    Executes tasks and measures completion time against targets.
    Supports both simulated and real executor runs.

    Example:
        >>> benchmark = TaskCompletionBenchmark()
        >>>
        >>> # Run with simulated executor
        >>> result = await benchmark.run_simulated(SIMPLE_TASK, sim_time=15.0)
        >>> print(result)
        >>>
        >>> # Run with real executor
        >>> result = await benchmark.run(SIMPLE_TASK, executor_fn=my_executor)
        >>> assert result.passes_target
    """

    def __init__(self, timeout: float = 300.0) -> None:
        """Initialize benchmark.

        Args:
            timeout: Maximum time to wait for task completion.
        """
        self.timeout = timeout
        self._results: list[TaskCompletionResult] = []

    async def run(
        self,
        task: StandardTask,
        executor_fn: Callable[[str], Coroutine[Any, Any, dict[str, Any]]],
        work_dir: Path | None = None,
    ) -> TaskCompletionResult:
        """Run a task completion benchmark.

        Args:
            task: The standard task to benchmark.
            executor_fn: Async function that executes the roadmap.
                Should accept roadmap path and return result dict.
            work_dir: Working directory for the task.

        Returns:
            TaskCompletionResult with timing data.
        """
        import tempfile
        from pathlib import Path

        # Create working directory if needed
        if work_dir is None:
            temp_dir = tempfile.mkdtemp(prefix=f"benchmark_{task.name}_")
            work_dir = Path(temp_dir)

        # Write roadmap
        roadmap_path = work_dir / "ROADMAP.md"
        roadmap_path.write_text(task.roadmap_content)

        # Execute and time
        start = time.perf_counter()
        status = "completed"
        files_created: list[str] = []

        try:
            async with asyncio.timeout(self.timeout):
                exec_result = await executor_fn(str(roadmap_path))
                status = exec_result.get("status", "completed")
        except TimeoutError:
            status = "timeout"
        except Exception as e:
            status = f"failed: {e!s}"

        elapsed = time.perf_counter() - start

        # Check for created files
        if work_dir.exists():
            files_created = [
                f.name for f in work_dir.iterdir() if f.is_file() and f.name != "ROADMAP.md"
            ]

        result = TaskCompletionResult(
            task_name=task.name,
            complexity=task.complexity,
            elapsed_seconds=elapsed,
            target_seconds=task.target.target_seconds,
            status=status,
            files_created=files_created,
        )

        self._results.append(result)
        return result

    async def run_simulated(
        self,
        task: StandardTask,
        sim_time: float | None = None,
        sim_status: str = "completed",
    ) -> TaskCompletionResult:
        """Run a simulated task completion benchmark.

        Useful for testing the benchmark infrastructure without
        requiring a real executor.

        Args:
            task: The standard task to benchmark.
            sim_time: Simulated completion time (None = use target * 0.8).
            sim_status: Simulated status result.

        Returns:
            TaskCompletionResult with simulated timing.
        """
        # Default to 80% of target time
        if sim_time is None:
            sim_time = task.target.target_seconds * 0.8

        start = time.perf_counter()
        await asyncio.sleep(min(sim_time, 0.1))  # Cap actual sleep for tests
        elapsed = time.perf_counter() - start

        # Use simulated time for result, not actual elapsed
        result = TaskCompletionResult(
            task_name=task.name,
            complexity=task.complexity,
            elapsed_seconds=sim_time,
            target_seconds=task.target.target_seconds,
            status=sim_status,
            files_created=list(task.expected_files),
            metadata={"simulated": True},
        )

        self._results.append(result)
        return result

    async def run_all_standard(
        self,
        executor_fn: Callable[[str], Coroutine[Any, Any, dict[str, Any]]] | None = None,
        simulated: bool = True,
    ) -> list[TaskCompletionResult]:
        """Run all standard tasks.

        Args:
            executor_fn: Executor function (required if simulated=False).
            simulated: Whether to use simulated execution.

        Returns:
            List of results for all standard tasks.
        """
        results = []
        for task in STANDARD_TASKS:
            if simulated:
                result = await self.run_simulated(task)
            else:
                if executor_fn is None:
                    msg = "executor_fn required when simulated=False"
                    raise ValueError(msg)
                result = await self.run(task, executor_fn)
            results.append(result)
        return results

    def get_results(self) -> list[TaskCompletionResult]:
        """Get all benchmark results."""
        return self._results.copy()

    def summary(self) -> str:
        """Get summary of all results."""
        if not self._results:
            return "No results"

        lines = ["Task Completion Benchmark Results:", "=" * 40]

        passed = sum(1 for r in self._results if r.passes_target)
        total = len(self._results)
        lines.append(f"Overall: {passed}/{total} passed\n")

        for result in self._results:
            lines.append(str(result))
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Comparison Utilities
# =============================================================================


def compare_task_results(
    baseline: TaskCompletionResult,
    current: TaskCompletionResult,
) -> dict[str, Any]:
    """Compare two task completion results.

    Args:
        baseline: Previous result.
        current: Current result.

    Returns:
        Dictionary with comparison metrics.
    """
    time_diff = current.elapsed_seconds - baseline.elapsed_seconds
    time_diff_pct = (
        (time_diff / baseline.elapsed_seconds * 100) if baseline.elapsed_seconds > 0 else 0
    )

    return {
        "task_name": current.task_name,
        "baseline_seconds": baseline.elapsed_seconds,
        "current_seconds": current.elapsed_seconds,
        "diff_seconds": round(time_diff, 3),
        "diff_percent": round(time_diff_pct, 2),
        "baseline_passed": baseline.passes_target,
        "current_passed": current.passes_target,
        "is_regression": time_diff > 0 and current.elapsed_seconds > current.target_seconds,
        "is_improvement": time_diff < 0,
    }


def summarize_all_results(results: list[TaskCompletionResult]) -> dict[str, Any]:
    """Summarize multiple task completion results.

    Args:
        results: List of task completion results.

    Returns:
        Summary dictionary with aggregate metrics.
    """
    if not results:
        return {"total": 0, "passed": 0, "failed": 0}

    passed = [r for r in results if r.passes_target]
    failed = [r for r in results if not r.passes_target]

    by_complexity: dict[str, list[TaskCompletionResult]] = {}
    for r in results:
        key = r.complexity.value
        if key not in by_complexity:
            by_complexity[key] = []
        by_complexity[key].append(r)

    return {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": len(passed) / len(results) * 100,
        "by_complexity": {
            k: {
                "total": len(v),
                "passed": sum(1 for r in v if r.passes_target),
                "avg_seconds": sum(r.elapsed_seconds for r in v) / len(v),
            }
            for k, v in by_complexity.items()
        },
        "total_time_seconds": sum(r.elapsed_seconds for r in results),
    }
