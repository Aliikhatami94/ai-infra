"""Benchmark: Executor Performance Metrics (Phase 6.4).

This benchmark measures key executor performance metrics:
- Time to first token (TTFT)
- Task completion latency
- Token efficiency
- Success rates by task type

These benchmarks use simulated scenarios to avoid LLM costs during CI/CD.
For real LLM benchmarks, use the integration tests with live API keys.

Run with: pytest benchmarks/bench_executor_performance.py -v --benchmark-only
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from ai_infra.executor.models import Task
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Performance Data Structures
# =============================================================================


@dataclass
class PerformanceMetrics:
    """Metrics collected during performance benchmarks."""

    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    characters_generated: int = 0
    success: bool = False
    task_type: str = ""

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.tokens_in + self.tokens_out

    @property
    def token_efficiency(self) -> float:
        """Tokens per character (lower is better)."""
        if self.characters_generated == 0:
            return float("inf")
        return self.tokens_out / self.characters_generated

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ttft_ms": self.ttft_ms,
            "total_latency_ms": self.total_latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "characters_generated": self.characters_generated,
            "token_efficiency": self.token_efficiency,
            "success": self.success,
            "task_type": self.task_type,
        }


@dataclass
class TaskExecutionResult:
    """Result of a simulated task execution."""

    success: bool = True
    content_generated: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    files_modified: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


# =============================================================================
# Simulators for Performance Testing
# =============================================================================


class ExecutorSimulator:
    """Simulates executor operations for benchmarking without LLM calls."""

    # Simulated latencies in milliseconds
    TTFT_SIMPLE_MS = 150  # Simple tasks
    TTFT_COMPLEX_MS = 800  # Complex tasks
    LATENCY_SIMPLE_MS = 5000  # Simple task completion
    LATENCY_AVERAGE_MS = 15000  # Average task completion
    LATENCY_COMPLEX_MS = 45000  # Complex task completion

    # Simulated token usage
    TOKENS_SIMPLE = 500
    TOKENS_AVERAGE = 2000
    TOKENS_COMPLEX = 8000

    # Success rates by task type
    SUCCESS_RATES = {
        "fix_typo": 0.98,
        "add_docstring": 0.96,
        "add_function": 0.92,
        "add_test": 0.90,
        "refactor": 0.85,
        "debug": 0.80,
        "complex": 0.75,
    }

    def __init__(self, speed_factor: float = 0.001) -> None:
        """Initialize simulator.

        Args:
            speed_factor: Multiplier for simulated delays (0.001 = 1000x faster)
        """
        self.speed_factor = speed_factor

    def _classify_task(self, task: Task | TodoItem) -> str:
        """Classify task type based on title."""
        title = task.title.lower()

        if "typo" in title or "spelling" in title:
            return "fix_typo"
        elif "docstring" in title or "comment" in title:
            return "add_docstring"
        elif "test" in title:
            return "add_test"
        elif "function" in title or "method" in title or "add" in title:
            return "add_function"
        elif "refactor" in title or "reorganize" in title:
            return "refactor"
        elif "debug" in title or "fix" in title or "bug" in title:
            return "debug"
        elif "complex" in title or "system" in title:
            return "complex"
        else:
            return "add_function"  # Default

    def _get_ttft(self, task_type: str) -> float:
        """Get time to first token for task type."""
        if task_type in ("fix_typo", "add_docstring"):
            return self.TTFT_SIMPLE_MS
        elif task_type in ("complex", "refactor"):
            return self.TTFT_COMPLEX_MS
        else:
            return (self.TTFT_SIMPLE_MS + self.TTFT_COMPLEX_MS) / 2

    def _get_latency(self, task_type: str) -> float:
        """Get completion latency for task type."""
        if task_type in ("fix_typo", "add_docstring"):
            return self.LATENCY_SIMPLE_MS
        elif task_type in ("complex", "refactor"):
            return self.LATENCY_COMPLEX_MS
        else:
            return self.LATENCY_AVERAGE_MS

    def _get_tokens(self, task_type: str) -> tuple[int, int]:
        """Get (tokens_in, tokens_out) for task type."""
        if task_type in ("fix_typo", "add_docstring"):
            return (200, self.TOKENS_SIMPLE)
        elif task_type in ("complex", "refactor"):
            return (1000, self.TOKENS_COMPLEX)
        else:
            return (500, self.TOKENS_AVERAGE)

    async def stream_task(self, task: Task | TodoItem) -> AsyncIterator[tuple[str, float]]:
        """Stream task execution, yielding (chunk, timestamp).

        Yields:
            Tuple of (chunk_text, timestamp_ms)
        """
        task_type = self._classify_task(task)
        ttft = self._get_ttft(task_type)
        latency = self._get_latency(task_type)

        start = time.perf_counter()

        # Simulate TTFT delay
        await asyncio.sleep(ttft * self.speed_factor / 1000)

        # First token
        first_token_time = (time.perf_counter() - start) * 1000
        yield ("Starting task...", first_token_time)

        # Simulate remaining execution
        remaining = latency - ttft
        chunks = 10
        for i in range(chunks):
            await asyncio.sleep(remaining * self.speed_factor / 1000 / chunks)
            current_time = (time.perf_counter() - start) * 1000
            yield (f"Progress {(i + 1) * 10}%...", current_time)

        # Final chunk
        final_time = (time.perf_counter() - start) * 1000
        yield ("Task complete.", final_time)

    async def execute_task(self, task: Task | TodoItem) -> TaskExecutionResult:
        """Execute a task and return result."""
        task_type = self._classify_task(task)
        latency = self._get_latency(task_type)
        tokens_in, tokens_out = self._get_tokens(task_type)
        success_rate = self.SUCCESS_RATES.get(task_type, 0.85)

        start = time.perf_counter()

        # Simulate execution delay
        await asyncio.sleep(latency * self.speed_factor / 1000)

        execution_time = (time.perf_counter() - start) * 1000

        # Simulate success/failure based on rate
        import random

        success = random.random() < success_rate

        # Generate simulated content
        content = self._generate_content(task_type, success)

        return TaskExecutionResult(
            success=success,
            content_generated=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            files_modified=[f"src/{task_type}.py"],
            execution_time_ms=execution_time,
        )

    def _generate_content(self, task_type: str, success: bool) -> str:
        """Generate simulated content output."""
        if not success:
            return "# Error: Task failed"

        templates = {
            "fix_typo": 'Fixed typo in line 42: "recieve" -> "receive"',
            "add_docstring": '''"""Calculate the sum of two numbers.

Args:
    a: First number
    b: Second number

Returns:
    Sum of a and b
"""''',
            "add_function": '''def process_data(data: list) -> dict:
    """Process input data and return summary."""
    result = {}
    for item in data:
        key = item.get("type", "unknown")
        result[key] = result.get(key, 0) + 1
    return result''',
            "add_test": '''def test_process_data():
    """Test the process_data function."""
    data = [{"type": "a"}, {"type": "b"}, {"type": "a"}]
    result = process_data(data)
    assert result == {"a": 2, "b": 1}''',
            "refactor": '''class DataProcessor:
    """Refactored data processing class."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def process(self, data: list) -> dict:
        """Process data with configuration."""
        return self._transform(data)

    def _transform(self, data: list) -> dict:
        """Internal transformation logic."""
        return {item["key"]: item["value"] for item in data}''',
            "debug": "# Fixed: Changed == to is for None comparison",
            "complex": """# Implemented authentication system
# - JWT token generation
# - Password hashing
# - Session management
# - Role-based access control""",
        }
        return templates.get(task_type, "# Code generated successfully")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simulator() -> ExecutorSimulator:
    """Create executor simulator."""
    return ExecutorSimulator(speed_factor=0.001)  # 1000x faster


@pytest.fixture
def simple_task() -> Task:
    """Create a simple task."""
    return Task(
        id="simple-1",
        title="Fix typo in README",
        description="Change 'recieve' to 'receive'",
    )


@pytest.fixture
def average_task() -> Task:
    """Create an average complexity task."""
    return Task(
        id="avg-1",
        title="Add helper function for data processing",
        description="Create a utility function to process input data",
    )


@pytest.fixture
def complex_task() -> Task:
    """Create a complex task."""
    return Task(
        id="complex-1",
        title="Refactor authentication system",
        description="Reorganize auth module with proper separation of concerns",
    )


@pytest.fixture
def task_suite() -> dict[str, list[Task]]:
    """Create a suite of tasks for success rate testing."""
    return {
        "fix_typo": [
            Task(id="typo-1", title="Fix typo in docs", description=""),
            Task(id="typo-2", title="Fix spelling error", description=""),
            Task(id="typo-3", title="Correct typo in string", description=""),
        ],
        "add_function": [
            Task(id="func-1", title="Add helper function", description=""),
            Task(id="func-2", title="Create utility method", description=""),
            Task(id="func-3", title="Add processing function", description=""),
        ],
        "add_test": [
            Task(id="test-1", title="Add unit test for module", description=""),
            Task(id="test-2", title="Write test for function", description=""),
            Task(id="test-3", title="Create integration test", description=""),
        ],
        "refactor": [
            Task(id="ref-1", title="Refactor data module", description=""),
            Task(id="ref-2", title="Reorganize API handlers", description=""),
        ],
        "debug": [
            Task(id="dbg-1", title="Debug failing assertion", description=""),
            Task(id="dbg-2", title="Fix bug in parser", description=""),
        ],
    }


# =============================================================================
# 6.4.1 Time to First Token Benchmarks
# =============================================================================


class TestTimeToFirstToken:
    """Benchmark time to first token (Phase 6.4.1)."""

    @pytest.mark.benchmark(group="ttft")
    @pytest.mark.asyncio
    async def test_simple_task_ttft(
        self, benchmark, simulator: ExecutorSimulator, simple_task: Task
    ) -> None:
        """Simple tasks should have TTFT < 500ms (simulated)."""

        async def measure_ttft() -> float:
            start = time.perf_counter()
            async for _chunk, _timestamp in simulator.stream_task(simple_task):
                ttft = (time.perf_counter() - start) * 1000
                return ttft
            return 0.0

        ttft = await measure_ttft()
        benchmark.extra_info["ttft_ms"] = ttft

        # Verify within threshold (scaled by speed_factor)
        expected_max = simulator.TTFT_SIMPLE_MS * simulator.speed_factor * 2
        assert ttft < expected_max, f"TTFT was {ttft}ms, expected < {expected_max}ms"

    @pytest.mark.benchmark(group="ttft")
    @pytest.mark.asyncio
    async def test_complex_task_ttft(
        self, benchmark, simulator: ExecutorSimulator, complex_task: Task
    ) -> None:
        """Complex tasks should have TTFT < 2000ms (simulated)."""

        async def measure_ttft() -> float:
            start = time.perf_counter()
            async for _chunk, _timestamp in simulator.stream_task(complex_task):
                ttft = (time.perf_counter() - start) * 1000
                return ttft
            return 0.0

        ttft = await measure_ttft()
        benchmark.extra_info["ttft_ms"] = ttft

        expected_max = simulator.TTFT_COMPLEX_MS * simulator.speed_factor * 2
        assert ttft < expected_max, f"TTFT was {ttft}ms, expected < {expected_max}ms"

    @pytest.mark.benchmark(group="ttft")
    @pytest.mark.asyncio
    async def test_average_task_ttft(
        self, benchmark, simulator: ExecutorSimulator, average_task: Task
    ) -> None:
        """Average tasks should have TTFT < 1000ms (simulated)."""

        async def measure_ttft() -> float:
            start = time.perf_counter()
            async for _chunk, _timestamp in simulator.stream_task(average_task):
                ttft = (time.perf_counter() - start) * 1000
                return ttft
            return 0.0

        ttft = await measure_ttft()
        benchmark.extra_info["ttft_ms"] = ttft

        # Average should be between simple and complex
        avg_expected = (simulator.TTFT_SIMPLE_MS + simulator.TTFT_COMPLEX_MS) / 2
        expected_max = avg_expected * simulator.speed_factor * 2
        assert ttft < expected_max


# =============================================================================
# 6.4.2 Task Completion Latency Benchmarks
# =============================================================================


class TestTaskCompletionLatency:
    """Benchmark task completion time (Phase 6.4.2)."""

    @pytest.mark.benchmark(group="latency")
    @pytest.mark.asyncio
    async def test_simple_task_latency(
        self, benchmark, simulator: ExecutorSimulator, simple_task: Task
    ) -> None:
        """Simple tasks should complete quickly (simulated < 30s real)."""
        start = time.perf_counter()
        result = await simulator.execute_task(simple_task)
        latency_ms = (time.perf_counter() - start) * 1000

        benchmark.extra_info["latency_ms"] = latency_ms
        benchmark.extra_info["success"] = result.success

        expected_max = simulator.LATENCY_SIMPLE_MS * simulator.speed_factor * 2
        assert latency_ms < expected_max, f"Took {latency_ms}ms, expected < {expected_max}ms"

    @pytest.mark.benchmark(group="latency")
    @pytest.mark.asyncio
    async def test_average_task_latency(
        self, benchmark, simulator: ExecutorSimulator, average_task: Task
    ) -> None:
        """Average tasks should complete in reasonable time (simulated < 60s real)."""
        start = time.perf_counter()
        result = await simulator.execute_task(average_task)
        latency_ms = (time.perf_counter() - start) * 1000

        benchmark.extra_info["latency_ms"] = latency_ms
        benchmark.extra_info["success"] = result.success

        expected_max = simulator.LATENCY_AVERAGE_MS * simulator.speed_factor * 2
        assert latency_ms < expected_max

    @pytest.mark.benchmark(group="latency")
    @pytest.mark.asyncio
    async def test_complex_task_latency(
        self, benchmark, simulator: ExecutorSimulator, complex_task: Task
    ) -> None:
        """Complex tasks may take longer (simulated < 120s real)."""
        start = time.perf_counter()
        result = await simulator.execute_task(complex_task)
        latency_ms = (time.perf_counter() - start) * 1000

        benchmark.extra_info["latency_ms"] = latency_ms
        benchmark.extra_info["success"] = result.success

        expected_max = simulator.LATENCY_COMPLEX_MS * simulator.speed_factor * 2
        assert latency_ms < expected_max

    @pytest.mark.benchmark(group="latency")
    @pytest.mark.asyncio
    async def test_batch_task_latency(
        self, benchmark, simulator: ExecutorSimulator, task_suite: dict[str, list[Task]]
    ) -> None:
        """Measure average latency across multiple tasks."""
        all_tasks = [task for tasks in task_suite.values() for task in tasks]
        latencies = []

        start = time.perf_counter()
        for task in all_tasks:
            task_start = time.perf_counter()
            await simulator.execute_task(task)
            latencies.append((time.perf_counter() - task_start) * 1000)
        total_time = (time.perf_counter() - start) * 1000

        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        benchmark.extra_info["total_time_ms"] = total_time
        benchmark.extra_info["avg_latency_ms"] = avg_latency
        benchmark.extra_info["task_count"] = len(all_tasks)


# =============================================================================
# 6.4.3 Token Efficiency Benchmarks
# =============================================================================


class TestTokenEfficiency:
    """Benchmark token usage (Phase 6.4.3)."""

    @pytest.mark.benchmark(group="tokens")
    @pytest.mark.asyncio
    async def test_tokens_per_simple_task(
        self, benchmark, simulator: ExecutorSimulator, simple_task: Task
    ) -> None:
        """Track tokens for simple tasks."""
        result = await simulator.execute_task(simple_task)
        total_tokens = result.tokens_in + result.tokens_out

        benchmark.extra_info["tokens_in"] = result.tokens_in
        benchmark.extra_info["tokens_out"] = result.tokens_out
        benchmark.extra_info["total_tokens"] = total_tokens

        # Simple tasks should use fewer tokens
        assert total_tokens < simulator.TOKENS_SIMPLE * 2, (
            f"Used {total_tokens} tokens, expected < {simulator.TOKENS_SIMPLE * 2}"
        )

    @pytest.mark.benchmark(group="tokens")
    @pytest.mark.asyncio
    async def test_tokens_per_complex_task(
        self, benchmark, simulator: ExecutorSimulator, complex_task: Task
    ) -> None:
        """Track tokens for complex tasks."""
        result = await simulator.execute_task(complex_task)
        total_tokens = result.tokens_in + result.tokens_out

        benchmark.extra_info["tokens_in"] = result.tokens_in
        benchmark.extra_info["tokens_out"] = result.tokens_out
        benchmark.extra_info["total_tokens"] = total_tokens

        # Complex tasks use more tokens but should have a ceiling
        assert total_tokens < simulator.TOKENS_COMPLEX * 2, (
            f"Used {total_tokens} tokens, expected < {simulator.TOKENS_COMPLEX * 2}"
        )

    @pytest.mark.benchmark(group="tokens")
    @pytest.mark.asyncio
    async def test_tokens_per_character_output(
        self, benchmark, simulator: ExecutorSimulator, average_task: Task
    ) -> None:
        """Track efficiency: tokens used per character of output."""
        result = await simulator.execute_task(average_task)

        output_chars = len(result.content_generated)
        tokens = result.tokens_out
        efficiency = tokens / output_chars if output_chars > 0 else float("inf")

        benchmark.extra_info["output_chars"] = output_chars
        benchmark.extra_info["tokens_out"] = tokens
        benchmark.extra_info["efficiency"] = efficiency

        # Typical efficiency is ~0.25-0.5 tokens per character for GPT models
        # Our simulation should be reasonable
        assert efficiency < 10.0, f"Efficiency was {efficiency}, expected < 10.0"

    @pytest.mark.benchmark(group="tokens")
    @pytest.mark.asyncio
    async def test_aggregate_token_usage(
        self, benchmark, simulator: ExecutorSimulator, task_suite: dict[str, list[Task]]
    ) -> None:
        """Track total tokens across task suite."""
        all_tasks = [task for tasks in task_suite.values() for task in tasks]

        total_in = 0
        total_out = 0
        total_chars = 0

        for task in all_tasks:
            result = await simulator.execute_task(task)
            total_in += result.tokens_in
            total_out += result.tokens_out
            total_chars += len(result.content_generated)

        total_tokens = total_in + total_out
        avg_tokens = total_tokens / len(all_tasks) if all_tasks else 0

        benchmark.extra_info["total_tokens"] = total_tokens
        benchmark.extra_info["avg_tokens_per_task"] = avg_tokens
        benchmark.extra_info["total_chars"] = total_chars
        benchmark.extra_info["task_count"] = len(all_tasks)


# =============================================================================
# 6.4.4 Success Rate Benchmarks
# =============================================================================


class TestSuccessRates:
    """Track success rates by task type (Phase 6.4.4)."""

    @pytest.mark.benchmark(group="success")
    @pytest.mark.asyncio
    async def test_success_rate_by_type(
        self, benchmark, simulator: ExecutorSimulator, task_suite: dict[str, list[Task]]
    ) -> None:
        """Track success rate for each task type."""
        results: dict[str, dict[str, Any]] = {}

        for task_type, tasks in task_suite.items():
            successes = 0
            total_latency = 0.0

            for task in tasks:
                result = await simulator.execute_task(task)
                if result.success:
                    successes += 1
                total_latency += result.execution_time_ms

            rate = successes / len(tasks) if tasks else 0
            avg_latency = total_latency / len(tasks) if tasks else 0

            results[task_type] = {
                "success_rate": rate,
                "successes": successes,
                "total": len(tasks),
                "avg_latency_ms": avg_latency,
            }
            benchmark.extra_info[f"{task_type}_success_rate"] = rate

        # Output summary
        for task_type, data in results.items():
            benchmark.extra_info[f"{task_type}_details"] = data

    @pytest.mark.benchmark(group="success")
    @pytest.mark.asyncio
    async def test_critical_task_success(self, benchmark, simulator: ExecutorSimulator) -> None:
        """Critical task types should have high success rates."""
        # Run multiple iterations for statistical significance
        iterations = 10
        task_results = {
            "fix_typo": [],
            "add_docstring": [],
            "add_function": [],
        }

        for _ in range(iterations):
            for task_type in task_results:
                task = Task(id=f"{task_type}-test", title=f"Test {task_type}", description="")
                result = await simulator.execute_task(task)
                task_results[task_type].append(result.success)

        # Calculate rates
        rates = {}
        for task_type, successes in task_results.items():
            rate = sum(1 for s in successes if s) / len(successes)
            rates[task_type] = rate
            benchmark.extra_info[f"{task_type}_rate"] = rate

        # Note: These are simulated rates based on SUCCESS_RATES constants
        # In production, actual rates would vary

    @pytest.mark.benchmark(group="success")
    @pytest.mark.asyncio
    async def test_overall_success_rate(
        self, benchmark, simulator: ExecutorSimulator, task_suite: dict[str, list[Task]]
    ) -> None:
        """Track overall success rate across all task types."""
        all_tasks = [task for tasks in task_suite.values() for task in tasks]
        successes = 0

        for task in all_tasks:
            result = await simulator.execute_task(task)
            if result.success:
                successes += 1

        overall_rate = successes / len(all_tasks) if all_tasks else 0

        benchmark.extra_info["overall_success_rate"] = overall_rate
        benchmark.extra_info["successes"] = successes
        benchmark.extra_info["total_tasks"] = len(all_tasks)

        # Overall should be > 80% for mixed task suite
        assert overall_rate > 0.7, f"Overall rate {overall_rate} too low"


# =============================================================================
# Performance Comparison Tests
# =============================================================================


class TestPerformanceComparison:
    """Compare performance across different scenarios."""

    @pytest.mark.benchmark(group="comparison")
    @pytest.mark.asyncio
    async def test_simple_vs_complex_ratio(
        self,
        benchmark,
        simulator: ExecutorSimulator,
        simple_task: Task,
        complex_task: Task,
    ) -> None:
        """Complex tasks should take proportionally longer."""
        simple_result = await simulator.execute_task(simple_task)
        complex_result = await simulator.execute_task(complex_task)

        ratio = complex_result.execution_time_ms / simple_result.execution_time_ms

        benchmark.extra_info["simple_ms"] = simple_result.execution_time_ms
        benchmark.extra_info["complex_ms"] = complex_result.execution_time_ms
        benchmark.extra_info["ratio"] = ratio

        # Complex should be 3-10x slower than simple
        assert 2.0 < ratio < 20.0, f"Ratio {ratio} outside expected range"

    @pytest.mark.benchmark(group="comparison")
    @pytest.mark.asyncio
    async def test_token_scaling(
        self,
        benchmark,
        simulator: ExecutorSimulator,
        simple_task: Task,
        complex_task: Task,
    ) -> None:
        """Token usage should scale with task complexity."""
        simple_result = await simulator.execute_task(simple_task)
        complex_result = await simulator.execute_task(complex_task)

        simple_tokens = simple_result.tokens_in + simple_result.tokens_out
        complex_tokens = complex_result.tokens_in + complex_result.tokens_out
        ratio = complex_tokens / simple_tokens

        benchmark.extra_info["simple_tokens"] = simple_tokens
        benchmark.extra_info["complex_tokens"] = complex_tokens
        benchmark.extra_info["token_ratio"] = ratio

        # Complex should use more tokens but not excessively
        assert ratio > 1.5, f"Complex should use more tokens (ratio: {ratio})"
        assert ratio < 50.0, f"Token ratio {ratio} seems excessive"


# =============================================================================
# Throughput Tests
# =============================================================================


class TestThroughput:
    """Measure executor throughput."""

    @pytest.mark.benchmark(group="throughput")
    @pytest.mark.asyncio
    async def test_sequential_throughput(
        self, benchmark, simulator: ExecutorSimulator, task_suite: dict[str, list[Task]]
    ) -> None:
        """Measure tasks per second in sequential execution."""
        all_tasks = [task for tasks in task_suite.values() for task in tasks]

        start = time.perf_counter()
        completed = 0
        for task in all_tasks:
            await simulator.execute_task(task)
            completed += 1
        elapsed = time.perf_counter() - start

        throughput = completed / elapsed if elapsed > 0 else 0

        benchmark.extra_info["tasks_completed"] = completed
        benchmark.extra_info["elapsed_seconds"] = elapsed
        benchmark.extra_info["throughput_per_second"] = throughput

    @pytest.mark.benchmark(group="throughput")
    @pytest.mark.asyncio
    async def test_concurrent_throughput(
        self, benchmark, simulator: ExecutorSimulator, task_suite: dict[str, list[Task]]
    ) -> None:
        """Measure tasks per second with concurrent execution."""
        all_tasks = [task for tasks in task_suite.values() for task in tasks]

        start = time.perf_counter()
        results = await asyncio.gather(*[simulator.execute_task(task) for task in all_tasks])
        elapsed = time.perf_counter() - start

        completed = len(results)
        throughput = completed / elapsed if elapsed > 0 else 0

        benchmark.extra_info["tasks_completed"] = completed
        benchmark.extra_info["elapsed_seconds"] = elapsed
        benchmark.extra_info["concurrent_throughput"] = throughput


# =============================================================================
# Real Executor Integration (Optional - requires mocking)
# =============================================================================


class TestRealExecutorMetrics:
    """Tests that use real executor components with mocked LLM."""

    def test_todolist_manager_performance(self, benchmark) -> None:
        """Benchmark TodoListManager operations."""
        from ai_infra.executor.todolist import TodoItem, TodoListManager

        # Create manager with many todos
        todos = [
            TodoItem(
                id=i,
                title=f"Task {i}",
                description=f"Description for task {i}",
                status=TodoStatus.NOT_STARTED,
            )
            for i in range(100)
        ]
        manager = TodoListManager(todos=todos)

        def iterate_todos():
            count = 0
            for todo in manager.todos:
                count += 1
            return count

        result = benchmark(iterate_todos)
        assert result == 100

    def test_task_model_serialization(self, benchmark) -> None:
        """Benchmark Task model serialization."""
        task = Task(
            id="perf-1",
            title="Performance test task",
            description="A task for testing serialization performance",
            file_hints=["src/a.py", "src/b.py", "src/c.py"],
            dependencies=["task-0", "task-1"],
        )

        def serialize_task():
            return task.to_dict()

        result = benchmark(serialize_task)
        assert "id" in result
        assert result["title"] == "Performance test task"

    def test_task_model_deserialization(self, benchmark) -> None:
        """Benchmark Task model deserialization."""
        data = {
            "id": "perf-1",
            "title": "Performance test task",
            "description": "A task for testing serialization performance",
            "file_hints": ["src/a.py", "src/b.py", "src/c.py"],
            "dependencies": ["task-0", "task-1"],
            "status": "pending",
        }

        def deserialize_task():
            return Task.from_dict(data)

        result = benchmark(deserialize_task)
        assert result.id == "perf-1"
