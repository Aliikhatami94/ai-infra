"""Benchmark harness for LLM performance testing.

Phase 12.1 - Benchmark Framework.

This module provides a standardized benchmark harness for measuring LLM
performance metrics including:
- Time to first token (TTFT)
- Task completion latency
- Throughput measurements

Example:
    >>> from ai_infra.benchmarks.harness import benchmark, BenchmarkResult
    >>>
    >>> async def my_operation():
    ...     await some_async_work()
    >>>
    >>> result = await benchmark("my_operation", my_operation, iterations=10)
    >>> print(result)
    my_operation:
      Mean: 150.0ms, Median: 145.0ms
      P95: 180.0ms, P99: 200.0ms
      Min: 120.0ms, Max: 210.0ms
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_infra.benchmarks.harness import (
    BenchmarkResult,
    BenchmarkSuite,
    benchmark,
    benchmark_sync,
    compare_results,
    run_benchmark_suite,
)
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
from ai_infra.benchmarks.ttft import (
    TTFT_TARGET_DEFAULT,
    TTFT_TARGET_FAST,
    TTFT_TARGET_STREAMING,
    TTFTBenchmark,
    TTFTMeasurement,
    TTFTProfile,
    TTFTProfileResult,
    TTFTTarget,
    measure_ttft,
)

if TYPE_CHECKING:
    pass


__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "benchmark",
    "benchmark_sync",
    "compare_results",
    "run_benchmark_suite",
    # TTFT exports
    "TTFTBenchmark",
    "TTFTMeasurement",
    "TTFTProfile",
    "TTFTProfileResult",
    "TTFTTarget",
    "TTFT_TARGET_DEFAULT",
    "TTFT_TARGET_FAST",
    "TTFT_TARGET_STREAMING",
    "measure_ttft",
    # Task completion exports
    "COMPLEX_TASK",
    "MEDIUM_TASK",
    "SIMPLE_TASK",
    "STANDARD_TASKS",
    "StandardTask",
    "TaskComplexity",
    "TaskCompletionBenchmark",
    "TaskCompletionResult",
    "TaskCompletionTarget",
    "compare_task_results",
    "get_target_for_complexity",
    "summarize_all_results",
    # Throughput exports
    "STANDARD_WORKLOADS",
    "THROUGHPUT_TARGET_BASELINE",
    "THROUGHPUT_TARGET_BATCH",
    "THROUGHPUT_TARGET_FAST",
    "ThroughputBenchmark",
    "ThroughputResult",
    "ThroughputTarget",
    "WORKLOAD_LARGE",
    "WORKLOAD_MEDIUM",
    "WORKLOAD_SMALL",
    "WorkloadConfig",
    "compare_throughput_results",
    "generate_multi_task_roadmap",
    "generate_varied_task_roadmap",
    "summarize_throughput_results",
]
