"""Phase 16.5.11.6.4-5: Orchestrator Performance Benchmarks.

This module provides benchmarks for measuring orchestrator performance:
- Latency benchmark (<2s target per routing decision)
- Token cost benchmark (<500 tokens target per routing decision)

Run with: pytest benchmarks/bench_orchestrator.py -v --benchmark-enable
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ai_infra.executor.agents.orchestrator import (
    OrchestratorAgent,
    RoutingContext,
    RoutingDecision,
)
from ai_infra.executor.agents.registry import SubAgentType
from ai_infra.executor.todolist import TodoItem

# =============================================================================
# Benchmark Fixtures
# =============================================================================


@pytest.fixture
def benchmark_tasks() -> list[TodoItem]:
    """Generate diverse tasks for benchmarking."""
    return [
        TodoItem(id=1, title="Create src/user.py with User class", description=""),
        TodoItem(id=2, title="Write tests for authentication", description=""),
        TodoItem(id=3, title="Run pytest to verify", description=""),
        TodoItem(id=4, title="Fix the ImportError in main.py", description=""),
        TodoItem(id=5, title="Refactor service for performance", description=""),
        TodoItem(id=6, title="Research OAuth best practices", description=""),
        TodoItem(id=7, title="Create tests/test_api.py", description=""),
        TodoItem(id=8, title="Debug authentication failure", description=""),
        TodoItem(id=9, title="Implement payment service", description=""),
        TodoItem(id=10, title="Review code quality", description=""),
    ]


@pytest.fixture
def benchmark_context() -> RoutingContext:
    """Create a realistic routing context for benchmarks."""
    return RoutingContext(
        workspace=Path("/project"),
        completed_tasks=[
            "Initialize project",
            "Create basic structure",
            "Add configuration",
        ],
        existing_files=[
            "src/main.py",
            "src/config.py",
            "src/utils.py",
            "tests/test_main.py",
            "README.md",
            "pyproject.toml",
        ],
        project_type="python",
        previous_agent="coder",
    )


# =============================================================================
# Phase 16.5.11.6.4: Latency Benchmarks
# =============================================================================


class TestOrchestratorLatencyBenchmarks:
    """Benchmarks for orchestrator routing latency.

    Target: <2s per routing decision (with mocked LLM).
    """

    @pytest.mark.asyncio
    async def test_single_routing_latency(
        self,
        benchmark_tasks: list[TodoItem],
        benchmark_context: RoutingContext,
    ) -> None:
        """Benchmark single routing decision latency.

        Target: <2000ms with mocked LLM.
        """
        orchestrator = OrchestratorAgent()
        task = benchmark_tasks[0]

        latencies = []

        with patch.object(orchestrator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = RoutingDecision(
                agent_type=SubAgentType.CODER,
                confidence=0.9,
                reasoning="Implementation task",
            )

            for _ in range(10):
                start = time.perf_counter()
                await orchestrator.route(task, benchmark_context)
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print("\n=== Single Routing Latency Benchmark ===")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"Max: {max_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")

        # Target: <2000ms (mocked should be <100ms)
        assert avg_latency < 2000, f"Average latency {avg_latency:.0f}ms exceeds 2s"
        assert max_latency < 2000, f"Max latency {max_latency:.0f}ms exceeds 2s"

    @pytest.mark.asyncio
    async def test_batch_routing_latency(
        self,
        benchmark_tasks: list[TodoItem],
        benchmark_context: RoutingContext,
    ) -> None:
        """Benchmark routing 10 tasks sequentially.

        Target: <20s total (2s per task average).
        """
        orchestrator = OrchestratorAgent()

        with patch.object(orchestrator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = RoutingDecision(
                agent_type=SubAgentType.CODER,
                confidence=0.9,
                reasoning="Task routing",
            )

            start = time.perf_counter()
            for task in benchmark_tasks:
                await orchestrator.route(task, benchmark_context)
            total_time_ms = (time.perf_counter() - start) * 1000

        avg_per_task = total_time_ms / len(benchmark_tasks)

        print("\n=== Batch Routing Latency Benchmark ===")
        print(f"Total: {total_time_ms:.2f}ms")
        print(f"Per task: {avg_per_task:.2f}ms")

        # Target: <2000ms average per task
        assert avg_per_task < 2000, f"Per-task latency {avg_per_task:.0f}ms exceeds 2s"

    def test_prompt_building_latency(
        self,
        benchmark_tasks: list[TodoItem],
        benchmark_context: RoutingContext,
    ) -> None:
        """Benchmark prompt building latency (non-async).

        Target: <50ms per prompt.
        """
        orchestrator = OrchestratorAgent()

        latencies = []

        for task in benchmark_tasks:
            start = time.perf_counter()
            _ = orchestrator._build_routing_prompt(task, benchmark_context)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)

        print("\n=== Prompt Building Latency Benchmark ===")
        print(f"Average: {avg_latency:.3f}ms")
        print(f"Max: {max_latency:.3f}ms")

        # Target: <50ms for prompt building
        assert avg_latency < 50, f"Prompt building {avg_latency:.1f}ms exceeds 50ms"

    def test_keyword_fallback_latency(
        self,
        benchmark_tasks: list[TodoItem],
    ) -> None:
        """Benchmark keyword fallback latency.

        Target: <10ms (fast fallback).
        """
        orchestrator = OrchestratorAgent()

        latencies = []

        for task in benchmark_tasks:
            start = time.perf_counter()
            _ = orchestrator._keyword_fallback(task)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)

        print("\n=== Keyword Fallback Latency Benchmark ===")
        print(f"Average: {avg_latency:.3f}ms")
        print(f"Max: {max_latency:.3f}ms")

        # Target: <10ms for keyword fallback
        assert avg_latency < 10, f"Keyword fallback {avg_latency:.1f}ms exceeds 10ms"


# =============================================================================
# Phase 16.5.11.6.5: Token Cost Benchmarks
# =============================================================================


class TestOrchestratorTokenCostBenchmarks:
    """Benchmarks for orchestrator token usage.

    Target: <500 tokens per routing decision.
    """

    CHARS_PER_TOKEN = 4  # Rough estimate for English text

    def test_routing_prompt_token_count(
        self,
        benchmark_tasks: list[TodoItem],
        benchmark_context: RoutingContext,
    ) -> None:
        """Benchmark routing prompt token count.

        Target: <500 tokens per prompt.
        """
        orchestrator = OrchestratorAgent()

        token_counts = []

        for task in benchmark_tasks:
            prompt = orchestrator._build_routing_prompt(task, benchmark_context)
            estimated_tokens = len(prompt) / self.CHARS_PER_TOKEN
            token_counts.append(estimated_tokens)

        avg_tokens = statistics.mean(token_counts)
        max_tokens = max(token_counts)

        print("\n=== Routing Prompt Token Benchmark ===")
        print(f"Average: ~{avg_tokens:.0f} tokens")
        print(f"Max: ~{max_tokens:.0f} tokens")

        # Target: <500 tokens per routing prompt
        assert avg_tokens < 500, f"Average prompt ~{avg_tokens:.0f} tokens exceeds 500"
        assert max_tokens < 500, f"Max prompt ~{max_tokens:.0f} tokens exceeds 500"

    def test_system_prompt_token_count(self) -> None:
        """Benchmark system prompt token count.

        Target: <1000 tokens for system prompt.
        """
        from ai_infra.executor.agents.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

        estimated_tokens = len(ORCHESTRATOR_SYSTEM_PROMPT) / self.CHARS_PER_TOKEN

        print("\n=== System Prompt Token Benchmark ===")
        print(f"System prompt: ~{estimated_tokens:.0f} tokens")

        # Target: <1000 tokens for system prompt
        assert estimated_tokens < 1000, f"System prompt ~{estimated_tokens:.0f} exceeds 1000"

    def test_total_request_token_count(
        self,
        benchmark_context: RoutingContext,
    ) -> None:
        """Benchmark total request token count (system + user).

        Target: <1500 tokens total per request.
        """
        from ai_infra.executor.agents.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

        orchestrator = OrchestratorAgent()
        task = TodoItem(
            id=1,
            title="Create tests for user authentication",
            description="Write comprehensive unit tests",
        )

        routing_prompt = orchestrator._build_routing_prompt(task, benchmark_context)
        system_tokens = len(ORCHESTRATOR_SYSTEM_PROMPT) / self.CHARS_PER_TOKEN
        user_tokens = len(routing_prompt) / self.CHARS_PER_TOKEN
        total_tokens = system_tokens + user_tokens

        print("\n=== Total Request Token Benchmark ===")
        print(f"System: ~{system_tokens:.0f} tokens")
        print(f"User: ~{user_tokens:.0f} tokens")
        print(f"Total: ~{total_tokens:.0f} tokens")

        # Target: <1500 tokens total per request
        assert total_tokens < 1500, f"Total ~{total_tokens:.0f} tokens exceeds 1500"

    def test_response_token_estimate(self) -> None:
        """Estimate response token count.

        Target: <100 tokens per response.
        """
        # Typical routing response
        typical_response = """{
            "agent_type": "CODER",
            "confidence": 0.95,
            "reasoning": "Task involves creating a new Python file with implementation code"
        }"""

        estimated_tokens = len(typical_response) / self.CHARS_PER_TOKEN

        print("\n=== Response Token Benchmark ===")
        print(f"Typical response: ~{estimated_tokens:.0f} tokens")

        # Target: <100 tokens per response
        assert estimated_tokens < 100, f"Response ~{estimated_tokens:.0f} tokens exceeds 100"


# =============================================================================
# Combined Performance Report
# =============================================================================


class TestOrchestratorPerformanceReport:
    """Generate combined performance report."""

    @pytest.mark.asyncio
    async def test_generate_performance_report(
        self,
        benchmark_tasks: list[TodoItem],
        benchmark_context: RoutingContext,
    ) -> None:
        """Generate comprehensive performance report.

        Phase 16.5.11.6.7: Combined latency and token metrics.
        """
        from ai_infra.executor.agents.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

        orchestrator = OrchestratorAgent()
        chars_per_token = 4

        # Collect metrics
        latencies = []
        token_counts = []
        keyword_latencies = []

        with patch.object(orchestrator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = RoutingDecision(
                agent_type=SubAgentType.CODER,
                confidence=0.9,
                reasoning="Routing decision",
            )

            for task in benchmark_tasks:
                # Route latency
                start = time.perf_counter()
                await orchestrator.route(task, benchmark_context)
                latencies.append((time.perf_counter() - start) * 1000)

                # Keyword fallback latency
                start = time.perf_counter()
                _ = orchestrator._keyword_fallback(task)
                keyword_latencies.append((time.perf_counter() - start) * 1000)

                # Token count
                prompt = orchestrator._build_routing_prompt(task, benchmark_context)
                token_counts.append(len(prompt) / chars_per_token)

        system_tokens = len(ORCHESTRATOR_SYSTEM_PROMPT) / chars_per_token

        # Generate report
        report = f"""
========================================
ORCHESTRATOR PERFORMANCE REPORT
========================================

LATENCY METRICS (Target: <2000ms)
---------------------------------
Routing (mocked LLM):
  Average: {statistics.mean(latencies):.2f}ms
  Max: {max(latencies):.2f}ms
  P50: {sorted(latencies)[len(latencies) // 2]:.2f}ms

Keyword Fallback:
  Average: {statistics.mean(keyword_latencies):.3f}ms
  Max: {max(keyword_latencies):.3f}ms

TOKEN METRICS (Target: <500 per request)
----------------------------------------
System Prompt: ~{system_tokens:.0f} tokens
User Prompt:
  Average: ~{statistics.mean(token_counts):.0f} tokens
  Max: ~{max(token_counts):.0f} tokens

Total per Request:
  Average: ~{system_tokens + statistics.mean(token_counts):.0f} tokens
  Max: ~{system_tokens + max(token_counts):.0f} tokens

TARGETS
-------
[{"PASS" if statistics.mean(latencies) < 2000 else "FAIL"}] Latency <2000ms: {statistics.mean(latencies):.0f}ms
[{"PASS" if statistics.mean(token_counts) < 500 else "FAIL"}] Tokens <500: ~{statistics.mean(token_counts):.0f}
========================================
"""
        print(report)

        # Assertions
        assert statistics.mean(latencies) < 2000, "Latency exceeds 2s target"
        assert statistics.mean(token_counts) < 500, "Token count exceeds 500 target"
