"""Benchmarks for LLM operations.

Run with:
    make benchmark
    pytest benchmarks/bench_llm.py --benchmark-only
"""

from __future__ import annotations


class TestLLMInitialization:
    """Benchmark LLM initialization and setup."""

    def test_llm_import(self, benchmark):
        """Benchmark LLM module import time."""

        def import_llm():
            import importlib

            import ai_infra.llm

            importlib.reload(ai_infra.llm)

        benchmark(import_llm)

    def test_llm_init(self, benchmark, sample_messages):
        """Benchmark LLM class instantiation."""
        from ai_infra import LLM

        def create_llm():
            return LLM(provider="openai", model="gpt-4o-mini")

        result = benchmark(create_llm)
        assert result is not None


class TestMemoryOperations:
    """Benchmark memory and context management."""

    def test_count_tokens(self, benchmark, sample_messages):
        """Benchmark token counting."""
        from ai_infra.llm import count_tokens_approximate

        text = " ".join(m["content"] for m in sample_messages if m.get("content"))

        result = benchmark(count_tokens_approximate, text)
        assert result > 0

    def test_fit_context(self, benchmark, sample_messages):
        """Benchmark context fitting."""
        from ai_infra.llm import fit_context

        # Create many messages to test fitting
        messages = sample_messages * 100

        def run_fit():
            return fit_context(messages, max_tokens=4096)

        result = benchmark(run_fit)
        assert isinstance(result.messages, list)

    def test_memory_store_operations(self, benchmark):
        """Benchmark MemoryStore add/search."""
        from ai_infra.llm import MemoryItem, MemoryStore

        store = MemoryStore(max_items=1000)
        items = [
            MemoryItem(
                content=f"Memory item {i} with some content about topic {i % 10}",
                metadata={"index": i},
            )
            for i in range(100)
        ]

        # Pre-populate
        for item in items:
            store.add(item)

        def search_store():
            return store.search("topic 5", limit=10)

        result = benchmark(search_store)
        assert len(result) > 0


class TestAgentOperations:
    """Benchmark Agent class operations."""

    def test_agent_init(self, benchmark):
        """Benchmark Agent instantiation."""
        from ai_infra import Agent

        def create_agent():
            return Agent(
                name="benchmark_agent",
                instructions="You are a helpful assistant.",
                model="gpt-4o-mini",
            )

        result = benchmark(create_agent)
        assert result is not None
        assert result.name == "benchmark_agent"

    def test_subagent_compilation(self, benchmark):
        """Benchmark SubAgent compilation."""
        from ai_infra.llm import Agent, SubAgent

        parent = Agent(name="parent", instructions="Parent agent")
        sub = SubAgent(
            name="math_helper",
            instructions="You help with math problems.",
            handoff_keywords=["calculate", "math", "compute"],
        )

        def compile_subagent():
            return sub.compile(parent)

        result = benchmark(compile_subagent)
        assert result is not None
