"""Tests for per-node metrics tracking.

Phase 2.4.3: Unit tests for NodeMetrics, track_node_metrics decorator,
and aggregation/formatting functions.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_infra.executor.metrics import (
    AggregatedNodeMetrics,
    NodeMetrics,
    TokenTracker,
    aggregate_node_metrics,
    format_run_summary_with_nodes,
    track_node_metrics,
)
from ai_infra.executor.state import ExecutorGraphState

# =============================================================================
# NodeMetrics Tests
# =============================================================================


class TestNodeMetrics:
    """Tests for NodeMetrics dataclass."""

    def test_default_values(self):
        """Test default values for NodeMetrics."""
        metrics = NodeMetrics()
        assert metrics.tokens_in == 0
        assert metrics.tokens_out == 0
        assert metrics.duration_ms == 0
        assert metrics.llm_calls == 0
        assert metrics.invocations == 0

    def test_total_tokens(self):
        """Test total_tokens property."""
        metrics = NodeMetrics(tokens_in=100, tokens_out=200)
        assert metrics.total_tokens == 300

    def test_to_dict(self):
        """Test converting NodeMetrics to dict."""
        metrics = NodeMetrics(
            tokens_in=100,
            tokens_out=200,
            duration_ms=1500,
            llm_calls=3,
            invocations=2,
        )
        result = metrics.to_dict()

        assert result == {
            "tokens_in": 100,
            "tokens_out": 200,
            "duration_ms": 1500,
            "llm_calls": 3,
            "invocations": 2,
            "total_tokens": 300,
        }

    def test_from_dict(self):
        """Test creating NodeMetrics from dict."""
        data = {
            "tokens_in": 150,
            "tokens_out": 250,
            "duration_ms": 2000,
            "llm_calls": 5,
            "invocations": 3,
        }
        metrics = NodeMetrics.from_dict(data)

        assert metrics.tokens_in == 150
        assert metrics.tokens_out == 250
        assert metrics.duration_ms == 2000
        assert metrics.llm_calls == 5
        assert metrics.invocations == 3

    def test_from_dict_with_missing_keys(self):
        """Test from_dict handles missing keys gracefully."""
        data: dict[str, Any] = {}
        metrics = NodeMetrics.from_dict(data)

        assert metrics.tokens_in == 0
        assert metrics.tokens_out == 0
        assert metrics.duration_ms == 0
        assert metrics.llm_calls == 0
        assert metrics.invocations == 0

    def test_merge(self):
        """Test merging two NodeMetrics."""
        m1 = NodeMetrics(
            tokens_in=100,
            tokens_out=200,
            duration_ms=1000,
            llm_calls=2,
            invocations=1,
        )
        m2 = NodeMetrics(
            tokens_in=50,
            tokens_out=100,
            duration_ms=500,
            llm_calls=1,
            invocations=1,
        )

        merged = m1.merge(m2)

        assert merged.tokens_in == 150
        assert merged.tokens_out == 300
        assert merged.duration_ms == 1500
        assert merged.llm_calls == 3
        assert merged.invocations == 2

    def test_roundtrip(self):
        """Test to_dict and from_dict are consistent."""
        original = NodeMetrics(
            tokens_in=123,
            tokens_out=456,
            duration_ms=789,
            llm_calls=10,
            invocations=5,
        )
        roundtrip = NodeMetrics.from_dict(original.to_dict())

        assert roundtrip.tokens_in == original.tokens_in
        assert roundtrip.tokens_out == original.tokens_out
        assert roundtrip.duration_ms == original.duration_ms
        assert roundtrip.llm_calls == original.llm_calls
        assert roundtrip.invocations == original.invocations


# =============================================================================
# AggregatedNodeMetrics Tests
# =============================================================================


class TestAggregatedNodeMetrics:
    """Tests for AggregatedNodeMetrics."""

    def test_empty_metrics(self):
        """Test empty aggregated metrics."""
        agg = AggregatedNodeMetrics()

        assert agg.total_tokens == 0
        assert agg.total_tokens_in == 0
        assert agg.total_tokens_out == 0
        assert agg.total_duration_ms == 0
        assert agg.total_llm_calls == 0

    def test_total_calculations(self):
        """Test total calculations across nodes."""
        agg = AggregatedNodeMetrics(
            node_metrics={
                "execute_task": NodeMetrics(tokens_in=500, tokens_out=300, duration_ms=2000),
                "verify_task": NodeMetrics(tokens_in=100, tokens_out=50, duration_ms=500),
                "build_context": NodeMetrics(tokens_in=200, tokens_out=100, duration_ms=1000),
            }
        )

        assert agg.total_tokens == 1250  # 800 + 150 + 300
        assert agg.total_tokens_in == 800
        assert agg.total_tokens_out == 450
        assert agg.total_duration_ms == 3500

    def test_get_node_percentage(self):
        """Test percentage calculation for a node."""
        agg = AggregatedNodeMetrics(
            node_metrics={
                "execute_task": NodeMetrics(tokens_in=400, tokens_out=400),  # 800 = 80%
                "verify_task": NodeMetrics(tokens_in=100, tokens_out=100),  # 200 = 20%
            }
        )

        assert agg.get_node_percentage("execute_task") == 80.0
        assert agg.get_node_percentage("verify_task") == 20.0
        assert agg.get_node_percentage("nonexistent") == 0.0

    def test_get_node_percentage_empty(self):
        """Test percentage with no tokens."""
        agg = AggregatedNodeMetrics()
        assert agg.get_node_percentage("execute_task") == 0.0

    def test_to_dict(self):
        """Test converting to dict."""
        agg = AggregatedNodeMetrics(
            node_metrics={
                "execute_task": NodeMetrics(tokens_in=100, tokens_out=200, duration_ms=1000),
            }
        )
        result = agg.to_dict()

        assert "nodes" in result
        assert "execute_task" in result["nodes"]
        assert "totals" in result
        assert result["totals"]["total_tokens"] == 300
        assert result["totals"]["duration_ms"] == 1000

    def test_format_breakdown(self):
        """Test formatted breakdown output."""
        agg = AggregatedNodeMetrics(
            node_metrics={
                "execute_task": NodeMetrics(tokens_in=700, tokens_out=300, duration_ms=5000),
                "verify_task": NodeMetrics(tokens_in=100, tokens_out=100, duration_ms=1000),
            }
        )

        output = agg.format_breakdown()

        assert "Per-Node Breakdown" in output
        assert "execute_task" in output
        assert "verify_task" in output
        assert "1,000" in output  # total tokens formatted
        assert "Optimization target" in output  # highest consumer marked

    def test_format_breakdown_empty(self):
        """Test formatted breakdown with no metrics."""
        agg = AggregatedNodeMetrics()
        output = agg.format_breakdown()
        assert "No node metrics collected" in output


# =============================================================================
# TokenTracker Tests
# =============================================================================


class TestTokenTracker:
    """Tests for TokenTracker class."""

    def test_reset(self):
        """Test resetting token tracker."""
        TokenTracker.add_usage(tokens_in=100, tokens_out=200, llm_calls=5)
        TokenTracker.reset()

        tokens_in, tokens_out, llm_calls = TokenTracker.get_usage()
        assert tokens_in == 0
        assert tokens_out == 0
        assert llm_calls == 0

    def test_add_usage(self):
        """Test adding usage to tracker."""
        TokenTracker.reset()
        TokenTracker.add_usage(tokens_in=50, tokens_out=100, llm_calls=1)
        TokenTracker.add_usage(tokens_in=30, tokens_out=60, llm_calls=2)

        tokens_in, tokens_out, llm_calls = TokenTracker.get_usage()
        assert tokens_in == 80
        assert tokens_out == 160
        assert llm_calls == 3

    def test_get_usage(self):
        """Test getting current usage."""
        TokenTracker.reset()
        TokenTracker.add_usage(tokens_in=123, tokens_out=456, llm_calls=7)

        result = TokenTracker.get_usage()
        assert result == (123, 456, 7)


# =============================================================================
# track_node_metrics Decorator Tests
# =============================================================================


class TestTrackNodeMetricsDecorator:
    """Tests for track_node_metrics decorator."""

    @pytest.fixture
    def base_state(self) -> ExecutorGraphState:
        """Create base state for testing."""
        return ExecutorGraphState(
            roadmap_path="ROADMAP.md",
            todos=[],
            current_task=None,
            context="",
            prompt="",
            agent_result=None,
            files_modified=[],
            verified=False,
            last_checkpoint_sha=None,
            error=None,
            retry_count=0,
            replan_count=0,
            completed_count=0,
            max_tasks=None,
            should_continue=True,
            interrupt_requested=False,
            run_memory={},
            node_metrics={},
            enable_node_metrics=True,
        )

    @pytest.mark.asyncio
    async def test_decorator_tracks_duration(self, base_state: ExecutorGraphState):
        """Test that decorator tracks execution duration."""

        @track_node_metrics("test_node")
        async def slow_node(state: ExecutorGraphState) -> ExecutorGraphState:
            await asyncio.sleep(0.1)  # 100ms
            return {**state}

        result = await slow_node(base_state)

        assert "node_metrics" in result
        assert "test_node" in result["node_metrics"]
        # Duration should be at least 100ms
        assert result["node_metrics"]["test_node"]["duration_ms"] >= 100

    @pytest.mark.asyncio
    async def test_decorator_tracks_invocations(self, base_state: ExecutorGraphState):
        """Test that decorator tracks invocation count."""

        @track_node_metrics("counting_node")
        async def counting_node(state: ExecutorGraphState) -> ExecutorGraphState:
            return {**state}

        # Run twice
        result1 = await counting_node(base_state)
        result2 = await counting_node(result1)

        assert result2["node_metrics"]["counting_node"]["invocations"] == 2

    @pytest.mark.asyncio
    async def test_decorator_merges_with_existing(self, base_state: ExecutorGraphState):
        """Test that decorator merges with existing metrics."""
        state_with_metrics = {
            **base_state,
            "node_metrics": {
                "existing_node": NodeMetrics(tokens_in=100).to_dict(),
            },
        }

        @track_node_metrics("new_node")
        async def new_node(state: ExecutorGraphState) -> ExecutorGraphState:
            return {**state}

        result = await new_node(state_with_metrics)

        # Both nodes should be present
        assert "existing_node" in result["node_metrics"]
        assert "new_node" in result["node_metrics"]

    @pytest.mark.asyncio
    async def test_decorator_skips_when_disabled(self, base_state: ExecutorGraphState):
        """Test that decorator skips tracking when disabled."""
        state = {**base_state, "enable_node_metrics": False}

        @track_node_metrics("skipped_node")
        async def skipped_node(state: ExecutorGraphState) -> ExecutorGraphState:
            return {**state, "modified": True}

        result = await skipped_node(state)

        # Should return original state without node_metrics update
        assert result.get("modified") is True
        # node_metrics should not be updated with new node
        assert "skipped_node" not in result.get("node_metrics", {})


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_aggregate_node_metrics(self):
        """Test aggregating raw node metrics dict."""
        raw_metrics = {
            "execute_task": {
                "tokens_in": 500,
                "tokens_out": 300,
                "duration_ms": 2000,
                "llm_calls": 5,
                "invocations": 1,
            },
            "verify_task": {
                "tokens_in": 100,
                "tokens_out": 50,
                "duration_ms": 500,
                "llm_calls": 1,
                "invocations": 1,
            },
        }

        result = aggregate_node_metrics(raw_metrics)

        assert isinstance(result, AggregatedNodeMetrics)
        assert result.total_tokens == 950
        assert len(result.node_metrics) == 2

    def test_aggregate_node_metrics_none(self):
        """Test aggregating None returns empty."""
        result = aggregate_node_metrics(None)

        assert isinstance(result, AggregatedNodeMetrics)
        assert result.total_tokens == 0

    def test_format_run_summary_with_nodes(self):
        """Test formatting run summary with node metrics."""
        node_metrics = {
            "execute_task": {"tokens_in": 500, "tokens_out": 300, "duration_ms": 2000},
        }

        output = format_run_summary_with_nodes(
            total_duration_ms=5000,
            total_tokens=1000,
            node_metrics=node_metrics,
        )

        assert "Run Summary" in output
        assert "5,000ms" in output
        assert "1,000" in output
        assert "Per-Node Breakdown" in output
        assert "execute_task" in output

    def test_format_run_summary_without_nodes(self):
        """Test formatting run summary without node metrics."""
        output = format_run_summary_with_nodes(
            total_duration_ms=5000,
            total_tokens=1000,
            node_metrics=None,
        )

        assert "Run Summary" in output
        assert "5,000ms" in output
        assert "1,000" in output
        assert "Per-Node Breakdown" not in output
