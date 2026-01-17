"""Tests for Phase 16.5.1: Subagent Token Tracking.

Tests that token usage is properly captured and propagated from subagent
execution back to the main executor's progress tracking.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_infra.callbacks import LLMEndEvent, MetricsCallbacks
from ai_infra.executor.agents.base import (
    ExecutionContext,
    SubAgentResult,
)
from ai_infra.executor.progress import ProgressSummary, ProgressTracker, TaskProgress

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_context(tmp_path: Path) -> ExecutionContext:
    """Create a sample execution context."""
    return ExecutionContext(
        workspace=tmp_path,
        files_modified=["src/app.py"],
        project_type="python",
        summary="Test project",
    )


@pytest.fixture
def metrics_callback() -> MetricsCallbacks:
    """Create a MetricsCallbacks instance."""
    return MetricsCallbacks()


# =============================================================================
# MetricsCallbacks Tests
# =============================================================================


class TestMetricsCallbacks:
    """Tests for MetricsCallbacks token accumulation."""

    def test_initial_state(self, metrics_callback: MetricsCallbacks) -> None:
        """Callback should start with zero tokens."""
        assert metrics_callback.total_tokens == 0
        assert metrics_callback.llm_calls == 0

    def test_single_llm_end_event(self, metrics_callback: MetricsCallbacks) -> None:
        """Single LLM call should accumulate tokens."""
        event = LLMEndEvent(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            response="Test response",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=1000.0,
        )
        metrics_callback.on_llm_end(event)

        assert metrics_callback.total_tokens == 150
        assert metrics_callback.llm_calls == 1

    def test_multiple_llm_end_events(self, metrics_callback: MetricsCallbacks) -> None:
        """Multiple LLM calls should accumulate tokens."""
        events = [
            LLMEndEvent(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                response="Response 1",
                total_tokens=100,
            ),
            LLMEndEvent(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                response="Response 2",
                total_tokens=200,
            ),
            LLMEndEvent(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                response="Response 3",
                total_tokens=150,
            ),
        ]

        for event in events:
            metrics_callback.on_llm_end(event)

        assert metrics_callback.total_tokens == 450
        assert metrics_callback.llm_calls == 3

    def test_get_summary(self, metrics_callback: MetricsCallbacks) -> None:
        """Get summary should return proper structure."""
        event = LLMEndEvent(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            response="Test",
            total_tokens=500,
            latency_ms=2000.0,
        )
        metrics_callback.on_llm_end(event)

        summary = metrics_callback.get_summary()
        assert summary["llm"]["calls"] == 1
        assert summary["llm"]["total_tokens"] == 500


# =============================================================================
# SubAgentResult Tests
# =============================================================================


class TestSubAgentResultMetrics:
    """Tests for SubAgentResult metrics field."""

    def test_metrics_with_token_info(self) -> None:
        """SubAgentResult should store token metrics."""
        result = SubAgentResult(
            success=True,
            output="Task completed",
            files_created=["src/new_file.py"],
            metrics={
                "duration_ms": 5000,
                "tokens_in": 1000,
                "tokens_out": 0,
                "total_tokens": 1000,
                "model": "claude-sonnet-4-20250514",
                "agent_type": "Coder",
                "llm_calls": 3,
            },
        )

        assert result.metrics["total_tokens"] == 1000
        assert result.metrics["agent_type"] == "Coder"
        assert result.metrics["llm_calls"] == 3

    def test_to_dict_includes_metrics(self) -> None:
        """to_dict should include full metrics."""
        result = SubAgentResult(
            success=True,
            output="Done",
            metrics={
                "total_tokens": 2500,
                "model": "gpt-5-mini",
            },
        )

        d = result.to_dict()
        assert d["metrics"]["total_tokens"] == 2500
        assert d["metrics"]["model"] == "gpt-5-mini"


# =============================================================================
# TaskProgress Tests
# =============================================================================


class TestTaskProgressSubagentTokens:
    """Tests for TaskProgress subagent_tokens field."""

    def test_subagent_tokens_default(self) -> None:
        """subagent_tokens should default to 0."""
        progress = TaskProgress(
            task_id="1.1",
            task_title="Test task",
        )
        assert progress.subagent_tokens == 0

    def test_subagent_tokens_set(self) -> None:
        """subagent_tokens should be settable."""
        progress = TaskProgress(
            task_id="1.1",
            task_title="Test task",
            subagent_tokens=5000,
        )
        assert progress.subagent_tokens == 5000

    def test_total_tokens_includes_subagent(self) -> None:
        """total_tokens property should include subagent tokens."""
        progress = TaskProgress(
            task_id="1.1",
            task_title="Test task",
            tokens_in=1000,
            tokens_out=500,
            subagent_tokens=3000,
        )
        assert progress.total_tokens == 4500  # 1000 + 500 + 3000

    def test_to_dict_includes_subagent_tokens(self) -> None:
        """to_dict should include subagent_tokens."""
        progress = TaskProgress(
            task_id="1.1",
            task_title="Test task",
            subagent_tokens=2500,
        )
        d = progress.to_dict()
        assert d["subagent_tokens"] == 2500


# =============================================================================
# ProgressSummary Tests
# =============================================================================


class TestProgressSummarySubagentTokens:
    """Tests for ProgressSummary subagent_tokens field."""

    def test_subagent_tokens_default(self) -> None:
        """subagent_tokens should default to 0."""
        summary = ProgressSummary(
            total=5,
            completed=3,
            in_progress=1,
            pending=1,
            failed=0,
            skipped=0,
            percent=60.0,
            tokens_in=1000,
            tokens_out=500,
        )
        assert summary.subagent_tokens == 0

    def test_subagent_tokens_set(self) -> None:
        """subagent_tokens should be settable."""
        summary = ProgressSummary(
            total=5,
            completed=3,
            in_progress=1,
            pending=1,
            failed=0,
            skipped=0,
            percent=60.0,
            tokens_in=1000,
            tokens_out=500,
            subagent_tokens=10000,
        )
        assert summary.subagent_tokens == 10000

    def test_total_tokens_includes_subagent(self) -> None:
        """total_tokens property should include subagent tokens."""
        summary = ProgressSummary(
            total=5,
            completed=5,
            in_progress=0,
            pending=0,
            failed=0,
            skipped=0,
            percent=100.0,
            tokens_in=2000,
            tokens_out=1000,
            subagent_tokens=5000,
        )
        assert summary.total_tokens == 8000  # 2000 + 1000 + 5000

    def test_to_dict_includes_subagent_tokens(self) -> None:
        """to_dict should include subagent_tokens and total_tokens."""
        summary = ProgressSummary(
            total=1,
            completed=1,
            in_progress=0,
            pending=0,
            failed=0,
            skipped=0,
            percent=100.0,
            tokens_in=100,
            tokens_out=50,
            subagent_tokens=500,
        )
        d = summary.to_dict()
        assert d["subagent_tokens"] == 500
        assert d["total_tokens"] == 650


# =============================================================================
# ProgressTracker Integration Tests
# =============================================================================


class TestProgressTrackerSubagentTokens:
    """Tests for ProgressTracker subagent token aggregation."""

    def test_get_summary_aggregates_subagent_tokens(self) -> None:
        """get_summary should aggregate subagent_tokens across tasks."""
        # Create a simple roadmap mock that uses all_tasks()
        roadmap = MagicMock()
        roadmap.title = "Test Roadmap"
        roadmap.all_tasks.return_value = [
            MagicMock(id="1", title="Task 1"),
            MagicMock(id="2", title="Task 2"),
            MagicMock(id="3", title="Task 3"),
        ]

        tracker = ProgressTracker(roadmap)

        # Simulate task completion with subagent tokens
        tracker.tasks["1"].status = "completed"
        tracker.tasks["1"].subagent_tokens = 1000
        tracker.tasks["1"].duration = timedelta(seconds=10)

        tracker.tasks["2"].status = "completed"
        tracker.tasks["2"].subagent_tokens = 2000
        tracker.tasks["2"].duration = timedelta(seconds=15)

        tracker.tasks["3"].status = "completed"
        tracker.tasks["3"].subagent_tokens = 1500
        tracker.tasks["3"].duration = timedelta(seconds=12)

        summary = tracker.get_summary()
        assert summary.subagent_tokens == 4500  # 1000 + 2000 + 1500

    def test_get_summary_mixed_tokens(self) -> None:
        """get_summary should correctly sum all token types."""
        roadmap = MagicMock()
        roadmap.title = "Test Roadmap"
        roadmap.all_tasks.return_value = [
            MagicMock(id="1", title="Task 1"),
        ]

        tracker = ProgressTracker(roadmap)

        # Set various token types
        tracker.tasks["1"].status = "completed"
        tracker.tasks["1"].tokens_in = 500
        tracker.tasks["1"].tokens_out = 200
        tracker.tasks["1"].subagent_tokens = 3000
        tracker.tasks["1"].duration = timedelta(seconds=30)

        summary = tracker.get_summary()
        assert summary.tokens_in == 500
        assert summary.tokens_out == 200
        assert summary.subagent_tokens == 3000
        assert summary.total_tokens == 3700  # 500 + 200 + 3000
