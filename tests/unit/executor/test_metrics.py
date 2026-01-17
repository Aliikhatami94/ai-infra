"""Tests for Phase 16.5.13: Orchestrator Observability and Metrics.

This module tests the metrics and observability functionality:
- OrchestratorMetrics dataclass
- RoutingRecord dataclass
- RoutingOutcome dataclass
- MetricsCollector class
- Misrouting detection heuristics
- Summary formatting
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from ai_infra.executor.agents.metrics import (
    MetricsCollector,
    OrchestratorMetrics,
    RoutingOutcome,
    RoutingRecord,
    RoutingTimer,
    check_routing_mismatch,
    format_metrics_summary,
)
from ai_infra.executor.agents.registry import SubAgentType

# =============================================================================
# OrchestratorMetrics Tests (16.5.13.1.1)
# =============================================================================


class TestOrchestratorMetrics:
    """Tests for OrchestratorMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values are initialized correctly."""
        metrics = OrchestratorMetrics()

        assert metrics.total_routings == 0
        assert metrics.routing_latency_ms == []
        assert metrics.routing_tokens == []
        assert metrics.confidence_scores == []
        assert metrics.fallback_count == 0
        assert metrics.agent_distribution == {}
        assert metrics.misroute_count == 0
        assert metrics.successful_tasks == 0
        assert metrics.failed_tasks == 0

    def test_avg_latency_empty(self) -> None:
        """Test avg_latency returns 0 when empty."""
        metrics = OrchestratorMetrics()
        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self) -> None:
        """Test average latency calculation."""
        metrics = OrchestratorMetrics(routing_latency_ms=[100.0, 200.0, 300.0])
        assert metrics.avg_latency_ms == 200.0

    def test_min_max_latency(self) -> None:
        """Test min and max latency calculations."""
        metrics = OrchestratorMetrics(routing_latency_ms=[100.0, 250.0, 300.0, 150.0])
        assert metrics.min_latency_ms == 100.0
        assert metrics.max_latency_ms == 300.0

    def test_avg_tokens_empty(self) -> None:
        """Test avg_tokens returns 0 when empty."""
        metrics = OrchestratorMetrics()
        assert metrics.avg_tokens == 0.0

    def test_avg_tokens_calculation(self) -> None:
        """Test average tokens calculation."""
        metrics = OrchestratorMetrics(routing_tokens=[300, 400, 500])
        assert metrics.avg_tokens == 400.0

    def test_avg_confidence_empty(self) -> None:
        """Test avg_confidence returns 0 when empty."""
        metrics = OrchestratorMetrics()
        assert metrics.avg_confidence == 0.0

    def test_avg_confidence_calculation(self) -> None:
        """Test average confidence calculation."""
        metrics = OrchestratorMetrics(confidence_scores=[0.8, 0.9, 1.0])
        assert metrics.avg_confidence == 0.9

    def test_fallback_rate_zero_routings(self) -> None:
        """Test fallback_rate returns 0 when no routings."""
        metrics = OrchestratorMetrics()
        assert metrics.fallback_rate == 0.0

    def test_fallback_rate_calculation(self) -> None:
        """Test fallback rate calculation."""
        metrics = OrchestratorMetrics(
            total_routings=10,
            fallback_count=2,
        )
        assert metrics.fallback_rate == 0.2

    def test_success_rate_no_outcomes(self) -> None:
        """Test success_rate returns 0 when no outcomes."""
        metrics = OrchestratorMetrics()
        assert metrics.success_rate == 0.0

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        metrics = OrchestratorMetrics(
            successful_tasks=8,
            failed_tasks=2,
        )
        assert metrics.success_rate == 0.8

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        metrics = OrchestratorMetrics(
            total_routings=5,
            routing_latency_ms=[100.0, 200.0],
            routing_tokens=[300, 400],
            confidence_scores=[0.9, 0.95],
            fallback_count=1,
            agent_distribution={"coder": 3, "testwriter": 2},
            misroute_count=1,
            successful_tasks=4,
            failed_tasks=1,
        )

        result = metrics.to_dict()

        assert result["total_routings"] == 5
        assert result["avg_latency_ms"] == 150.0
        assert result["min_latency_ms"] == 100.0
        assert result["max_latency_ms"] == 200.0
        assert result["avg_tokens"] == 350.0
        assert result["avg_confidence"] == 0.925
        assert result["fallback_count"] == 1
        assert result["fallback_rate"] == 0.2
        assert result["agent_distribution"] == {"coder": 3, "testwriter": 2}
        assert result["misroute_count"] == 1
        assert result["successful_tasks"] == 4
        assert result["failed_tasks"] == 1
        assert result["success_rate"] == 0.8


# =============================================================================
# RoutingOutcome Tests
# =============================================================================


class TestRoutingOutcome:
    """Tests for RoutingOutcome dataclass."""

    def test_successful_outcome(self) -> None:
        """Test creating a successful outcome."""
        outcome = RoutingOutcome(success=True)

        assert outcome.success is True
        assert outcome.error_message is None
        assert outcome.might_be_misrouted is False
        assert outcome.actual_work_done is None

    def test_failed_outcome_with_details(self) -> None:
        """Test creating a failed outcome with details."""
        outcome = RoutingOutcome(
            success=False,
            error_message="Test failed: assertion error",
            might_be_misrouted=True,
            actual_work_done="Created test file but tests failed",
        )

        assert outcome.success is False
        assert outcome.error_message == "Test failed: assertion error"
        assert outcome.might_be_misrouted is True
        assert outcome.actual_work_done == "Created test file but tests failed"


# =============================================================================
# RoutingRecord Tests
# =============================================================================


class TestRoutingRecord:
    """Tests for RoutingRecord dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic routing record."""
        record = RoutingRecord(
            task_id=1,
            task_title="Create tests for user.py",
            task_description="Write unit tests",
            agent_type=SubAgentType.TESTWRITER,
            confidence=0.92,
            latency_ms=1250.5,
            tokens_used=380,
            used_fallback=False,
            reasoning="Task mentions creating tests",
        )

        assert record.task_id == 1
        assert record.task_title == "Create tests for user.py"
        assert record.task_description == "Write unit tests"
        assert record.agent_type == SubAgentType.TESTWRITER
        assert record.confidence == 0.92
        assert record.latency_ms == 1250.5
        assert record.tokens_used == 380
        assert record.used_fallback is False
        assert record.reasoning == "Task mentions creating tests"
        assert isinstance(record.timestamp, datetime)
        assert record.outcome is None

    def test_to_dict_without_outcome(self) -> None:
        """Test serialization without outcome."""
        record = RoutingRecord(
            task_id=1,
            task_title="Test task",
            task_description=None,
            agent_type=SubAgentType.CODER,
            confidence=0.85,
            latency_ms=500.0,
            tokens_used=200,
            used_fallback=True,
            reasoning="Keyword match",
        )

        result = record.to_dict()

        assert result["task_id"] == 1
        assert result["task_title"] == "Test task"
        assert result["task_description"] is None
        assert result["agent_type"] == "coder"
        assert result["confidence"] == 0.85
        assert result["latency_ms"] == 500.0
        assert result["tokens_used"] == 200
        assert result["used_fallback"] is True
        assert result["reasoning"] == "Keyword match"
        assert "timestamp" in result
        assert "outcome" not in result

    def test_to_dict_with_outcome(self) -> None:
        """Test serialization with outcome."""
        record = RoutingRecord(
            task_id=1,
            task_title="Test task",
            task_description=None,
            agent_type=SubAgentType.CODER,
            confidence=0.85,
            latency_ms=500.0,
            tokens_used=200,
            used_fallback=False,
            reasoning="LLM decision",
        )
        record.outcome = RoutingOutcome(
            success=False,
            error_message="Syntax error",
            might_be_misrouted=True,
        )

        result = record.to_dict()

        assert "outcome" in result
        assert result["outcome"]["success"] is False
        assert result["outcome"]["error_message"] == "Syntax error"
        assert result["outcome"]["might_be_misrouted"] is True


# =============================================================================
# Misrouting Detection Tests (16.5.13.2.1)
# =============================================================================


class TestCheckRoutingMismatch:
    """Tests for check_routing_mismatch function."""

    def test_test_task_to_coder_detected(self) -> None:
        """Test that test task routed to Coder is detected as mismatch."""
        assert (
            check_routing_mismatch(
                "Create tests for user.py",
                None,
                SubAgentType.CODER,
            )
            is True
        )

    def test_test_task_to_testwriter_ok(self) -> None:
        """Test that test task routed to TestWriter is not a mismatch."""
        assert (
            check_routing_mismatch(
                "Create tests for user.py",
                None,
                SubAgentType.TESTWRITER,
            )
            is False
        )

    def test_run_tests_to_coder_detected(self) -> None:
        """Test that run tests task routed to Coder is detected."""
        assert (
            check_routing_mismatch(
                "Run tests to verify changes",
                None,
                SubAgentType.CODER,
            )
            is True
        )

    def test_run_tests_to_tester_ok(self) -> None:
        """Test that run tests task routed to Tester is not a mismatch."""
        assert (
            check_routing_mismatch(
                "Run tests to verify changes",
                None,
                SubAgentType.TESTER,
            )
            is False
        )

    def test_fix_bug_to_coder_detected(self) -> None:
        """Test that fix bug task routed to Coder is detected."""
        assert (
            check_routing_mismatch(
                "Fix the bug in user module",
                None,
                SubAgentType.CODER,
            )
            is True
        )

    def test_fix_bug_to_debugger_ok(self) -> None:
        """Test that fix bug task routed to Debugger is not a mismatch."""
        assert (
            check_routing_mismatch(
                "Fix the bug in user module",
                None,
                SubAgentType.DEBUGGER,
            )
            is False
        )

    def test_review_task_to_coder_detected(self) -> None:
        """Test that review task routed to Coder is detected."""
        assert (
            check_routing_mismatch(
                "Review and refactor the auth module",
                None,
                SubAgentType.CODER,
            )
            is True
        )

    def test_review_task_to_reviewer_ok(self) -> None:
        """Test that review task routed to Reviewer is not a mismatch."""
        assert (
            check_routing_mismatch(
                "Review and refactor the auth module",
                None,
                SubAgentType.REVIEWER,
            )
            is False
        )

    def test_implementation_task_to_coder_ok(self) -> None:
        """Test that implementation task routed to Coder is not a mismatch."""
        assert (
            check_routing_mismatch(
                "Implement user authentication",
                "Create login and registration functions",
                SubAgentType.CODER,
            )
            is False
        )

    def test_description_also_checked(self) -> None:
        """Test that description is also checked for keywords."""
        assert (
            check_routing_mismatch(
                "Update user module",
                "Fix the bug causing login failures",
                SubAgentType.CODER,
            )
            is True
        )


# =============================================================================
# MetricsCollector Tests (16.5.13.1.2)
# =============================================================================


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_initial_state(self) -> None:
        """Test collector starts with empty metrics."""
        collector = MetricsCollector()

        assert collector.metrics.total_routings == 0
        assert collector.records == []

    def test_record_routing_basic(self) -> None:
        """Test basic routing recording."""
        collector = MetricsCollector()

        collector.record_routing(
            task_id=1,
            task_title="Create user module",
            agent_type=SubAgentType.CODER,
            confidence=0.9,
            latency_ms=1000.0,
            tokens_used=400,
            used_fallback=False,
            reasoning="LLM decision",
        )

        metrics = collector.metrics
        assert metrics.total_routings == 1
        assert metrics.routing_latency_ms == [1000.0]
        assert metrics.routing_tokens == [400]
        assert metrics.confidence_scores == [0.9]
        assert metrics.fallback_count == 0
        assert metrics.agent_distribution == {"coder": 1}

    def test_record_multiple_routings(self) -> None:
        """Test recording multiple routing decisions."""
        collector = MetricsCollector()

        collector.record_routing(
            task_id=1,
            task_title="Task 1",
            agent_type=SubAgentType.CODER,
            confidence=0.9,
            latency_ms=1000.0,
        )
        collector.record_routing(
            task_id=2,
            task_title="Task 2",
            agent_type=SubAgentType.TESTWRITER,
            confidence=0.85,
            latency_ms=1200.0,
            used_fallback=True,
        )
        collector.record_routing(
            task_id=3,
            task_title="Task 3",
            agent_type=SubAgentType.CODER,
            confidence=0.95,
            latency_ms=800.0,
        )

        metrics = collector.metrics
        assert metrics.total_routings == 3
        assert metrics.fallback_count == 1
        assert metrics.agent_distribution == {"coder": 2, "testwriter": 1}
        assert metrics.avg_confidence == pytest.approx(0.9, abs=0.001)

    def test_record_routing_with_feedback_disabled(self) -> None:
        """Test that records are not stored when feedback is disabled."""
        collector = MetricsCollector(record_routing_feedback=False)

        collector.record_routing(
            task_id=1,
            task_title="Task 1",
            agent_type=SubAgentType.CODER,
            confidence=0.9,
            latency_ms=1000.0,
        )

        assert collector.records == []

    def test_record_routing_with_feedback_enabled(self) -> None:
        """Test that records are stored when feedback is enabled."""
        collector = MetricsCollector(record_routing_feedback=True)

        collector.record_routing(
            task_id=1,
            task_title="Task 1",
            task_description="Test description",
            agent_type=SubAgentType.CODER,
            confidence=0.9,
            latency_ms=1000.0,
            reasoning="Test reasoning",
        )

        assert len(collector.records) == 1
        record = collector.records[0]
        assert record.task_id == 1
        assert record.task_title == "Task 1"
        assert record.task_description == "Test description"
        assert record.reasoning == "Test reasoning"

    def test_record_outcome_success(self) -> None:
        """Test recording successful outcome."""
        collector = MetricsCollector(record_routing_feedback=True)

        collector.record_routing(
            task_id=1,
            task_title="Task 1",
            agent_type=SubAgentType.CODER,
            confidence=0.9,
            latency_ms=1000.0,
        )
        collector.record_outcome(task_id=1, success=True)

        assert collector.metrics.successful_tasks == 1
        assert collector.metrics.failed_tasks == 0
        assert collector.records[0].outcome is not None
        assert collector.records[0].outcome.success is True

    def test_record_outcome_failure(self) -> None:
        """Test recording failed outcome."""
        collector = MetricsCollector(record_routing_feedback=True)

        collector.record_routing(
            task_id=1,
            task_title="Task 1",
            agent_type=SubAgentType.CODER,
            confidence=0.9,
            latency_ms=1000.0,
        )
        collector.record_outcome(
            task_id=1,
            success=False,
            error_message="Compilation error",
        )

        assert collector.metrics.successful_tasks == 0
        assert collector.metrics.failed_tasks == 1
        record = collector.records[0]
        assert record.outcome is not None
        assert record.outcome.success is False
        assert record.outcome.error_message == "Compilation error"

    def test_record_outcome_detects_misroute(self) -> None:
        """Test that misrouting is detected on failure."""
        collector = MetricsCollector(record_routing_feedback=True)

        # Record a test task routed to Coder (potential misroute)
        collector.record_routing(
            task_id=1,
            task_title="Create tests for user.py",
            agent_type=SubAgentType.CODER,  # Wrong agent
            confidence=0.8,
            latency_ms=1000.0,
        )
        collector.record_outcome(task_id=1, success=False)

        assert collector.metrics.misroute_count == 1
        record = collector.records[0]
        assert record.outcome is not None
        assert record.outcome.might_be_misrouted is True

    def test_get_metrics_returns_copy(self) -> None:
        """Test that get_metrics returns a copy."""
        collector = MetricsCollector()

        collector.record_routing(
            task_id=1,
            task_title="Task 1",
            agent_type=SubAgentType.CODER,
            confidence=0.9,
            latency_ms=1000.0,
        )

        metrics1 = collector.get_metrics()
        metrics2 = collector.get_metrics()

        # Modify one copy
        metrics1.total_routings = 999

        # Other copy should be unchanged
        assert metrics2.total_routings == 1
        assert collector.metrics.total_routings == 1

    def test_reset(self) -> None:
        """Test resetting the collector."""
        collector = MetricsCollector(record_routing_feedback=True)

        collector.record_routing(
            task_id=1,
            task_title="Task 1",
            agent_type=SubAgentType.CODER,
            confidence=0.9,
            latency_ms=1000.0,
        )

        collector.reset()

        assert collector.metrics.total_routings == 0
        assert collector.records == []

    def test_format_summary(self) -> None:
        """Test format_summary returns formatted string."""
        collector = MetricsCollector()

        collector.record_routing(
            task_id=1,
            task_title="Task 1",
            agent_type=SubAgentType.CODER,
            confidence=0.9,
            latency_ms=1000.0,
        )

        summary = collector.format_summary()

        assert "Orchestrator Routing Summary" in summary
        assert "Routings:" in summary
        assert "1 total" in summary

    def test_export_records(self, tmp_path: Path) -> None:
        """Test exporting records to JSON file."""
        collector = MetricsCollector(record_routing_feedback=True)

        collector.record_routing(
            task_id=1,
            task_title="Task 1",
            agent_type=SubAgentType.CODER,
            confidence=0.9,
            latency_ms=1000.0,
            reasoning="Test",
        )

        export_path = tmp_path / "metrics" / "routing.json"
        collector.export_records(export_path)

        assert export_path.exists()

        import json

        data = json.loads(export_path.read_text())

        assert "metrics" in data
        assert "records" in data
        assert len(data["records"]) == 1
        assert data["records"][0]["task_id"] == 1


# =============================================================================
# RoutingTimer Tests
# =============================================================================


class TestRoutingTimer:
    """Tests for RoutingTimer context manager."""

    def test_timing_context(self) -> None:
        """Test basic timing functionality."""
        import time

        timer = RoutingTimer()

        with timer:
            time.sleep(0.01)  # Sleep 10ms

        # Should be at least 10ms
        assert timer.elapsed_ms >= 10.0
        # But not too long (allow some margin)
        assert timer.elapsed_ms < 100.0

    def test_initial_elapsed_is_zero(self) -> None:
        """Test that elapsed is 0 before use."""
        timer = RoutingTimer()
        assert timer.elapsed_ms == 0.0


# =============================================================================
# Summary Formatting Tests (16.5.13.1.3)
# =============================================================================


class TestFormatMetricsSummary:
    """Tests for format_metrics_summary function."""

    def test_empty_metrics(self) -> None:
        """Test formatting empty metrics."""
        metrics = OrchestratorMetrics()
        result = format_metrics_summary(metrics)

        assert result == "No routing decisions recorded."

    def test_basic_format(self) -> None:
        """Test basic formatting with metrics."""
        metrics = OrchestratorMetrics(
            total_routings=3,
            routing_latency_ms=[1000.0, 1200.0, 800.0],
            confidence_scores=[0.9, 0.85, 0.95],
            fallback_count=1,
            agent_distribution={"coder": 2, "testwriter": 1},
        )

        result = format_metrics_summary(metrics)

        assert "Orchestrator Routing Summary" in result
        assert "3 total" in result
        assert "1 fallbacks" in result
        assert "Coder: 2 tasks" in result
        assert "Testwriter: 1 task" in result

    def test_includes_tokens_when_available(self) -> None:
        """Test that tokens are shown when available."""
        metrics = OrchestratorMetrics(
            total_routings=1,
            routing_latency_ms=[1000.0],
            routing_tokens=[400],
            confidence_scores=[0.9],
            agent_distribution={"coder": 1},
        )

        result = format_metrics_summary(metrics)

        assert "Tokens:" in result
        assert "400" in result

    def test_includes_success_rate(self) -> None:
        """Test that success rate is shown when outcomes recorded."""
        metrics = OrchestratorMetrics(
            total_routings=10,
            routing_latency_ms=[1000.0] * 10,
            confidence_scores=[0.9] * 10,
            agent_distribution={"coder": 10},
            successful_tasks=8,
            failed_tasks=2,
        )

        result = format_metrics_summary(metrics)

        assert "Success:" in result
        assert "80%" in result

    def test_includes_misroute_count(self) -> None:
        """Test that misroute count is shown when detected."""
        metrics = OrchestratorMetrics(
            total_routings=5,
            routing_latency_ms=[1000.0] * 5,
            confidence_scores=[0.9] * 5,
            agent_distribution={"coder": 5},
            misroute_count=2,
        )

        result = format_metrics_summary(metrics)

        assert "Misroutes:" in result
        assert "2 detected" in result

    def test_box_formatting(self) -> None:
        """Test that output has box formatting."""
        metrics = OrchestratorMetrics(
            total_routings=1,
            routing_latency_ms=[1000.0],
            confidence_scores=[0.9],
            agent_distribution={"coder": 1},
        )

        result = format_metrics_summary(metrics)

        # Check box characters
        assert "╭" in result
        assert "╯" in result
        assert "│" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetricsIntegration:
    """Integration tests for metrics module."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: record, outcome, export."""
        collector = MetricsCollector(record_routing_feedback=True)

        # Record several routing decisions
        for i in range(5):
            agent = SubAgentType.TESTWRITER if i == 2 else SubAgentType.CODER
            collector.record_routing(
                task_id=i,
                task_title=f"Task {i}",
                agent_type=agent,
                confidence=0.85 + i * 0.02,
                latency_ms=1000.0 + i * 100,
                tokens_used=300 + i * 20,
            )

        # Record outcomes
        for i in range(5):
            success = i != 3  # Task 3 fails
            collector.record_outcome(task_id=i, success=success)

        # Check final metrics
        metrics = collector.get_metrics()
        assert metrics.total_routings == 5
        assert metrics.successful_tasks == 4
        assert metrics.failed_tasks == 1
        assert metrics.agent_distribution == {"coder": 4, "testwriter": 1}

        # Check summary
        summary = collector.format_summary()
        assert "5 total" in summary
        assert "80%" in summary  # 4/5 success rate
