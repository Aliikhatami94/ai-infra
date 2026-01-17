"""Tests for executor observability module.

Tests structured logging, metrics collection, and tracing integration.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_infra.callbacks import LLMEndEvent, LLMErrorEvent, ToolEndEvent, ToolErrorEvent
from ai_infra.executor.tracing import (
    ExecutorCallbacks,
    ExecutorMetrics,
    TaskMetrics,
    get_executor_tracer,
    log_recovery_action,
    log_task_context,
    log_verification_result,
)

# =============================================================================
# TaskMetrics Tests
# =============================================================================


class TestTaskMetrics:
    """Tests for TaskMetrics dataclass."""

    def test_create_task_metrics(self) -> None:
        """Test creating task metrics with default values."""
        metrics = TaskMetrics(task_id="task-1")
        assert metrics.task_id == "task-1"
        assert metrics.started_at is None
        assert metrics.completed_at is None
        assert metrics.duration_ms == 0.0
        assert metrics.llm_calls == 0
        assert metrics.llm_tokens == 0
        assert metrics.tool_calls == 0
        assert metrics.tool_duration_ms == 0.0
        assert metrics.files_modified == 0
        assert not metrics.success
        assert metrics.error is None

    def test_task_metrics_with_values(self) -> None:
        """Test creating task metrics with custom values."""
        started = datetime.now(UTC)
        completed = datetime.now(UTC)
        metrics = TaskMetrics(
            task_id="task-2",
            started_at=started,
            completed_at=completed,
            duration_ms=1500.0,
            llm_calls=5,
            llm_tokens=2500,
            tool_calls=10,
            tool_duration_ms=500.0,
            files_modified=3,
            success=True,
        )
        assert metrics.task_id == "task-2"
        assert metrics.duration_ms == 1500.0
        assert metrics.llm_calls == 5
        assert metrics.llm_tokens == 2500
        assert metrics.success

    def test_task_metrics_to_dict(self) -> None:
        """Test converting task metrics to dictionary."""
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        metrics = TaskMetrics(
            task_id="task-3",
            started_at=started,
            duration_ms=1000.0,
            success=True,
        )
        d = metrics.to_dict()
        assert d["task_id"] == "task-3"
        assert d["started_at"] == started.isoformat()
        assert d["completed_at"] is None
        assert d["duration_ms"] == 1000.0
        assert d["success"] is True


# =============================================================================
# ExecutorMetrics Tests
# =============================================================================


class TestExecutorMetrics:
    """Tests for ExecutorMetrics dataclass."""

    def test_create_executor_metrics(self) -> None:
        """Test creating executor metrics with default values."""
        metrics = ExecutorMetrics()
        assert metrics.run_id is None
        assert metrics.tasks_started == 0
        assert metrics.tasks_completed == 0
        assert metrics.tasks_failed == 0
        assert metrics.tasks_skipped == 0
        assert metrics.total_llm_calls == 0
        assert metrics.total_tokens == 0
        assert metrics.checkpoints_created == 0
        assert metrics.rollbacks_performed == 0

    def test_avg_task_duration_empty(self) -> None:
        """Test average duration with no tasks."""
        metrics = ExecutorMetrics()
        assert metrics.avg_task_duration_ms == 0.0

    def test_avg_task_duration_with_tasks(self) -> None:
        """Test average duration with completed tasks."""
        metrics = ExecutorMetrics(
            tasks_completed=3,
            tasks_failed=1,
            total_duration_ms=4000.0,
        )
        assert metrics.avg_task_duration_ms == 1000.0

    def test_avg_llm_calls_per_task(self) -> None:
        """Test average LLM calls per task."""
        metrics = ExecutorMetrics(
            tasks_completed=2,
            tasks_failed=2,
            total_llm_calls=20,
        )
        assert metrics.avg_llm_calls_per_task == 5.0

    def test_avg_tokens_per_task(self) -> None:
        """Test average tokens per task."""
        metrics = ExecutorMetrics(
            tasks_completed=4,
            total_tokens=10000,
        )
        assert metrics.avg_tokens_per_task == 2500.0

    def test_success_rate_empty(self) -> None:
        """Test success rate with no tasks."""
        metrics = ExecutorMetrics()
        assert metrics.success_rate == 0.0

    def test_success_rate_all_success(self) -> None:
        """Test success rate with all successful tasks."""
        metrics = ExecutorMetrics(
            tasks_completed=10,
            tasks_failed=0,
        )
        assert metrics.success_rate == 100.0

    def test_success_rate_mixed(self) -> None:
        """Test success rate with mixed results."""
        metrics = ExecutorMetrics(
            tasks_completed=7,
            tasks_failed=3,
        )
        assert metrics.success_rate == 70.0

    def test_to_dict(self) -> None:
        """Test converting metrics to dictionary."""
        metrics = ExecutorMetrics(
            run_id="run-123",
            tasks_completed=5,
            tasks_failed=2,
            total_tokens=5000,
        )
        d = metrics.to_dict()
        assert d["run_id"] == "run-123"
        assert d["tasks"]["completed"] == 5
        assert d["tasks"]["failed"] == 2
        assert d["llm"]["total_tokens"] == 5000
        assert "task_metrics" in d

    def test_summary_format(self) -> None:
        """Test human-readable summary format."""
        metrics = ExecutorMetrics(
            run_id="run-summary",
            tasks_completed=3,
            tasks_failed=1,
            total_tokens=3000,
            total_duration_ms=5000.0,
        )
        summary = metrics.summary()
        assert "Executor Metrics Summary" in summary
        assert "run-summary" in summary
        assert "Completed: 3" in summary
        assert "Failed:    1" in summary
        assert "3,000" in summary  # formatted tokens


# =============================================================================
# ExecutorCallbacks Tests
# =============================================================================


class TestExecutorCallbacks:
    """Tests for ExecutorCallbacks class."""

    def test_init(self) -> None:
        """Test callback initialization."""
        callbacks = ExecutorCallbacks()
        assert callbacks._current_task is None
        metrics = callbacks.get_metrics()
        assert metrics.run_id is None
        assert metrics.tasks_started == 0

    def test_set_run_context(self) -> None:
        """Test setting run context."""
        callbacks = ExecutorCallbacks()
        callbacks.set_run_context("test-run-123")
        metrics = callbacks.get_metrics()
        assert metrics.run_id == "test-run-123"
        assert metrics.started_at is not None

    def test_on_run_end(self) -> None:
        """Test run end callback."""
        callbacks = ExecutorCallbacks()
        callbacks.set_run_context("test-run")
        callbacks.on_run_end()
        metrics = callbacks.get_metrics()
        assert metrics.completed_at is not None

    def test_on_task_start(self) -> None:
        """Test task start callback."""
        callbacks = ExecutorCallbacks()
        callbacks.on_task_start("task-1", "Test Task")
        metrics = callbacks.get_metrics()
        assert metrics.tasks_started == 1
        current = callbacks.get_current_task_metrics()
        assert current is not None
        assert current.task_id == "task-1"

    def test_on_task_end_success(self) -> None:
        """Test task end callback for success."""
        callbacks = ExecutorCallbacks()
        callbacks.on_task_start("task-1", "Test Task")
        callbacks.on_task_end("task-1", success=True, files_modified=5)
        metrics = callbacks.get_metrics()
        assert metrics.tasks_completed == 1
        assert metrics.tasks_failed == 0
        assert len(metrics.task_metrics) == 1
        assert metrics.task_metrics[0].success
        assert metrics.task_metrics[0].files_modified == 5

    def test_on_task_end_failure(self) -> None:
        """Test task end callback for failure."""
        callbacks = ExecutorCallbacks()
        callbacks.on_task_start("task-2", "Failing Task")
        callbacks.on_task_end("task-2", success=False, error="Test error")
        metrics = callbacks.get_metrics()
        assert metrics.tasks_completed == 0
        assert metrics.tasks_failed == 1
        assert len(metrics.task_metrics) == 1
        assert not metrics.task_metrics[0].success
        assert metrics.task_metrics[0].error == "Test error"

    def test_on_task_skip(self) -> None:
        """Test task skip callback."""
        callbacks = ExecutorCallbacks()
        callbacks.on_task_skip("task-3", "dry_run")
        metrics = callbacks.get_metrics()
        assert metrics.tasks_skipped == 1

    def test_on_llm_end(self) -> None:
        """Test LLM end callback."""
        callbacks = ExecutorCallbacks()
        callbacks.on_task_start("task-1", "Test")
        event = LLMEndEvent(
            provider="anthropic",
            model="claude-3",
            response="Test response",
            latency_ms=100.0,
            total_tokens=500,
        )
        callbacks.on_llm_end(event)
        metrics = callbacks.get_metrics()
        assert metrics.total_llm_calls == 1
        assert metrics.total_tokens == 500
        current = callbacks.get_current_task_metrics()
        assert current is not None
        assert current.llm_calls == 1
        assert current.llm_tokens == 500

    def test_on_llm_error(self) -> None:
        """Test LLM error callback."""
        callbacks = ExecutorCallbacks()
        callbacks.on_task_start("task-1", "Test")
        event = LLMErrorEvent(
            provider="anthropic",
            model="claude-3",
            latency_ms=50.0,
            error=Exception("API error"),
        )
        # Should not raise
        callbacks.on_llm_error(event)

    def test_on_tool_end(self) -> None:
        """Test tool end callback."""
        callbacks = ExecutorCallbacks()
        callbacks.on_task_start("task-1", "Test")
        event = ToolEndEvent(
            tool_name="read_file",
            latency_ms=50.0,
            result={"content": "data"},
        )
        callbacks.on_tool_end(event)
        metrics = callbacks.get_metrics()
        assert metrics.total_tool_calls == 1
        current = callbacks.get_current_task_metrics()
        assert current is not None
        assert current.tool_calls == 1
        assert current.tool_duration_ms == 50.0

    def test_on_tool_error(self) -> None:
        """Test tool error callback."""
        callbacks = ExecutorCallbacks()
        callbacks.on_task_start("task-1", "Test")
        event = ToolErrorEvent(
            tool_name="write_file",
            error=PermissionError("Access denied"),
            arguments={"path": "/etc/passwd"},
            latency_ms=10.0,
        )
        # Should not raise
        callbacks.on_tool_error(event)

    def test_on_checkpoint_created(self) -> None:
        """Test checkpoint created callback."""
        callbacks = ExecutorCallbacks()
        callbacks.on_checkpoint_created("abc123", "task-1")
        metrics = callbacks.get_metrics()
        assert metrics.checkpoints_created == 1

    def test_on_rollback(self) -> None:
        """Test rollback callback."""
        callbacks = ExecutorCallbacks()
        callbacks.on_rollback("abc123", "verification failed")
        metrics = callbacks.get_metrics()
        assert metrics.rollbacks_performed == 1

    def test_on_dependency_warning(self) -> None:
        """Test dependency warning callback - should not raise."""
        callbacks = ExecutorCallbacks()
        callbacks.on_dependency_warning(
            file_path="src/utils.py",
            impact="high",
            affected_files=["src/main.py", "src/app.py"],
        )
        # Just verify it doesn't raise

    def test_reset(self) -> None:
        """Test resetting callbacks."""
        callbacks = ExecutorCallbacks()
        callbacks.set_run_context("run-1")
        callbacks.on_task_start("task-1", "Test")
        callbacks.on_task_end("task-1", success=True)
        callbacks.on_checkpoint_created("abc", "task-1")

        callbacks.reset()
        metrics = callbacks.get_metrics()
        assert metrics.run_id is None
        assert metrics.tasks_started == 0
        assert metrics.tasks_completed == 0
        assert metrics.checkpoints_created == 0

    def test_multiple_tasks(self) -> None:
        """Test tracking multiple tasks."""
        callbacks = ExecutorCallbacks()
        callbacks.set_run_context("multi-task-run")

        # Task 1 - success
        callbacks.on_task_start("task-1", "First Task")
        callbacks.on_llm_end(
            LLMEndEvent(
                provider="anthropic",
                model="claude-3",
                response="Response 1",
                latency_ms=100.0,
                total_tokens=1000,
            )
        )
        callbacks.on_task_end("task-1", success=True, files_modified=2)

        # Task 2 - failure
        callbacks.on_task_start("task-2", "Second Task")
        callbacks.on_llm_end(
            LLMEndEvent(
                provider="anthropic",
                model="claude-3",
                response="Response 2",
                latency_ms=200.0,
                total_tokens=500,
            )
        )
        callbacks.on_task_end("task-2", success=False, error="Failed")

        # Task 3 - success
        callbacks.on_task_start("task-3", "Third Task")
        callbacks.on_task_end("task-3", success=True, files_modified=1)

        callbacks.on_run_end()

        metrics = callbacks.get_metrics()
        assert metrics.tasks_started == 3
        assert metrics.tasks_completed == 2
        assert metrics.tasks_failed == 1
        assert metrics.total_llm_calls == 2
        assert metrics.total_tokens == 1500
        assert len(metrics.task_metrics) == 3
        assert metrics.success_rate == pytest.approx(66.67, rel=0.01)


# =============================================================================
# Tracing Utilities Tests
# =============================================================================


class TestTracingUtilities:
    """Tests for tracing utility functions."""

    def test_get_executor_tracer(self) -> None:
        """Test getting executor tracer."""
        tracer = get_executor_tracer()
        assert tracer is not None
        # Should have span method
        assert hasattr(tracer, "span")
        assert hasattr(tracer, "start_span")
        assert hasattr(tracer, "end_span")


# =============================================================================
# Logging Utilities Tests
# =============================================================================


class TestLoggingUtilities:
    """Tests for logging utility functions."""

    def test_log_task_context(self) -> None:
        """Test logging task context - should not raise."""
        log_task_context(
            task_id="task-1",
            context_tokens=5000,
            files_included=10,
            file_hints=["src/main.py", "src/utils.py"],
        )

    def test_log_task_context_no_hints(self) -> None:
        """Test logging task context without file hints."""
        log_task_context(
            task_id="task-2",
            context_tokens=3000,
            files_included=5,
        )

    def test_log_verification_result_passed(self) -> None:
        """Test logging verification result - passed."""
        log_verification_result(
            task_id="task-1",
            passed=True,
            verification_level="tests",
            duration_ms=1500.0,
        )

    def test_log_verification_result_failed(self) -> None:
        """Test logging verification result - failed."""
        log_verification_result(
            task_id="task-2",
            passed=False,
            verification_level="syntax",
            duration_ms=100.0,
            details="Syntax error on line 42",
        )

    def test_log_recovery_action_success(self) -> None:
        """Test logging recovery action - success."""
        log_recovery_action(
            action="rollback",
            checkpoint_id="abc123",
            files_affected=5,
            success=True,
        )

    def test_log_recovery_action_failure(self) -> None:
        """Test logging recovery action - failure."""
        log_recovery_action(
            action="restore",
            checkpoint_id="def456",
            files_affected=0,
            success=False,
            error="Checkpoint not found",
        )


# =============================================================================
# Integration Tests
# =============================================================================


class TestObservabilityIntegration:
    """Integration tests for observability features."""

    def test_full_run_lifecycle(self) -> None:
        """Test complete run lifecycle with all callbacks."""
        callbacks = ExecutorCallbacks()

        # Start run
        callbacks.set_run_context("integration-run")

        # Execute multiple tasks with various events
        for i in range(3):
            task_id = f"task-{i + 1}"
            callbacks.on_task_start(task_id, f"Task {i + 1}")

            # Simulate LLM calls
            for j in range(2):
                callbacks.on_llm_end(
                    LLMEndEvent(
                        provider="anthropic",
                        model="claude-3",
                        response=f"Response {i}-{j}",
                        latency_ms=50.0,
                        total_tokens=100,
                    )
                )

            # Simulate tool calls
            for _ in range(3):
                callbacks.on_tool_end(
                    ToolEndEvent(
                        tool_name="test_tool",
                        latency_ms=10.0,
                        result={},
                    )
                )

            # Complete task
            callbacks.on_task_end(task_id, success=True, files_modified=i + 1)

            # Simulate checkpoint
            callbacks.on_checkpoint_created(f"commit-{i}", task_id)

        # End run
        callbacks.on_run_end()

        # Verify metrics
        metrics = callbacks.get_metrics()
        assert metrics.run_id == "integration-run"
        assert metrics.tasks_started == 3
        assert metrics.tasks_completed == 3
        assert metrics.tasks_failed == 0
        assert metrics.total_llm_calls == 6  # 2 per task
        assert metrics.total_tokens == 600  # 100 * 2 * 3
        assert metrics.total_tool_calls == 9  # 3 per task
        assert metrics.checkpoints_created == 3
        assert metrics.success_rate == 100.0

    def test_metrics_serialization(self) -> None:
        """Test that metrics can be serialized to dict/JSON."""
        callbacks = ExecutorCallbacks()
        callbacks.set_run_context("serialize-test")
        callbacks.on_task_start("task-1", "Test Task")
        callbacks.on_task_end("task-1", success=True)
        callbacks.on_run_end()

        metrics = callbacks.get_metrics()
        d = metrics.to_dict()

        # Should be serializable
        import json

        json_str = json.dumps(d)
        assert "serialize-test" in json_str
        assert "task-1" in json_str

    def test_callbacks_without_current_task(self) -> None:
        """Test callbacks work even without current task context."""
        callbacks = ExecutorCallbacks()

        # These should not raise even without current task
        callbacks.on_llm_end(
            LLMEndEvent(
                provider="test",
                model="test",
                response="test response",
                latency_ms=10.0,
                total_tokens=50,
            )
        )
        callbacks.on_tool_end(
            ToolEndEvent(
                tool_name="test",
                latency_ms=5.0,
                result={},
            )
        )

        metrics = callbacks.get_metrics()
        assert metrics.total_llm_calls == 1
        assert metrics.total_tool_calls == 1
