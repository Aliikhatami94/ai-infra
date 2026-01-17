"""Tests for Phase 6.1.1: Consolidated Streaming.

Tests for executor streaming events including task lifecycle,
progress updates, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ai_infra.executor.streaming import (
    ExecutorStreamEvent,
    StreamEventType,
    StreamingConfig,
    create_interrupt_event,
    create_node_end_event,
    create_node_error_event,
    create_node_start_event,
    create_resume_event,
    create_run_end_event,
    create_run_start_event,
    create_task_complete_event,
    create_task_failed_event,
    create_task_start_event,
)

# =============================================================================
# ExecutorStreamEvent Tests
# =============================================================================


class TestExecutorStreamEvent:
    """Tests for ExecutorStreamEvent dataclass."""

    def test_basic_event_creation(self) -> None:
        """Test creating a basic event."""
        event = ExecutorStreamEvent(
            event_type=StreamEventType.PROGRESS,
            message="Processing...",
        )
        assert event.event_type == StreamEventType.PROGRESS
        assert event.message == "Processing..."
        assert event.node_name is None
        assert event.duration_ms is None

    def test_event_with_all_fields(self) -> None:
        """Test creating an event with all fields."""
        event = ExecutorStreamEvent(
            event_type=StreamEventType.NODE_END,
            node_name="executor_node",
            data={"files_changed": 3},
            state_snapshot={"phase": "execution"},
            duration_ms=1500.0,
            task={"id": "1.1", "title": "Add feature"},
            message="Node completed",
        )
        assert event.event_type == StreamEventType.NODE_END
        assert event.node_name == "executor_node"
        assert event.data["files_changed"] == 3
        assert event.duration_ms == 1500.0
        assert event.task["id"] == "1.1"

    def test_event_to_dict(self) -> None:
        """Test event serialization."""
        event = ExecutorStreamEvent(
            event_type=StreamEventType.TASK_START,
            node_name="planner",
            message="Starting task",
            data={"task_id": "1.1"},
        )
        d = event.to_dict()

        assert d["event_type"] == "task_start"
        assert d["node_name"] == "planner"
        assert d["message"] == "Starting task"
        assert d["data"]["task_id"] == "1.1"
        assert "timestamp" in d

    def test_event_timestamp_is_utc(self) -> None:
        """Test event timestamp is UTC."""
        event = ExecutorStreamEvent(event_type=StreamEventType.PROGRESS)
        assert event.timestamp.tzinfo is not None


# =============================================================================
# StreamEventType Tests
# =============================================================================


class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_all_event_types_exist(self) -> None:
        """Test all expected event types are defined."""
        expected = [
            "RUN_START",
            "RUN_END",
            "NODE_START",
            "NODE_END",
            "NODE_ERROR",
            "STATE_UPDATE",
            "TASK_START",
            "TASK_COMPLETE",
            "TASK_FAILED",
            "TASK_SKIPPED",
            "PROGRESS",
            "INTERRUPT",
            "RESUME",
        ]
        for name in expected:
            assert hasattr(StreamEventType, name), f"Missing: {name}"

    def test_event_type_values_are_strings(self) -> None:
        """Test event type values are lowercase strings."""
        assert StreamEventType.RUN_START.value == "run_start"
        assert StreamEventType.TASK_COMPLETE.value == "task_complete"
        assert StreamEventType.PROGRESS.value == "progress"


# =============================================================================
# Task Event Factory Tests
# =============================================================================


class TestTaskStartEvent:
    """Tests for create_task_start_event."""

    def test_basic_task_start(self) -> None:
        """Test basic task start event."""
        task = MagicMock()
        task.id = "1.1"
        task.title = "Add authentication"
        task.description = "Implement JWT auth"

        event = create_task_start_event(task)

        assert event.event_type == StreamEventType.TASK_START
        assert "Add authentication" in event.message
        assert event.task is not None

    def test_task_start_with_progress(self) -> None:
        """Test task start with progress numbers."""
        task = MagicMock()
        task.id = "1.2"
        task.title = "Write tests"

        event = create_task_start_event(task, task_number=2, total_tasks=5)

        assert event.event_type == StreamEventType.TASK_START
        assert event.data["task_number"] == 2
        assert event.data["total_tasks"] == 5
        assert "[2/5]" in event.message


class TestTaskCompleteEvent:
    """Tests for create_task_complete_event."""

    def test_basic_task_complete(self) -> None:
        """Test basic task complete event."""
        task = MagicMock()
        task.id = "1.1"
        task.title = "Add feature"

        event = create_task_complete_event(task, duration_ms=5000.0)

        assert event.event_type == StreamEventType.TASK_COMPLETE
        assert event.task is not None
        assert event.duration_ms == 5000.0

    def test_task_complete_with_files_modified(self) -> None:
        """Test task complete with files modified count."""
        task = MagicMock()
        task.id = "1.1"
        task.title = "Refactor"

        event = create_task_complete_event(task, duration_ms=2500.0, files_modified=3)

        assert event.duration_ms == 2500.0
        assert event.data["files_modified"] == 3


class TestTaskFailedEvent:
    """Tests for create_task_failed_event."""

    def test_basic_task_failed(self) -> None:
        """Test basic task failed event."""
        task = MagicMock()
        task.id = "1.1"
        task.title = "Broken task"

        event = create_task_failed_event(task, error="Syntax error in output", duration_ms=1000.0)

        assert event.event_type == StreamEventType.TASK_FAILED
        assert event.data["error"] == "Syntax error in output"
        assert "Syntax error" in event.message
        assert event.duration_ms == 1000.0


# =============================================================================
# Run Lifecycle Event Tests
# =============================================================================


class TestRunLifecycleEvents:
    """Tests for run start/end events."""

    def test_run_start_event(self) -> None:
        """Test run start event creation."""
        event = create_run_start_event(roadmap_path="/project/ROADMAP.md")

        assert event.event_type == StreamEventType.RUN_START
        assert "ROADMAP" in event.message
        assert event.data["roadmap_path"] == "/project/ROADMAP.md"

    def test_run_start_with_task_count(self) -> None:
        """Test run start with total tasks."""
        event = create_run_start_event(roadmap_path="ROADMAP.md", total_tasks=10)

        assert event.data["total_tasks"] == 10

    def test_run_end_event_success(self) -> None:
        """Test run end event for successful run."""
        event = create_run_end_event(completed=5, failed=0, skipped=1, duration_ms=10000.0)

        assert event.event_type == StreamEventType.RUN_END
        assert event.duration_ms == 10000.0
        assert event.data["completed"] == 5
        assert event.data["failed"] == 0
        assert "5 completed" in event.message

    def test_run_end_event_with_failures(self) -> None:
        """Test run end event with failures."""
        event = create_run_end_event(completed=3, failed=2, skipped=0, duration_ms=15000.0)

        assert event.data["failed"] == 2
        assert "2 failed" in event.message


# =============================================================================
# Node Event Tests
# =============================================================================


class TestNodeEvents:
    """Tests for node lifecycle events."""

    def test_node_start_event(self) -> None:
        """Test node start event."""
        event = create_node_start_event("planner")

        assert event.event_type == StreamEventType.NODE_START
        assert event.node_name == "planner"
        assert "planner" in event.message

    def test_node_end_event(self) -> None:
        """Test node end event."""
        event = create_node_end_event("executor", duration_ms=5000.0)

        assert event.event_type == StreamEventType.NODE_END
        assert event.node_name == "executor"
        assert event.duration_ms == 5000.0

    def test_node_error_event(self) -> None:
        """Test node error event."""
        error = ValueError("Verification failed")
        event = create_node_error_event("verifier", error=error, duration_ms=1000.0)

        assert event.event_type == StreamEventType.NODE_ERROR
        assert event.node_name == "verifier"
        assert "Verification failed" in event.message
        assert event.duration_ms == 1000.0


# =============================================================================
# HITL Event Tests
# =============================================================================


class TestHITLEvents:
    """Tests for HITL interrupt/resume events."""

    def test_interrupt_event(self) -> None:
        """Test interrupt event."""
        event = create_interrupt_event(node_name="hitl_node")

        assert event.event_type == StreamEventType.INTERRUPT
        assert event.node_name == "hitl_node"
        assert "awaiting human" in event.message.lower()

    def test_interrupt_event_with_task(self) -> None:
        """Test interrupt event with task."""
        task = MagicMock()
        task.id = "1.1"
        task.title = "Need approval"

        event = create_interrupt_event(node_name="hitl_node", task=task)

        assert event.event_type == StreamEventType.INTERRUPT
        assert event.task is not None

    def test_resume_event(self) -> None:
        """Test resume event."""
        event = create_resume_event(decision="approve")

        assert event.event_type == StreamEventType.RESUME
        assert event.data["decision"] == "approve"
        assert "approve" in event.message.lower()

    def test_resume_event_with_node(self) -> None:
        """Test resume event with node name."""
        event = create_resume_event(decision="continue", node_name="executor")

        assert event.node_name == "executor"


# =============================================================================
# StreamingConfig Tests
# =============================================================================


class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_default_config(self) -> None:
        """Test default streaming configuration."""
        config = StreamingConfig()
        assert config.enabled is True
        assert config.show_task_progress is True
        assert config.show_node_transitions is True

    def test_verbose_config(self) -> None:
        """Test verbose streaming configuration."""
        config = StreamingConfig.verbose()
        assert config.enabled is True
        assert config.include_state_snapshot is True
        assert config.show_timing is True
        assert config.stream_tokens is True

    def test_minimal_config(self) -> None:
        """Test minimal streaming configuration."""
        config = StreamingConfig.minimal()
        assert config.enabled is True
        assert config.show_node_transitions is False
        assert config.show_task_progress is True
        assert config.stream_tokens is False

    def test_json_config(self) -> None:
        """Test JSON output configuration."""
        config = StreamingConfig.json_output()
        assert config.enabled is True
        assert config.colors_enabled is False
        assert config.include_state_snapshot is True

    def test_disabled_config(self) -> None:
        """Test disabled configuration."""
        config = StreamingConfig.disabled()
        assert config.enabled is False

    def test_tokens_only_config(self) -> None:
        """Test tokens only configuration."""
        config = StreamingConfig.tokens_only()
        assert config.enabled is True
        assert config.stream_tokens is True
        assert config.show_llm_thinking is False
        assert config.token_visibility == "minimal"


# =============================================================================
# Progress Event Tests
# =============================================================================


class TestProgressEvents:
    """Tests for progress tracking events."""

    def test_progress_event_type_exists(self) -> None:
        """Test progress event type exists."""
        assert StreamEventType.PROGRESS.value == "progress"

    def test_create_progress_event_manually(self) -> None:
        """Test creating a progress event manually."""
        event = ExecutorStreamEvent(
            event_type=StreamEventType.PROGRESS,
            message="Task 3 of 10 completed",
            data={"completed": 3, "total": 10, "percent": 30},
        )

        assert event.event_type == StreamEventType.PROGRESS
        assert event.data["percent"] == 30
        assert event.data["completed"] == 3

    def test_progress_event_with_current_task(self) -> None:
        """Test progress event with current task info."""
        event = ExecutorStreamEvent(
            event_type=StreamEventType.PROGRESS,
            message="Working on authentication",
            data={"current_task_id": "2.1", "phase": "implementation"},
            task={"id": "2.1", "title": "Add auth"},
        )

        assert event.task is not None
        assert event.task["id"] == "2.1"


# =============================================================================
# Event Filtering Tests
# =============================================================================


class TestEventFiltering:
    """Tests for event filtering based on config."""

    def test_task_events_enabled_by_default(self) -> None:
        """Test task events are enabled by default."""
        config = StreamingConfig()
        assert config.show_task_progress is True

    def test_node_events_can_be_disabled(self) -> None:
        """Test node events can be disabled."""
        config = StreamingConfig(show_node_transitions=False)
        assert config.show_node_transitions is False

    def test_timing_can_be_disabled(self) -> None:
        """Test timing can be disabled."""
        config = StreamingConfig(show_timing=False)
        assert config.show_timing is False


# =============================================================================
# Integration-Style Tests
# =============================================================================


class TestStreamingIntegration:
    """Integration-style tests for streaming."""

    def test_full_task_lifecycle_events(self) -> None:
        """Test creating events for a full task lifecycle."""
        task = MagicMock()
        task.id = "1.1"
        task.title = "Add user endpoint"

        # Task start
        start_event = create_task_start_event(task, task_number=1, total_tasks=5)
        assert start_event.event_type == StreamEventType.TASK_START
        assert "[1/5]" in start_event.message

        # Task complete
        complete_event = create_task_complete_event(task, duration_ms=5000.0, files_modified=2)
        assert complete_event.event_type == StreamEventType.TASK_COMPLETE
        assert complete_event.duration_ms == 5000.0

    def test_full_run_lifecycle_events(self) -> None:
        """Test creating events for a full run lifecycle."""
        # Run start
        start_event = create_run_start_event(roadmap_path="ROADMAP.md", total_tasks=5)
        assert start_event.event_type == StreamEventType.RUN_START

        # Node start
        node_start = create_node_start_event("planner")
        assert node_start.event_type == StreamEventType.NODE_START

        # Node end
        node_end = create_node_end_event("planner", duration_ms=1000.0)
        assert node_end.event_type == StreamEventType.NODE_END

        # Run end
        end_event = create_run_end_event(completed=5, failed=0, skipped=0, duration_ms=30000.0)
        assert end_event.event_type == StreamEventType.RUN_END

    def test_error_handling_lifecycle(self) -> None:
        """Test error events in the lifecycle."""
        task = MagicMock()
        task.id = "1.1"
        task.title = "Broken task"

        # Node error
        error = RuntimeError("Execution failed")
        node_error = create_node_error_event("executor", error=error, duration_ms=1000.0)
        assert node_error.event_type == StreamEventType.NODE_ERROR

        # Task failed
        task_failed = create_task_failed_event(task, error="Syntax error", duration_ms=500.0)
        assert task_failed.event_type == StreamEventType.TASK_FAILED

        # Run end with failures
        run_end = create_run_end_event(completed=2, failed=1, skipped=0, duration_ms=10000.0)
        assert run_end.event_type == StreamEventType.RUN_END
        assert run_end.data["failed"] == 1

    def test_hitl_interrupt_resume_cycle(self) -> None:
        """Test HITL interrupt and resume cycle."""
        task = MagicMock()
        task.id = "2.1"
        task.title = "Needs review"

        # Interrupt
        interrupt = create_interrupt_event(node_name="hitl_node", task=task)
        assert interrupt.event_type == StreamEventType.INTERRUPT

        # Resume
        resume = create_resume_event(decision="approve", node_name="executor")
        assert resume.event_type == StreamEventType.RESUME
        assert resume.data["decision"] == "approve"
