"""Tests for Phase 1.6: Streaming & Observability.

Tests for:
- 1.6.1: Tracing via ai_infra.tracing
- 1.6.2: Streaming output
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from ai_infra.executor.streaming import (
    ExecutorStreamEvent,
    JsonFormatter,
    MinimalFormatter,
    OutputFormat,
    PlainFormatter,
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
    get_formatter,
    stream_to_console,
)
from ai_infra.executor.tracing import (
    ExecutorTracingCallbacks,
    TracingConfig,
    create_traced_nodes,
    create_tracing_callbacks,
    traced_node,
)

# =============================================================================
# Tests for StreamEventType
# =============================================================================


class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_event_types_exist(self) -> None:
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
            assert hasattr(StreamEventType, name)

    def test_event_type_values(self) -> None:
        """Test event types have string values."""
        assert StreamEventType.RUN_START.value == "run_start"
        assert StreamEventType.NODE_END.value == "node_end"
        assert StreamEventType.TASK_COMPLETE.value == "task_complete"


# =============================================================================
# Tests for ExecutorStreamEvent
# =============================================================================


class TestExecutorStreamEvent:
    """Tests for ExecutorStreamEvent dataclass."""

    def test_default_creation(self) -> None:
        """Test event creation with defaults."""
        event = ExecutorStreamEvent(event_type=StreamEventType.RUN_START)
        assert event.event_type == StreamEventType.RUN_START
        assert event.node_name is None
        assert event.timestamp is not None
        assert event.data == {}
        assert event.state_snapshot is None
        assert event.duration_ms is None
        assert event.task is None
        assert event.message == ""

    def test_full_creation(self) -> None:
        """Test event creation with all fields."""
        event = ExecutorStreamEvent(
            event_type=StreamEventType.NODE_END,
            node_name="parse_roadmap",
            data={"key": "value"},
            state_snapshot={"completed_count": 1},
            duration_ms=150.5,
            task={"id": "1", "title": "Task 1"},
            message="Completed parsing",
        )
        assert event.event_type == StreamEventType.NODE_END
        assert event.node_name == "parse_roadmap"
        assert event.data["key"] == "value"
        assert event.state_snapshot["completed_count"] == 1
        assert event.duration_ms == 150.5
        assert event.task["title"] == "Task 1"
        assert event.message == "Completed parsing"

    def test_to_dict(self) -> None:
        """Test event serialization to dict."""
        event = ExecutorStreamEvent(
            event_type=StreamEventType.TASK_START,
            node_name="pick_task",
            message="Starting task",
            duration_ms=10.0,
        )
        d = event.to_dict()
        assert d["event_type"] == "task_start"
        assert d["node_name"] == "pick_task"
        assert d["message"] == "Starting task"
        assert d["duration_ms"] == 10.0
        assert "timestamp" in d


# =============================================================================
# Tests for StreamingConfig
# =============================================================================


class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = StreamingConfig()
        assert config.enabled is True
        assert config.include_state_snapshot is False
        assert config.output_format == OutputFormat.PLAIN
        assert config.show_node_transitions is True
        assert config.show_task_progress is True
        assert config.show_timing is True
        assert config.colors_enabled is True

    def test_verbose_preset(self) -> None:
        """Test verbose configuration preset."""
        config = StreamingConfig.verbose()
        assert config.enabled is True
        assert config.include_state_snapshot is True
        assert config.show_node_transitions is True
        assert config.show_task_progress is True

    def test_minimal_preset(self) -> None:
        """Test minimal configuration preset."""
        config = StreamingConfig.minimal()
        assert config.enabled is True
        assert config.output_format == OutputFormat.MINIMAL
        assert config.show_node_transitions is False
        assert config.show_task_progress is True

    def test_json_output_preset(self) -> None:
        """Test JSON output configuration preset."""
        config = StreamingConfig.json_output()
        assert config.enabled is True
        assert config.output_format == OutputFormat.JSON
        assert config.include_state_snapshot is True
        assert config.colors_enabled is False

    def test_disabled_preset(self) -> None:
        """Test disabled configuration preset."""
        config = StreamingConfig.disabled()
        assert config.enabled is False


# =============================================================================
# Tests for Event Builders
# =============================================================================


class TestEventBuilders:
    """Tests for event builder functions."""

    def test_create_run_start_event(self) -> None:
        """Test run start event creation."""
        event = create_run_start_event("ROADMAP.md", total_tasks=5)
        assert event.event_type == StreamEventType.RUN_START
        assert "ROADMAP.md" in event.message
        assert event.data["roadmap_path"] == "ROADMAP.md"
        assert event.data["total_tasks"] == 5

    def test_create_run_end_event(self) -> None:
        """Test run end event creation."""
        event = create_run_end_event(
            completed=3,
            failed=1,
            skipped=0,
            duration_ms=5000.0,
        )
        assert event.event_type == StreamEventType.RUN_END
        assert event.data["completed"] == 3
        assert event.data["failed"] == 1
        assert event.duration_ms == 5000.0

    def test_create_node_start_event(self) -> None:
        """Test node start event creation."""
        event = create_node_start_event("parse_roadmap")
        assert event.event_type == StreamEventType.NODE_START
        assert event.node_name == "parse_roadmap"
        assert "parse_roadmap" in event.message

    def test_create_node_end_event(self) -> None:
        """Test node end event creation."""
        event = create_node_end_event("execute_task", duration_ms=500.0)
        assert event.event_type == StreamEventType.NODE_END
        assert event.node_name == "execute_task"
        assert event.duration_ms == 500.0

    def test_create_node_error_event(self) -> None:
        """Test node error event creation."""
        error = ValueError("Test error")
        event = create_node_error_event("verify_task", error, duration_ms=100.0)
        assert event.event_type == StreamEventType.NODE_ERROR
        assert event.node_name == "verify_task"
        assert event.data["error_type"] == "ValueError"
        assert "Test error" in event.data["error_message"]

    def test_create_task_start_event(self) -> None:
        """Test task start event creation."""
        task = {"id": "1", "title": "Implement feature"}
        event = create_task_start_event(task, task_number=2, total_tasks=5)
        assert event.event_type == StreamEventType.TASK_START
        assert event.task == task
        assert "[2/5]" in event.message

    def test_create_task_complete_event(self) -> None:
        """Test task complete event creation."""
        task = {"id": "1", "title": "Implement feature"}
        event = create_task_complete_event(task, duration_ms=2000.0, files_modified=3)
        assert event.event_type == StreamEventType.TASK_COMPLETE
        assert event.duration_ms == 2000.0
        assert event.data["files_modified"] == 3

    def test_create_task_failed_event(self) -> None:
        """Test task failed event creation."""
        task = {"id": "1", "title": "Broken feature"}
        event = create_task_failed_event(task, error="Compilation error", duration_ms=500.0)
        assert event.event_type == StreamEventType.TASK_FAILED
        assert "Compilation error" in event.data["error"]

    def test_create_interrupt_event(self) -> None:
        """Test interrupt event creation."""
        event = create_interrupt_event("execute_task")
        assert event.event_type == StreamEventType.INTERRUPT
        assert event.node_name == "execute_task"
        assert "awaiting human decision" in event.message

    def test_create_resume_event(self) -> None:
        """Test resume event creation."""
        event = create_resume_event("approve", node_name="execute_task")
        assert event.event_type == StreamEventType.RESUME
        assert event.data["decision"] == "approve"


# =============================================================================
# Tests for Formatters
# =============================================================================


class TestPlainFormatter:
    """Tests for PlainFormatter."""

    def test_format_node_start(self) -> None:
        """Test formatting node start event."""
        formatter = PlainFormatter(colors_enabled=False, show_timing=False)
        event = create_node_start_event("parse_roadmap")
        output = formatter.format(event)
        assert "-->" in output
        assert "parse_roadmap" in output

    def test_format_task_complete(self) -> None:
        """Test formatting task complete event."""
        formatter = PlainFormatter(colors_enabled=False, show_timing=True)
        task = {"id": "1", "title": "Test task"}
        event = create_task_complete_event(task, duration_ms=1500.0)
        output = formatter.format(event)
        assert "[+]" in output
        assert "1500ms" in output

    def test_format_with_colors(self) -> None:
        """Test formatting with colors enabled."""
        formatter = PlainFormatter(colors_enabled=True, show_timing=False)
        event = create_task_failed_event(
            {"title": "Failed task"},
            error="Error",
            duration_ms=100.0,
        )
        output = formatter.format(event)
        # Should contain ANSI escape codes
        assert "\033[" in output


class TestMinimalFormatter:
    """Tests for MinimalFormatter."""

    def test_format_task_start(self) -> None:
        """Test minimal formatting of task start."""
        formatter = MinimalFormatter(colors_enabled=False)
        event = create_task_start_event({"title": "My Task"})
        output = formatter.format(event)
        assert "Starting" in output
        assert "My Task" in output

    def test_format_run_end(self) -> None:
        """Test minimal formatting of run end."""
        formatter = MinimalFormatter(colors_enabled=False)
        event = create_run_end_event(completed=3, failed=1, skipped=0, duration_ms=1000.0)
        output = formatter.format(event)
        assert "Done" in output
        assert "3" in output
        assert "1" in output

    def test_format_node_event_returns_empty(self) -> None:
        """Test node events return empty string in minimal mode."""
        formatter = MinimalFormatter(colors_enabled=False)
        event = create_node_start_event("parse_roadmap")
        output = formatter.format(event)
        assert output == ""


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_format_as_json(self) -> None:
        """Test JSON formatting."""
        import json

        formatter = JsonFormatter()
        event = create_run_start_event("ROADMAP.md", total_tasks=3)
        output = formatter.format(event)

        # Should be valid JSON
        parsed = json.loads(output)
        assert parsed["event_type"] == "run_start"
        assert parsed["data"]["roadmap_path"] == "ROADMAP.md"


class TestGetFormatter:
    """Tests for get_formatter function."""

    def test_get_plain_formatter(self) -> None:
        """Test getting plain formatter."""
        config = StreamingConfig(output_format=OutputFormat.PLAIN)
        formatter = get_formatter(config)
        assert isinstance(formatter, PlainFormatter)

    def test_get_minimal_formatter(self) -> None:
        """Test getting minimal formatter."""
        config = StreamingConfig(output_format=OutputFormat.MINIMAL)
        formatter = get_formatter(config)
        assert isinstance(formatter, MinimalFormatter)

    def test_get_json_formatter(self) -> None:
        """Test getting JSON formatter."""
        config = StreamingConfig(output_format=OutputFormat.JSON)
        formatter = get_formatter(config)
        assert isinstance(formatter, JsonFormatter)


# =============================================================================
# Tests for stream_to_console
# =============================================================================


class TestStreamToConsole:
    """Tests for stream_to_console function."""

    def test_stream_to_custom_output(self) -> None:
        """Test streaming to custom output stream."""
        output = io.StringIO()
        config = StreamingConfig(
            output_format=OutputFormat.MINIMAL,
            colors_enabled=False,
            output_stream=output,
        )
        event = create_task_start_event({"title": "Test"})

        stream_to_console(event, config=config)

        output.seek(0)
        content = output.read()
        assert "Test" in content

    def test_stream_disabled(self) -> None:
        """Test streaming when disabled."""
        output = io.StringIO()
        config = StreamingConfig.disabled()
        config.output_stream = output
        event = create_task_start_event({"title": "Test"})

        stream_to_console(event, config=config)

        output.seek(0)
        content = output.read()
        assert content == ""


# =============================================================================
# Tests for TracingConfig
# =============================================================================


class TestTracingConfig:
    """Tests for TracingConfig."""

    def test_default_config(self) -> None:
        """Test default tracing configuration."""
        config = TracingConfig()
        assert config.enabled is True
        assert config.include_state_details is True
        assert config.console_output is False

    def test_development_preset(self) -> None:
        """Test development configuration preset."""
        config = TracingConfig.development()
        assert config.enabled is True
        assert config.console_output is True

    def test_production_preset(self) -> None:
        """Test production configuration preset."""
        config = TracingConfig.production()
        assert config.enabled is True
        assert config.include_state_details is False
        assert config.console_output is False

    def test_disabled_preset(self) -> None:
        """Test disabled configuration preset."""
        config = TracingConfig.disabled()
        assert config.enabled is False


# =============================================================================
# Tests for ExecutorTracingCallbacks
# =============================================================================


class TestExecutorTracingCallbacks:
    """Tests for ExecutorTracingCallbacks (now ExecutorCallbacks)."""

    def test_init(self) -> None:
        """Test callbacks initialization."""
        callbacks = ExecutorTracingCallbacks()
        assert callbacks._run_span is None
        assert callbacks._node_spans == {}

    def test_set_run_context(self) -> None:
        """Test setting run context (replaces start_run)."""
        callbacks = ExecutorTracingCallbacks()
        callbacks.set_run_context("test-run-123", roadmap_path="ROADMAP.md")

        assert callbacks._run_span is not None
        assert callbacks._metrics.run_id == "test-run-123"

    def test_on_run_end(self) -> None:
        """Test ending a run trace."""
        callbacks = ExecutorTracingCallbacks()
        callbacks.set_run_context("test-run-123")

        callbacks.on_run_end()

        assert callbacks._run_span is None

    def test_on_graph_node_start(self) -> None:
        """Test handling node start event."""
        from ai_infra.callbacks import GraphNodeStartEvent

        callbacks = ExecutorTracingCallbacks()
        callbacks.set_run_context("test-run")

        event = GraphNodeStartEvent(
            node_id="parse_roadmap",
            node_type="executor_node",
            inputs={"roadmap_path": "ROADMAP.md"},
            step=1,
        )

        callbacks.on_graph_node_start(event)

        assert "parse_roadmap" in callbacks._node_spans

    def test_on_graph_node_end(self) -> None:
        """Test handling node end event."""
        from ai_infra.callbacks import GraphNodeEndEvent, GraphNodeStartEvent

        callbacks = ExecutorTracingCallbacks()
        callbacks.set_run_context("test-run")

        # Start the node first
        start_event = GraphNodeStartEvent(
            node_id="parse_roadmap",
            node_type="executor_node",
            inputs={},
            step=1,
        )
        callbacks.on_graph_node_start(start_event)

        # End the node
        end_event = GraphNodeEndEvent(
            node_id="parse_roadmap",
            node_type="executor_node",
            outputs={"completed_count": 1},
            step=1,
            latency_ms=150.0,
        )
        callbacks.on_graph_node_end(end_event)

        assert "parse_roadmap" not in callbacks._node_spans

    def test_on_graph_node_error(self) -> None:
        """Test handling node error event."""
        from ai_infra.callbacks import GraphNodeErrorEvent, GraphNodeStartEvent

        callbacks = ExecutorTracingCallbacks()
        callbacks.set_run_context("test-run")

        # Start the node first
        start_event = GraphNodeStartEvent(
            node_id="execute_task",
            node_type="executor_node",
            inputs={},
            step=1,
        )
        callbacks.on_graph_node_start(start_event)

        # Error in the node
        error_event = GraphNodeErrorEvent(
            node_id="execute_task",
            node_type="executor_node",
            error=ValueError("Test error"),
            step=1,
            latency_ms=50.0,
        )
        callbacks.on_graph_node_error(error_event)

        assert "execute_task" not in callbacks._node_spans


# =============================================================================
# Tests for traced_node decorator
# =============================================================================


class TestTracedNodeDecorator:
    """Tests for traced_node decorator."""

    @pytest.mark.asyncio
    async def test_traced_node_basic(self) -> None:
        """Test basic traced node execution."""

        @traced_node(name="test.node")
        async def test_node(state: dict[str, Any]) -> dict[str, Any]:
            return {**state, "processed": True}

        result = await test_node({"input": "value"})

        assert result["processed"] is True
        assert result["input"] == "value"

    @pytest.mark.asyncio
    async def test_traced_node_with_error(self) -> None:
        """Test traced node that raises error."""

        @traced_node(name="test.error_node")
        async def error_node(state: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await error_node({})

    @pytest.mark.asyncio
    async def test_traced_node_with_task_info(self) -> None:
        """Test traced node with task in state."""

        class MockTask:
            id = "task-123"
            title = "Test Task"

        @traced_node()
        async def task_aware_node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        result = await task_aware_node(
            {
                "current_task": MockTask(),
                "completed_count": 5,
            }
        )

        assert result is not None


# =============================================================================
# Tests for create_traced_nodes
# =============================================================================


class TestCreateTracedNodes:
    """Tests for create_traced_nodes factory."""

    def test_creates_all_nodes(self) -> None:
        """Test that all expected nodes are created."""
        traced_nodes = create_traced_nodes()

        expected_nodes = [
            "parse_roadmap",
            "pick_task",
            "build_context",
            "execute_task",
            "verify_task",
            "checkpoint",
            "rollback",
            "handle_failure",
            "decide_next",
        ]

        for node_name in expected_nodes:
            assert node_name in traced_nodes, f"Missing traced node: {node_name}"

    def test_nodes_are_callable(self) -> None:
        """Test that traced nodes are callable."""
        traced_nodes = create_traced_nodes()

        for name, node_fn in traced_nodes.items():
            assert callable(node_fn), f"Node {name} is not callable"


# =============================================================================
# Tests for create_tracing_callbacks
# =============================================================================


class TestCreateTracingCallbacks:
    """Tests for create_tracing_callbacks factory."""

    def test_create_default(self) -> None:
        """Test creating callbacks with defaults."""
        callbacks = create_tracing_callbacks()
        assert isinstance(callbacks, ExecutorTracingCallbacks)

    def test_create_with_run_id(self) -> None:
        """Test creating callbacks with immediate run start."""
        callbacks = create_tracing_callbacks(
            run_id="my-run-123",
            roadmap_path="ROADMAP.md",
        )

        assert callbacks._run_span is not None

    def test_create_disabled(self) -> None:
        """Test creating callbacks when tracing is disabled."""
        config = TracingConfig.disabled()
        callbacks = create_tracing_callbacks(config=config)

        # Should still return valid callbacks, just no-op
        assert isinstance(callbacks, ExecutorTracingCallbacks)


# =============================================================================
# Integration Tests
# =============================================================================


class TestStreamingTracingIntegration:
    """Integration tests for streaming and tracing."""

    @pytest.mark.asyncio
    async def test_streaming_with_mock_graph(self) -> None:
        """Test streaming events with a mock graph execution."""
        events: list[ExecutorStreamEvent] = []

        # Simulate streaming events
        events.append(create_run_start_event("ROADMAP.md", total_tasks=2))
        events.append(create_node_start_event("parse_roadmap"))
        events.append(create_node_end_event("parse_roadmap", duration_ms=50.0))
        events.append(create_node_start_event("pick_task"))
        events.append(create_task_start_event({"title": "Task 1"}, task_number=1, total_tasks=2))
        events.append(create_node_end_event("pick_task", duration_ms=10.0))
        events.append(create_task_complete_event({"title": "Task 1"}, duration_ms=1000.0))
        events.append(create_run_end_event(completed=1, failed=0, skipped=1, duration_ms=2000.0))

        # All events should be valid
        assert len(events) == 8
        assert events[0].event_type == StreamEventType.RUN_START
        assert events[-1].event_type == StreamEventType.RUN_END

    def test_full_formatter_pipeline(self) -> None:
        """Test complete formatter pipeline."""
        output = io.StringIO()
        config = StreamingConfig.verbose()
        config.output_stream = output
        config.colors_enabled = False

        events = [
            create_run_start_event("ROADMAP.md"),
            create_task_start_event({"title": "Feature X"}),
            create_task_complete_event({"title": "Feature X"}, duration_ms=500.0),
            create_run_end_event(completed=1, failed=0, skipped=0, duration_ms=1000.0),
        ]

        for event in events:
            stream_to_console(event, config=config)

        output.seek(0)
        content = output.read()

        assert "ROADMAP.md" in content
        assert "Feature X" in content


# =============================================================================
# Import Tests
# =============================================================================


class TestPhase16Imports:
    """Test that all Phase 1.6 components can be imported."""

    def test_streaming_imports(self) -> None:
        """Test streaming module imports."""
        from ai_infra.executor import (
            StreamEventType,
            StreamingConfig,
        )

        # Verify imports work
        assert StreamEventType.RUN_START.value == "run_start"
        assert StreamingConfig.verbose().enabled is True

    def test_tracing_imports(self) -> None:
        """Test tracing module imports."""
        from ai_infra.executor import (
            TracingConfig,
            traced_node,
        )

        # Verify imports work
        assert TracingConfig.production().enabled is True
        assert callable(traced_node)
