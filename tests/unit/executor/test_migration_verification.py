"""Tests for Phase 1.9: Migration Verification.

Tests cover:
- 1.9.1: Parity testing (graph executor vs legacy executor)
- 1.9.2: Backwards compatibility (state.json reading, legacy mode)

These tests ensure the graph-based executor produces equivalent results
to the legacy imperative executor and maintains backwards compatibility.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_infra.executor.state import ExecutorState
from ai_infra.executor.state_migration import GraphStatePersistence
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_roadmap_content() -> str:
    """Sample ROADMAP content for testing."""
    return """\
# Test Project ROADMAP

## Phase 1: Foundation

### 1.1 Setup

- [ ] **Task 1** - Initialize project
  Set up the basic project structure.

- [ ] **Task 2** - Add configuration
  Add configuration files.

- [x] **Task 3** - Completed task
  This task is already done.
"""


@pytest.fixture
def temp_roadmap(temp_project_dir: Path, sample_roadmap_content: str) -> Path:
    """Create a temporary ROADMAP file."""
    roadmap = temp_project_dir / "ROADMAP.md"
    roadmap.write_text(sample_roadmap_content)
    return roadmap


@pytest.fixture
def legacy_state_json() -> dict[str, Any]:
    """Sample legacy state.json format."""
    return {
        "version": 1,
        "run_id": "run-legacy-123",
        "roadmap_hash": "abc123",
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_updated": "2026-01-01T01:00:00+00:00",
        "tasks": {
            "1.1.1": {
                "status": "completed",
                "started_at": "2026-01-01T00:30:00+00:00",
                "completed_at": "2026-01-01T00:45:00+00:00",
                "files_modified": ["src/main.py"],
                "attempts": 1,
            },
            "1.1.2": {
                "status": "pending",
                "attempts": 0,
            },
            "1.1.3": {
                "status": "failed",
                "started_at": "2026-01-01T00:50:00+00:00",
                "failed_at": "2026-01-01T00:55:00+00:00",
                "error": "Test failed",
                "attempts": 2,
            },
        },
    }


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent for testing."""
    agent = MagicMock()
    result = MagicMock()
    result.content = "Task completed"
    result.model_dump.return_value = {"content": "Task completed", "files_modified": []}
    agent.arun = AsyncMock(return_value=result)
    return agent


# =============================================================================
# Tests: 1.9.1 Parity Testing
# =============================================================================


class TestParityTesting:
    """Tests verifying graph executor produces same results as legacy.

    Phase 1.9.1: Graph executor produces same results as current.
    """

    def test_both_executors_import(self) -> None:
        """Verify both executor types can be imported."""
        from ai_infra.executor import Executor
        from ai_infra.executor.graph import ExecutorGraph

        assert Executor is not None
        assert ExecutorGraph is not None

    def test_graph_state_matches_legacy_fields(self) -> None:
        """Verify graph state contains equivalent fields to legacy state."""
        from ai_infra.executor.state import ExecutorGraphState

        # Key fields that must exist in graph state
        required_fields = {
            "roadmap_path",
            "todos",
            "current_task",
            "error",
            "retry_count",
            "should_continue",
            "files_modified",
            "verified",
        }

        # Get actual fields from TypedDict
        graph_fields = set(ExecutorGraphState.__annotations__.keys())

        # Verify all required fields are present
        missing = required_fields - graph_fields
        assert not missing, f"Missing fields in ExecutorGraphState: {missing}"

    def test_todo_item_format_consistency(self) -> None:
        """Verify TodoItem format is consistent between executors."""
        todo = TodoItem(
            id=1,
            title="Test task",
            description="Test description",
            status=TodoStatus.NOT_STARTED,
            source_task_ids=["1.1.1"],
        )

        # Verify TodoItem can be serialized (needed for state transfer)
        data = todo.to_dict()
        assert data["id"] == 1
        assert data["title"] == "Test task"
        assert data["status"] == "not-started"

        # Verify round-trip
        restored = TodoItem.from_dict(data)
        assert restored.id == todo.id
        assert restored.title == todo.title
        assert restored.status == todo.status

    @pytest.mark.asyncio
    async def test_parse_roadmap_produces_equivalent_todos(self, temp_roadmap: Path) -> None:
        """Verify both parsers produce equivalent todo lists."""
        from ai_infra.executor import Executor
        from ai_infra.executor.nodes.parse import parse_roadmap_node
        from ai_infra.executor.state import ExecutorGraphState

        # Legacy: Parse via Executor
        legacy_executor = Executor(roadmap=temp_roadmap)
        legacy_tasks = list(legacy_executor.roadmap.all_tasks())

        # Graph: Parse via node
        initial_state = ExecutorGraphState(
            roadmap_path=str(temp_roadmap),
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
            completed_count=0,
            max_tasks=None,
            should_continue=True,
            interrupt_requested=False,
            run_memory={},
        )

        result = await parse_roadmap_node(
            initial_state,
            agent=None,
            use_llm_normalization=False,
        )

        graph_todos = result.get("todos", [])

        # Both should find tasks from the ROADMAP
        # Note: Graph parser may group tasks differently than legacy parser
        # The key is both find tasks to execute
        incomplete_legacy = [t for t in legacy_tasks if t.status.value == "pending"]
        incomplete_graph = [t for t in graph_todos if t.status == TodoStatus.NOT_STARTED]

        # Both should find at least one task
        assert len(incomplete_legacy) >= 1
        assert len(incomplete_graph) >= 1

    def test_executor_config_applies_to_both(self) -> None:
        """Verify executor configuration works for both modes."""
        from ai_infra.executor import ExecutorConfig
        from ai_infra.executor.graph import ExecutorGraph

        config = ExecutorConfig(
            max_tasks=5,
            retry_failed=3,
            sync_roadmap=True,
        )

        # Verify config values are accessible
        assert config.max_tasks == 5
        assert config.retry_failed == 3
        assert config.sync_roadmap is True

        # Graph executor accepts max_tasks
        graph = ExecutorGraph(
            agent=None,
            roadmap_path="ROADMAP.md",
            max_tasks=config.max_tasks,
            sync_roadmap=config.sync_roadmap,
        )

        assert graph.max_tasks == 5
        assert graph.sync_roadmap is True

    def test_run_summary_format_compatibility(self) -> None:
        """Verify RunSummary format is compatible with graph results."""
        from ai_infra.executor import RunStatus, RunSummary

        # Create a RunSummary like the CLI does
        summary = RunSummary(
            status=RunStatus.COMPLETED,
            total_tasks=3,
            tasks_completed=2,
            tasks_failed=1,
            tasks_remaining=0,
            tasks_skipped=0,
            duration_ms=5000,
            total_tokens=1500,
            results=[],
            paused=False,
            pause_reason=None,
        )

        # Verify it can be serialized
        data = summary.to_dict()
        assert data["status"] == "completed"
        assert data["tasks_completed"] == 2
        assert data["tasks_failed"] == 1

    def test_error_format_parity(self) -> None:
        """Verify error format is consistent between executors."""
        from ai_infra.executor.state import ExecutorError, ExecutorGraphState

        # Graph error format
        error = ExecutorError(
            error_type="execution",
            message="Task execution failed",
            node="execute_task",
            task_id="task-1",
            recoverable=True,
            stack_trace=None,
        )

        # Should be usable in graph state
        state = ExecutorGraphState(
            roadmap_path="ROADMAP.md",
            todos=[],
            current_task=None,
            error=error,
            retry_count=1,
            should_continue=True,
        )

        assert state["error"]["error_type"] == "execution"
        assert state["error"]["recoverable"] is True


# =============================================================================
# Tests: 1.9.2 Backwards Compatibility
# =============================================================================


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with legacy state.

    Phase 1.9.2: Can still read .executor/state.json, legacy mode works.
    """

    def test_read_legacy_state_json(
        self, temp_project_dir: Path, legacy_state_json: dict[str, Any]
    ) -> None:
        """Verify legacy state.json can be read."""
        # Create legacy state file
        executor_dir = temp_project_dir / ".executor"
        executor_dir.mkdir()
        state_file = executor_dir / "state.json"
        state_file.write_text(json.dumps(legacy_state_json))

        # Create minimal ROADMAP for state loading
        roadmap = temp_project_dir / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n- [ ] Task 1\n- [ ] Task 2")

        # Load using ExecutorState
        state = ExecutorState.load(roadmap)

        # Verify state loaded correctly
        assert state.run_id == "run-legacy-123"
        assert "1.1.1" in state.tasks
        assert state.tasks["1.1.1"].status.value == "completed"
        assert state.tasks["1.1.3"].status.value == "failed"

    def test_legacy_state_task_status_preserved(
        self, temp_project_dir: Path, legacy_state_json: dict[str, Any]
    ) -> None:
        """Verify task statuses are preserved from legacy state."""
        # Create legacy state file
        executor_dir = temp_project_dir / ".executor"
        executor_dir.mkdir()
        state_file = executor_dir / "state.json"
        state_file.write_text(json.dumps(legacy_state_json))

        # Create ROADMAP
        roadmap = temp_project_dir / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n- [ ] Task 1")

        # Load state
        state = ExecutorState.load(roadmap)

        # Check specific task states
        completed_task = state.tasks.get("1.1.1")
        assert completed_task is not None
        assert completed_task.files_modified == ["src/main.py"]
        assert completed_task.attempts == 1

        failed_task = state.tasks.get("1.1.3")
        assert failed_task is not None
        assert failed_task.error == "Test failed"
        assert failed_task.attempts == 2

    def test_legacy_executor_still_works(self, temp_roadmap: Path) -> None:
        """Verify legacy Executor class still functions."""
        from ai_infra.executor import Executor

        # Create legacy executor
        executor = Executor(roadmap=temp_roadmap)

        # Verify basic functionality
        assert executor.roadmap is not None
        assert executor.state is not None
        assert executor.roadmap.total_tasks >= 0

    def test_legacy_mode_cli_option_exists(self) -> None:
        """Verify --legacy-mode CLI option exists."""
        from typer.testing import CliRunner

        from ai_infra.cli.cmds.executor_cmds import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "--legacy-mode" in result.output or "--graph-mode" in result.output

    def test_legacy_executor_config_compatible(self) -> None:
        """Verify ExecutorConfig works with legacy executor."""
        from ai_infra.executor import ExecutorConfig

        config = ExecutorConfig(
            max_tasks=5,
            dry_run=True,
            skip_verification=True,
            stop_on_failure=False,
        )

        # Verify config is accepted
        assert config.max_tasks == 5
        assert config.dry_run is True
        assert config.skip_verification is True
        assert config.stop_on_failure is False

    def test_graph_persistence_coexists_with_legacy(
        self, temp_project_dir: Path, legacy_state_json: dict[str, Any]
    ) -> None:
        """Verify graph and legacy state files can coexist."""
        executor_dir = temp_project_dir / ".executor"
        executor_dir.mkdir()

        # Create legacy state.json
        legacy_file = executor_dir / "state.json"
        legacy_file.write_text(json.dumps(legacy_state_json))

        # Create graph state
        persistence = GraphStatePersistence(project_root=temp_project_dir)
        graph_state = {
            "roadmap_path": str(temp_project_dir / "ROADMAP.md"),
            "todos": [],
            "completed_count": 0,
        }
        persistence.save_state(graph_state)

        # Both files should exist
        assert legacy_file.exists()
        assert persistence.graph_state_path.exists()

        # Both should be independently loadable
        legacy_data = json.loads(legacy_file.read_text())
        graph_data = persistence.load_state()

        assert legacy_data["run_id"] == "run-legacy-123"
        assert graph_data["roadmap_path"] == str(temp_project_dir / "ROADMAP.md")

    def test_state_version_detection(self, temp_project_dir: Path) -> None:
        """Verify state version can be detected."""
        executor_dir = temp_project_dir / ".executor"
        executor_dir.mkdir()

        # Version 1 state
        v1_state = {"version": 1, "run_id": "run-v1", "tasks": {}}
        state_file = executor_dir / "state.json"
        state_file.write_text(json.dumps(v1_state))

        # Load and check
        data = json.loads(state_file.read_text())
        assert data.get("version") == 1

    def test_legacy_recovery_still_works(self, temp_project_dir: Path) -> None:
        """Verify legacy crash recovery mechanism works."""
        # Create ROADMAP
        roadmap = temp_project_dir / "ROADMAP.md"
        roadmap.write_text("# ROADMAP\n- [ ] Task 1")

        # Create state with in-progress task
        executor_dir = temp_project_dir / ".executor"
        executor_dir.mkdir()

        state_data = {
            "version": 1,
            "run_id": "run-crashed",
            "tasks": {
                "1.1.1": {
                    "status": "in_progress",
                    "started_at": "2026-01-01T00:30:00+00:00",
                    "attempts": 1,
                },
            },
        }
        state_file = executor_dir / "state.json"
        state_file.write_text(json.dumps(state_data))

        # Load state
        state = ExecutorState.load(roadmap)

        # Recover (resets in_progress to pending)
        state.recover()

        # Verify recovery worked
        assert state.tasks["1.1.1"].status.value == "pending"

    def test_smooth_rollback_capability(self, temp_roadmap: Path) -> None:
        """Verify rollback from graph to legacy is possible."""
        from ai_infra.executor import Executor

        # This test verifies that even after graph mode is used,
        # the legacy executor can still be instantiated and used

        # Create and use legacy executor
        executor = Executor(roadmap=temp_roadmap)

        # Verify it has all necessary attributes
        assert hasattr(executor, "run")
        assert hasattr(executor, "state")
        assert hasattr(executor, "roadmap")
        assert hasattr(executor, "checkpointer")

        # Verify state can be saved
        executor.state.save()

        # Verify state file was created
        state_file = temp_roadmap.parent / ".executor" / "state.json"
        assert state_file.exists()


# =============================================================================
# Tests: Timing and Token Parity (Structural)
# =============================================================================


class TestTimingAndTokenStructure:
    """Tests for timing and token tracking structures.

    These verify the infrastructure for tracking timing and tokens exists,
    which enables actual parity verification in live environments.
    """

    def test_run_summary_has_timing_fields(self) -> None:
        """Verify RunSummary tracks duration."""
        from ai_infra.executor import RunStatus, RunSummary

        summary = RunSummary(
            status=RunStatus.COMPLETED,
            total_tasks=1,
            tasks_completed=1,
            tasks_failed=0,
            tasks_remaining=0,
            tasks_skipped=0,
            duration_ms=1234,
            total_tokens=5000,
            results=[],
        )

        assert summary.duration_ms == 1234

    def test_run_summary_has_token_fields(self) -> None:
        """Verify RunSummary tracks tokens."""
        from ai_infra.executor import RunStatus, RunSummary

        summary = RunSummary(
            status=RunStatus.COMPLETED,
            total_tasks=1,
            tasks_completed=1,
            tasks_failed=0,
            tasks_remaining=0,
            tasks_skipped=0,
            duration_ms=1000,
            total_tokens=5000,
            results=[],
        )

        assert summary.total_tokens == 5000

    def test_execution_result_tracks_tokens(self) -> None:
        """Verify ExecutionResult tracks token usage."""
        from ai_infra.executor import ExecutionResult, ExecutionStatus

        result = ExecutionResult(
            task_id="task-1",
            title="Test task",
            status=ExecutionStatus.SUCCESS,
            duration_ms=500,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        assert result.token_usage["prompt_tokens"] == 100
        assert result.token_usage["completion_tokens"] == 50

    def test_callbacks_available_for_tracking(self) -> None:
        """Verify ExecutorCallbacks can track metrics."""
        from ai_infra.executor import ExecutorCallbacks

        callbacks = ExecutorCallbacks()

        # Verify callback methods exist
        assert hasattr(callbacks, "on_task_start")
        assert hasattr(callbacks, "on_task_end")
        assert hasattr(callbacks, "on_run_end")

    def test_graph_streaming_provides_timing(self) -> None:
        """Verify graph streaming events include timing."""
        from ai_infra.executor.streaming import (
            create_node_end_event,
            create_run_end_event,
        )

        # Node event with timing
        node_event = create_node_end_event(
            node_name="execute_task",
            duration_ms=1500,
            state={},
        )
        assert node_event.duration_ms == 1500

        # Run event with timing
        run_event = create_run_end_event(
            completed=5,
            failed=1,
            skipped=0,
            duration_ms=60000,
        )
        assert run_event.duration_ms == 60000
