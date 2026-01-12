"""Tests for ExecutorState."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_infra.executor import (
    ExecutorState,
    FailureCategory,
    StateSummary,
    TaskState,
    TaskStatus,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_roadmap_content() -> str:
    """Sample ROADMAP content."""
    return """\
# Test Project ROADMAP

## Phase 0: Foundation

> **Goal**: Establish core infrastructure
> **Priority**: HIGH

### 0.1 Setup

**Files**: `pyproject.toml`

- [x] **Initialize project**
  Set up the project structure.

- [ ] **Configure linting**
  Set up ruff and mypy.

### 0.2 Core

**Files**: `src/core.py`

- [ ] **Implement core logic**
  Main business logic.

- [ ] **Add tests**
  Unit tests for core.

## Phase 1: Features

### 1.1 API

- [ ] **Create API endpoints**
  REST API implementation.
"""


@pytest.fixture
def temp_roadmap(sample_roadmap_content: str) -> Path:
    """Create a temporary ROADMAP file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        roadmap_path = Path(tmpdir) / "ROADMAP.md"
        roadmap_path.write_text(sample_roadmap_content)
        yield roadmap_path


@pytest.fixture
def temp_roadmap_dir(sample_roadmap_content: str):
    """Create a temporary directory with ROADMAP file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        roadmap_path = Path(tmpdir) / "ROADMAP.md"
        roadmap_path.write_text(sample_roadmap_content)
        yield Path(tmpdir), roadmap_path


# =============================================================================
# TaskState Tests
# =============================================================================


class TestTaskState:
    """Test TaskState dataclass."""

    def test_default_values(self) -> None:
        """Test default task state values."""
        state = TaskState()

        assert state.status == TaskStatus.PENDING
        assert state.started_at is None
        assert state.completed_at is None
        assert state.failed_at is None
        assert state.files_modified == []
        assert state.error is None
        assert state.failure_category is None
        assert state.token_usage == {}
        assert state.attempts == 0
        assert state.agent_run_id is None

    def test_to_dict_minimal(self) -> None:
        """Test serialization with minimal values."""
        state = TaskState()
        data = state.to_dict()

        assert data["status"] == "pending"
        assert data["attempts"] == 0
        assert "started_at" not in data
        assert "error" not in data

    def test_to_dict_full(self) -> None:
        """Test serialization with all values."""
        now = datetime.now(UTC)
        state = TaskState(
            status=TaskStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            files_modified=["src/foo.py"],
            token_usage={"prompt": 100, "completion": 50},
            attempts=1,
            agent_run_id="run_123",
        )
        data = state.to_dict()

        assert data["status"] == "completed"
        assert data["started_at"] == now.isoformat()
        assert data["completed_at"] == now.isoformat()
        assert data["files_modified"] == ["src/foo.py"]
        assert data["token_usage"] == {"prompt": 100, "completion": 50}
        assert data["attempts"] == 1
        assert data["agent_run_id"] == "run_123"

    def test_from_dict(self) -> None:
        """Test deserialization."""
        now = datetime.now(UTC)
        data = {
            "status": "failed",
            "started_at": now.isoformat(),
            "failed_at": now.isoformat(),
            "error": "Something went wrong",
            "failure_category": "syntax_error",
            "attempts": 2,
        }

        state = TaskState.from_dict(data)

        assert state.status == TaskStatus.FAILED
        assert state.started_at is not None
        assert state.failed_at is not None
        assert state.error == "Something went wrong"
        assert state.failure_category == FailureCategory.SYNTAX_ERROR
        assert state.attempts == 2

    def test_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        now = datetime.now(UTC)
        original = TaskState(
            status=TaskStatus.IN_PROGRESS,
            started_at=now,
            attempts=3,
            agent_run_id="run_abc",
        )

        data = original.to_dict()
        restored = TaskState.from_dict(data)

        assert restored.status == original.status
        assert restored.attempts == original.attempts
        assert restored.agent_run_id == original.agent_run_id


# =============================================================================
# StateSummary Tests
# =============================================================================


class TestStateSummary:
    """Test StateSummary dataclass."""

    def test_default_values(self) -> None:
        """Test default summary values."""
        summary = StateSummary()

        assert summary.pending == 0
        assert summary.in_progress == 0
        assert summary.completed == 0
        assert summary.failed == 0
        assert summary.total_tokens == 0
        assert summary.total_attempts == 0

    def test_total_property(self) -> None:
        """Test total count calculation."""
        summary = StateSummary(pending=5, in_progress=1, completed=3, failed=1)

        assert summary.total == 10

    def test_progress_property(self) -> None:
        """Test progress calculation."""
        summary = StateSummary(pending=6, completed=4)

        assert summary.progress == 0.4

    def test_progress_empty(self) -> None:
        """Test progress with no tasks."""
        summary = StateSummary()

        assert summary.progress == 1.0

    def test_to_dict(self) -> None:
        """Test serialization."""
        summary = StateSummary(
            pending=5,
            completed=3,
            total_tokens=1000,
            total_attempts=10,
        )
        data = summary.to_dict()

        assert data["pending"] == 5
        assert data["completed"] == 3
        assert data["total_tokens"] == 1000

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "pending": 2,
            "in_progress": 1,
            "completed": 7,
            "failed": 0,
            "total_tokens": 5000,
            "total_attempts": 8,
        }

        summary = StateSummary.from_dict(data)

        assert summary.pending == 2
        assert summary.completed == 7
        assert summary.total_tokens == 5000


# =============================================================================
# ExecutorState Creation Tests
# =============================================================================


class TestExecutorStateCreation:
    """Test ExecutorState creation and initialization."""

    def test_init_basic(self, temp_roadmap: Path) -> None:
        """Test basic initialization."""
        state = ExecutorState(temp_roadmap)

        assert state.roadmap_path == temp_roadmap.resolve()
        assert state.run_id.startswith("run_")
        assert state.roadmap_hash != ""

    def test_init_with_run_id(self, temp_roadmap: Path) -> None:
        """Test initialization with custom run ID."""
        state = ExecutorState(temp_roadmap, run_id="custom_run_123")

        assert state.run_id == "custom_run_123"

    def test_from_roadmap(self, temp_roadmap: Path) -> None:
        """Test creation from ROADMAP file."""
        state = ExecutorState.from_roadmap(temp_roadmap)

        # Should have tasks from the roadmap
        assert len(state.tasks) > 0

        # Should have correct initial states
        # Task 0.1.1 is marked [x] in the sample
        task_state = state.get_task_state("0.1.1")
        assert task_state is not None
        assert task_state.status == TaskStatus.COMPLETED

        # Task 0.1.2 is marked [ ] in the sample
        task_state = state.get_task_state("0.1.2")
        assert task_state is not None
        assert task_state.status == TaskStatus.PENDING

    def test_from_roadmap_nonexistent(self) -> None:
        """Test creation from nonexistent file."""
        state = ExecutorState.from_roadmap("/nonexistent/ROADMAP.md")

        assert len(state.tasks) == 0

    def test_state_dir_property(self, temp_roadmap: Path) -> None:
        """Test state directory path."""
        state = ExecutorState(temp_roadmap)

        # Use resolve() on expected to handle macOS /var -> /private/var symlinks
        expected = (temp_roadmap.parent / ".executor").resolve()
        assert state.state_dir == expected

    def test_state_file_property(self, temp_roadmap: Path) -> None:
        """Test state file path."""
        state = ExecutorState(temp_roadmap)

        # Use resolve() on expected to handle macOS /var -> /private/var symlinks
        expected = (temp_roadmap.parent / ".executor" / "state.json").resolve()
        assert state.state_file == expected


# =============================================================================
# ExecutorState Updates Tests
# =============================================================================


class TestExecutorStateUpdates:
    """Test ExecutorState update operations."""

    def test_mark_started(self, temp_roadmap: Path) -> None:
        """Test marking a task as started."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_started("0.1.2")

        task_state = state.get_task_state("0.1.2")
        assert task_state is not None
        assert task_state.status == TaskStatus.IN_PROGRESS
        assert task_state.started_at is not None
        assert task_state.attempts == 1

    def test_mark_started_increments_attempts(self, temp_roadmap: Path) -> None:
        """Test that starting again increments attempts."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_started("0.1.2")
        state.mark_started("0.1.2")

        task_state = state.get_task_state("0.1.2")
        assert task_state is not None
        assert task_state.attempts == 2

    def test_mark_started_with_agent_run_id(self, temp_roadmap: Path) -> None:
        """Test marking started with agent run ID."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_started("0.1.2", agent_run_id="agent_abc123")

        task_state = state.get_task_state("0.1.2")
        assert task_state is not None
        assert task_state.agent_run_id == "agent_abc123"

    def test_mark_completed(self, temp_roadmap: Path) -> None:
        """Test marking a task as completed."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_started("0.1.2")
        state.mark_completed("0.1.2", files_modified=["src/lint.py"])

        task_state = state.get_task_state("0.1.2")
        assert task_state is not None
        assert task_state.status == TaskStatus.COMPLETED
        assert task_state.completed_at is not None
        assert task_state.files_modified == ["src/lint.py"]

    def test_mark_completed_with_tokens(self, temp_roadmap: Path) -> None:
        """Test marking completed with token usage."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_completed("0.1.2", token_usage={"prompt": 500, "completion": 200})

        task_state = state.get_task_state("0.1.2")
        assert task_state is not None
        assert task_state.token_usage == {"prompt": 500, "completion": 200}

    def test_mark_failed(self, temp_roadmap: Path) -> None:
        """Test marking a task as failed."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_started("0.1.2")
        state.mark_failed(
            "0.1.2",
            error="Syntax error in generated code",
            category=FailureCategory.SYNTAX_ERROR,
        )

        task_state = state.get_task_state("0.1.2")
        assert task_state is not None
        assert task_state.status == TaskStatus.FAILED
        assert task_state.failed_at is not None
        assert task_state.error == "Syntax error in generated code"
        assert task_state.failure_category == FailureCategory.SYNTAX_ERROR

    def test_reset_task(self, temp_roadmap: Path) -> None:
        """Test resetting a task."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_started("0.1.2")
        state.mark_failed("0.1.2", error="Error")
        state.reset_task("0.1.2")

        task_state = state.get_task_state("0.1.2")
        assert task_state is not None
        assert task_state.status == TaskStatus.PENDING
        assert task_state.started_at is None
        assert task_state.error is None
        # Attempts should be preserved
        assert task_state.attempts == 1


# =============================================================================
# ExecutorState Query Tests
# =============================================================================


class TestExecutorStateQueries:
    """Test ExecutorState query methods."""

    def test_get_task_state_exists(self, temp_roadmap: Path) -> None:
        """Test getting existing task state."""
        state = ExecutorState.from_roadmap(temp_roadmap)

        task_state = state.get_task_state("0.1.1")
        assert task_state is not None

    def test_get_task_state_missing(self, temp_roadmap: Path) -> None:
        """Test getting nonexistent task state."""
        state = ExecutorState.from_roadmap(temp_roadmap)

        task_state = state.get_task_state("99.99.99")
        assert task_state is None

    def test_get_status(self, temp_roadmap: Path) -> None:
        """Test getting task status."""
        state = ExecutorState.from_roadmap(temp_roadmap)

        # Existing completed task
        assert state.get_status("0.1.1") == TaskStatus.COMPLETED

        # Existing pending task
        assert state.get_status("0.1.2") == TaskStatus.PENDING

        # Nonexistent task defaults to PENDING
        assert state.get_status("99.99.99") == TaskStatus.PENDING

    def test_get_summary(self, temp_roadmap: Path) -> None:
        """Test getting summary statistics."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_started("0.1.2")
        state.mark_failed("0.2.1", error="Failed")

        summary = state.get_summary()

        assert summary.completed >= 1  # 0.1.1 from roadmap
        assert summary.in_progress == 1  # 0.1.2
        assert summary.failed == 1  # 0.2.1
        assert summary.total > 0

    def test_get_in_progress_tasks(self, temp_roadmap: Path) -> None:
        """Test getting in-progress task IDs."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_started("0.1.2")
        state.mark_started("0.2.1")

        in_progress = state.get_in_progress_tasks()

        assert "0.1.2" in in_progress
        assert "0.2.1" in in_progress
        assert len(in_progress) == 2

    def test_get_failed_tasks(self, temp_roadmap: Path) -> None:
        """Test getting failed task IDs."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_failed("0.1.2", error="Error 1")
        state.mark_failed("0.2.1", error="Error 2")

        failed = state.get_failed_tasks()

        assert "0.1.2" in failed
        assert "0.2.1" in failed
        assert len(failed) == 2

    def test_get_completed_tasks(self, temp_roadmap: Path) -> None:
        """Test getting completed task IDs."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_completed("0.1.2")

        completed = state.get_completed_tasks()

        # 0.1.1 was already completed in roadmap
        assert "0.1.1" in completed
        assert "0.1.2" in completed


# =============================================================================
# ExecutorState Recovery Tests
# =============================================================================


class TestExecutorStateRecovery:
    """Test ExecutorState crash recovery."""

    def test_recover_resets_in_progress(self, temp_roadmap: Path) -> None:
        """Test recovery resets in-progress tasks."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_started("0.1.2")
        state.mark_started("0.2.1")

        # Verify in-progress
        assert state.get_status("0.1.2") == TaskStatus.IN_PROGRESS
        assert state.get_status("0.2.1") == TaskStatus.IN_PROGRESS

        recovered = state.recover()

        # Should be reset to pending
        assert state.get_status("0.1.2") == TaskStatus.PENDING
        assert state.get_status("0.2.1") == TaskStatus.PENDING
        assert "0.1.2" in recovered
        assert "0.2.1" in recovered

    def test_recover_leaves_completed(self, temp_roadmap: Path) -> None:
        """Test recovery doesn't affect completed tasks."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_completed("0.1.2")

        recovered = state.recover()

        assert state.get_status("0.1.2") == TaskStatus.COMPLETED
        assert "0.1.2" not in recovered

    def test_recover_leaves_failed(self, temp_roadmap: Path) -> None:
        """Test recovery doesn't affect failed tasks."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_failed("0.1.2", error="Error")

        recovered = state.recover()

        assert state.get_status("0.1.2") == TaskStatus.FAILED
        assert "0.1.2" not in recovered

    def test_recover_returns_empty_when_none(self, temp_roadmap: Path) -> None:
        """Test recovery returns empty list when nothing to recover."""
        state = ExecutorState.from_roadmap(temp_roadmap)

        recovered = state.recover()

        assert recovered == []


# =============================================================================
# ExecutorState Persistence Tests
# =============================================================================


class TestExecutorStatePersistence:
    """Test ExecutorState save/load operations."""

    def test_save_creates_directory(self, temp_roadmap: Path) -> None:
        """Test save creates state directory."""
        state = ExecutorState.from_roadmap(temp_roadmap)

        state_file = state.save()

        assert state.state_dir.exists()
        assert state_file.exists()

    def test_save_creates_gitignore(self, temp_roadmap: Path) -> None:
        """Test save creates .gitignore."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.save()

        gitignore = state.state_dir / ".gitignore"
        assert gitignore.exists()
        assert "*" in gitignore.read_text()

    def test_save_format(self, temp_roadmap: Path) -> None:
        """Test saved state format."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_started("0.1.2")
        state.mark_completed("0.1.2", files_modified=["test.py"])

        state.save()

        with open(state.state_file) as f:
            data = json.load(f)

        assert data["version"] == 1
        assert "roadmap_path" in data
        assert "roadmap_hash" in data
        assert "run_id" in data
        assert "tasks" in data
        assert "0.1.2" in data["tasks"]
        assert data["tasks"]["0.1.2"]["status"] == "completed"

    def test_load_existing(self, temp_roadmap: Path) -> None:
        """Test loading existing state."""
        # Create and save state
        state1 = ExecutorState.from_roadmap(temp_roadmap)
        state1.mark_started("0.1.2")
        state1.mark_completed("0.1.2", files_modified=["test.py"])
        state1.save()

        # Load state
        state2 = ExecutorState.load(temp_roadmap)

        assert state2.get_status("0.1.2") == TaskStatus.COMPLETED
        task_state = state2.get_task_state("0.1.2")
        assert task_state is not None
        assert task_state.files_modified == ["test.py"]

    def test_load_nonexistent_creates_fresh(self, temp_roadmap: Path) -> None:
        """Test loading nonexistent state creates fresh."""
        state = ExecutorState.load(temp_roadmap)

        # Should have tasks from roadmap
        assert len(state.tasks) > 0

    def test_load_corrupt_creates_fresh(self, temp_roadmap: Path) -> None:
        """Test loading corrupt state creates fresh."""
        # Create corrupt state file
        state_dir = temp_roadmap.parent / ".executor"
        state_dir.mkdir()
        state_file = state_dir / "state.json"
        state_file.write_text("{ invalid json")

        state = ExecutorState.load(temp_roadmap)

        # Should create fresh state
        assert len(state.tasks) > 0

    def test_roundtrip(self, temp_roadmap: Path) -> None:
        """Test save/load roundtrip."""
        state1 = ExecutorState.from_roadmap(temp_roadmap)
        state1.mark_started("0.1.2")
        state1.mark_completed("0.1.2", files_modified=["a.py", "b.py"])
        state1.mark_failed("0.2.1", error="Test error", category=FailureCategory.IMPORT_ERROR)
        state1.save()

        state2 = ExecutorState.load(temp_roadmap)

        assert state2.get_status("0.1.2") == TaskStatus.COMPLETED
        assert state2.get_status("0.2.1") == TaskStatus.FAILED

        task1 = state2.get_task_state("0.1.2")
        assert task1 is not None
        assert task1.files_modified == ["a.py", "b.py"]

        task2 = state2.get_task_state("0.2.1")
        assert task2 is not None
        assert task2.error == "Test error"
        assert task2.failure_category == FailureCategory.IMPORT_ERROR


# =============================================================================
# ExecutorState ROADMAP Sync Tests
# =============================================================================


class TestExecutorStateSync:
    """Test ExecutorState ROADMAP synchronization."""

    def test_sync_to_roadmap_updates_checkbox(self, temp_roadmap: Path) -> None:
        """Test syncing updates checkboxes."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_completed("0.1.2")  # "Configure linting"

        updated = state.sync_to_roadmap()

        assert updated == 1

        # Verify the file was updated
        content = temp_roadmap.read_text()
        assert "- [x] **Configure linting**" in content

    def test_sync_to_roadmap_multiple(self, temp_roadmap: Path) -> None:
        """Test syncing multiple completed tasks."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_completed("0.1.2")
        state.mark_completed("0.2.1")

        updated = state.sync_to_roadmap()

        assert updated == 2

    def test_sync_to_roadmap_idempotent(self, temp_roadmap: Path) -> None:
        """Test syncing is idempotent."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_completed("0.1.2")

        updated1 = state.sync_to_roadmap()
        updated2 = state.sync_to_roadmap()

        assert updated1 == 1
        assert updated2 == 0  # Already updated

    def test_sync_to_roadmap_preserves_formatting(self, temp_roadmap: Path) -> None:
        """Test syncing preserves other formatting."""
        temp_roadmap.read_text()
        state = ExecutorState.from_roadmap(temp_roadmap)
        state.mark_completed("0.1.2")

        state.sync_to_roadmap()

        updated = temp_roadmap.read_text()
        # Phase headers should be preserved
        assert "## Phase 0: Foundation" in updated
        assert "## Phase 1: Features" in updated

    def test_sync_to_roadmap_nonexistent_file(self) -> None:
        """Test syncing nonexistent file raises error."""
        state = ExecutorState("/nonexistent/ROADMAP.md")

        with pytest.raises(FileNotFoundError):
            state.sync_to_roadmap()

    def test_check_roadmap_changed(self, temp_roadmap: Path) -> None:
        """Test detecting roadmap changes."""
        state = ExecutorState.from_roadmap(temp_roadmap)

        # Initially unchanged
        assert state.check_roadmap_changed() is False

        # Modify the file
        content = temp_roadmap.read_text()
        temp_roadmap.write_text(content + "\n\n- [ ] **New task**\n")

        # Should detect change
        assert state.check_roadmap_changed() is True

    def test_refresh_from_roadmap_adds_new_tasks(self, temp_roadmap: Path) -> None:
        """Test refreshing adds new tasks."""
        state = ExecutorState.from_roadmap(temp_roadmap)
        initial_count = len(state.tasks)

        # Add a new task to roadmap
        content = temp_roadmap.read_text()
        new_content = content.replace(
            "### 1.1 API",
            "### 1.1 API\n\n- [ ] **Brand new task**\n  Description here.\n",
        )
        temp_roadmap.write_text(new_content)

        updated = state.refresh_from_roadmap()

        assert updated >= 1
        assert len(state.tasks) > initial_count

    def test_refresh_from_roadmap_updates_completed(self, temp_roadmap: Path) -> None:
        """Test refreshing updates manually completed tasks."""
        state = ExecutorState.from_roadmap(temp_roadmap)

        # Manually mark task as complete in ROADMAP
        content = temp_roadmap.read_text()
        new_content = content.replace("- [ ] **Configure linting**", "- [x] **Configure linting**")
        temp_roadmap.write_text(new_content)

        updated = state.refresh_from_roadmap()

        assert updated >= 1
        assert state.get_status("0.1.2") == TaskStatus.COMPLETED


# =============================================================================
# ExecutorState Repr Tests
# =============================================================================


class TestExecutorStateRepr:
    """Test ExecutorState string representation."""

    def test_repr(self, temp_roadmap: Path) -> None:
        """Test string representation."""
        state = ExecutorState.from_roadmap(temp_roadmap)

        repr_str = repr(state)

        assert "ExecutorState" in repr_str
        assert "ROADMAP.md" in repr_str
        assert "tasks=" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================


class TestExecutorStateIntegration:
    """Integration tests for ExecutorState."""

    def test_full_workflow(self, temp_roadmap: Path) -> None:
        """Test complete workflow."""
        # Load state
        state = ExecutorState.load(temp_roadmap)

        # Execute first task
        state.mark_started("0.1.2", agent_run_id="agent_001")
        state.mark_completed(
            "0.1.2",
            files_modified=["pyproject.toml"],
            token_usage={"prompt": 500, "completion": 200},
        )

        # Execute second task (fails)
        state.mark_started("0.2.1", agent_run_id="agent_002")
        state.mark_failed(
            "0.2.1",
            error="Import error in generated code",
            category=FailureCategory.IMPORT_ERROR,
        )

        # Save and sync
        state.save()
        state.sync_to_roadmap()

        # Verify state
        summary = state.get_summary()
        assert summary.completed >= 2  # 0.1.1 + 0.1.2
        assert summary.failed == 1
        assert summary.total_tokens == 700

        # Reload and verify persistence
        state2 = ExecutorState.load(temp_roadmap)
        assert state2.get_status("0.1.2") == TaskStatus.COMPLETED
        assert state2.get_status("0.2.1") == TaskStatus.FAILED

    def test_crash_recovery_workflow(self, temp_roadmap: Path) -> None:
        """Test crash recovery workflow."""
        # Simulate crash during execution
        state1 = ExecutorState.load(temp_roadmap)
        state1.mark_started("0.1.2")
        state1.mark_started("0.2.1")
        state1.save()

        # Simulate restart
        state2 = ExecutorState.load(temp_roadmap)

        # Both should be in progress (as persisted)
        assert state2.get_status("0.1.2") == TaskStatus.IN_PROGRESS
        assert state2.get_status("0.2.1") == TaskStatus.IN_PROGRESS

        # Recover
        recovered = state2.recover()

        # Should be back to pending
        assert "0.1.2" in recovered
        assert "0.2.1" in recovered
        assert state2.get_status("0.1.2") == TaskStatus.PENDING
        assert state2.get_status("0.2.1") == TaskStatus.PENDING

    def test_multiple_runs(self, temp_roadmap: Path) -> None:
        """Test multiple execution runs."""
        # First run
        state1 = ExecutorState.load(temp_roadmap)
        run_id_1 = state1.run_id
        state1.mark_completed("0.1.2")
        state1.save()

        # Second run (fresh load)
        state2 = ExecutorState.load(temp_roadmap)

        # Should preserve run_id from saved state
        assert state2.run_id == run_id_1
        # Should preserve completed status
        assert state2.get_status("0.1.2") == TaskStatus.COMPLETED
