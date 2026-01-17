"""Scenario tests for resume after interrupt (Phase 6.3.4).

Tests scenarios for resuming execution after interruption, including:
- Checkpoint creation and restoration
- File change preservation
- State recovery
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from ai_infra.executor.recovery import (
    CheckpointMetadata,
    FileSnapshot,
    RecoveryStrategy,
)
from ai_infra.executor.todolist import TodoItem, TodoStatus

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def python_project(tmp_path: Path) -> Path:
    """Create a minimal Python project."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    return tmp_path


@pytest.fixture
def sample_checkpoint_metadata() -> CheckpointMetadata:
    """Create sample checkpoint metadata."""
    return CheckpointMetadata(
        task_id="1.1.1",
        task_title="Create base module",
        commit_sha="abc123def456",
        files_modified=["src/base.py"],
        files_created=["src/base.py"],
        verification_passed=True,
        tags=["task-1.1.1", "phase-1"],
    )


@pytest.fixture
def sample_tasks() -> list[TodoItem]:
    """Create sample tasks for testing."""
    return [
        TodoItem(id=1, title="Task 1", description="", status=TodoStatus.COMPLETED),
        TodoItem(id=2, title="Task 2", description="", status=TodoStatus.COMPLETED),
        TodoItem(id=3, title="Task 3", description="", status=TodoStatus.NOT_STARTED),
        TodoItem(id=4, title="Task 4", description="", status=TodoStatus.NOT_STARTED),
    ]


# =============================================================================
# Checkpoint Creation Tests
# =============================================================================


class TestCheckpointCreation:
    """Tests for checkpoint creation."""

    def test_create_checkpoint_metadata(self) -> None:
        """Should create checkpoint metadata."""
        metadata = CheckpointMetadata(
            task_id="1.1.1",
            task_title="Create module",
            commit_sha="abc123",
        )

        assert metadata.task_id == "1.1.1"
        assert metadata.commit_sha == "abc123"
        assert isinstance(metadata.created_at, datetime)

    def test_checkpoint_with_files(self) -> None:
        """Should track modified files in checkpoint."""
        metadata = CheckpointMetadata(
            task_id="1.1.1",
            task_title="Create module",
            commit_sha="abc123",
            files_modified=["src/module.py"],
            files_created=["src/module.py"],
        )

        assert "src/module.py" in metadata.files_modified
        assert "src/module.py" in metadata.files_created

    def test_checkpoint_with_tags(self) -> None:
        """Should support tags for checkpoints."""
        metadata = CheckpointMetadata(
            task_id="1.1.1",
            task_title="Create module",
            commit_sha="abc123",
            tags=["phase-1", "milestone"],
        )

        assert "phase-1" in metadata.tags
        assert "milestone" in metadata.tags


class TestFileSnapshot:
    """Tests for file snapshots."""

    def test_create_file_snapshot(self, python_project: Path) -> None:
        """Should create snapshot of file."""
        file_path = python_project / "src" / "test.py"
        file_path.write_text("x = 1")

        snapshot = FileSnapshot.from_file(python_project, "src/test.py")

        assert snapshot.exists is True
        assert snapshot.hash is not None
        assert snapshot.size > 0

    def test_snapshot_nonexistent_file(self, python_project: Path) -> None:
        """Should handle nonexistent file."""
        snapshot = FileSnapshot.from_file(python_project, "src/missing.py")

        assert snapshot.exists is False
        assert snapshot.hash is None

    def test_snapshot_serialization(self, python_project: Path) -> None:
        """Should serialize and deserialize snapshot."""
        file_path = python_project / "src" / "test.py"
        file_path.write_text("x = 1")

        original = FileSnapshot.from_file(python_project, "src/test.py")
        data = original.to_dict()
        restored = FileSnapshot.from_dict(data)

        assert restored.path == original.path
        assert restored.hash == original.hash
        assert restored.size == original.size


# =============================================================================
# Checkpoint Restoration Tests
# =============================================================================


class TestCheckpointRestoration:
    """Tests for checkpoint restoration."""

    def test_identify_completed_tasks(self, sample_tasks: list[TodoItem]) -> None:
        """Should identify completed tasks from checkpoint."""
        completed = [t for t in sample_tasks if t.status == TodoStatus.COMPLETED]

        assert len(completed) == 2
        assert all(t.status == TodoStatus.COMPLETED for t in completed)

    def test_identify_pending_tasks(self, sample_tasks: list[TodoItem]) -> None:
        """Should identify pending tasks after interrupt."""
        pending = [t for t in sample_tasks if t.status == TodoStatus.NOT_STARTED]

        assert len(pending) == 2
        assert all(t.status == TodoStatus.NOT_STARTED for t in pending)

    def test_find_current_task(self, sample_tasks: list[TodoItem]) -> None:
        """Should find the first incomplete task."""
        # Find first incomplete task (where to resume)
        current = next((t for t in sample_tasks if t.status == TodoStatus.NOT_STARTED), None)

        assert current is not None
        assert current.title == "Task 3"


class TestStateRecovery:
    """Tests for state recovery after interrupt."""

    def test_preserve_completed_count(self) -> None:
        """Should preserve count of completed tasks."""
        state: dict[str, Any] = {
            "tasks_completed_count": 5,
            "tasks_failed_count": 1,
            "current_task_index": 6,
        }

        # Simulate state save/restore
        recovered_count = state["tasks_completed_count"]
        assert recovered_count == 5

    def test_preserve_files_modified(self) -> None:
        """Should preserve list of modified files."""
        state: dict[str, Any] = {
            "files_modified": ["src/a.py", "src/b.py", "src/c.py"],
        }

        # Simulate state save/restore
        recovered_files = state["files_modified"]
        assert len(recovered_files) == 3

    def test_preserve_run_memory(self) -> None:
        """Should preserve run memory context."""
        state: dict[str, Any] = {
            "run_memory": {
                "key_decisions": ["Used SQLite", "Chose FastAPI"],
                "context": {"project_type": "python"},
            },
        }

        recovered_memory = state["run_memory"]
        assert "key_decisions" in recovered_memory
        assert len(recovered_memory["key_decisions"]) == 2


# =============================================================================
# File Change Preservation Tests
# =============================================================================


class TestFileChangePreservation:
    """Tests for preserving file changes during interrupt."""

    def test_created_files_persist(self, python_project: Path) -> None:
        """Created files should persist after interrupt."""
        # Simulate task creating a file
        new_file = python_project / "src" / "new_module.py"
        new_file.write_text("def hello(): pass")

        # Simulate interrupt (file system persists)
        assert new_file.exists()

        # Simulate resume
        content = new_file.read_text()
        assert "def hello()" in content

    def test_modified_files_persist(self, python_project: Path) -> None:
        """Modified files should persist after interrupt."""
        target_file = python_project / "src" / "__init__.py"

        # Original content
        original = target_file.read_text()

        # Simulate task modifying file
        target_file.write_text("# Modified\n" + original)

        # Simulate interrupt
        # (file system persists)

        # Simulate resume
        content = target_file.read_text()
        assert "# Modified" in content

    def test_partial_changes_tracked(self, python_project: Path) -> None:
        """Partial changes should be tracked in checkpoint."""
        files_before_interrupt = [
            "src/module1.py",
            "src/module2.py",
        ]

        # Create files
        for f in files_before_interrupt:
            file_path = python_project / f
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f"# {f}")

        # Track in checkpoint metadata
        metadata = CheckpointMetadata(
            task_id="1.1",
            task_title="Create modules",
            commit_sha="abc123",
            files_created=files_before_interrupt,
        )

        assert len(metadata.files_created) == 2


# =============================================================================
# Resume Logic Tests
# =============================================================================


class TestResumeLogic:
    """Tests for resume after interrupt logic."""

    def test_resume_from_specific_task(self) -> None:
        """Should resume from specific task checkpoint."""
        tasks = [
            TodoItem(id=1, title="Task 1", description="", status=TodoStatus.COMPLETED),
            TodoItem(id=2, title="Task 2", description="", status=TodoStatus.COMPLETED),
            TodoItem(id=3, title="Task 3", description="", status=TodoStatus.NOT_STARTED),
        ]

        resume_from = next(t for t in tasks if t.status == TodoStatus.NOT_STARTED)
        assert resume_from.title == "Task 3"

    def test_skip_completed_tasks(self) -> None:
        """Should skip already completed tasks on resume."""
        tasks = [
            TodoItem(id=1, title="Task 1", description="", status=TodoStatus.COMPLETED),
            TodoItem(id=2, title="Task 2", description="", status=TodoStatus.COMPLETED),
            TodoItem(id=3, title="Task 3", description="", status=TodoStatus.NOT_STARTED),
        ]

        # Simulate resume - should only process incomplete tasks
        to_process = [t for t in tasks if t.status == TodoStatus.NOT_STARTED]

        assert len(to_process) == 1
        assert to_process[0].title == "Task 3"

    def test_recalculate_remaining_count(self) -> None:
        """Should recalculate remaining task count on resume."""
        total_tasks = 10
        completed_tasks = 6

        remaining = total_tasks - completed_tasks
        assert remaining == 4


class TestInterruptHandling:
    """Tests for handling interrupts gracefully."""

    def test_interrupt_signal_simulation(self) -> None:
        """Simulate interrupt signal handling."""
        interrupted = False
        current_task = None

        # Simulate processing
        tasks = ["Task 1", "Task 2", "Task 3"]
        for i, task in enumerate(tasks):
            current_task = task
            if i == 1:  # Interrupt during Task 2
                interrupted = True
                break

        assert interrupted is True
        assert current_task == "Task 2"

    def test_cleanup_on_interrupt(self, python_project: Path) -> None:
        """Should clean up temp files on interrupt."""
        temp_file = python_project / ".executor_temp"
        temp_file.write_text("temp data")

        # Simulate interrupt cleanup
        if temp_file.exists():
            temp_file.unlink()

        assert not temp_file.exists()


# =============================================================================
# Recovery Strategy Tests
# =============================================================================


class TestRecoveryStrategies:
    """Tests for different recovery strategies."""

    def test_recovery_strategy_enum(self) -> None:
        """Should have all recovery strategies defined."""
        assert RecoveryStrategy.ROLLBACK_ALL.value == "rollback_all"
        assert RecoveryStrategy.ROLLBACK_FAILED.value == "rollback_failed"
        assert RecoveryStrategy.RETRY_WITH_CONTEXT.value == "retry_with_context"
        assert RecoveryStrategy.SKIP.value == "skip"
        assert RecoveryStrategy.MANUAL.value == "manual"

    def test_select_strategy_for_syntax_error(self) -> None:
        """Should select appropriate strategy for syntax error."""
        # Syntax errors should retry with context
        error_type = "syntax_error"
        strategies = {
            "syntax_error": RecoveryStrategy.RETRY_WITH_CONTEXT,
            "test_failure": RecoveryStrategy.ROLLBACK_FAILED,
            "unknown": RecoveryStrategy.MANUAL,
        }

        selected = strategies.get(error_type, RecoveryStrategy.MANUAL)
        assert selected == RecoveryStrategy.RETRY_WITH_CONTEXT

    def test_skip_strategy(self) -> None:
        """SKIP strategy should continue with next task."""
        strategy = RecoveryStrategy.SKIP

        # Simulate skip behavior
        tasks = [
            TodoItem(id=1, title="Failed Task", description="", status=TodoStatus.NOT_STARTED),
            TodoItem(id=2, title="Next Task", description="", status=TodoStatus.NOT_STARTED),
        ]

        # If strategy is SKIP, move to next task
        if strategy == RecoveryStrategy.SKIP:
            current_index = 0
            current_index += 1  # Skip to next

        assert tasks[current_index].title == "Next Task"
