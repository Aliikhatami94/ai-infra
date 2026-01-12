"""Tests for Phase 1.3: State Migration.

Tests cover:
- 1.3.1: Port ExecutorState to graph state
- 1.3.2: Integrate memory layers
- 1.3.3: State pruning strategy
- 1.3.4: Graph-aware checkpointing
- 1.3.5: Checkpoint triggers
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ai_infra.executor.state_migration import (
    MAX_RUN_MEMORY_ENTRIES,
    CheckpointTrigger,
    GraphStatePersistence,
    MemoryIntegration,
    StatePruning,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def persistence(temp_project_dir: Path) -> GraphStatePersistence:
    """Create a GraphStatePersistence instance."""
    return GraphStatePersistence(project_root=temp_project_dir)


@pytest.fixture
def memory_integration(temp_project_dir: Path) -> MemoryIntegration:
    """Create a MemoryIntegration instance."""
    return MemoryIntegration(project_root=temp_project_dir)


@pytest.fixture
def pruner() -> StatePruning:
    """Create a StatePruning instance."""
    return StatePruning()


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Create a sample ExecutorGraphState-like dict."""
    return {
        "roadmap_path": "/project/ROADMAP.md",
        "run_id": "run-123",
        "todos": [],
        "current_task": None,
        "context": "sample context",
        "prompt": "sample prompt",
        "agent_result": None,
        "error": None,
        "run_memory": {
            "completed_tasks": [],
            "started_at": datetime.now(UTC).isoformat(),
        },
        "iteration": 0,
        "max_iterations": 100,
        "should_continue": True,
    }


# =============================================================================
# Tests: 1.3.1 Port ExecutorState to graph state
# =============================================================================


class TestGraphStatePersistence:
    """Tests for GraphStatePersistence class."""

    def test_ensure_executor_dir(self, persistence: GraphStatePersistence) -> None:
        """Test .executor directory creation."""
        assert not persistence.executor_dir.exists()
        persistence.ensure_executor_dir()
        assert persistence.executor_dir.exists()

    def test_save_state(
        self, persistence: GraphStatePersistence, sample_state: dict[str, Any]
    ) -> None:
        """Test saving graph state to JSON."""
        path = persistence.save_state(sample_state)

        assert path.exists()
        assert path.name == "graph_state.json"

        # Verify content
        data = json.loads(path.read_text())
        assert data["roadmap_path"] == "/project/ROADMAP.md"
        assert data["run_id"] == "run-123"
        assert "__metadata__" in data
        assert "saved_at" in data["__metadata__"]

    def test_load_state(
        self, persistence: GraphStatePersistence, sample_state: dict[str, Any]
    ) -> None:
        """Test loading graph state from JSON."""
        # First save
        persistence.save_state(sample_state)

        # Then load
        loaded = persistence.load_state()

        assert loaded is not None
        assert loaded["roadmap_path"] == "/project/ROADMAP.md"
        assert loaded["run_id"] == "run-123"
        assert "__metadata__" not in loaded  # Metadata stripped

    def test_load_state_no_file(self, persistence: GraphStatePersistence) -> None:
        """Test loading when no state file exists."""
        loaded = persistence.load_state()
        assert loaded is None

    def test_get_or_create_thread_id_new(self, persistence: GraphStatePersistence) -> None:
        """Test creating new thread ID."""
        thread_id = persistence.get_or_create_thread_id()

        assert thread_id.startswith("executor-")
        assert len(thread_id) > 12
        assert persistence.thread_id_path.exists()

    def test_get_or_create_thread_id_existing(self, persistence: GraphStatePersistence) -> None:
        """Test returning existing thread ID."""
        # Create first
        first_id = persistence.get_or_create_thread_id()

        # Get again
        second_id = persistence.get_or_create_thread_id()

        assert first_id == second_id

    def test_reset_thread_id(self, persistence: GraphStatePersistence) -> None:
        """Test resetting thread ID."""
        first_id = persistence.get_or_create_thread_id()
        new_id = persistence.reset_thread_id()

        assert new_id != first_id
        assert new_id.startswith("executor-")

    def test_clear_state(
        self, persistence: GraphStatePersistence, sample_state: dict[str, Any]
    ) -> None:
        """Test clearing saved state."""
        persistence.save_state(sample_state)
        assert persistence.graph_state_path.exists()

        persistence.clear_state()
        assert not persistence.graph_state_path.exists()


# =============================================================================
# Tests: 1.3.2 Integrate memory layers
# =============================================================================


class TestMemoryIntegration:
    """Tests for MemoryIntegration class."""

    def test_load_memories_empty(self, memory_integration: MemoryIntegration) -> None:
        """Test loading memories when nothing exists."""
        memories = memory_integration.load_memories()

        assert "run_memory" in memories
        assert "project_memory" in memories
        assert "learning_store" in memories
        assert memories["run_memory"] == {}

    def test_save_run_memory(self, memory_integration: MemoryIntegration) -> None:
        """Test saving run memory backup."""
        run_memory = {
            "completed_tasks": [{"id": "task-1", "status": "completed"}],
            "started_at": datetime.now(UTC).isoformat(),
        }

        path = memory_integration.save_run_memory(run_memory)

        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["completed_tasks"]) == 1

    def test_load_memories_with_backup(self, memory_integration: MemoryIntegration) -> None:
        """Test loading run memory from backup."""
        # Save a backup first
        run_memory = {"completed_tasks": [{"id": "task-1"}]}
        memory_integration.save_run_memory(run_memory)

        # Load memories
        memories = memory_integration.load_memories()

        assert memories["run_memory"]["completed_tasks"][0]["id"] == "task-1"

    def test_update_run_memory_from_outcome(self, memory_integration: MemoryIntegration) -> None:
        """Test updating run memory with a task outcome."""
        run_memory: dict[str, Any] = {}

        updated = memory_integration.update_run_memory_from_outcome(
            run_memory,
            task_id="task-1",
            title="Test task",
            status="completed",
            files_modified=["src/app.py"],
            summary="Implemented feature X",
        )

        assert "completed_tasks" in updated
        assert len(updated["completed_tasks"]) == 1
        assert updated["completed_tasks"][0]["id"] == "task-1"
        assert updated["completed_tasks"][0]["title"] == "Test task"
        assert updated["completed_tasks"][0]["files"] == ["src/app.py"]


# =============================================================================
# Tests: 1.3.3 State pruning strategy
# =============================================================================


class TestStatePruning:
    """Tests for StatePruning class."""

    def test_get_state_size(self, pruner: StatePruning) -> None:
        """Test calculating state size."""
        state = {"key": "value" * 100}
        size = pruner.get_state_size(state)

        assert size > 0
        assert size < 10000  # Should be reasonably small

    def test_prune_run_memory_no_action(self, pruner: StatePruning) -> None:
        """Test pruning when under limit."""
        state = {"run_memory": {"completed_tasks": [{"id": f"task-{i}"} for i in range(10)]}}

        pruned = pruner.prune_if_needed(state)

        # Should not change when under limit
        assert len(pruned["run_memory"]["completed_tasks"]) == 10
        assert "pruned_summary" not in pruned["run_memory"]

    def test_prune_run_memory_over_limit(self, pruner: StatePruning) -> None:
        """Test pruning when over limit."""
        # Create state with more than MAX_RUN_MEMORY_ENTRIES
        state = {
            "run_memory": {
                "completed_tasks": [
                    {"id": f"task-{i}", "status": "completed", "files": [f"file{i}.py"]}
                    for i in range(MAX_RUN_MEMORY_ENTRIES + 20)
                ]
            }
        }

        pruned = pruner.prune_if_needed(state)

        # Should be trimmed to limit
        assert len(pruned["run_memory"]["completed_tasks"]) == MAX_RUN_MEMORY_ENTRIES
        assert "pruned_summary" in pruned["run_memory"]
        assert pruned["run_memory"]["pruned_count"] == 20

    def test_prune_run_memory_keeps_recent(self, pruner: StatePruning) -> None:
        """Test that pruning keeps most recent entries."""
        total = MAX_RUN_MEMORY_ENTRIES + 10
        state = {
            "run_memory": {
                "completed_tasks": [
                    {"id": f"task-{i}", "status": "completed"} for i in range(total)
                ]
            }
        }

        pruned = pruner.prune_if_needed(state)

        # Check that most recent are kept
        kept = pruned["run_memory"]["completed_tasks"]
        assert kept[0]["id"] == "task-10"  # First kept
        assert kept[-1]["id"] == f"task-{total - 1}"  # Last kept

    def test_summarize_entries(self, pruner: StatePruning) -> None:
        """Test entry summarization."""
        entries = [
            {"status": "completed", "files": ["a.py", "b.py"]},
            {"status": "completed", "files": ["c.py"]},
            {"status": "failed", "files": []},
        ]

        summary = pruner._summarize_entries(entries)

        assert "2 tasks" in summary  # 2 completed
        assert "1 failure" in summary
        assert "3 files" in summary

    def test_aggressive_prune(self, pruner: StatePruning) -> None:
        """Test aggressive pruning for large states."""
        # Create large state
        state = {
            "context": "x" * 100000,
            "prompt": "y" * 100000,
            "agent_result": "z" * 100000,
            "run_memory": {"completed_tasks": [{"id": f"task-{i}"} for i in range(100)]},
        }

        pruned = pruner._aggressive_prune(state)

        assert pruned["context"] == ""
        assert pruned["prompt"] == ""
        assert pruned["agent_result"] is None
        assert len(pruned["run_memory"]["completed_tasks"]) == 10


# =============================================================================
# Tests: 1.3.5 Checkpoint triggers
# =============================================================================


class TestCheckpointTrigger:
    """Tests for CheckpointTrigger class."""

    def test_should_checkpoint_checkpoint_node(self) -> None:
        """Test checkpoint trigger for checkpoint node."""
        assert CheckpointTrigger.should_checkpoint("checkpoint") is True

    def test_should_checkpoint_handle_failure_node(self) -> None:
        """Test checkpoint trigger for handle_failure node."""
        assert CheckpointTrigger.should_checkpoint("handle_failure") is True

    def test_should_checkpoint_regular_node(self) -> None:
        """Test checkpoint trigger for regular nodes."""
        assert CheckpointTrigger.should_checkpoint("execute_task") is False
        assert CheckpointTrigger.should_checkpoint("build_context") is False

    def test_get_checkpoint_type(self) -> None:
        """Test getting checkpoint type."""
        assert CheckpointTrigger.get_checkpoint_type("checkpoint") == "full"
        assert CheckpointTrigger.get_checkpoint_type("handle_failure") == "graph"
        assert CheckpointTrigger.get_checkpoint_type("decide_next") == "graph"
        assert CheckpointTrigger.get_checkpoint_type("other") is None

    def test_should_git_checkpoint(self) -> None:
        """Test git checkpoint triggers."""
        assert CheckpointTrigger.should_git_checkpoint("checkpoint") is True
        assert CheckpointTrigger.should_git_checkpoint("handle_failure") is False

    def test_should_sync_roadmap(self) -> None:
        """Test ROADMAP sync triggers."""
        assert CheckpointTrigger.should_sync_roadmap("checkpoint") is True
        assert CheckpointTrigger.should_sync_roadmap("handle_failure") is False


# =============================================================================
# Tests: 1.3.4 Graph-aware checkpointing
# =============================================================================


class TestExecutorCheckpointer:
    """Tests for ExecutorCheckpointer class."""

    def test_put_and_get(self, temp_project_dir: Path) -> None:
        """Test saving and retrieving checkpoints."""
        from ai_infra.executor.checkpointer import ExecutorCheckpointer

        checkpointer = ExecutorCheckpointer(temp_project_dir)
        config = {"configurable": {"thread_id": "test-thread"}}

        checkpoint = {
            "v": 1,
            "id": "checkpoint-1",
            "ts": datetime.now(UTC).isoformat(),
            "channel_values": {"roadmap_path": "/project/ROADMAP.md"},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata = {"source": "test"}

        # Save checkpoint
        result = checkpointer.put(config, checkpoint, metadata)

        assert "configurable" in result
        assert result["configurable"]["thread_id"] == "test-thread"

        # Retrieve checkpoint
        retrieved = checkpointer.get_tuple(config)

        assert retrieved is not None
        assert retrieved.checkpoint["id"] == "checkpoint-1"
        assert retrieved.checkpoint["channel_values"]["roadmap_path"] == "/project/ROADMAP.md"

    def test_list_checkpoints(self, temp_project_dir: Path) -> None:
        """Test listing multiple checkpoints."""
        from ai_infra.executor.checkpointer import ExecutorCheckpointer

        checkpointer = ExecutorCheckpointer(temp_project_dir)
        config = {"configurable": {"thread_id": "list-thread"}}

        # Save multiple checkpoints
        for i in range(3):
            checkpoint = {
                "v": 1,
                "id": f"checkpoint-{i}",
                "ts": datetime.now(UTC).isoformat(),
                "channel_values": {"iteration": i},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            checkpointer.put(config, checkpoint, {})

        # List all checkpoints
        checkpoints = list(checkpointer.list(config))

        assert len(checkpoints) == 3
        # Should be newest first
        assert checkpoints[0].checkpoint["id"] == "checkpoint-2"

    def test_max_checkpoints_cleanup(self, temp_project_dir: Path) -> None:
        """Test that old checkpoints are cleaned up."""
        from ai_infra.executor.checkpointer import ExecutorCheckpointer

        checkpointer = ExecutorCheckpointer(temp_project_dir, max_checkpoints=3)
        config = {"configurable": {"thread_id": "cleanup-thread"}}

        # Save more than max checkpoints
        for i in range(5):
            checkpoint = {
                "v": 1,
                "id": f"checkpoint-{i}",
                "ts": datetime.now(UTC).isoformat(),
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            checkpointer.put(config, checkpoint, {})

        # Should only have 3 remaining
        checkpoints = list(checkpointer.list(config))
        assert len(checkpoints) == 3

    def test_clear_thread(self, temp_project_dir: Path) -> None:
        """Test clearing all checkpoints for a thread."""
        from ai_infra.executor.checkpointer import ExecutorCheckpointer

        checkpointer = ExecutorCheckpointer(temp_project_dir)
        config = {"configurable": {"thread_id": "clear-thread"}}

        # Save a checkpoint
        checkpoint = {
            "v": 1,
            "id": "checkpoint-1",
            "ts": datetime.now(UTC).isoformat(),
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        checkpointer.put(config, checkpoint, {})

        # Clear thread
        checkpointer.clear_thread("clear-thread")

        # Should be empty
        retrieved = checkpointer.get_tuple(config)
        assert retrieved is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestStateMigrationIntegration:
    """Integration tests for state migration components."""

    def test_full_persistence_cycle(self, temp_project_dir: Path) -> None:
        """Test full save/load cycle with all components."""
        persistence = GraphStatePersistence(temp_project_dir)
        memory = MemoryIntegration(temp_project_dir)
        pruner = StatePruning()

        # Create initial state
        state = {
            "roadmap_path": str(temp_project_dir / "ROADMAP.md"),
            "run_id": "run-test",
            "todos": [],
            "current_task": None,
            "run_memory": {},
        }

        # Add some outcomes
        for i in range(5):
            state["run_memory"] = memory.update_run_memory_from_outcome(
                state["run_memory"],
                task_id=f"task-{i}",
                title=f"Task {i}",
                status="completed",
                files_modified=[f"file{i}.py"],
            )

        # Prune if needed
        state = pruner.prune_if_needed(state)

        # Save state
        persistence.save_state(state)
        memory.save_run_memory(state["run_memory"])

        # Load state fresh
        loaded = persistence.load_state()

        assert loaded is not None
        assert loaded["run_id"] == "run-test"
        assert len(loaded["run_memory"]["completed_tasks"]) == 5

    def test_resume_from_checkpoint(self, temp_project_dir: Path) -> None:
        """Test resuming execution from saved state."""
        from ai_infra.executor.checkpointer import ExecutorCheckpointer

        persistence = GraphStatePersistence(temp_project_dir)
        checkpointer = ExecutorCheckpointer(temp_project_dir)

        # Get thread ID
        thread_id = persistence.get_or_create_thread_id()
        config = {"configurable": {"thread_id": thread_id}}

        # Save a checkpoint
        checkpoint = {
            "v": 1,
            "id": "checkpoint-resume",
            "ts": datetime.now(UTC).isoformat(),
            "channel_values": {
                "iteration": 5,
                "current_task": {"id": "task-5", "title": "Resume here"},
            },
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        checkpointer.put(config, checkpoint, {"step": "execute_task"})

        # Simulate new run resuming
        thread_id_2 = persistence.get_or_create_thread_id()
        assert thread_id_2 == thread_id  # Same thread ID

        # Load checkpoint
        retrieved = checkpointer.get_tuple(config)

        assert retrieved is not None
        assert retrieved.checkpoint["channel_values"]["iteration"] == 5
        assert retrieved.checkpoint["channel_values"]["current_task"]["title"] == "Resume here"
