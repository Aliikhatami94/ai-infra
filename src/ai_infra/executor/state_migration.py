"""State migration for graph-based executor.

Phase 1.3: State Migration for ExecutorGraph.

This module provides:
- GraphStatePersistence: Persist/restore graph state to .executor/
- MemoryIntegration: Integrate RunMemory, ProjectMemory, LearningStore
- StatePruning: Limit state size and prune old data

The graph state is persisted to:
- `.executor/graph_state.json` - Current graph state
- `.executor/thread_id` - Thread ID for resume
- `.executor/run_memory.json` - Current run memory (ephemeral backup)
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.learning import LearningStore
    from ai_infra.executor.project_memory import ProjectMemory
    from ai_infra.executor.state import ExecutorGraphState

logger = get_logger("executor.state_migration")


# =============================================================================
# Constants
# =============================================================================

EXECUTOR_DIR = ".executor"
GRAPH_STATE_FILE = "graph_state.json"
THREAD_ID_FILE = "thread_id"
RUN_MEMORY_FILE = "run_memory.json"
PROJECT_MEMORY_FILE = "project_memory.json"
LEARNING_STORE_DIR = "learning"

# Pruning limits
MAX_RUN_MEMORY_ENTRIES = 50  # Maximum tool calls / outcomes to track
MAX_STATE_SIZE_BYTES = 1_000_000  # 1MB warning threshold
STATE_PRUNE_TARGET = 800_000  # Target size after pruning (800KB)


# =============================================================================
# Graph State Persistence
# =============================================================================


@dataclass
class GraphStatePersistence:
    """Handles persistence of graph state to .executor/ directory.

    Phase 1.3.1: Port ExecutorState to graph state.

    This class manages:
    - Saving/loading graph state to .executor/graph_state.json
    - Thread ID persistence for resume capability
    - State serialization with proper type handling

    Example:
        ```python
        persistence = GraphStatePersistence(Path("/project"))

        # Save state after node execution
        persistence.save_state(current_state)

        # Resume from saved state
        restored = persistence.load_state()

        # Get thread ID for continuation
        thread_id = persistence.get_or_create_thread_id()
        ```
    """

    project_root: Path
    """Root directory of the project (parent of .executor/)."""

    @property
    def executor_dir(self) -> Path:
        """Get the .executor directory path."""
        return self.project_root / EXECUTOR_DIR

    @property
    def graph_state_path(self) -> Path:
        """Get the graph state file path."""
        return self.executor_dir / GRAPH_STATE_FILE

    @property
    def thread_id_path(self) -> Path:
        """Get the thread ID file path."""
        return self.executor_dir / THREAD_ID_FILE

    def ensure_executor_dir(self) -> None:
        """Create .executor directory if it doesn't exist."""
        self.executor_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: ExecutorGraphState) -> Path:
        """Save graph state to .executor/graph_state.json.

        Args:
            state: The ExecutorGraphState to persist.

        Returns:
            Path to the saved state file.
        """
        self.ensure_executor_dir()

        # Serialize state with type conversion
        serialized = _serialize_graph_state(state)

        # Add metadata
        serialized["__metadata__"] = {
            "saved_at": datetime.now(UTC).isoformat(),
            "version": "1.0",
        }

        self.graph_state_path.write_text(
            json.dumps(serialized, indent=2, default=str),
            encoding="utf-8",
        )

        logger.debug(f"Saved graph state to {self.graph_state_path}")
        return self.graph_state_path

    def load_state(self) -> dict[str, Any] | None:
        """Load graph state from .executor/graph_state.json.

        Returns:
            Deserialized state dict, or None if no state exists.
        """
        if not self.graph_state_path.exists():
            logger.debug("No saved graph state found")
            return None

        try:
            data = json.loads(self.graph_state_path.read_text(encoding="utf-8"))
            # Remove metadata before returning
            data.pop("__metadata__", None)
            logger.debug(f"Loaded graph state from {self.graph_state_path}")
            return _deserialize_graph_state(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load graph state: {e}")
            return None

    def get_or_create_thread_id(self) -> str:
        """Get existing thread ID or create new one.

        Thread IDs are used by LangGraph for checkpointing and resume.

        Returns:
            The thread ID string.
        """
        self.ensure_executor_dir()

        if self.thread_id_path.exists():
            thread_id = self.thread_id_path.read_text(encoding="utf-8").strip()
            if thread_id:
                return thread_id

        # Generate new thread ID
        thread_id = f"executor-{uuid.uuid4().hex[:12]}"
        self.thread_id_path.write_text(thread_id, encoding="utf-8")
        logger.info(f"Created new thread ID: {thread_id}")
        return thread_id

    def reset_thread_id(self) -> str:
        """Reset thread ID for fresh execution.

        Returns:
            The new thread ID.
        """
        if self.thread_id_path.exists():
            self.thread_id_path.unlink()
        return self.get_or_create_thread_id()

    def clear_state(self) -> None:
        """Clear all saved graph state."""
        if self.graph_state_path.exists():
            self.graph_state_path.unlink()
            logger.info("Cleared graph state")


# =============================================================================
# Memory Integration
# =============================================================================


@dataclass
class MemoryIntegration:
    """Integrates memory layers with graph state.

    Phase 1.3.2: Integrate memory layers.

    Memory layers:
    - RunMemory: In-graph, per-run context (state["run_memory"])
    - ProjectMemory: Persisted outside graph (.executor/project_memory.json)
    - LearningStore: Persisted outside graph (.executor/learning/)

    Example:
        ```python
        integration = MemoryIntegration(Path("/project"))

        # Load all memories at run start
        memories = integration.load_memories()
        state["run_memory"] = memories["run_memory"]

        # Save memories at checkpoint
        integration.save_memories(state)
        ```
    """

    project_root: Path
    """Root directory of the project."""

    @property
    def executor_dir(self) -> Path:
        """Get the .executor directory path."""
        return self.project_root / EXECUTOR_DIR

    def load_memories(self) -> dict[str, Any]:
        """Load all memory layers.

        RunMemory is loaded from backup if available (for resume).
        ProjectMemory and LearningStore are loaded from their standard paths.

        Returns:
            Dict with keys: run_memory, project_memory, learning_store
        """
        from ai_infra.executor.learning import LearningStore
        from ai_infra.executor.project_memory import ProjectMemory

        result: dict[str, Any] = {}

        # Load RunMemory from backup (or create fresh)
        run_memory_path = self.executor_dir / RUN_MEMORY_FILE
        if run_memory_path.exists():
            try:
                data = json.loads(run_memory_path.read_text(encoding="utf-8"))
                result["run_memory"] = data
                logger.debug("Loaded run memory from backup")
            except (json.JSONDecodeError, KeyError):
                result["run_memory"] = {}
        else:
            result["run_memory"] = {}

        # Load ProjectMemory
        result["project_memory"] = ProjectMemory.load(self.project_root)

        # Load LearningStore
        learning_dir = self.executor_dir / LEARNING_STORE_DIR
        result["learning_store"] = LearningStore(data_dir=learning_dir)

        return result

    def save_run_memory(self, run_memory: dict[str, Any]) -> Path:
        """Save run memory to backup file.

        Args:
            run_memory: The run memory dict from graph state.

        Returns:
            Path to saved file.
        """
        self.executor_dir.mkdir(parents=True, exist_ok=True)
        path = self.executor_dir / RUN_MEMORY_FILE

        path.write_text(
            json.dumps(run_memory, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def save_project_memory(self, project_memory: ProjectMemory) -> Path:
        """Save project memory.

        Args:
            project_memory: ProjectMemory instance to save.

        Returns:
            Path to saved file.
        """
        return project_memory.save()

    def save_learning_store(self, learning_store: LearningStore) -> None:
        """Save learning store.

        Args:
            learning_store: LearningStore instance to save.
        """
        learning_store.save()

    def update_run_memory_from_outcome(
        self,
        run_memory: dict[str, Any],
        task_id: str,
        title: str,
        status: str,
        files_modified: list[str],
        summary: str = "",
    ) -> dict[str, Any]:
        """Update run memory with a task outcome.

        Args:
            run_memory: Current run memory dict.
            task_id: ID of completed task.
            title: Task title.
            status: Completion status.
            files_modified: List of file paths modified.
            summary: Optional summary of what was done.

        Returns:
            Updated run memory dict.
        """
        if "completed_tasks" not in run_memory:
            run_memory["completed_tasks"] = []

        run_memory["completed_tasks"].append(
            {
                "id": task_id,
                "title": title,
                "status": status,
                "files": files_modified,
                "summary": summary,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return run_memory


# =============================================================================
# State Pruning
# =============================================================================


@dataclass
class StatePruning:
    """Handles state size management and pruning.

    Phase 1.3.3: Implement state pruning strategy.

    Features:
    - Limits run_memory to MAX_RUN_MEMORY_ENTRIES (50)
    - Summarizes old entries before pruning
    - Warns if state exceeds MAX_STATE_SIZE_BYTES (1MB)
    - Prunes to STATE_PRUNE_TARGET (800KB) when needed

    Example:
        ```python
        pruner = StatePruning()

        # Check and prune state
        state = pruner.prune_if_needed(state)

        # Check state size
        size = pruner.get_state_size(state)
        if size > MAX_STATE_SIZE_BYTES:
            logger.warning(f"State size {size} exceeds limit")
        ```
    """

    max_run_memory_entries: int = MAX_RUN_MEMORY_ENTRIES
    max_state_size: int = MAX_STATE_SIZE_BYTES
    prune_target: int = STATE_PRUNE_TARGET

    def get_state_size(self, state: dict[str, Any]) -> int:
        """Calculate approximate size of state in bytes.

        Args:
            state: The graph state dict.

        Returns:
            Approximate size in bytes.
        """
        try:
            return len(json.dumps(state, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            return sys.getsizeof(state)

    def prune_if_needed(self, state: dict[str, Any]) -> dict[str, Any]:
        """Prune state if it exceeds size limits.

        Args:
            state: The graph state dict.

        Returns:
            Pruned state dict.
        """
        # Always prune run_memory entries
        state = self._prune_run_memory(state)

        # Check overall size
        size = self.get_state_size(state)
        if size > self.max_state_size:
            logger.warning(f"State size ({size:,} bytes) exceeds {self.max_state_size:,} limit")
            state = self._aggressive_prune(state)

        return state

    def _prune_run_memory(self, state: dict[str, Any]) -> dict[str, Any]:
        """Prune run_memory to max entries.

        Keeps most recent entries and creates a summary of pruned ones.

        Args:
            state: The graph state dict.

        Returns:
            State with pruned run_memory.
        """
        run_memory = state.get("run_memory", {})
        if not run_memory:
            return state

        completed_tasks = run_memory.get("completed_tasks", [])
        if len(completed_tasks) <= self.max_run_memory_entries:
            return state

        # Summarize old entries
        old_entries = completed_tasks[: -self.max_run_memory_entries]
        summary = self._summarize_entries(old_entries)

        # Keep recent entries
        recent_entries = completed_tasks[-self.max_run_memory_entries :]

        # Update run_memory
        run_memory = {
            **run_memory,
            "completed_tasks": recent_entries,
            "pruned_summary": summary,
            "pruned_count": len(old_entries),
        }

        logger.info(
            f"Pruned {len(old_entries)} old entries from run_memory (kept {len(recent_entries)})"
        )

        return {**state, "run_memory": run_memory}

    def _summarize_entries(self, entries: list[dict[str, Any]]) -> str:
        """Create a summary of pruned entries.

        Args:
            entries: List of task outcome dicts.

        Returns:
            Summary string.
        """
        if not entries:
            return ""

        completed = sum(1 for e in entries if e.get("status") == "completed")
        failed = sum(1 for e in entries if e.get("status") == "failed")

        # Collect unique files
        all_files: set[str] = set()
        for entry in entries:
            all_files.update(entry.get("files", []))

        summary_parts = [
            f"Previously completed {completed} tasks",
        ]
        if failed:
            summary_parts.append(f"with {failed} failures")
        if all_files:
            summary_parts.append(f"modifying {len(all_files)} files")

        return ". ".join(summary_parts) + "."

    def _aggressive_prune(self, state: dict[str, Any]) -> dict[str, Any]:
        """Aggressively prune state to meet size target.

        Args:
            state: The graph state dict.

        Returns:
            Aggressively pruned state.
        """
        # Clear large fields that can be regenerated
        pruned = {**state}

        # Clear context and prompt (regenerated each task)
        pruned["context"] = ""
        pruned["prompt"] = ""

        # Clear agent_result (transient)
        pruned["agent_result"] = None

        # Reduce run_memory further
        run_memory = pruned.get("run_memory", {})
        if run_memory.get("completed_tasks"):
            # Keep only last 10 in aggressive mode
            run_memory["completed_tasks"] = run_memory["completed_tasks"][-10:]
            pruned["run_memory"] = run_memory

        size = self.get_state_size(pruned)
        logger.info(f"Aggressive pruning reduced state to {size:,} bytes")

        return pruned


# =============================================================================
# Checkpoint Triggers
# =============================================================================


class CheckpointTrigger:
    """Defines when to trigger different types of checkpoints.

    Phase 1.3.5: Define checkpoint triggers.

    Checkpoint types:
    - GRAPH: Save graph state only (for resume)
    - GIT: Create git commit (for code changes)
    - FULL: Graph + git + ROADMAP sync

    Triggers:
    - checkpoint_node: FULL (git commit + graph + ROADMAP)
    - handle_failure_node: GRAPH (preserve error state)
    - decide_next_node: GRAPH (loop boundary)
    """

    GRAPH = "graph"
    GIT = "git"
    FULL = "full"

    # Node -> checkpoint type mapping
    NODE_TRIGGERS: dict[str, str] = {
        "checkpoint": "full",  # Git commit + graph checkpoint + ROADMAP sync
        "handle_failure": "graph",  # Graph checkpoint only (preserve error state)
        "decide_next": "graph",  # Graph checkpoint (loop boundary)
    }

    @classmethod
    def should_checkpoint(cls, node_name: str) -> bool:
        """Check if a node should trigger checkpointing.

        Args:
            node_name: Name of the node.

        Returns:
            True if checkpointing should occur.
        """
        return node_name in cls.NODE_TRIGGERS

    @classmethod
    def get_checkpoint_type(cls, node_name: str) -> str | None:
        """Get the checkpoint type for a node.

        Args:
            node_name: Name of the node.

        Returns:
            Checkpoint type or None if no checkpoint.
        """
        return cls.NODE_TRIGGERS.get(node_name)

    @classmethod
    def should_git_checkpoint(cls, node_name: str) -> bool:
        """Check if node should trigger git checkpoint.

        Args:
            node_name: Name of the node.

        Returns:
            True if git checkpoint should occur.
        """
        checkpoint_type = cls.get_checkpoint_type(node_name)
        return checkpoint_type in (cls.GIT, cls.FULL)

    @classmethod
    def should_sync_roadmap(cls, node_name: str) -> bool:
        """Check if node should sync to ROADMAP.

        Args:
            node_name: Name of the node.

        Returns:
            True if ROADMAP sync should occur.
        """
        return cls.get_checkpoint_type(node_name) == cls.FULL


# =============================================================================
# Serialization Helpers
# =============================================================================


def _serialize_graph_state(state: dict[str, Any]) -> dict[str, Any]:
    """Serialize graph state for JSON storage.

    Handles:
    - TodoItem objects -> dicts
    - Path objects -> strings
    - datetime objects -> ISO strings

    Args:
        state: The graph state dict.

    Returns:
        JSON-serializable dict.
    """
    serialized = {}

    for key, value in state.items():
        if key == "todos" and value:
            # Serialize TodoItem list
            serialized[key] = [
                item.to_dict() if hasattr(item, "to_dict") else item for item in value
            ]
        elif key == "current_task" and value is not None:
            # Serialize current TodoItem
            serialized[key] = value.to_dict() if hasattr(value, "to_dict") else value
        elif key == "error" and value is not None:
            # Error is already a dict (ExecutorError TypedDict)
            serialized[key] = value
        elif isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value

    return serialized


def _deserialize_graph_state(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize graph state from JSON storage.

    Args:
        data: The JSON-loaded dict.

    Returns:
        Graph state dict with proper types.
    """
    from ai_infra.executor.todolist import TodoItem

    deserialized = dict(data)

    # Deserialize todos
    if data.get("todos"):
        deserialized["todos"] = [
            TodoItem.from_dict(item) if isinstance(item, dict) else item for item in data["todos"]
        ]

    # Deserialize current_task
    if "current_task" in data and data["current_task"] is not None:
        if isinstance(data["current_task"], dict):
            deserialized["current_task"] = TodoItem.from_dict(data["current_task"])

    return deserialized


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "GraphStatePersistence",
    "MemoryIntegration",
    "StatePruning",
    "CheckpointTrigger",
    "MAX_RUN_MEMORY_ENTRIES",
    "MAX_STATE_SIZE_BYTES",
    "EXECUTOR_DIR",
    "GRAPH_STATE_FILE",
    "THREAD_ID_FILE",
]
