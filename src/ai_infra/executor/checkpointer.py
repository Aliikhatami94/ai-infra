"""Graph checkpointer for executor state persistence.

Phase 1.3.4: Implement graph-aware checkpointing.

This module provides a custom checkpointer that persists graph state
to `.executor/checkpoints/` directory as JSON files.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from ai_infra.logging import get_logger

logger = get_logger("executor.checkpointer")


class ExecutorCheckpointer(BaseCheckpointSaver):
    """JSON-based checkpointer for executor graph state.

    Persists checkpoints to `.executor/checkpoints/` directory with:
    - Thread-based organization: `{thread_id}/`
    - Sequential checkpoint files: `checkpoint_{n}.json`
    - Metadata: `checkpoint_{n}.meta.json`

    Example:
        ```python
        checkpointer = ExecutorCheckpointer(Path("/project"))

        # Use with Graph
        graph = Graph(
            nodes={...},
            edges=[...],
            checkpointer=checkpointer,
        )

        # Resume from checkpoint
        config = {"configurable": {"thread_id": "executor-abc123"}}
        latest = checkpointer.get(config)
        ```
    """

    def __init__(
        self,
        project_root: Path | str,
        max_checkpoints: int = 10,
    ):
        """Initialize the checkpointer.

        Args:
            project_root: Root directory of the project.
            max_checkpoints: Maximum checkpoints to retain per thread.
        """
        super().__init__()
        self.project_root = Path(project_root)
        self.max_checkpoints = max_checkpoints
        self._checkpoint_dir = self.project_root / ".executor" / "checkpoints"

    def _get_thread_dir(self, thread_id: str) -> Path:
        """Get directory for a specific thread."""
        return self._checkpoint_dir / thread_id

    def _list_checkpoint_files(self, thread_id: str) -> list[Path]:
        """List all checkpoint files for a thread, sorted by number."""
        thread_dir = self._get_thread_dir(thread_id)
        if not thread_dir.exists():
            return []

        files = sorted(
            [f for f in thread_dir.glob("checkpoint_*.json") if not f.stem.endswith(".meta")],
            key=lambda f: int(f.stem.split("_")[1]) if f.stem.split("_")[1].isdigit() else 0,
        )
        return files

    def _get_next_checkpoint_number(self, thread_id: str) -> int:
        """Get the next checkpoint number for a thread."""
        files = self._list_checkpoint_files(thread_id)
        if not files:
            return 1

        last_file = files[-1]
        try:
            last_num = int(last_file.stem.split("_")[1])
            return last_num + 1
        except (IndexError, ValueError):
            return len(files) + 1

    def _cleanup_old_checkpoints(self, thread_id: str) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        files = self._list_checkpoint_files(thread_id)
        if len(files) <= self.max_checkpoints:
            return

        # Remove oldest checkpoints
        to_remove = files[: len(files) - self.max_checkpoints]
        for f in to_remove:
            f.unlink(missing_ok=True)
            meta_file = f.with_suffix(".meta.json")
            meta_file.unlink(missing_ok=True)
            logger.debug(f"Removed old checkpoint: {f.name}")

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save a checkpoint.

        Args:
            config: Configuration with thread_id.
            checkpoint: The checkpoint data.
            metadata: Checkpoint metadata.
            new_versions: Channel versions (optional).

        Returns:
            Updated config with checkpoint_id.
        """
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        thread_dir = self._get_thread_dir(thread_id)
        thread_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_num = self._get_next_checkpoint_number(thread_id)
        checkpoint_file = thread_dir / f"checkpoint_{checkpoint_num}.json"
        meta_file = thread_dir / f"checkpoint_{checkpoint_num}.meta.json"

        # Serialize checkpoint
        checkpoint_data = {
            "v": checkpoint.get("v", 1),
            "id": checkpoint.get("id"),
            "ts": checkpoint.get("ts") or datetime.now(UTC).isoformat(),
            "channel_values": self._serialize_values(checkpoint.get("channel_values", {})),
            "channel_versions": checkpoint.get("channel_versions", {}),
            "versions_seen": checkpoint.get("versions_seen", {}),
            "pending_sends": checkpoint.get("pending_sends", []),
        }

        # Write checkpoint
        checkpoint_file.write_text(
            json.dumps(checkpoint_data, indent=2, default=str),
            encoding="utf-8",
        )

        # Write metadata
        meta_data = {
            **metadata,
            "saved_at": datetime.now(UTC).isoformat(),
            "checkpoint_num": checkpoint_num,
        }
        meta_file.write_text(
            json.dumps(meta_data, indent=2, default=str),
            encoding="utf-8",
        )

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(thread_id)

        logger.debug(f"Saved checkpoint {checkpoint_num} for thread {thread_id}")

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint.get("id"),
                "checkpoint_ns": config.get("configurable", {}).get("checkpoint_ns", ""),
            }
        }

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: tuple[str | int, ...] | str = (),
    ) -> None:
        """Save intermediate writes (not implemented for JSON checkpointer).

        For simplicity, we only save full checkpoints, not intermediate writes.
        """
        pass  # Not implemented for file-based checkpointer

    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        """Get the latest checkpoint tuple for a thread.

        Args:
            config: Configuration with thread_id.

        Returns:
            CheckpointTuple or None if no checkpoint exists.
        """
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        files = self._list_checkpoint_files(thread_id)

        if not files:
            return None

        # Get latest checkpoint
        checkpoint_file = files[-1]
        meta_file = checkpoint_file.with_name(checkpoint_file.stem + ".meta.json")

        try:
            checkpoint_data = json.loads(checkpoint_file.read_text(encoding="utf-8"))

            # Build checkpoint dict
            checkpoint: Checkpoint = {
                "v": checkpoint_data.get("v", 1),
                "id": checkpoint_data.get("id"),
                "ts": checkpoint_data.get("ts"),
                "channel_values": self._deserialize_values(
                    checkpoint_data.get("channel_values", {})
                ),
                "channel_versions": checkpoint_data.get("channel_versions", {}),
                "versions_seen": checkpoint_data.get("versions_seen", {}),
                "pending_sends": checkpoint_data.get("pending_sends", []),
            }

            # Load metadata
            metadata: CheckpointMetadata = {}
            if meta_file.exists():
                metadata = json.loads(meta_file.read_text(encoding="utf-8"))

            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=None,
                pending_writes=None,
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def list(
        self,
        config: dict[str, Any] | None = None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List all checkpoints for a thread.

        Args:
            config: Configuration with thread_id.
            filter: Optional filter (not implemented).
            before: Checkpoint to list before (not implemented).
            limit: Maximum number to return.

        Yields:
            CheckpointTuple for each checkpoint.
        """
        if config is None:
            return

        thread_id = config.get("configurable", {}).get("thread_id", "default")
        files = self._list_checkpoint_files(thread_id)

        # Reverse for newest first
        files = list(reversed(files))

        if limit:
            files = files[:limit]

        for checkpoint_file in files:
            try:
                checkpoint_data = json.loads(checkpoint_file.read_text(encoding="utf-8"))

                checkpoint: Checkpoint = {
                    "v": checkpoint_data.get("v", 1),
                    "id": checkpoint_data.get("id"),
                    "ts": checkpoint_data.get("ts"),
                    "channel_values": self._deserialize_values(
                        checkpoint_data.get("channel_values", {})
                    ),
                    "channel_versions": checkpoint_data.get("channel_versions", {}),
                    "versions_seen": checkpoint_data.get("versions_seen", {}),
                    "pending_sends": checkpoint_data.get("pending_sends", []),
                }

                meta_file = checkpoint_file.with_name(checkpoint_file.stem + ".meta.json")
                metadata: CheckpointMetadata = {}
                if meta_file.exists():
                    metadata = json.loads(meta_file.read_text(encoding="utf-8"))

                yield CheckpointTuple(
                    config=config,
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=None,
                    pending_writes=None,
                )

            except (json.JSONDecodeError, KeyError):
                continue

    async def aget_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        """Async version of get_tuple."""
        return self.get_tuple(config)

    async def alist(
        self,
        config: dict[str, Any] | None = None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ):
        """Async version of list."""
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async version of put."""
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: tuple[str | int, ...] | str = (),
    ) -> None:
        """Async version of put_writes."""
        self.put_writes(config, writes, task_id, task_path)

    def _serialize_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Serialize channel values for JSON storage."""
        serialized = {}
        for key, value in values.items():
            if hasattr(value, "to_dict"):
                serialized[key] = value.to_dict()
            elif hasattr(value, "__dict__"):
                serialized[key] = value.__dict__
            else:
                serialized[key] = value
        return serialized

    def _deserialize_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Deserialize channel values from JSON storage."""
        # For now, return as-is. Type reconstruction happens in graph state.
        return values

    def clear_thread(self, thread_id: str) -> None:
        """Clear all checkpoints for a thread.

        Args:
            thread_id: The thread to clear.
        """
        import shutil

        thread_dir = self._get_thread_dir(thread_id)
        if thread_dir.exists():
            shutil.rmtree(thread_dir)
            logger.info(f"Cleared checkpoints for thread {thread_id}")


__all__ = ["ExecutorCheckpointer"]
