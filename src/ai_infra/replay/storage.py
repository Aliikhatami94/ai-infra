"""
Storage backends for workflow recordings.

Provides different storage options:
- MemoryStorage: In-memory (default for development)
- SQLiteStorage: Local file persistence
- PostgresStorage: Production persistence
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from ai_infra.replay.recorder import WorkflowStep


class Storage(ABC):
    """Abstract base class for workflow storage backends."""

    @abstractmethod
    def save(self, record_id: str, steps: list[WorkflowStep]) -> None:
        """
        Save workflow steps to storage.

        Args:
            record_id: Unique identifier for the workflow
            steps: List of workflow steps to save
        """
        pass

    @abstractmethod
    def load(self, record_id: str) -> list[WorkflowStep] | None:
        """
        Load workflow steps from storage.

        Args:
            record_id: Unique identifier for the workflow

        Returns:
            List of workflow steps, or None if not found
        """
        pass

    @abstractmethod
    def exists(self, record_id: str) -> bool:
        """
        Check if a workflow recording exists.

        Args:
            record_id: Unique identifier for the workflow

        Returns:
            True if recording exists
        """
        pass

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """
        Delete a workflow recording.

        Args:
            record_id: Unique identifier for the workflow

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def list_recordings(self) -> list[str]:
        """
        List all recording IDs.

        Returns:
            List of record IDs
        """
        pass


class MemoryStorage(Storage):
    """
    In-memory storage backend.

    Suitable for development and testing. Recordings are lost
    when the process exits.

    Example:
        ```python
        storage = MemoryStorage()
        recorder = WorkflowRecorder("test", storage)
        ```
    """

    def __init__(self):
        self._recordings: dict[str, list[WorkflowStep]] = {}

    def save(self, record_id: str, steps: list[WorkflowStep]) -> None:
        """Save workflow to memory."""
        self._recordings[record_id] = steps.copy()

    def load(self, record_id: str) -> list[WorkflowStep] | None:
        """Load workflow from memory."""
        steps = self._recordings.get(record_id)
        return steps.copy() if steps else None

    def exists(self, record_id: str) -> bool:
        """Check if recording exists in memory."""
        return record_id in self._recordings

    def delete(self, record_id: str) -> bool:
        """Delete recording from memory."""
        if record_id in self._recordings:
            del self._recordings[record_id]
            return True
        return False

    def list_recordings(self) -> list[str]:
        """List all recording IDs."""
        return list(self._recordings.keys())


class SQLiteStorage(Storage):
    """
    SQLite-based storage backend.

    Persists recordings to a local SQLite database file.

    Example:
        ```python
        storage = SQLiteStorage("workflows.db")
        recorder = WorkflowRecorder("test", storage)
        ```
    """

    def __init__(self, db_path: str | Path = "workflows.db"):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        import sqlite3

        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self._db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workflows (
                record_id TEXT PRIMARY KEY,
                steps TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self._conn.commit()

    def _serialize_steps(self, steps: list[WorkflowStep]) -> str:
        """Serialize steps to JSON string."""
        return json.dumps([s.to_dict() for s in steps])

    def _deserialize_steps(self, data: str) -> list[WorkflowStep]:
        """Deserialize steps from JSON string."""
        return [WorkflowStep.from_dict(d) for d in json.loads(data)]

    def save(self, record_id: str, steps: list[WorkflowStep]) -> None:
        """Save workflow to SQLite."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO workflows (record_id, steps, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (record_id, self._serialize_steps(steps)),
        )
        self._conn.commit()

    def load(self, record_id: str) -> list[WorkflowStep] | None:
        """Load workflow from SQLite."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT steps FROM workflows WHERE record_id = ?", (record_id,))
        row = cursor.fetchone()
        return self._deserialize_steps(row[0]) if row else None

    def exists(self, record_id: str) -> bool:
        """Check if recording exists in SQLite."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM workflows WHERE record_id = ?", (record_id,))
        return cursor.fetchone() is not None

    def delete(self, record_id: str) -> bool:
        """Delete recording from SQLite."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM workflows WHERE record_id = ?", (record_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_recordings(self) -> list[str]:
        """List all recording IDs."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT record_id FROM workflows ORDER BY created_at DESC")
        return [row[0] for row in cursor.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()


# Global default storage (can be overridden)
_default_storage: Storage | None = None


def get_default_storage() -> Storage:
    """Get the default storage backend."""
    global _default_storage
    if _default_storage is None:
        _default_storage = MemoryStorage()
    return _default_storage


def set_default_storage(storage: Storage) -> None:
    """Set the default storage backend."""
    global _default_storage
    _default_storage = storage
