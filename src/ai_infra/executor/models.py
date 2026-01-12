"""Data models for the Executor module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    """Status of a task in the execution queue."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class Task:
    """A task parsed from a ROADMAP.md file.

    Attributes:
        id: Unique identifier (e.g., "1.1.1")
        title: Short task title
        description: Full task description
        status: Current execution status
        file_hints: List of file paths mentioned in the task
        dependencies: List of task IDs this task depends on
        phase: The phase this task belongs to (e.g., "1")
        section: The section within the phase (e.g., "1.1")
        metadata: Additional metadata from the ROADMAP
    """

    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    file_hints: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    phase: str = ""
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "file_hints": self.file_hints,
            "dependencies": self.dependencies,
            "phase": self.phase,
            "section": self.section,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create a Task from a dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=TaskStatus(data.get("status", "pending")),
            file_hints=data.get("file_hints", []),
            dependencies=data.get("dependencies", []),
            phase=data.get("phase", ""),
            section=data.get("section", ""),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.utcnow(),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            error=data.get("error"),
        )


# =============================================================================
# File Write Verification Models (Phase 5.11.4)
# =============================================================================


@dataclass
class FileWriteRecord:
    """Record of a file write for verification.

    Phase 5.11.4: Tracks file creation with checksum and size for validation.

    Attributes:
        path: Relative path to the file from project root.
        absolute_path: Absolute path to the file.
        task_id: ID of the task that created/modified the file.
        size_bytes: Size of the file in bytes.
        checksum: MD5 checksum of the file content.
        created_at: When the file was written.
        verified: Whether the file was verified to exist.
    """

    path: str
    absolute_path: str
    task_id: str
    size_bytes: int
    checksum: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "absolute_path": self.absolute_path,
            "task_id": self.task_id,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileWriteRecord:
        """Create a FileWriteRecord from a dictionary."""
        return cls(
            path=data["path"],
            absolute_path=data["absolute_path"],
            task_id=data["task_id"],
            size_bytes=data["size_bytes"],
            checksum=data["checksum"],
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.utcnow(),
            verified=data.get("verified", False),
        )


@dataclass
class FileWriteSummary:
    """Summary of file writes for a run.

    Phase 5.11.4: Aggregated stats for file write verification.

    Attributes:
        total_expected: Number of files expected to be created.
        total_created: Number of files actually created.
        total_verified: Number of files verified with checksum.
        missing_files: List of files that were not created.
        verified_files: List of successfully verified file records.
        failed_files: List of files that failed verification.
    """

    total_expected: int = 0
    total_created: int = 0
    total_verified: int = 0
    missing_files: list[str] = field(default_factory=list)
    verified_files: list[FileWriteRecord] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)

    def summary_text(self) -> str:
        """Generate human-readable summary text."""
        lines = [
            "",
            "=" * 60,
            "FILE WRITE VERIFICATION SUMMARY",
            "=" * 60,
            "",
            f"Expected files:  {self.total_expected}",
            f"Created files:   {self.total_created}",
            f"Verified files:  {self.total_verified}",
            "",
        ]

        if self.missing_files:
            lines.append("MISSING FILES:")
            for f in self.missing_files:
                lines.append(f"  - {f}")
            lines.append("")

        if self.failed_files:
            lines.append("FAILED VERIFICATION:")
            for f in self.failed_files:
                lines.append(f"  - {f}")
            lines.append("")

        if self.total_expected == self.total_verified and self.total_expected > 0:
            lines.append("✓ All files verified successfully!")
        elif self.total_expected == 0:
            lines.append("(No files expected from tasks)")
        else:
            missing = self.total_expected - self.total_verified
            lines.append(f"⚠ {missing} file(s) could not be verified")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_expected": self.total_expected,
            "total_created": self.total_created,
            "total_verified": self.total_verified,
            "missing_files": self.missing_files,
            "verified_files": [f.to_dict() for f in self.verified_files],
            "failed_files": self.failed_files,
        }
