"""Enhanced rollback and recovery for the Executor module.

This module provides Phase 4.2 capabilities:
- Enhanced checkpoint metadata (file hashes, verification status, tags)
- Selective rollback (by file, by tag, preview mode)
- Recovery strategies (retry, partial rollback, branch-and-merge)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ai_infra.logging import get_logger

from .checkpoint import (
    Checkpointer,
    CheckpointResult,
    GitOperationError,
    RollbackResult,
)

logger = get_logger("executor.recovery")


# =============================================================================
# Recovery Strategy
# =============================================================================


class RecoveryStrategy(str, Enum):
    """Strategy for recovering from task failures."""

    ROLLBACK_ALL = "rollback_all"
    """Rollback all changes from the failed task."""

    ROLLBACK_FAILED = "rollback_failed"
    """Rollback only files that caused failures, keep successful changes."""

    RETRY_WITH_CONTEXT = "retry_with_context"
    """Retry the task with additional context about the failure."""

    BRANCH_AND_MERGE = "branch_and_merge"
    """Create a branch for risky changes, merge if successful."""

    SKIP = "skip"
    """Skip the failed task and continue with next."""

    MANUAL = "manual"
    """Pause and wait for manual intervention."""


# =============================================================================
# Checkpoint Metadata
# =============================================================================


@dataclass
class FileSnapshot:
    """Snapshot of a file's state at checkpoint time.

    Attributes:
        path: Relative file path.
        hash: SHA256 hash of file contents.
        size: File size in bytes.
        exists: Whether the file exists.
        is_new: Whether the file was created (didn't exist before).
    """

    path: str
    hash: str | None = None
    size: int = 0
    exists: bool = True
    is_new: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "hash": self.hash,
            "size": self.size,
            "exists": self.exists,
            "is_new": self.is_new,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileSnapshot:
        """Create from dictionary."""
        return cls(
            path=data["path"],
            hash=data.get("hash"),
            size=data.get("size", 0),
            exists=data.get("exists", True),
            is_new=data.get("is_new", False),
        )

    @classmethod
    def from_file(cls, workspace: Path, path: str) -> FileSnapshot:
        """Create snapshot from a file."""
        file_path = workspace / path
        if not file_path.exists():
            return cls(path=path, exists=False)

        content = file_path.read_bytes()
        return cls(
            path=path,
            hash=hashlib.sha256(content).hexdigest(),
            size=len(content),
            exists=True,
        )


@dataclass
class CheckpointMetadata:
    """Enhanced metadata for a checkpoint.

    Attributes:
        task_id: The task ID this checkpoint is for.
        task_title: Human-readable task title.
        commit_sha: The git commit SHA.
        created_at: When the checkpoint was created.
        files_modified: Files that were modified.
        files_created: Files that were created.
        files_deleted: Files that were deleted.
        file_snapshots: Snapshots of file states before changes.
        verification_passed: Whether verification passed for this task.
        verification_level: The verification level used.
        parent_checkpoint: SHA of the parent checkpoint (if any).
        tags: Named tags for this checkpoint.
        recovery_strategy: Strategy used if this was a recovery.
        metadata: Additional custom metadata.
    """

    task_id: str
    task_title: str
    commit_sha: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    file_snapshots: list[FileSnapshot] = field(default_factory=list)
    verification_passed: bool | None = None
    verification_level: str | None = None
    parent_checkpoint: str | None = None
    tags: list[str] = field(default_factory=list)
    recovery_strategy: RecoveryStrategy | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_files(self) -> list[str]:
        """All files affected by this checkpoint."""
        return list(set(self.files_modified + self.files_created + self.files_deleted))

    def has_tag(self, tag: str) -> bool:
        """Check if checkpoint has a specific tag."""
        return tag in self.tags

    def add_tag(self, tag: str) -> None:
        """Add a tag to the checkpoint."""
        if tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "task_id": self.task_id,
            "task_title": self.task_title,
            "commit_sha": self.commit_sha,
            "created_at": self.created_at.isoformat(),
            "files_modified": self.files_modified,
            "files_created": self.files_created,
            "files_deleted": self.files_deleted,
            "file_snapshots": [s.to_dict() for s in self.file_snapshots],
            "verification_passed": self.verification_passed,
            "verification_level": self.verification_level,
            "parent_checkpoint": self.parent_checkpoint,
            "tags": self.tags,
            "recovery_strategy": (self.recovery_strategy.value if self.recovery_strategy else None),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(UTC)

        recovery_strategy = data.get("recovery_strategy")
        if isinstance(recovery_strategy, str):
            recovery_strategy = RecoveryStrategy(recovery_strategy)

        return cls(
            task_id=data["task_id"],
            task_title=data["task_title"],
            commit_sha=data["commit_sha"],
            created_at=created_at,
            files_modified=data.get("files_modified", []),
            files_created=data.get("files_created", []),
            files_deleted=data.get("files_deleted", []),
            file_snapshots=[FileSnapshot.from_dict(s) for s in data.get("file_snapshots", [])],
            verification_passed=data.get("verification_passed"),
            verification_level=data.get("verification_level"),
            parent_checkpoint=data.get("parent_checkpoint"),
            tags=data.get("tags", []),
            recovery_strategy=recovery_strategy,
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Rollback Preview
# =============================================================================


@dataclass
class RollbackPreview:
    """Preview of what a rollback would change.

    Attributes:
        target_sha: The commit SHA that would be restored.
        commits_to_revert: Number of commits that would be reverted.
        files_to_restore: Files that would be restored to previous state.
        files_to_delete: Files that would be deleted (were created after target).
        files_to_recreate: Files that would be recreated (were deleted after target).
        tasks_affected: Task IDs that would be rolled back.
        warnings: Any warnings about the rollback.
    """

    target_sha: str
    commits_to_revert: int = 0
    files_to_restore: list[str] = field(default_factory=list)
    files_to_delete: list[str] = field(default_factory=list)
    files_to_recreate: list[str] = field(default_factory=list)
    tasks_affected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        """Whether the rollback appears safe (no warnings)."""
        return len(self.warnings) == 0

    @property
    def total_files_affected(self) -> int:
        """Total number of files affected."""
        return len(self.files_to_restore) + len(self.files_to_delete) + len(self.files_to_recreate)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_sha": self.target_sha,
            "commits_to_revert": self.commits_to_revert,
            "files_to_restore": self.files_to_restore,
            "files_to_delete": self.files_to_delete,
            "files_to_recreate": self.files_to_recreate,
            "tasks_affected": self.tasks_affected,
            "warnings": self.warnings,
            "is_safe": self.is_safe,
            "total_files_affected": self.total_files_affected,
        }


@dataclass
class SelectiveRollbackResult:
    """Result of a selective rollback operation.

    Attributes:
        success: Whether the rollback succeeded.
        files_restored: Files that were restored.
        files_deleted: Files that were deleted.
        files_failed: Files that failed to restore.
        message: Status message.
        error: Error message if failed.
    """

    success: bool
    files_restored: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    files_failed: list[str] = field(default_factory=list)
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "files_restored": self.files_restored,
            "files_deleted": self.files_deleted,
            "files_failed": self.files_failed,
            "message": self.message,
            "error": self.error,
        }


@dataclass
class RecoveryResult:
    """Result of a recovery operation.

    Attributes:
        success: Whether recovery succeeded.
        strategy: The strategy that was used.
        rollback_result: Result of rollback if performed.
        branch_name: Name of recovery branch if created.
        retry_count: Number of retries attempted.
        message: Status message.
        error: Error message if failed.
    """

    success: bool
    strategy: RecoveryStrategy
    rollback_result: RollbackResult | SelectiveRollbackResult | None = None
    branch_name: str | None = None
    retry_count: int = 0
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "strategy": self.strategy.value,
            "rollback_result": (self.rollback_result.to_dict() if self.rollback_result else None),
            "branch_name": self.branch_name,
            "retry_count": self.retry_count,
            "message": self.message,
            "error": self.error,
        }


# =============================================================================
# Recovery Manager
# =============================================================================

METADATA_DIR = ".executor/checkpoints"
TAGS_FILE = ".executor/checkpoint-tags.json"


class RecoveryManager:
    """Manager for enhanced rollback and recovery operations.

    Extends the basic Checkpointer with:
    - Enhanced metadata storage
    - Selective rollback by file or tag
    - Preview mode
    - Recovery strategies

    Example:
        >>> recovery = RecoveryManager(checkpointer)
        >>> # Preview a rollback
        >>> preview = recovery.preview_rollback(task_id="1.1.1")
        >>> print(f"Would revert {preview.commits_to_revert} commits")
        >>> # Rollback specific files
        >>> result = recovery.rollback_files(["src/main.py"], to_sha="abc123")
        >>> # Rollback to a named tag
        >>> result = recovery.rollback_to_tag("stable")
    """

    def __init__(
        self,
        checkpointer: Checkpointer,
        *,
        metadata_dir: str | Path | None = None,
    ):
        """Initialize the recovery manager.

        Args:
            checkpointer: The git checkpointer to use.
            metadata_dir: Directory to store metadata (default: .executor/checkpoints).
        """
        self._checkpointer = checkpointer
        self._workspace = checkpointer.workspace

        # Set up metadata directory
        if metadata_dir is None:
            self._metadata_dir = self._workspace / METADATA_DIR
        else:
            self._metadata_dir = Path(metadata_dir)

        self._tags_file = self._workspace / TAGS_FILE

        # Ensure directories exist
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._tags_file.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def checkpointer(self) -> Checkpointer:
        """The underlying git checkpointer."""
        return self._checkpointer

    @property
    def workspace(self) -> Path:
        """The workspace path."""
        return self._workspace

    # =========================================================================
    # Enhanced Checkpoint Operations
    # =========================================================================

    def create_checkpoint(
        self,
        task_id: str,
        task_title: str,
        *,
        files_modified: list[str] | None = None,
        files_created: list[str] | None = None,
        files_deleted: list[str] | None = None,
        verification_passed: bool | None = None,
        verification_level: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        capture_snapshots: bool = True,
    ) -> tuple[CheckpointResult, CheckpointMetadata | None]:
        """Create a checkpoint with enhanced metadata.

        Args:
            task_id: The task ID.
            task_title: Human-readable task title.
            files_modified: Files that were modified.
            files_created: Files that were created.
            files_deleted: Files that were deleted.
            verification_passed: Whether verification passed.
            verification_level: The verification level used.
            tags: Tags to apply to this checkpoint.
            metadata: Additional custom metadata.
            capture_snapshots: Whether to capture file snapshots.

        Returns:
            Tuple of (CheckpointResult, CheckpointMetadata or None).
        """
        files_modified = files_modified or []
        files_created = files_created or []
        files_deleted = files_deleted or []
        tags = tags or []
        metadata = metadata or {}

        # Capture file snapshots before committing
        snapshots: list[FileSnapshot] = []
        if capture_snapshots:
            all_files = files_modified + files_created
            for file_path in all_files:
                snapshot = FileSnapshot.from_file(self._workspace, file_path)
                if file_path in files_created:
                    snapshot.is_new = True
                snapshots.append(snapshot)

        # Get parent checkpoint
        parent_sha = self._checkpointer.current_sha

        # Create the git checkpoint
        result = self._checkpointer.checkpoint(
            task_id=task_id,
            task_title=task_title,
            files_modified=files_modified,
            files_created=files_created,
        )

        if not result.success or not result.commit_sha:
            return result, None

        # Create enhanced metadata
        checkpoint_meta = CheckpointMetadata(
            task_id=task_id,
            task_title=task_title,
            commit_sha=result.commit_sha,
            files_modified=files_modified,
            files_created=files_created,
            files_deleted=files_deleted,
            file_snapshots=snapshots,
            verification_passed=verification_passed,
            verification_level=verification_level,
            parent_checkpoint=parent_sha,
            tags=tags,
            metadata=metadata,
        )

        # Save metadata
        self._save_metadata(checkpoint_meta)

        # Save tags
        for tag in tags:
            self._save_tag(tag, result.commit_sha)

        logger.info(
            f"Created checkpoint {result.commit_sha[:7]} for task {task_id}"
            + (f" with tags: {tags}" if tags else "")
        )

        return result, checkpoint_meta

    def get_checkpoint_metadata(self, commit_sha: str) -> CheckpointMetadata | None:
        """Get metadata for a checkpoint.

        Args:
            commit_sha: The commit SHA.

        Returns:
            CheckpointMetadata if found, None otherwise.
        """
        return self._load_metadata(commit_sha)

    def add_tag(self, tag: str, commit_sha: str | None = None) -> bool:
        """Add a tag to a checkpoint.

        Args:
            tag: The tag name.
            commit_sha: The commit SHA (default: current HEAD).

        Returns:
            True if tag was added.
        """
        if commit_sha is None:
            commit_sha = self._checkpointer.current_sha
            if commit_sha is None:
                return False

        self._save_tag(tag, commit_sha)

        # Update metadata if exists
        meta = self._load_metadata(commit_sha)
        if meta:
            meta.add_tag(tag)
            self._save_metadata(meta)

        logger.info(f"Added tag '{tag}' to checkpoint {commit_sha[:7]}")
        return True

    def get_checkpoint_by_tag(self, tag: str) -> str | None:
        """Get the commit SHA for a tag.

        Args:
            tag: The tag name.

        Returns:
            Commit SHA if found, None otherwise.
        """
        tags = self._load_tags()
        return tags.get(tag)

    def list_tags(self) -> dict[str, str]:
        """List all checkpoint tags.

        Returns:
            Dictionary mapping tag names to commit SHAs.
        """
        return self._load_tags()

    # =========================================================================
    # Preview Operations
    # =========================================================================

    def preview_rollback(
        self,
        *,
        task_id: str | None = None,
        tag: str | None = None,
        commit_sha: str | None = None,
    ) -> RollbackPreview | None:
        """Preview what a rollback would change.

        Exactly one of task_id, tag, or commit_sha must be provided.

        Args:
            task_id: Rollback to before this task.
            tag: Rollback to this tagged checkpoint.
            commit_sha: Rollback to this specific commit.

        Returns:
            RollbackPreview or None if target not found.
        """
        # Resolve target SHA
        target_sha = self._resolve_rollback_target(task_id=task_id, tag=tag, commit_sha=commit_sha)
        if target_sha is None:
            return None

        # Get current SHA
        current_sha = self._checkpointer.current_sha
        if current_sha is None:
            return None

        # Count commits to revert
        commits_to_revert = self._checkpointer._count_commits_between(target_sha, current_sha)

        # Get tasks affected
        tasks_affected = self._checkpointer.get_tasks_since(target_sha)

        # Collect files affected from metadata
        files_to_restore: list[str] = []
        files_to_delete: list[str] = []
        files_to_recreate: list[str] = []
        warnings: list[str] = []

        for tid in tasks_affected:
            commit = self._checkpointer.get_commit_for_task(tid)
            if commit:
                meta = self._load_metadata(commit.sha)
                if meta:
                    files_to_restore.extend(meta.files_modified)
                    files_to_delete.extend(meta.files_created)
                    files_to_recreate.extend(meta.files_deleted)

        # Deduplicate
        files_to_restore = list(set(files_to_restore))
        files_to_delete = list(set(files_to_delete))
        files_to_recreate = list(set(files_to_recreate))

        # Check for potential issues
        if commits_to_revert > 10:
            warnings.append(f"Rolling back {commits_to_revert} commits - this is a large rollback")

        # Check if any files have uncommitted changes
        uncommitted = self._checkpointer.get_uncommitted_files()
        for f in uncommitted:
            if f in files_to_restore or f in files_to_delete:
                warnings.append(f"File '{f}' has uncommitted changes that will be lost")

        return RollbackPreview(
            target_sha=target_sha,
            commits_to_revert=commits_to_revert,
            files_to_restore=files_to_restore,
            files_to_delete=files_to_delete,
            files_to_recreate=files_to_recreate,
            tasks_affected=tasks_affected,
            warnings=warnings,
        )

    # =========================================================================
    # Selective Rollback Operations
    # =========================================================================

    def rollback_to_tag(
        self,
        tag: str,
        *,
        hard: bool = False,
        preview: bool = False,
    ) -> RollbackResult | RollbackPreview:
        """Rollback to a tagged checkpoint.

        Args:
            tag: The tag name.
            hard: If True, discard uncommitted changes.
            preview: If True, return preview instead of performing rollback.

        Returns:
            RollbackResult or RollbackPreview if preview=True.
        """
        commit_sha = self.get_checkpoint_by_tag(tag)
        if commit_sha is None:
            if preview:
                return RollbackPreview(target_sha="", warnings=[f"Tag '{tag}' not found"])
            return RollbackResult(
                success=False,
                error=f"Tag '{tag}' not found",
                message=f"Cannot rollback: tag '{tag}' not found",
            )

        if preview:
            result = self.preview_rollback(commit_sha=commit_sha)
            if result is None:
                return RollbackPreview(target_sha="", warnings=["Could not generate preview"])
            return result

        return self._checkpointer.rollback_to_sha(commit_sha, hard=hard)

    def rollback_files(
        self,
        files: list[str],
        *,
        to_sha: str | None = None,
        to_task_id: str | None = None,
        to_tag: str | None = None,
    ) -> SelectiveRollbackResult:
        """Rollback specific files to a previous state.

        Args:
            files: Files to rollback.
            to_sha: Commit SHA to restore from.
            to_task_id: Task ID to restore to (before that task).
            to_tag: Tag to restore from.

        Returns:
            SelectiveRollbackResult with details.
        """
        # Resolve target
        target_sha = self._resolve_rollback_target(
            task_id=to_task_id, tag=to_tag, commit_sha=to_sha
        )

        if target_sha is None:
            return SelectiveRollbackResult(
                success=False,
                error="Could not resolve rollback target",
                message="No valid target specified for selective rollback",
            )

        files_restored: list[str] = []
        files_deleted: list[str] = []
        files_failed: list[str] = []

        for file_path in files:
            try:
                # Try to restore file from target commit
                result = self._checkpointer._run_git(
                    ["show", f"{target_sha}:{file_path}"],
                    check=False,
                )

                if result is not None:
                    # File existed at target - restore it
                    full_path = self._workspace / file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(result)
                    files_restored.append(file_path)
                else:
                    # File didn't exist at target - delete it if it exists now
                    full_path = self._workspace / file_path
                    if full_path.exists():
                        full_path.unlink()
                        files_deleted.append(file_path)
                    else:
                        # File doesn't exist anywhere - nothing to do
                        files_restored.append(file_path)

            except (GitOperationError, OSError) as e:
                logger.warning(f"Failed to restore file {file_path}: {e}")
                files_failed.append(file_path)

        success = len(files_failed) == 0
        total_affected = len(files_restored) + len(files_deleted)

        return SelectiveRollbackResult(
            success=success,
            files_restored=files_restored,
            files_deleted=files_deleted,
            files_failed=files_failed,
            message=(
                f"Restored {total_affected} file(s) to state at {target_sha[:7]}"
                if success
                else f"Partially restored files, {len(files_failed)} failed"
            ),
            error=None if success else f"Failed to restore: {files_failed}",
        )

    # =========================================================================
    # Recovery Strategies
    # =========================================================================

    def recover(
        self,
        task_id: str,
        strategy: RecoveryStrategy,
        *,
        failed_files: list[str] | None = None,
        error_context: str | None = None,
    ) -> RecoveryResult:
        """Apply a recovery strategy for a failed task.

        Args:
            task_id: The failed task ID.
            strategy: The recovery strategy to apply.
            failed_files: Files that caused the failure (for ROLLBACK_FAILED).
            error_context: Error information for RETRY_WITH_CONTEXT.

        Returns:
            RecoveryResult with details.
        """
        logger.info(f"Applying recovery strategy {strategy.value} for task {task_id}")

        if strategy == RecoveryStrategy.ROLLBACK_ALL:
            return self._recover_rollback_all(task_id)

        elif strategy == RecoveryStrategy.ROLLBACK_FAILED:
            return self._recover_rollback_failed(task_id, failed_files or [])

        elif strategy == RecoveryStrategy.RETRY_WITH_CONTEXT:
            return self._recover_retry_with_context(task_id, error_context)

        elif strategy == RecoveryStrategy.BRANCH_AND_MERGE:
            return self._recover_branch_and_merge(task_id)

        elif strategy == RecoveryStrategy.SKIP:
            return RecoveryResult(
                success=True,
                strategy=strategy,
                message=f"Skipped recovery for task {task_id}",
            )

        elif strategy == RecoveryStrategy.MANUAL:
            return RecoveryResult(
                success=True,
                strategy=strategy,
                message=f"Manual intervention required for task {task_id}",
            )

        return RecoveryResult(
            success=False,
            strategy=strategy,
            error=f"Unknown recovery strategy: {strategy}",
            message="Recovery failed: unknown strategy",
        )

    def _recover_rollback_all(self, task_id: str) -> RecoveryResult:
        """Rollback all changes from a failed task."""
        result = self._checkpointer.rollback(task_id, hard=True)
        return RecoveryResult(
            success=result.success,
            strategy=RecoveryStrategy.ROLLBACK_ALL,
            rollback_result=result,
            message=result.message,
            error=result.error,
        )

    def _recover_rollback_failed(self, task_id: str, failed_files: list[str]) -> RecoveryResult:
        """Rollback only the files that caused failures."""
        if not failed_files:
            return RecoveryResult(
                success=False,
                strategy=RecoveryStrategy.ROLLBACK_FAILED,
                error="No failed files specified",
                message="Cannot perform partial rollback without specifying failed files",
            )

        # Find the commit before the task
        commit = self._checkpointer.get_commit_for_task(task_id)
        if commit is None:
            return RecoveryResult(
                success=False,
                strategy=RecoveryStrategy.ROLLBACK_FAILED,
                error=f"No commit found for task {task_id}",
                message="Cannot find commit to rollback from",
            )

        parent_sha = self._checkpointer._get_parent_sha(commit.sha)
        if parent_sha is None:
            return RecoveryResult(
                success=False,
                strategy=RecoveryStrategy.ROLLBACK_FAILED,
                error="Cannot determine parent commit",
                message="Failed to find parent commit for partial rollback",
            )

        # Rollback only the failed files
        result = self.rollback_files(failed_files, to_sha=parent_sha)

        return RecoveryResult(
            success=result.success,
            strategy=RecoveryStrategy.ROLLBACK_FAILED,
            rollback_result=result,
            message=result.message,
            error=result.error,
        )

    def _recover_retry_with_context(
        self, task_id: str, error_context: str | None
    ) -> RecoveryResult:
        """Prepare for retry with additional context.

        Note: This doesn't actually retry - it just prepares the context.
        The actual retry is handled by the Executor loop.
        """
        # Store error context in metadata for the next attempt
        meta_path = self._metadata_dir / f"retry-context-{task_id}.json"
        context_data = {
            "task_id": task_id,
            "error_context": error_context,
            "retry_requested_at": datetime.now(UTC).isoformat(),
        }
        meta_path.write_text(json.dumps(context_data, indent=2))

        return RecoveryResult(
            success=True,
            strategy=RecoveryStrategy.RETRY_WITH_CONTEXT,
            retry_count=1,
            message=f"Prepared retry context for task {task_id}",
        )

    def _recover_branch_and_merge(self, task_id: str) -> RecoveryResult:
        """Create a recovery branch for risky changes."""
        try:
            # Get current branch
            current_branch = self._checkpointer.current_branch

            # Create recovery branch
            branch_name = f"executor-recovery-{task_id}"
            self._checkpointer._run_git(["checkout", "-b", branch_name])

            logger.info(f"Created recovery branch: {branch_name}")

            return RecoveryResult(
                success=True,
                strategy=RecoveryStrategy.BRANCH_AND_MERGE,
                branch_name=branch_name,
                message=f"Created recovery branch '{branch_name}' from '{current_branch}'",
            )

        except GitOperationError as e:
            return RecoveryResult(
                success=False,
                strategy=RecoveryStrategy.BRANCH_AND_MERGE,
                error=str(e),
                message=f"Failed to create recovery branch: {e}",
            )

    def get_retry_context(self, task_id: str) -> dict[str, Any] | None:
        """Get stored retry context for a task.

        Args:
            task_id: The task ID.

        Returns:
            Context dictionary if exists, None otherwise.
        """
        meta_path = self._metadata_dir / f"retry-context-{task_id}.json"
        if not meta_path.exists():
            return None

        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def clear_retry_context(self, task_id: str) -> None:
        """Clear stored retry context for a task."""
        meta_path = self._metadata_dir / f"retry-context-{task_id}.json"
        if meta_path.exists():
            meta_path.unlink()

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _resolve_rollback_target(
        self,
        *,
        task_id: str | None = None,
        tag: str | None = None,
        commit_sha: str | None = None,
    ) -> str | None:
        """Resolve a rollback target to a commit SHA."""
        if commit_sha:
            return self._checkpointer._resolve_ref(commit_sha)

        if tag:
            return self.get_checkpoint_by_tag(tag)

        if task_id:
            commit = self._checkpointer.get_commit_for_task(task_id)
            if commit:
                # Return parent (state before the task)
                return self._checkpointer._get_parent_sha(commit.sha)

        return None

    def _save_metadata(self, meta: CheckpointMetadata) -> None:
        """Save checkpoint metadata to disk."""
        # Use first 7 chars of SHA for filename
        filename = f"{meta.commit_sha[:7]}.json"
        meta_path = self._metadata_dir / filename
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2))

    def _load_metadata(self, commit_sha: str) -> CheckpointMetadata | None:
        """Load checkpoint metadata from disk."""
        # Try short SHA first, then full
        for sha_prefix in [commit_sha[:7], commit_sha]:
            meta_path = self._metadata_dir / f"{sha_prefix}.json"
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text())
                    return CheckpointMetadata.from_dict(data)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to load metadata for {commit_sha}: {e}")
                    return None
        return None

    def _save_tag(self, tag: str, commit_sha: str) -> None:
        """Save a tag mapping."""
        tags = self._load_tags()
        tags[tag] = commit_sha
        self._tags_file.write_text(json.dumps(tags, indent=2))

    def _load_tags(self) -> dict[str, str]:
        """Load tag mappings from disk."""
        if not self._tags_file.exists():
            return {}
        try:
            return json.loads(self._tags_file.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def for_workspace(cls, workspace: str | Path) -> RecoveryManager | None:
        """Create a recovery manager for a workspace.

        Args:
            workspace: Path to the workspace.

        Returns:
            RecoveryManager if workspace is a git repo, None otherwise.
        """
        checkpointer = Checkpointer.for_workspace(workspace)
        if checkpointer is None:
            return None
        return cls(checkpointer)
