"""Git checkpointing for task execution.

This module provides:
- Automatic git commits after task completion
- Rollback to previous states
- Tracking of which commits belong to which tasks
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_infra.logging import get_logger

logger = get_logger("executor.checkpoint")


# =============================================================================
# Exceptions
# =============================================================================


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""

    pass


class GitNotFoundError(CheckpointError):
    """Git is not installed or not in PATH."""

    pass


class NotAGitRepoError(CheckpointError):
    """The workspace is not a git repository."""

    pass


class GitOperationError(CheckpointError):
    """A git operation failed."""

    def __init__(self, message: str, command: str, returncode: int, stderr: str):
        super().__init__(message)
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class CommitInfo:
    """Information about a git commit.

    Attributes:
        sha: Full commit SHA.
        short_sha: Short (7-char) commit SHA.
        message: Commit message.
        task_id: Task ID if this is an executor commit.
        timestamp: Commit timestamp.
        author: Commit author.
        files_changed: Number of files changed.
        insertions: Lines inserted.
        deletions: Lines deleted.
    """

    sha: str
    short_sha: str
    message: str
    task_id: str | None = None
    timestamp: datetime | None = None
    author: str = ""
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def is_executor_commit(self) -> bool:
        """Whether this commit was created by the executor."""
        return self.task_id is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sha": self.sha,
            "short_sha": self.short_sha,
            "message": self.message,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "author": self.author,
            "files_changed": self.files_changed,
            "insertions": self.insertions,
            "deletions": self.deletions,
        }


@dataclass
class CheckpointResult:
    """Result of a checkpoint operation.

    Attributes:
        success: Whether the checkpoint succeeded.
        commit_sha: The commit SHA if created.
        message: Status message.
        files_staged: Files that were staged.
        error: Error message if failed.
    """

    success: bool
    commit_sha: str | None = None
    message: str = ""
    files_staged: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "commit_sha": self.commit_sha,
            "message": self.message,
            "files_staged": self.files_staged,
            "error": self.error,
        }


@dataclass
class RollbackResult:
    """Result of a rollback operation.

    Attributes:
        success: Whether the rollback succeeded.
        target_sha: The commit we rolled back to.
        commits_reverted: Number of commits reverted.
        message: Status message.
        error: Error message if failed.
    """

    success: bool
    target_sha: str | None = None
    commits_reverted: int = 0
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "target_sha": self.target_sha,
            "commits_reverted": self.commits_reverted,
            "message": self.message,
            "error": self.error,
        }


# =============================================================================
# Checkpointer
# =============================================================================

# Pattern to extract task ID from executor commit messages
# Format: executor(<task_id>): <message>
EXECUTOR_COMMIT_PATTERN = re.compile(r"^executor\(([^)]+)\):\s*(.*)$")


class Checkpointer:
    """Git checkpointing for task execution.

    Creates git commits after task completion and supports rollback
    to previous states.

    Example:
        >>> checkpointer = Checkpointer(workspace="/path/to/repo")
        >>> result = checkpointer.checkpoint(
        ...     task_id="1.1.1",
        ...     task_title="Implement feature",
        ...     files_modified=["src/main.py"],
        ... )
        >>> if result.success:
        ...     print(f"Created commit: {result.commit_sha}")
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        author_name: str = "Executor",
        author_email: str = "executor@nfrax.io",
        auto_stage: bool = True,
        sign_commits: bool = False,
    ):
        """Initialize the checkpointer.

        Args:
            workspace: Path to the git repository.
            author_name: Name for commit author.
            author_email: Email for commit author.
            auto_stage: Automatically stage modified files.
            sign_commits: Sign commits with GPG.
        """
        self._workspace = Path(workspace).resolve()
        self._author_name = author_name
        self._author_email = author_email
        self._auto_stage = auto_stage
        self._sign_commits = sign_commits
        self._git_dir: Path | None = None

        # Validate git is available and this is a repo
        self._validate()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def workspace(self) -> Path:
        """The workspace path."""
        return self._workspace

    @property
    def git_dir(self) -> Path:
        """The .git directory path."""
        if self._git_dir is None:
            result = self._run_git(["rev-parse", "--git-dir"])
            # result is not None when check=True (default)
            assert result is not None
            self._git_dir = Path(result.strip()).resolve()
        return self._git_dir

    @property
    def is_repo(self) -> bool:
        """Whether the workspace is a git repository."""
        try:
            self._run_git(["rev-parse", "--git-dir"])
            return True
        except (GitOperationError, NotAGitRepoError):
            return False

    @property
    def current_branch(self) -> str:
        """The current branch name."""
        try:
            result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
            assert result is not None
            return result.strip()
        except GitOperationError:
            return "HEAD"

    @property
    def current_sha(self) -> str | None:
        """The current HEAD commit SHA."""
        try:
            result = self._run_git(["rev-parse", "HEAD"])
            assert result is not None
            return result.strip()
        except GitOperationError:
            return None

    # =========================================================================
    # Checkpoint Operations
    # =========================================================================

    def checkpoint(
        self,
        task_id: str,
        task_title: str,
        files_modified: list[str] | None = None,
        files_created: list[str] | None = None,
        *,
        message: str | None = None,
        stage_all: bool = False,
    ) -> CheckpointResult:
        """Create a git commit for a completed task.

        Args:
            task_id: The task ID (e.g., "1.1.1").
            task_title: The task title (truncated in commit message).
            files_modified: Files that were modified.
            files_created: Files that were created.
            message: Custom commit message (uses default format if None).
            stage_all: Stage all changes instead of specific files.

        Returns:
            CheckpointResult with commit info.
        """
        files_modified = files_modified or []
        files_created = files_created or []
        all_files = files_modified + files_created

        try:
            # Check if there are any changes
            if not self.has_changes():
                return CheckpointResult(
                    success=True,
                    message="No changes to commit",
                )

            # Stage files
            if stage_all:
                staged = self._stage_all()
            elif all_files and self._auto_stage:
                staged = self._stage_files(all_files)
            else:
                staged = self._get_staged_files()

            if not staged:
                return CheckpointResult(
                    success=True,
                    message="No files staged for commit",
                )

            # Build commit message
            if message is None:
                # Truncate title to 50 chars for conventional commit format
                title_truncated = task_title[:50]
                if len(task_title) > 50:
                    title_truncated = title_truncated[:47] + "..."
                message = f"executor({task_id}): {title_truncated}"

            # Create commit
            commit_sha = self._commit(message)

            logger.info(f"Created checkpoint: {commit_sha[:7]} for task {task_id}")

            return CheckpointResult(
                success=True,
                commit_sha=commit_sha,
                message=f"Created commit {commit_sha[:7]}",
                files_staged=staged,
            )

        except GitOperationError as e:
            logger.error(f"Checkpoint failed: {e}")
            return CheckpointResult(
                success=False,
                error=str(e),
                message=f"Failed to create checkpoint: {e}",
            )

    def rollback(
        self,
        task_id: str,
        *,
        hard: bool = False,
    ) -> RollbackResult:
        """Rollback to the state before a task was executed.

        Finds the commit before the task's executor commit and
        resets to that state.

        Args:
            task_id: The task ID to rollback.
            hard: If True, discard all changes. If False, keep changes staged.

        Returns:
            RollbackResult with rollback info.
        """
        try:
            # Find the executor commit for this task
            task_commit = self._find_task_commit(task_id)
            if task_commit is None:
                return RollbackResult(
                    success=False,
                    error=f"No commit found for task {task_id}",
                    message=f"Cannot rollback: no commit found for task {task_id}",
                )

            # Find the parent commit
            parent_sha = self._get_parent_sha(task_commit.sha)
            if parent_sha is None:
                return RollbackResult(
                    success=False,
                    error="Cannot rollback past initial commit",
                    message="The task commit is the initial commit",
                )

            # Count commits being reverted
            commits_between = self._count_commits_between(parent_sha, "HEAD")

            # Perform reset
            mode = "--hard" if hard else "--soft"
            self._run_git(["reset", mode, parent_sha])

            logger.info(f"Rolled back to {parent_sha[:7]} (before task {task_id})")

            return RollbackResult(
                success=True,
                target_sha=parent_sha,
                commits_reverted=commits_between,
                message=f"Rolled back {commits_between} commit(s) to {parent_sha[:7]}",
            )

        except GitOperationError as e:
            logger.error(f"Rollback failed: {e}")
            return RollbackResult(
                success=False,
                error=str(e),
                message=f"Failed to rollback: {e}",
            )

    def rollback_to_sha(
        self,
        sha: str,
        *,
        hard: bool = False,
    ) -> RollbackResult:
        """Rollback to a specific commit SHA.

        Args:
            sha: The commit SHA to rollback to.
            hard: If True, discard all changes. If False, keep changes staged.

        Returns:
            RollbackResult with rollback info.
        """
        try:
            # Verify the commit exists
            full_sha = self._resolve_ref(sha)
            if full_sha is None:
                return RollbackResult(
                    success=False,
                    error=f"Commit {sha} not found",
                    message=f"Cannot rollback: commit {sha} not found",
                )

            # Count commits being reverted
            commits_between = self._count_commits_between(full_sha, "HEAD")

            # Perform reset
            mode = "--hard" if hard else "--soft"
            self._run_git(["reset", mode, full_sha])

            logger.info(f"Rolled back to {full_sha[:7]}")

            return RollbackResult(
                success=True,
                target_sha=full_sha,
                commits_reverted=commits_between,
                message=f"Rolled back {commits_between} commit(s) to {full_sha[:7]}",
            )

        except GitOperationError as e:
            logger.error(f"Rollback failed: {e}")
            return RollbackResult(
                success=False,
                error=str(e),
                message=f"Failed to rollback: {e}",
            )

    # =========================================================================
    # Query Operations
    # =========================================================================

    def has_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        # Check for staged changes
        result = self._run_git(["diff", "--cached", "--quiet"], check=False)
        if result is None:  # Non-zero exit = has changes
            return True

        # Check for unstaged changes
        result = self._run_git(["diff", "--quiet"], check=False)
        if result is None:
            return True

        # Check for untracked files
        result = self._run_git(["ls-files", "--others", "--exclude-standard"])
        return bool(result and result.strip())

    def has_staged_changes(self) -> bool:
        """Check if there are staged changes."""
        result = self._run_git(["diff", "--cached", "--quiet"], check=False)
        return result is None

    def get_uncommitted_files(self) -> list[str]:
        """Get list of uncommitted files."""
        files: list[str] = []

        # Staged files
        result = self._run_git(["diff", "--cached", "--name-only"])
        if result:
            files.extend(result.strip().split("\n") if result.strip() else [])

        # Modified but not staged
        result = self._run_git(["diff", "--name-only"])
        if result:
            files.extend(result.strip().split("\n") if result.strip() else [])

        # Untracked files
        result = self._run_git(["ls-files", "--others", "--exclude-standard"])
        if result:
            files.extend(result.strip().split("\n") if result.strip() else [])

        return list(set(files))

    def get_executor_commits(
        self,
        limit: int = 50,
    ) -> list[CommitInfo]:
        """Get recent executor commits.

        Args:
            limit: Maximum number of commits to return.

        Returns:
            List of CommitInfo for executor commits.
        """
        commits = self._get_recent_commits(limit=limit * 2)  # Fetch more to filter
        executor_commits = [c for c in commits if c.is_executor_commit]
        return executor_commits[:limit]

    def get_commit_for_task(self, task_id: str) -> CommitInfo | None:
        """Get the commit for a specific task.

        Args:
            task_id: The task ID to find.

        Returns:
            CommitInfo if found, None otherwise.
        """
        return self._find_task_commit(task_id)

    def get_tasks_since(self, sha: str) -> list[str]:
        """Get task IDs for commits since a given SHA.

        Args:
            sha: The commit SHA to start from.

        Returns:
            List of task IDs in order (oldest first).
        """
        try:
            # Get commits between sha and HEAD
            result = self._run_git(
                [
                    "log",
                    "--oneline",
                    "--reverse",
                    f"{sha}..HEAD",
                ]
            )

            if not result:
                return []

            task_ids = []
            for line in result.strip().split("\n"):
                if not line.strip():
                    continue
                # Parse commit message
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    message = parts[1]
                    match = EXECUTOR_COMMIT_PATTERN.match(message)
                    if match:
                        task_ids.append(match.group(1))

            return task_ids

        except GitOperationError:
            return []

    # =========================================================================
    # Internal Git Operations
    # =========================================================================

    def _validate(self) -> None:
        """Validate git is available and we're in a repo."""
        # Check git is installed
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            raise GitNotFoundError("Git is not installed or not in PATH")

        # Check we're in a git repo
        if not self._workspace.exists():
            raise NotAGitRepoError(f"Workspace does not exist: {self._workspace}")

        try:
            self._run_git(["rev-parse", "--git-dir"])
        except GitOperationError:
            raise NotAGitRepoError(f"Not a git repository: {self._workspace}")

    def _run_git(
        self,
        args: list[str],
        *,
        check: bool = True,
    ) -> str | None:
        """Run a git command.

        Args:
            args: Git command arguments (without 'git').
            check: If True, raise on non-zero exit.

        Returns:
            Command stdout, or None if check=False and command failed.
        """
        cmd = ["git", "-C", str(self._workspace)] + args

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Never raise, we handle it ourselves
        )

        if result.returncode != 0:
            if not check:
                return None
            raise GitOperationError(
                f"Git command failed: {' '.join(args)}",
                command=" ".join(cmd),
                returncode=result.returncode,
                stderr=result.stderr,
            )

        return result.stdout

    def _stage_files(self, files: list[str]) -> list[str]:
        """Stage specific files.

        Args:
            files: Files to stage.

        Returns:
            List of files that were staged.
        """
        staged = []
        for file in files:
            try:
                # Check if file exists or was deleted
                file_path = self._workspace / file
                if file_path.exists():
                    self._run_git(["add", file])
                    staged.append(file)
                else:
                    # File was deleted, stage the deletion
                    self._run_git(["add", file], check=False)
                    if self._is_file_staged(file):
                        staged.append(file)
            except GitOperationError:
                logger.warning(f"Could not stage file: {file}")
        return staged

    def _stage_all(self) -> list[str]:
        """Stage all changes.

        Returns:
            List of files that were staged.
        """
        self._run_git(["add", "-A"])
        return self._get_staged_files()

    def _get_staged_files(self) -> list[str]:
        """Get list of currently staged files."""
        result = self._run_git(["diff", "--cached", "--name-only"])
        if not result:
            return []
        files = result.strip().split("\n") if result.strip() else []
        return [f for f in files if f]

    def _is_file_staged(self, file: str) -> bool:
        """Check if a file is staged."""
        staged = self._get_staged_files()
        return file in staged

    def _commit(self, message: str) -> str:
        """Create a commit.

        Args:
            message: Commit message.

        Returns:
            The commit SHA.
        """
        args = [
            "commit",
            "-m",
            message,
            "--author",
            f"{self._author_name} <{self._author_email}>",
        ]
        if self._sign_commits:
            args.append("-S")

        self._run_git(args)

        # Get the commit SHA
        result = self._run_git(["rev-parse", "HEAD"])
        assert result is not None
        return result.strip()

    def _find_task_commit(self, task_id: str) -> CommitInfo | None:
        """Find the commit for a specific task."""
        commits = self._get_recent_commits(limit=100)
        for commit in commits:
            if commit.task_id == task_id:
                return commit
        return None

    def _get_parent_sha(self, sha: str) -> str | None:
        """Get the parent commit SHA."""
        try:
            result = self._run_git(["rev-parse", f"{sha}^"])
            assert result is not None
            return result.strip()
        except GitOperationError:
            return None

    def _resolve_ref(self, ref: str) -> str | None:
        """Resolve a ref to a full SHA."""
        try:
            result = self._run_git(["rev-parse", ref])
            assert result is not None
            return result.strip()
        except GitOperationError:
            return None

    def _count_commits_between(self, from_sha: str, to_sha: str) -> int:
        """Count commits between two refs."""
        try:
            result = self._run_git(
                [
                    "rev-list",
                    "--count",
                    f"{from_sha}..{to_sha}",
                ]
            )
            if not result:
                return 0
            return int(result.strip())
        except (GitOperationError, ValueError):
            return 0

    def _get_recent_commits(self, limit: int = 50) -> list[CommitInfo]:
        """Get recent commits.

        Args:
            limit: Maximum number of commits.

        Returns:
            List of CommitInfo objects.
        """
        try:
            # Format: sha|short_sha|author|timestamp|message
            result = self._run_git(
                [
                    "log",
                    f"-{limit}",
                    "--format=%H|%h|%an|%aI|%s",
                ]
            )

            if not result:
                return []

            commits = []
            for line in result.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split("|", 4)
                if len(parts) != 5:
                    continue

                sha, short_sha, author, timestamp_str, message = parts

                # Parse task ID from message
                task_id = None
                match = EXECUTOR_COMMIT_PATTERN.match(message)
                if match:
                    task_id = match.group(1)

                # Parse timestamp
                timestamp = None
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except ValueError:
                    pass

                commits.append(
                    CommitInfo(
                        sha=sha,
                        short_sha=short_sha,
                        message=message,
                        task_id=task_id,
                        timestamp=timestamp,
                        author=author,
                    )
                )

            return commits

        except GitOperationError:
            return []

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def for_workspace(cls, workspace: str | Path) -> Checkpointer | None:
        """Create a checkpointer if the workspace is a git repo.

        Args:
            workspace: Path to the workspace.

        Returns:
            Checkpointer if workspace is a git repo, None otherwise.
        """
        try:
            return cls(workspace)
        except (NotAGitRepoError, GitNotFoundError):
            return None
