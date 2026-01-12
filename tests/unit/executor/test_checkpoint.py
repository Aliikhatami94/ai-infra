"""Tests for the checkpoint module."""

from __future__ import annotations

import subprocess
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_infra.executor.checkpoint import (
    Checkpointer,
    CheckpointError,
    CheckpointResult,
    CommitInfo,
    GitNotFoundError,
    GitOperationError,
    NotAGitRepoError,
    RollbackResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary git repository."""
    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    # Configure git user for commits
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    # Create initial commit
    readme = tmp_path / "README.md"
    readme.write_text("# Test Project\n")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    yield tmp_path


@pytest.fixture
def checkpointer(temp_git_repo: Path) -> Checkpointer:
    """Create a checkpointer for the temp repo."""
    return Checkpointer(temp_git_repo)


# =============================================================================
# CommitInfo Tests
# =============================================================================


class TestCommitInfo:
    """Tests for CommitInfo dataclass."""

    def test_basic_commit_info(self) -> None:
        """Test basic CommitInfo creation."""
        info = CommitInfo(
            sha="abc123def456",
            short_sha="abc123d",
            message="test commit",
        )
        assert info.sha == "abc123def456"
        assert info.short_sha == "abc123d"
        assert info.message == "test commit"
        assert info.task_id is None
        assert not info.is_executor_commit

    def test_executor_commit_info(self) -> None:
        """Test CommitInfo with executor task ID."""
        info = CommitInfo(
            sha="abc123def456",
            short_sha="abc123d",
            message="executor(1.1.1): Implement feature",
            task_id="1.1.1",
        )
        assert info.task_id == "1.1.1"
        assert info.is_executor_commit

    def test_to_dict(self) -> None:
        """Test CommitInfo serialization."""
        now = datetime.now(UTC)
        info = CommitInfo(
            sha="abc123",
            short_sha="abc",
            message="test",
            task_id="1.0",
            timestamp=now,
            author="Test",
            files_changed=5,
            insertions=10,
            deletions=2,
        )
        d = info.to_dict()
        assert d["sha"] == "abc123"
        assert d["task_id"] == "1.0"
        assert d["files_changed"] == 5


# =============================================================================
# CheckpointResult Tests
# =============================================================================


class TestCheckpointResult:
    """Tests for CheckpointResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful checkpoint result."""
        result = CheckpointResult(
            success=True,
            commit_sha="abc123",
            message="Created commit abc123",
            files_staged=["src/main.py"],
        )
        assert result.success
        assert result.commit_sha == "abc123"
        assert len(result.files_staged) == 1

    def test_failure_result(self) -> None:
        """Test failed checkpoint result."""
        result = CheckpointResult(
            success=False,
            error="Git operation failed",
            message="Failed to create checkpoint",
        )
        assert not result.success
        assert result.error == "Git operation failed"
        assert result.commit_sha is None

    def test_to_dict(self) -> None:
        """Test CheckpointResult serialization."""
        result = CheckpointResult(
            success=True,
            commit_sha="abc",
            files_staged=["a.py", "b.py"],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["commit_sha"] == "abc"
        assert len(d["files_staged"]) == 2


# =============================================================================
# RollbackResult Tests
# =============================================================================


class TestRollbackResult:
    """Tests for RollbackResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful rollback result."""
        result = RollbackResult(
            success=True,
            target_sha="abc123",
            commits_reverted=3,
            message="Rolled back 3 commits",
        )
        assert result.success
        assert result.target_sha == "abc123"
        assert result.commits_reverted == 3

    def test_failure_result(self) -> None:
        """Test failed rollback result."""
        result = RollbackResult(
            success=False,
            error="No commit found for task",
        )
        assert not result.success
        assert result.error is not None

    def test_to_dict(self) -> None:
        """Test RollbackResult serialization."""
        result = RollbackResult(
            success=True,
            target_sha="abc",
            commits_reverted=2,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["commits_reverted"] == 2


# =============================================================================
# Checkpointer Initialization Tests
# =============================================================================


class TestCheckpointerInit:
    """Tests for Checkpointer initialization."""

    def test_init_with_git_repo(self, temp_git_repo: Path) -> None:
        """Test initialization with a valid git repo."""
        cp = Checkpointer(temp_git_repo)
        assert cp.workspace == temp_git_repo
        assert cp.is_repo

    def test_init_with_non_repo(self, tmp_path: Path) -> None:
        """Test initialization with a non-git directory."""
        with pytest.raises(NotAGitRepoError):
            Checkpointer(tmp_path)

    def test_init_with_nonexistent_path(self, tmp_path: Path) -> None:
        """Test initialization with non-existent path."""
        with pytest.raises(NotAGitRepoError):
            Checkpointer(tmp_path / "nonexistent")

    def test_for_workspace_with_repo(self, temp_git_repo: Path) -> None:
        """Test factory method with valid repo."""
        cp = Checkpointer.for_workspace(temp_git_repo)
        assert cp is not None
        assert cp.workspace == temp_git_repo

    def test_for_workspace_with_non_repo(self, tmp_path: Path) -> None:
        """Test factory method with non-git directory."""
        cp = Checkpointer.for_workspace(tmp_path)
        assert cp is None

    def test_custom_author(self, temp_git_repo: Path) -> None:
        """Test custom author configuration."""
        cp = Checkpointer(
            temp_git_repo,
            author_name="Custom Author",
            author_email="custom@example.com",
        )
        assert cp._author_name == "Custom Author"
        assert cp._author_email == "custom@example.com"


# =============================================================================
# Checkpointer Properties Tests
# =============================================================================


class TestCheckpointerProperties:
    """Tests for Checkpointer properties."""

    def test_workspace_property(self, checkpointer: Checkpointer) -> None:
        """Test workspace property."""
        assert checkpointer.workspace.exists()
        assert checkpointer.workspace.is_dir()

    def test_is_repo_property(self, checkpointer: Checkpointer) -> None:
        """Test is_repo property."""
        assert checkpointer.is_repo is True

    def test_current_branch(self, checkpointer: Checkpointer) -> None:
        """Test current_branch property."""
        # Default branch name varies (main/master)
        branch = checkpointer.current_branch
        assert branch in ["main", "master"]

    def test_current_sha(self, checkpointer: Checkpointer) -> None:
        """Test current_sha property."""
        sha = checkpointer.current_sha
        assert sha is not None
        assert len(sha) == 40  # Full SHA length

    def test_git_dir(self, checkpointer: Checkpointer) -> None:
        """Test git_dir property."""
        git_dir = checkpointer.git_dir
        assert git_dir.exists()


# =============================================================================
# Checkpoint Operation Tests
# =============================================================================


class TestCheckpointOperations:
    """Tests for checkpoint operations."""

    def test_checkpoint_no_changes(self, checkpointer: Checkpointer) -> None:
        """Test checkpoint with no changes."""
        result = checkpointer.checkpoint(
            task_id="1.1.1",
            task_title="Test task",
        )
        assert result.success
        assert result.commit_sha is None
        assert "No changes" in result.message

    def test_checkpoint_with_changes(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test checkpoint with actual changes."""
        # Create a new file
        test_file = temp_git_repo / "src" / "main.py"
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text("# Main module\n")

        result = checkpointer.checkpoint(
            task_id="1.1.1",
            task_title="Implement feature",
            files_created=["src/main.py"],
        )

        assert result.success
        assert result.commit_sha is not None
        assert len(result.commit_sha) == 40
        assert "src/main.py" in result.files_staged

    def test_checkpoint_commit_message_format(
        self, checkpointer: Checkpointer, temp_git_repo: Path
    ) -> None:
        """Test checkpoint creates proper commit message."""
        # Create a file
        test_file = temp_git_repo / "test.py"
        test_file.write_text("pass\n")

        result = checkpointer.checkpoint(
            task_id="2.1.3",
            task_title="Add unit tests",
            stage_all=True,
        )

        assert result.success

        # Verify commit message
        commit = checkpointer.get_commit_for_task("2.1.3")
        assert commit is not None
        assert commit.message.startswith("executor(2.1.3):")
        assert "Add unit tests" in commit.message

    def test_checkpoint_title_truncation(
        self, checkpointer: Checkpointer, temp_git_repo: Path
    ) -> None:
        """Test that long titles are truncated."""
        test_file = temp_git_repo / "test.py"
        test_file.write_text("pass\n")

        long_title = "A" * 100

        result = checkpointer.checkpoint(
            task_id="1.0.0",
            task_title=long_title,
            stage_all=True,
        )

        assert result.success
        commit = checkpointer.get_commit_for_task("1.0.0")
        assert commit is not None
        # Message should be truncated (50 chars max for title)
        assert len(commit.message) < 100

    def test_checkpoint_custom_message(
        self, checkpointer: Checkpointer, temp_git_repo: Path
    ) -> None:
        """Test checkpoint with custom message."""
        test_file = temp_git_repo / "test.py"
        test_file.write_text("pass\n")

        result = checkpointer.checkpoint(
            task_id="1.0.0",
            task_title="Original title",
            message="Custom commit message",
            stage_all=True,
        )

        assert result.success
        # Get the latest commit
        commits = checkpointer._get_recent_commits(limit=1)
        assert commits[0].message == "Custom commit message"

    def test_checkpoint_stage_all(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test checkpoint with stage_all option."""
        # Create multiple files
        (temp_git_repo / "a.py").write_text("a\n")
        (temp_git_repo / "b.py").write_text("b\n")
        (temp_git_repo / "c.py").write_text("c\n")

        result = checkpointer.checkpoint(
            task_id="1.0.0",
            task_title="Add files",
            stage_all=True,
        )

        assert result.success
        assert len(result.files_staged) >= 3

    def test_checkpoint_modified_files(
        self, checkpointer: Checkpointer, temp_git_repo: Path
    ) -> None:
        """Test checkpoint stages modified files."""
        # Modify existing file
        readme = temp_git_repo / "README.md"
        readme.write_text("# Updated\n")

        result = checkpointer.checkpoint(
            task_id="1.0.0",
            task_title="Update readme",
            stage_all=True,  # Use stage_all for simplicity
        )

        assert result.success
        assert "README.md" in result.files_staged


# =============================================================================
# Rollback Operation Tests
# =============================================================================


class TestRollbackOperations:
    """Tests for rollback operations."""

    def test_rollback_nonexistent_task(self, checkpointer: Checkpointer) -> None:
        """Test rollback for task with no commit."""
        result = checkpointer.rollback("nonexistent")
        assert not result.success
        assert "No commit found" in result.error or "no commit" in result.error.lower()

    def test_rollback_task(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test rollback to before a task."""
        # Get current SHA
        original_sha = checkpointer.current_sha

        # Create task commit
        test_file = temp_git_repo / "feature.py"
        test_file.write_text("# Feature\n")

        checkpointer.checkpoint(
            task_id="1.0.0",
            task_title="Add feature",
            files_created=["feature.py"],
        )

        # Verify file exists
        assert test_file.exists()

        # Rollback (soft - keeps changes staged)
        result = checkpointer.rollback("1.0.0")

        assert result.success
        assert result.target_sha == original_sha
        assert result.commits_reverted == 1

    def test_rollback_hard(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test hard rollback discards changes."""

        # Create and commit file
        test_file = temp_git_repo / "feature.py"
        test_file.write_text("# Feature\n")

        checkpointer.checkpoint(
            task_id="1.0.0",
            task_title="Add feature",
            files_created=["feature.py"],
        )

        # Hard rollback
        result = checkpointer.rollback("1.0.0", hard=True)

        assert result.success
        # File should be gone after hard reset
        assert not test_file.exists()

    def test_rollback_to_sha(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test rollback to specific SHA."""
        original_sha = checkpointer.current_sha

        # Create multiple commits
        for i in range(3):
            (temp_git_repo / f"file{i}.py").write_text(f"# File {i}\n")
            checkpointer.checkpoint(
                task_id=f"1.0.{i}",
                task_title=f"Add file {i}",
                stage_all=True,
            )

        # Rollback to original
        result = checkpointer.rollback_to_sha(original_sha)

        assert result.success
        assert result.commits_reverted == 3

    def test_rollback_to_invalid_sha(self, checkpointer: Checkpointer) -> None:
        """Test rollback to invalid SHA."""
        result = checkpointer.rollback_to_sha("invalid_sha_12345")
        assert not result.success
        assert "not found" in result.error.lower()


# =============================================================================
# Query Operation Tests
# =============================================================================


class TestQueryOperations:
    """Tests for query operations."""

    def test_has_changes_no_changes(self, checkpointer: Checkpointer) -> None:
        """Test has_changes with clean working tree."""
        assert not checkpointer.has_changes()

    def test_has_changes_with_changes(
        self, checkpointer: Checkpointer, temp_git_repo: Path
    ) -> None:
        """Test has_changes with modifications."""
        (temp_git_repo / "new.py").write_text("pass\n")
        assert checkpointer.has_changes()

    def test_has_staged_changes(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test has_staged_changes."""
        assert not checkpointer.has_staged_changes()

        # Stage a file
        new_file = temp_git_repo / "staged.py"
        new_file.write_text("pass\n")
        subprocess.run(
            ["git", "add", "staged.py"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        assert checkpointer.has_staged_changes()

    def test_get_uncommitted_files(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test get_uncommitted_files."""
        # Create untracked and modified files
        (temp_git_repo / "untracked.py").write_text("pass\n")
        (temp_git_repo / "README.md").write_text("# Modified\n")

        files = checkpointer.get_uncommitted_files()
        assert "untracked.py" in files
        assert "README.md" in files

    def test_get_executor_commits(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test get_executor_commits."""
        # Create some executor commits
        for i in range(3):
            (temp_git_repo / f"file{i}.py").write_text("pass\n")
            checkpointer.checkpoint(
                task_id=f"1.{i}.0",
                task_title=f"Task {i}",
                stage_all=True,
            )

        commits = checkpointer.get_executor_commits()
        assert len(commits) == 3
        assert all(c.is_executor_commit for c in commits)

    def test_get_commit_for_task(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test get_commit_for_task."""
        (temp_git_repo / "test.py").write_text("pass\n")
        checkpointer.checkpoint(
            task_id="2.1.3",
            task_title="Specific task",
            stage_all=True,
        )

        commit = checkpointer.get_commit_for_task("2.1.3")
        assert commit is not None
        assert commit.task_id == "2.1.3"
        assert commit.is_executor_commit

    def test_get_commit_for_task_not_found(self, checkpointer: Checkpointer) -> None:
        """Test get_commit_for_task with no match."""
        commit = checkpointer.get_commit_for_task("nonexistent")
        assert commit is None

    def test_get_tasks_since(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test get_tasks_since."""
        base_sha = checkpointer.current_sha

        # Create task commits
        for i in range(3):
            (temp_git_repo / f"file{i}.py").write_text("pass\n")
            checkpointer.checkpoint(
                task_id=f"1.{i}.0",
                task_title=f"Task {i}",
                stage_all=True,
            )

        tasks = checkpointer.get_tasks_since(base_sha)
        assert len(tasks) == 3
        assert tasks == ["1.0.0", "1.1.0", "1.2.0"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_git_not_found(self) -> None:
        """Test GitNotFoundError."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            with pytest.raises(GitNotFoundError):
                Checkpointer("/tmp")

    def test_git_operation_error_attributes(self) -> None:
        """Test GitOperationError attributes."""
        error = GitOperationError(
            "Command failed",
            command="git commit",
            returncode=1,
            stderr="error message",
        )
        assert error.command == "git commit"
        assert error.returncode == 1
        assert error.stderr == "error message"

    def test_checkpoint_error_hierarchy(self) -> None:
        """Test exception hierarchy."""
        assert issubclass(GitNotFoundError, CheckpointError)
        assert issubclass(NotAGitRepoError, CheckpointError)
        assert issubclass(GitOperationError, CheckpointError)


# =============================================================================
# Integration Tests
# =============================================================================


class TestCheckpointerIntegration:
    """Integration tests for Checkpointer."""

    def test_full_workflow(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test complete checkpoint and rollback workflow."""
        initial_sha = checkpointer.current_sha

        # Create and checkpoint multiple tasks
        task_ids = []
        for i in range(3):
            file_path = temp_git_repo / f"feature_{i}.py"
            file_path.write_text(f"# Feature {i}\n")

            result = checkpointer.checkpoint(
                task_id=f"1.0.{i}",
                task_title=f"Add feature {i}",
                files_created=[f"feature_{i}.py"],
            )
            assert result.success
            task_ids.append(f"1.0.{i}")

        # Verify all commits exist
        for task_id in task_ids:
            commit = checkpointer.get_commit_for_task(task_id)
            assert commit is not None

        # Verify tasks since initial
        tasks = checkpointer.get_tasks_since(initial_sha)
        assert len(tasks) == 3

        # Rollback to middle
        result = checkpointer.rollback("1.0.2")
        assert result.success

        # Verify rollback
        assert checkpointer.get_commit_for_task("1.0.2") is None
        assert checkpointer.get_commit_for_task("1.0.1") is not None

    def test_concurrent_file_modifications(
        self, checkpointer: Checkpointer, temp_git_repo: Path
    ) -> None:
        """Test checkpoint with many file changes."""
        # Create many files
        src_dir = temp_git_repo / "src"
        src_dir.mkdir()

        files = []
        for i in range(10):
            file_path = src_dir / f"module_{i}.py"
            file_path.write_text(f"# Module {i}\n")
            files.append(f"src/module_{i}.py")

        result = checkpointer.checkpoint(
            task_id="1.0.0",
            task_title="Add modules",
            files_created=files,
        )

        assert result.success
        assert len(result.files_staged) == 10

    def test_deleted_file_handling(self, checkpointer: Checkpointer, temp_git_repo: Path) -> None:
        """Test checkpoint handles deleted files."""
        # Create and commit a file
        test_file = temp_git_repo / "to_delete.py"
        test_file.write_text("pass\n")

        checkpointer.checkpoint(
            task_id="1.0.0",
            task_title="Add file",
            files_created=["to_delete.py"],
        )

        # Delete the file
        test_file.unlink()

        # Checkpoint the deletion
        result = checkpointer.checkpoint(
            task_id="1.0.1",
            task_title="Remove file",
            stage_all=True,  # Need stage_all for deletions
        )

        assert result.success
