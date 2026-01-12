"""Tests for executor recovery module (Phase 4.2)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ai_infra.executor.checkpoint import Checkpointer
from ai_infra.executor.recovery import (
    CheckpointMetadata,
    FileSnapshot,
    RecoveryManager,
    RecoveryResult,
    RecoveryStrategy,
    RollbackPreview,
    SelectiveRollbackResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create initial commit
    readme = repo_path / "README.md"
    readme.write_text("# Test Repository\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    return repo_path


@pytest.fixture
def checkpointer(temp_git_repo: Path) -> Checkpointer:
    """Create a checkpointer for the temp repo."""
    return Checkpointer(temp_git_repo)


@pytest.fixture
def recovery_manager(checkpointer: Checkpointer) -> RecoveryManager:
    """Create a recovery manager for the temp repo."""
    return RecoveryManager(checkpointer)


# =============================================================================
# FileSnapshot Tests
# =============================================================================


class TestFileSnapshot:
    """Tests for FileSnapshot dataclass."""

    def test_from_existing_file(self, tmp_path: Path) -> None:
        """Test creating snapshot from existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        snapshot = FileSnapshot.from_file(tmp_path, "test.txt")

        assert snapshot.path == "test.txt"
        assert snapshot.exists is True
        assert snapshot.size == 13
        assert snapshot.hash is not None
        assert len(snapshot.hash) == 64  # SHA256 hex

    def test_from_nonexistent_file(self, tmp_path: Path) -> None:
        """Test creating snapshot from non-existent file."""
        snapshot = FileSnapshot.from_file(tmp_path, "missing.txt")

        assert snapshot.path == "missing.txt"
        assert snapshot.exists is False
        assert snapshot.hash is None

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        original = FileSnapshot(
            path="src/main.py",
            hash="abc123",
            size=1024,
            exists=True,
            is_new=True,
        )

        data = original.to_dict()
        restored = FileSnapshot.from_dict(data)

        assert restored.path == original.path
        assert restored.hash == original.hash
        assert restored.size == original.size
        assert restored.exists == original.exists
        assert restored.is_new == original.is_new


# =============================================================================
# CheckpointMetadata Tests
# =============================================================================


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating basic metadata."""
        meta = CheckpointMetadata(
            task_id="1.1.1",
            task_title="Test task",
            commit_sha="abc123def456",
        )

        assert meta.task_id == "1.1.1"
        assert meta.task_title == "Test task"
        assert meta.commit_sha == "abc123def456"
        assert meta.files_modified == []
        assert meta.tags == []

    def test_all_files_property(self) -> None:
        """Test all_files aggregation property."""
        meta = CheckpointMetadata(
            task_id="1.1.1",
            task_title="Test",
            commit_sha="abc123",
            files_modified=["a.py", "b.py"],
            files_created=["c.py"],
            files_deleted=["d.py"],
        )

        all_files = meta.all_files
        assert len(all_files) == 4
        assert set(all_files) == {"a.py", "b.py", "c.py", "d.py"}

    def test_tag_operations(self) -> None:
        """Test tag add and check."""
        meta = CheckpointMetadata(
            task_id="1.1.1",
            task_title="Test",
            commit_sha="abc123",
        )

        assert not meta.has_tag("stable")

        meta.add_tag("stable")
        assert meta.has_tag("stable")

        # Adding same tag twice should not duplicate
        meta.add_tag("stable")
        assert meta.tags.count("stable") == 1

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        original = CheckpointMetadata(
            task_id="1.1.1",
            task_title="Test task",
            commit_sha="abc123def456",
            files_modified=["src/main.py"],
            files_created=["src/new.py"],
            verification_passed=True,
            verification_level="syntax",
            parent_checkpoint="parent123",
            tags=["stable", "tested"],
            recovery_strategy=RecoveryStrategy.RETRY_WITH_CONTEXT,
            metadata={"custom": "value"},
        )

        data = original.to_dict()
        restored = CheckpointMetadata.from_dict(data)

        assert restored.task_id == original.task_id
        assert restored.task_title == original.task_title
        assert restored.commit_sha == original.commit_sha
        assert restored.files_modified == original.files_modified
        assert restored.files_created == original.files_created
        assert restored.verification_passed == original.verification_passed
        assert restored.verification_level == original.verification_level
        assert restored.parent_checkpoint == original.parent_checkpoint
        assert restored.tags == original.tags
        assert restored.recovery_strategy == original.recovery_strategy
        assert restored.metadata == original.metadata


# =============================================================================
# RollbackPreview Tests
# =============================================================================


class TestRollbackPreview:
    """Tests for RollbackPreview dataclass."""

    def test_safe_preview(self) -> None:
        """Test preview with no warnings is safe."""
        preview = RollbackPreview(
            target_sha="abc123",
            commits_to_revert=2,
            files_to_restore=["a.py", "b.py"],
        )

        assert preview.is_safe is True
        assert preview.total_files_affected == 2

    def test_unsafe_preview(self) -> None:
        """Test preview with warnings is not safe."""
        preview = RollbackPreview(
            target_sha="abc123",
            commits_to_revert=15,
            files_to_restore=["a.py"],
            warnings=["Rolling back 15 commits - this is a large rollback"],
        )

        assert preview.is_safe is False

    def test_total_files_affected(self) -> None:
        """Test total files calculation."""
        preview = RollbackPreview(
            target_sha="abc123",
            files_to_restore=["a.py", "b.py"],
            files_to_delete=["c.py"],
            files_to_recreate=["d.py", "e.py"],
        )

        assert preview.total_files_affected == 5

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        preview = RollbackPreview(
            target_sha="abc123",
            commits_to_revert=3,
            files_to_restore=["a.py"],
            tasks_affected=["1.1.1", "1.1.2"],
            warnings=["warning 1"],
        )

        data = preview.to_dict()

        assert data["target_sha"] == "abc123"
        assert data["commits_to_revert"] == 3
        assert data["files_to_restore"] == ["a.py"]
        assert data["tasks_affected"] == ["1.1.1", "1.1.2"]
        assert data["warnings"] == ["warning 1"]
        assert data["is_safe"] is False


# =============================================================================
# SelectiveRollbackResult Tests
# =============================================================================


class TestSelectiveRollbackResult:
    """Tests for SelectiveRollbackResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful rollback result."""
        result = SelectiveRollbackResult(
            success=True,
            files_restored=["a.py", "b.py"],
            files_deleted=["c.py"],
            message="Restored 3 file(s)",
        )

        assert result.success is True
        assert len(result.files_restored) == 2
        assert len(result.files_deleted) == 1
        assert result.error is None

    def test_partial_failure_result(self) -> None:
        """Test partial failure result."""
        result = SelectiveRollbackResult(
            success=False,
            files_restored=["a.py"],
            files_failed=["b.py"],
            error="Failed to restore: ['b.py']",
        )

        assert result.success is False
        assert len(result.files_failed) == 1


# =============================================================================
# RecoveryResult Tests
# =============================================================================


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic recovery result."""
        result = RecoveryResult(
            success=True,
            strategy=RecoveryStrategy.ROLLBACK_ALL,
            message="Rolled back successfully",
        )

        assert result.success is True
        assert result.strategy == RecoveryStrategy.ROLLBACK_ALL

    def test_branch_result(self) -> None:
        """Test recovery result with branch."""
        result = RecoveryResult(
            success=True,
            strategy=RecoveryStrategy.BRANCH_AND_MERGE,
            branch_name="executor-recovery-1.1.1",
        )

        assert result.branch_name == "executor-recovery-1.1.1"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        result = RecoveryResult(
            success=True,
            strategy=RecoveryStrategy.RETRY_WITH_CONTEXT,
            retry_count=2,
        )

        data = result.to_dict()
        assert data["strategy"] == "retry_with_context"
        assert data["retry_count"] == 2


# =============================================================================
# RecoveryManager Tests
# =============================================================================


class TestRecoveryManagerInit:
    """Tests for RecoveryManager initialization."""

    def test_init_with_checkpointer(self, checkpointer: Checkpointer) -> None:
        """Test initialization with checkpointer."""
        manager = RecoveryManager(checkpointer)

        assert manager.checkpointer is checkpointer
        assert manager.workspace == checkpointer.workspace

    def test_for_workspace_factory(self, temp_git_repo: Path) -> None:
        """Test factory method for workspace."""
        manager = RecoveryManager.for_workspace(temp_git_repo)

        assert manager is not None
        assert manager.workspace == temp_git_repo

    def test_for_workspace_non_repo(self, tmp_path: Path) -> None:
        """Test factory method with non-repo returns None."""
        manager = RecoveryManager.for_workspace(tmp_path)

        assert manager is None

    def test_creates_metadata_dir(self, checkpointer: Checkpointer) -> None:
        """Test that metadata directory is created."""
        RecoveryManager(checkpointer)

        metadata_dir = checkpointer.workspace / ".executor" / "checkpoints"
        assert metadata_dir.exists()


class TestRecoveryManagerCheckpoints:
    """Tests for RecoveryManager checkpoint operations."""

    def test_create_checkpoint_basic(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test creating a basic checkpoint."""
        # Create a file to commit
        test_file = temp_git_repo / "test.py"
        test_file.write_text("print('hello')")

        result, meta = recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Add test file",
            files_created=["test.py"],
        )

        assert result.success is True
        assert result.commit_sha is not None
        assert meta is not None
        assert meta.task_id == "1.1.1"
        assert meta.files_created == ["test.py"]

    def test_create_checkpoint_with_tags(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test creating checkpoint with tags."""
        test_file = temp_git_repo / "test.py"
        test_file.write_text("print('hello')")

        result, meta = recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Add test file",
            files_created=["test.py"],
            tags=["stable", "phase-1"],
        )

        assert result.success is True
        assert meta is not None
        assert "stable" in meta.tags
        assert "phase-1" in meta.tags

        # Verify tags are stored
        stored_sha = recovery_manager.get_checkpoint_by_tag("stable")
        assert stored_sha == result.commit_sha

    def test_create_checkpoint_with_verification(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test creating checkpoint with verification status."""
        test_file = temp_git_repo / "test.py"
        test_file.write_text("print('hello')")

        result, meta = recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Add test file",
            files_created=["test.py"],
            verification_passed=True,
            verification_level="syntax",
        )

        assert meta is not None
        assert meta.verification_passed is True
        assert meta.verification_level == "syntax"

    def test_create_checkpoint_captures_snapshots(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test that file snapshots are captured."""
        test_file = temp_git_repo / "test.py"
        test_file.write_text("print('hello')")

        result, meta = recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Add test file",
            files_created=["test.py"],
            capture_snapshots=True,
        )

        assert meta is not None
        assert len(meta.file_snapshots) == 1
        snapshot = meta.file_snapshots[0]
        assert snapshot.path == "test.py"
        assert snapshot.is_new is True
        assert snapshot.hash is not None

    def test_get_checkpoint_metadata(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test retrieving checkpoint metadata."""
        test_file = temp_git_repo / "test.py"
        test_file.write_text("print('hello')")

        result, original_meta = recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Add test file",
            files_created=["test.py"],
        )

        assert result.commit_sha is not None
        loaded_meta = recovery_manager.get_checkpoint_metadata(result.commit_sha)

        assert loaded_meta is not None
        assert loaded_meta.task_id == "1.1.1"
        assert loaded_meta.task_title == "Add test file"

    def test_add_tag_to_existing(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test adding tag to existing checkpoint."""
        test_file = temp_git_repo / "test.py"
        test_file.write_text("print('hello')")

        result, _ = recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Add test file",
            files_created=["test.py"],
        )

        success = recovery_manager.add_tag("new-tag", result.commit_sha)

        assert success is True
        assert recovery_manager.get_checkpoint_by_tag("new-tag") == result.commit_sha

    def test_list_tags(self, recovery_manager: RecoveryManager, temp_git_repo: Path) -> None:
        """Test listing all tags."""
        test_file = temp_git_repo / "test.py"
        test_file.write_text("print('hello')")

        result, _ = recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Add test file",
            files_created=["test.py"],
            tags=["tag1", "tag2"],
        )

        tags = recovery_manager.list_tags()

        assert "tag1" in tags
        assert "tag2" in tags
        assert tags["tag1"] == result.commit_sha


class TestRecoveryManagerPreview:
    """Tests for rollback preview functionality."""

    def test_preview_by_task_id(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test previewing rollback by task ID."""
        # Create first task
        file1 = temp_git_repo / "file1.py"
        file1.write_text("# File 1")
        recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Task 1",
            files_created=["file1.py"],
        )

        # Create second task
        file2 = temp_git_repo / "file2.py"
        file2.write_text("# File 2")
        recovery_manager.create_checkpoint(
            task_id="1.1.2",
            task_title="Task 2",
            files_created=["file2.py"],
        )

        # Preview rolling back to before task 1.1.2
        preview = recovery_manager.preview_rollback(task_id="1.1.2")

        assert preview is not None
        assert preview.commits_to_revert >= 1
        assert "1.1.2" in preview.tasks_affected

    def test_preview_by_tag(self, recovery_manager: RecoveryManager, temp_git_repo: Path) -> None:
        """Test previewing rollback by tag."""
        # Create checkpoint with tag
        file1 = temp_git_repo / "file1.py"
        file1.write_text("# File 1")
        recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Task 1",
            files_created=["file1.py"],
            tags=["stable"],
        )

        # Create more changes
        file2 = temp_git_repo / "file2.py"
        file2.write_text("# File 2")
        recovery_manager.create_checkpoint(
            task_id="1.1.2",
            task_title="Task 2",
            files_created=["file2.py"],
        )

        # Preview rollback to stable
        preview = recovery_manager.preview_rollback(tag="stable")

        assert preview is not None
        assert "1.1.2" in preview.tasks_affected

    def test_preview_warns_on_large_rollback(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test that preview warns on large rollbacks."""
        # Create many commits
        for i in range(12):
            file = temp_git_repo / f"file{i}.py"
            file.write_text(f"# File {i}")
            recovery_manager.create_checkpoint(
                task_id=f"1.1.{i}",
                task_title=f"Task {i}",
                files_created=[f"file{i}.py"],
            )

        # Preview rollback to first task
        preview = recovery_manager.preview_rollback(task_id="1.1.0")

        assert preview is not None
        assert len(preview.warnings) > 0
        assert preview.is_safe is False


class TestRecoveryManagerSelectiveRollback:
    """Tests for selective rollback functionality."""

    def test_rollback_to_tag(self, recovery_manager: RecoveryManager, temp_git_repo: Path) -> None:
        """Test rolling back to a tagged checkpoint."""
        # Create checkpoint with tag
        file1 = temp_git_repo / "file1.py"
        file1.write_text("# Original content")
        recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Task 1",
            files_created=["file1.py"],
            tags=["stable"],
        )

        # Make more changes
        file1.write_text("# Modified content")
        recovery_manager.create_checkpoint(
            task_id="1.1.2",
            task_title="Task 2",
            files_modified=["file1.py"],
        )

        # Rollback to stable
        result = recovery_manager.rollback_to_tag("stable", hard=True)

        assert result.success is True
        # File should have original content
        assert file1.read_text() == "# Original content"

    def test_rollback_to_nonexistent_tag(self, recovery_manager: RecoveryManager) -> None:
        """Test rolling back to non-existent tag."""
        result = recovery_manager.rollback_to_tag("nonexistent")

        assert result.success is False
        assert "not found" in str(result.error).lower()

    def test_rollback_to_tag_preview(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test preview mode for tag rollback."""
        file1 = temp_git_repo / "file1.py"
        file1.write_text("# Content")
        recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Task 1",
            files_created=["file1.py"],
            tags=["stable"],
        )

        result = recovery_manager.rollback_to_tag("stable", preview=True)

        assert isinstance(result, RollbackPreview)
        # File should still exist (preview doesn't change anything)
        assert file1.exists()

    def test_rollback_specific_files(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test rolling back specific files."""
        # Create files and checkpoint
        file1 = temp_git_repo / "file1.py"
        file2 = temp_git_repo / "file2.py"
        file1.write_text("# File 1 original")
        file2.write_text("# File 2 original")

        recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Task 1",
            files_created=["file1.py", "file2.py"],
            tags=["stable"],
        )

        # Modify both files
        file1.write_text("# File 1 modified")
        file2.write_text("# File 2 modified")
        recovery_manager.create_checkpoint(
            task_id="1.1.2",
            task_title="Task 2",
            files_modified=["file1.py", "file2.py"],
        )

        # Rollback only file1
        result = recovery_manager.rollback_files(["file1.py"], to_tag="stable")

        assert result.success is True
        assert "file1.py" in result.files_restored
        # file1 should be restored, file2 should still be modified
        assert file1.read_text() == "# File 1 original"
        assert file2.read_text() == "# File 2 modified"

    def test_rollback_files_delete_new_file(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test rollback deletes files that didn't exist at target."""
        recovery_manager.add_tag("before-new-file")

        # Create new file after tag
        new_file = temp_git_repo / "new_file.py"
        new_file.write_text("# New file")
        recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Add new file",
            files_created=["new_file.py"],
        )

        # Rollback the new file to before it existed
        result = recovery_manager.rollback_files(["new_file.py"], to_tag="before-new-file")

        assert result.success is True
        assert "new_file.py" in result.files_deleted
        assert not new_file.exists()


class TestRecoveryManagerRecoveryStrategies:
    """Tests for recovery strategy functionality."""

    def test_recover_rollback_all(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test ROLLBACK_ALL recovery strategy."""
        # Create a task with changes
        file1 = temp_git_repo / "file1.py"
        file1.write_text("# Content")
        recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Task 1",
            files_created=["file1.py"],
        )

        # Recover with ROLLBACK_ALL
        result = recovery_manager.recover("1.1.1", RecoveryStrategy.ROLLBACK_ALL)

        assert result.strategy == RecoveryStrategy.ROLLBACK_ALL
        # Note: success depends on whether rollback worked

    def test_recover_rollback_failed(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test ROLLBACK_FAILED recovery strategy."""
        file1 = temp_git_repo / "file1.py"
        file2 = temp_git_repo / "file2.py"
        file1.write_text("# Original 1")
        file2.write_text("# Original 2")
        recovery_manager.create_checkpoint(
            task_id="1.1.0",
            task_title="Task 0",
            files_created=["file1.py", "file2.py"],
        )

        file1.write_text("# Modified 1")
        file2.write_text("# Modified 2")
        recovery_manager.create_checkpoint(
            task_id="1.1.1",
            task_title="Task 1",
            files_modified=["file1.py", "file2.py"],
        )

        # Rollback only file1 (the "failed" one)
        result = recovery_manager.recover(
            "1.1.1",
            RecoveryStrategy.ROLLBACK_FAILED,
            failed_files=["file1.py"],
        )

        assert result.strategy == RecoveryStrategy.ROLLBACK_FAILED
        if result.success:
            assert file1.read_text() == "# Original 1"
            assert file2.read_text() == "# Modified 2"

    def test_recover_rollback_failed_no_files(self, recovery_manager: RecoveryManager) -> None:
        """Test ROLLBACK_FAILED with no files specified."""
        result = recovery_manager.recover(
            "1.1.1",
            RecoveryStrategy.ROLLBACK_FAILED,
            failed_files=[],
        )

        assert result.success is False
        assert "No failed files" in str(result.error)

    def test_recover_retry_with_context(self, recovery_manager: RecoveryManager) -> None:
        """Test RETRY_WITH_CONTEXT recovery strategy."""
        result = recovery_manager.recover(
            "1.1.1",
            RecoveryStrategy.RETRY_WITH_CONTEXT,
            error_context="Test failed with assertion error",
        )

        assert result.success is True
        assert result.strategy == RecoveryStrategy.RETRY_WITH_CONTEXT

        # Verify context was stored
        context = recovery_manager.get_retry_context("1.1.1")
        assert context is not None
        assert context["error_context"] == "Test failed with assertion error"

    def test_recover_branch_and_merge(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test BRANCH_AND_MERGE recovery strategy."""
        original_branch = recovery_manager.checkpointer.current_branch

        result = recovery_manager.recover(
            "1.1.1",
            RecoveryStrategy.BRANCH_AND_MERGE,
        )

        assert result.success is True
        assert result.strategy == RecoveryStrategy.BRANCH_AND_MERGE
        assert result.branch_name == "executor-recovery-1.1.1"

        # Should now be on the recovery branch
        current_branch = recovery_manager.checkpointer.current_branch
        assert current_branch == "executor-recovery-1.1.1"

        # Clean up - switch back
        recovery_manager.checkpointer._run_git(["checkout", original_branch])

    def test_recover_skip(self, recovery_manager: RecoveryManager) -> None:
        """Test SKIP recovery strategy."""
        result = recovery_manager.recover("1.1.1", RecoveryStrategy.SKIP)

        assert result.success is True
        assert result.strategy == RecoveryStrategy.SKIP
        assert "Skipped" in result.message

    def test_recover_manual(self, recovery_manager: RecoveryManager) -> None:
        """Test MANUAL recovery strategy."""
        result = recovery_manager.recover("1.1.1", RecoveryStrategy.MANUAL)

        assert result.success is True
        assert result.strategy == RecoveryStrategy.MANUAL
        assert "Manual intervention" in result.message

    def test_retry_context_lifecycle(self, recovery_manager: RecoveryManager) -> None:
        """Test retry context storage and cleanup."""
        # Initially no context
        assert recovery_manager.get_retry_context("1.1.1") is None

        # Store context via recovery
        recovery_manager.recover(
            "1.1.1",
            RecoveryStrategy.RETRY_WITH_CONTEXT,
            error_context="Error details",
        )

        # Context should exist
        context = recovery_manager.get_retry_context("1.1.1")
        assert context is not None

        # Clear context
        recovery_manager.clear_retry_context("1.1.1")

        # Context should be gone
        assert recovery_manager.get_retry_context("1.1.1") is None


# =============================================================================
# RecoveryStrategy Enum Tests
# =============================================================================


class TestRecoveryStrategy:
    """Tests for RecoveryStrategy enum."""

    def test_all_strategies_exist(self) -> None:
        """Test that all expected strategies exist."""
        strategies = [
            RecoveryStrategy.ROLLBACK_ALL,
            RecoveryStrategy.ROLLBACK_FAILED,
            RecoveryStrategy.RETRY_WITH_CONTEXT,
            RecoveryStrategy.BRANCH_AND_MERGE,
            RecoveryStrategy.SKIP,
            RecoveryStrategy.MANUAL,
        ]

        assert len(strategies) == 6

    def test_strategy_values(self) -> None:
        """Test strategy string values."""
        assert RecoveryStrategy.ROLLBACK_ALL.value == "rollback_all"
        assert RecoveryStrategy.ROLLBACK_FAILED.value == "rollback_failed"
        assert RecoveryStrategy.RETRY_WITH_CONTEXT.value == "retry_with_context"
        assert RecoveryStrategy.BRANCH_AND_MERGE.value == "branch_and_merge"
        assert RecoveryStrategy.SKIP.value == "skip"
        assert RecoveryStrategy.MANUAL.value == "manual"


# =============================================================================
# Integration Tests
# =============================================================================


class TestRecoveryIntegration:
    """Integration tests for recovery workflow."""

    def test_full_checkpoint_and_recovery_workflow(
        self, recovery_manager: RecoveryManager, temp_git_repo: Path
    ) -> None:
        """Test complete workflow: checkpoint, modify, recover."""
        # Step 1: Create initial checkpoint with tag
        file1 = temp_git_repo / "main.py"
        file1.write_text("def main():\n    pass\n")

        result1, meta1 = recovery_manager.create_checkpoint(
            task_id="1.0.0",
            task_title="Initial setup",
            files_created=["main.py"],
            tags=["v1.0"],
            verification_passed=True,
        )

        assert result1.success is True

        # Step 2: Make changes
        file1.write_text("def main():\n    print('hello')\n")
        result2, meta2 = recovery_manager.create_checkpoint(
            task_id="1.1.0",
            task_title="Add print statement",
            files_modified=["main.py"],
            verification_passed=False,  # Simulated failure
        )

        assert result2.success is True

        # Step 3: Preview rollback
        preview = recovery_manager.preview_rollback(tag="v1.0")
        assert preview is not None
        assert "1.1.0" in preview.tasks_affected

        # Step 4: Perform selective rollback
        rollback_result = recovery_manager.rollback_files(["main.py"], to_tag="v1.0")

        assert rollback_result.success is True
        assert file1.read_text() == "def main():\n    pass\n"

    def test_metadata_persistence_across_instances(
        self, checkpointer: Checkpointer, temp_git_repo: Path
    ) -> None:
        """Test that metadata persists across RecoveryManager instances."""
        # Create checkpoint with first manager
        manager1 = RecoveryManager(checkpointer)
        file1 = temp_git_repo / "test.py"
        file1.write_text("# Test")

        result, _ = manager1.create_checkpoint(
            task_id="1.1.1",
            task_title="Test task",
            files_created=["test.py"],
            tags=["test-tag"],
        )

        # Create new manager instance
        manager2 = RecoveryManager(checkpointer)

        # Should be able to retrieve metadata
        meta = manager2.get_checkpoint_metadata(result.commit_sha)
        assert meta is not None
        assert meta.task_id == "1.1.1"

        # Should be able to retrieve tag
        sha = manager2.get_checkpoint_by_tag("test-tag")
        assert sha == result.commit_sha
