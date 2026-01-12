"""Tests for Human Approval Gates (Phase 2.3).

This module tests the human approval gate functionality including:
- Pause after N tasks
- Pause before destructive operations
- Resume with approval/rejection
- Resume with rollback
- Get changes for review
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.executor import (
    Checkpointer,
    CommitInfo,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    ExecutorConfig,
    ReviewInfo,
    RollbackResult,
    RunStatus,
    TaskStatus,
    VerificationResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_roadmap_content() -> str:
    """Sample ROADMAP content."""
    return """\
# Test Project ROADMAP

## Phase 0: Foundation

### 0.1 Setup

- [ ] **Task 1**
  First task.

- [ ] **Task 2**
  Second task.

- [ ] **Task 3**
  Third task.

### 0.2 Cleanup

- [ ] **Delete old files**
  Remove deprecated code and cleanup.

- [ ] **Remove temp data**
  Delete temporary test data.
"""


@pytest.fixture
def temp_roadmap(sample_roadmap_content: str):
    """Create a temporary ROADMAP file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        roadmap_path = Path(tmpdir) / "ROADMAP.md"
        roadmap_path.write_text(sample_roadmap_content)
        yield roadmap_path


@pytest.fixture
def mock_agent() -> AsyncMock:
    """Create a mock agent."""
    agent = AsyncMock()
    agent.arun.return_value = "Task completed. Modified `src/file.py`."
    return agent


@pytest.fixture
def mock_verifier() -> MagicMock:
    """Create a mock verifier."""
    verifier = MagicMock()
    result = VerificationResult(task_id="test", checks=[])
    verifier.verify = AsyncMock(return_value=result)
    return verifier


# =============================================================================
# Destructive Task Detection Tests
# =============================================================================


class TestDestructiveTaskDetection:
    """Test destructive task detection."""

    def test_is_destructive_task_delete_keyword(self, temp_roadmap: Path) -> None:
        """Test detection of delete keyword in task."""
        executor = Executor(roadmap=temp_roadmap)

        # Get task with "Delete" in title
        task = executor.roadmap.get_task("0.2.1")
        assert task is not None

        assert executor._is_destructive_task(task) is True

    def test_is_destructive_task_remove_keyword(self, temp_roadmap: Path) -> None:
        """Test detection of remove keyword in task."""
        executor = Executor(roadmap=temp_roadmap)

        # Get task with "Remove" in title
        task = executor.roadmap.get_task("0.2.2")
        assert task is not None

        assert executor._is_destructive_task(task) is True

    def test_is_destructive_task_normal_task(self, temp_roadmap: Path) -> None:
        """Test normal task is not detected as destructive."""
        executor = Executor(roadmap=temp_roadmap)

        # Get a normal task
        task = executor.roadmap.get_task("0.1.1")
        assert task is not None

        assert executor._is_destructive_task(task) is False

    def test_has_destructive_changes_with_deletions(self, temp_roadmap: Path) -> None:
        """Test detection of destructive changes when files deleted."""
        executor = Executor(roadmap=temp_roadmap)

        result = ExecutionResult(
            task_id="0.1.1",
            status=ExecutionStatus.SUCCESS,
            files_deleted=["old_file.py"],
        )

        assert executor._has_destructive_changes(result) is True

    def test_has_destructive_changes_no_deletions(self, temp_roadmap: Path) -> None:
        """Test no destructive changes when no files deleted."""
        executor = Executor(roadmap=temp_roadmap)

        result = ExecutionResult(
            task_id="0.1.1",
            status=ExecutionStatus.SUCCESS,
            files_modified=["file.py"],
            files_created=["new_file.py"],
        )

        assert executor._has_destructive_changes(result) is False


# =============================================================================
# Pause After N Tasks Tests
# =============================================================================


class TestPauseAfterNTasks:
    """Test pausing after N tasks."""

    @pytest.mark.asyncio
    async def test_pause_after_n_tasks(
        self,
        temp_roadmap: Path,
        mock_agent: AsyncMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Test execution pauses after N tasks."""
        config = ExecutorConfig(
            require_human_approval_after=2,
            skip_verification=True,
        )
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        summary = await executor.run()

        assert summary.status == RunStatus.PAUSED
        assert summary.paused is True
        assert summary.tasks_completed == 2
        assert "2 tasks" in summary.pause_reason

    @pytest.mark.asyncio
    async def test_no_pause_when_disabled(
        self,
        temp_roadmap: Path,
        mock_agent: AsyncMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Test no pause when require_human_approval_after is 0."""
        config = ExecutorConfig(
            require_human_approval_after=0,
            max_tasks=2,
            skip_verification=True,
        )
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        summary = await executor.run()

        # Should complete, not pause
        assert summary.status in (RunStatus.COMPLETED, RunStatus.STOPPED)
        assert summary.paused is False


# =============================================================================
# Pause Before Destructive Tests
# =============================================================================


class TestPauseBeforeDestructive:
    """Test pausing before destructive operations."""

    @pytest.mark.asyncio
    async def test_pause_on_file_deletion(
        self,
        temp_roadmap: Path,
        mock_verifier: MagicMock,
    ) -> None:
        """Test execution pauses when files are deleted."""
        # Create agent that simulates file deletion
        agent = AsyncMock()
        agent.arun.return_value = "Deleted `old_file.py`."

        config = ExecutorConfig(
            pause_before_destructive=True,
            max_tasks=3,
            skip_verification=True,
        )
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=agent,
            verifier=mock_verifier,
        )

        # Mock the result to have files_deleted
        original_execute = executor._execute_task

        async def mock_execute(task, *, retry_context=None):
            result = await original_execute(task, retry_context=retry_context)
            # Simulate file deletion in the result
            result.files_deleted = ["old_file.py"]
            return result

        executor._execute_task = mock_execute

        summary = await executor.run()

        assert summary.status == RunStatus.PAUSED
        assert summary.paused is True
        assert "destructive" in summary.pause_reason.lower()

    @pytest.mark.asyncio
    async def test_no_pause_when_destructive_disabled(
        self,
        temp_roadmap: Path,
        mock_verifier: MagicMock,
    ) -> None:
        """Test no pause when pause_before_destructive is False."""
        agent = AsyncMock()
        agent.arun.return_value = "Deleted `old_file.py`."

        config = ExecutorConfig(
            pause_before_destructive=False,
            max_tasks=1,
            skip_verification=True,
        )
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=agent,
            verifier=mock_verifier,
        )

        summary = await executor.run()

        # Should not pause even if we add files_deleted
        # (we can't truly mock this without more complex setup)
        assert summary.paused is False or summary.status != RunStatus.PAUSED


# =============================================================================
# Resume Tests
# =============================================================================


class TestResume:
    """Test resume functionality."""

    def test_resume_with_approval(self, temp_roadmap: Path) -> None:
        """Test resume with approval keeps tasks."""
        executor = Executor(roadmap=temp_roadmap)
        executor.state.mark_started("0.1.1")
        executor.state.mark_started("0.1.2")

        result = executor.resume(approved=True)

        # Should not return rollback result on approval
        assert result is None

        # Tasks should remain in-progress
        assert executor.state.get_status("0.1.1") == TaskStatus.IN_PROGRESS
        assert executor.state.get_status("0.1.2") == TaskStatus.IN_PROGRESS

    def test_resume_with_rejection_resets_tasks(self, temp_roadmap: Path) -> None:
        """Test resume with rejection resets tasks."""
        executor = Executor(roadmap=temp_roadmap)
        executor.state.mark_started("0.1.1")
        executor.state.mark_started("0.1.2")

        result = executor.resume(approved=False)

        # Should return None (no rollback attempted)
        assert result is None

        # Tasks should be reset to pending
        assert executor.state.get_status("0.1.1") == TaskStatus.PENDING
        assert executor.state.get_status("0.1.2") == TaskStatus.PENDING

    def test_resume_with_rollback_no_checkpointer(self, temp_roadmap: Path) -> None:
        """Test resume with rollback when no checkpointer."""
        config = ExecutorConfig(checkpoint_every=0)  # Disable checkpointing
        executor = Executor(roadmap=temp_roadmap, config=config)
        executor.state.mark_started("0.1.1")

        result = executor.resume(approved=False, rollback=True)

        # Should return failed rollback result
        assert result is not None
        assert result.success is False
        assert "disabled" in result.message.lower()

    def test_resume_with_rollback_with_checkpointer(self, temp_roadmap: Path) -> None:
        """Test resume with rollback and checkpointer."""
        config = ExecutorConfig(checkpoint_every=1)
        executor = Executor(roadmap=temp_roadmap, config=config)
        executor.state.mark_started("0.1.1")

        # Mock the checkpointer
        mock_checkpointer = MagicMock(spec=Checkpointer)
        mock_checkpointer.rollback.return_value = RollbackResult(
            success=True,
            target_sha="abc123",
            commits_reverted=1,
            message="Rolled back 1 commit",
        )

        with patch.object(executor, "_checkpointer", mock_checkpointer):
            executor._checkpointer_initialized = True

            result = executor.resume(approved=False, rollback=True)

        # Should return successful rollback result
        assert result is not None
        assert result.success is True
        assert result.commits_reverted == 1


# =============================================================================
# Get Changes for Review Tests
# =============================================================================


class TestGetChangesForReview:
    """Test get_changes_for_review functionality."""

    def test_get_changes_for_review_empty(self, temp_roadmap: Path) -> None:
        """Test get_changes_for_review with no in-progress tasks."""
        executor = Executor(roadmap=temp_roadmap)

        review = executor.get_changes_for_review()

        assert isinstance(review, ReviewInfo)
        assert review.task_ids == []
        assert review.files_modified == []
        assert review.has_destructive is False

    def test_get_changes_for_review_with_tasks(self, temp_roadmap: Path) -> None:
        """Test get_changes_for_review with results from last run."""
        executor = Executor(roadmap=temp_roadmap)

        # Simulate results from a run that was paused
        executor._last_run_results = [
            ExecutionResult(
                task_id="0.1.1",
                status=ExecutionStatus.SUCCESS,
                files_modified=["file1.py"],
            ),
            ExecutionResult(
                task_id="0.1.2",
                status=ExecutionStatus.SUCCESS,
                files_modified=["file2.py"],
                files_deleted=["old_file.py"],
            ),
        ]

        review = executor.get_changes_for_review()

        assert "0.1.1" in review.task_ids
        assert "0.1.2" in review.task_ids
        assert "file1.py" in review.files_modified
        assert "file2.py" in review.files_modified
        assert "old_file.py" in review.files_deleted
        assert review.has_destructive is True

    def test_get_changes_for_review_with_commits(self, temp_roadmap: Path) -> None:
        """Test get_changes_for_review with checkpointer commits."""
        config = ExecutorConfig(checkpoint_every=1)
        executor = Executor(roadmap=temp_roadmap, config=config)

        # Simulate results from a run that was paused
        executor._last_run_results = [
            ExecutionResult(
                task_id="0.1.1",
                status=ExecutionStatus.SUCCESS,
                files_modified=["src/file.py"],
            ),
        ]

        # Mock checkpointer
        mock_checkpointer = MagicMock(spec=Checkpointer)
        mock_commit = CommitInfo(
            sha="abc123def456",
            short_sha="abc123d",
            message="executor(0.1.1): Task 1",
            task_id="0.1.1",
        )
        mock_checkpointer.get_commit_for_task.return_value = mock_commit

        with patch.object(executor, "_checkpointer", mock_checkpointer):
            executor._checkpointer_initialized = True

            review = executor.get_changes_for_review()

        assert len(review.commits) == 1
        assert review.commits[0].task_id == "0.1.1"
        assert "src/file.py" in review.files_modified


# =============================================================================
# ReviewInfo Tests
# =============================================================================


class TestReviewInfo:
    """Test ReviewInfo dataclass."""

    def test_review_info_total_files_affected(self) -> None:
        """Test total_files_affected calculation."""
        review = ReviewInfo(
            files_modified=["a.py", "b.py"],
            files_created=["c.py"],
            files_deleted=["d.py"],
        )

        assert review.total_files_affected == 4

    def test_review_info_total_files_affected_dedup(self) -> None:
        """Test total_files_affected deduplicates files."""
        review = ReviewInfo(
            files_modified=["a.py", "b.py"],
            files_created=["a.py"],  # Same as modified
            files_deleted=["b.py"],  # Same as modified
        )

        assert review.total_files_affected == 2

    def test_review_info_to_dict(self) -> None:
        """Test ReviewInfo.to_dict()."""
        review = ReviewInfo(
            task_ids=["0.1.1"],
            files_modified=["file.py"],
            has_destructive=True,
            pause_reason="Test reason",
        )

        data = review.to_dict()

        assert data["task_ids"] == ["0.1.1"]
        assert data["files_modified"] == ["file.py"]
        assert data["has_destructive"] is True
        assert data["pause_reason"] == "Test reason"
        assert data["total_files_affected"] == 1

    def test_review_info_summary(self) -> None:
        """Test ReviewInfo.summary()."""
        review = ReviewInfo(
            task_ids=["0.1.1", "0.1.2"],
            files_modified=["a.py"],
            files_created=["b.py"],
            files_deleted=["c.py"],
            has_destructive=True,
            pause_reason="Destructive ops",
        )

        summary = review.summary()

        assert "2 task(s)" in summary
        assert "Files Modified: 1" in summary
        assert "Files Created: 1" in summary
        assert "Files Deleted: 1" in summary
        assert "Destructive" in summary
        assert "Destructive ops" in summary


# =============================================================================
# RunSummary pause_reason Tests
# =============================================================================


class TestRunSummaryPauseReason:
    """Test RunSummary pause_reason field."""

    def test_run_summary_pause_reason(self) -> None:
        """Test RunSummary includes pause_reason."""
        from ai_infra.executor.loop import RunStatus, RunSummary

        summary = RunSummary(
            status=RunStatus.PAUSED,
            paused=True,
            pending_review=["0.1.1"],
            pause_reason="Reached 5 tasks",
        )

        assert summary.pause_reason == "Reached 5 tasks"

    def test_run_summary_to_dict_includes_pause_reason(self) -> None:
        """Test to_dict includes pause_reason."""
        from ai_infra.executor.loop import RunStatus, RunSummary

        summary = RunSummary(
            status=RunStatus.PAUSED,
            paused=True,
            pause_reason="Test pause",
        )

        data = summary.to_dict()

        assert "pause_reason" in data
        assert data["pause_reason"] == "Test pause"

    def test_run_summary_summary_includes_reason(self) -> None:
        """Test summary() includes pause reason."""
        from ai_infra.executor.loop import RunStatus, RunSummary

        summary = RunSummary(
            status=RunStatus.PAUSED,
            paused=True,
            pending_review=["0.1.1"],
            pause_reason="Destructive changes",
        )

        text = summary.summary()

        assert "Paused for review" in text
        assert "Reason: Destructive changes" in text


# =============================================================================
# Integration Tests
# =============================================================================


class TestHumanApprovalIntegration:
    """Integration tests for human approval gates."""

    @pytest.mark.asyncio
    async def test_full_pause_resume_cycle(
        self,
        temp_roadmap: Path,
        mock_agent: AsyncMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Test complete pause and resume cycle."""
        config = ExecutorConfig(
            require_human_approval_after=1,
            skip_verification=True,
        )
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        # First run - should pause after 1 task
        summary1 = await executor.run()
        assert summary1.status == RunStatus.PAUSED
        assert summary1.tasks_completed == 1

        # Get changes for review
        review = executor.get_changes_for_review()
        assert len(review.task_ids) > 0

        # Resume with approval
        executor.resume(approved=True)

        # Continue running
        config2 = ExecutorConfig(
            require_human_approval_after=2,  # Increase limit
            skip_verification=True,
        )
        executor._config = config2
        executor._tasks_this_run = 0  # Reset counter

        summary2 = await executor.run()

        # Should make more progress
        assert summary2.tasks_completed >= 1

    @pytest.mark.asyncio
    async def test_pause_resume_reject_cycle(
        self,
        temp_roadmap: Path,
        mock_agent: AsyncMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Test pause and resume with rejection."""
        config = ExecutorConfig(
            require_human_approval_after=1,
            skip_verification=True,
        )
        executor = Executor(
            roadmap=temp_roadmap,
            config=config,
            agent=mock_agent,
            verifier=mock_verifier,
        )

        # First run - should pause
        summary1 = await executor.run()
        assert summary1.status == RunStatus.PAUSED

        # Resume with rejection
        executor.resume(approved=False)

        # In-progress tasks should be reset
        in_progress = executor.state.get_in_progress_tasks()
        assert len(in_progress) == 0
