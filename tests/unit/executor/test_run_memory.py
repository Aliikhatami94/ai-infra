"""Tests for the run_memory module (Phase 5.8.1).

This module tests:
- FileAction enum
- TaskOutcome dataclass and serialization
- RunMemory class with context generation and truncation
"""

from __future__ import annotations

from pathlib import Path

from ai_infra.executor.run_memory import (
    FileAction,
    RunMemory,
    TaskOutcome,
    _count_tokens_simple,
)

# =============================================================================
# Test FileAction Enum
# =============================================================================


class TestFileAction:
    """Tests for FileAction enum."""

    def test_file_action_values(self):
        """Test all file actions have correct values."""
        assert FileAction.CREATED.value == "created"
        assert FileAction.MODIFIED.value == "modified"
        assert FileAction.DELETED.value == "deleted"

    def test_file_action_is_string_enum(self):
        """Test FileAction can be used as string."""
        assert str(FileAction.CREATED) == "FileAction.CREATED"
        assert FileAction.CREATED == "created"

    def test_file_action_from_string(self):
        """Test creating FileAction from string."""
        assert FileAction("created") == FileAction.CREATED
        assert FileAction("modified") == FileAction.MODIFIED
        assert FileAction("deleted") == FileAction.DELETED


# =============================================================================
# Test TaskOutcome
# =============================================================================


class TestTaskOutcome:
    """Tests for TaskOutcome dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        outcome = TaskOutcome(
            task_id="1.1",
            title="Create utils module",
            status="completed",
        )
        assert outcome.task_id == "1.1"
        assert outcome.title == "Create utils module"
        assert outcome.status == "completed"
        assert outcome.files == {}
        assert outcome.key_decisions == []
        assert outcome.summary == ""
        assert outcome.duration_seconds == 0.0
        assert outcome.tokens_used == 0

    def test_create_full(self):
        """Test creating with all args."""
        files = {
            Path("src/utils.js"): FileAction.CREATED,
            Path("src/index.js"): FileAction.MODIFIED,
        }
        outcome = TaskOutcome(
            task_id="1.2",
            title="Create greeter module",
            status="completed",
            files=files,
            key_decisions=["Used ES6 modules", "Added type hints"],
            summary="Created greeter.js importing from utils",
            duration_seconds=15.5,
            tokens_used=2000,
        )
        assert len(outcome.files) == 2
        assert len(outcome.key_decisions) == 2
        assert outcome.duration_seconds == 15.5
        assert outcome.tokens_used == 2000

    def test_to_context_line_with_summary(self):
        """Test context line formatting uses summary."""
        outcome = TaskOutcome(
            task_id="1.1",
            title="Create utils",
            status="completed",
            files={Path("src/utils.js"): FileAction.CREATED},
            summary="Created formatName utility function",
        )
        line = outcome.to_context_line()
        assert "**1.1**" in line
        assert "Created formatName utility function" in line
        assert "utils.js (created)" in line

    def test_to_context_line_falls_back_to_title(self):
        """Test context line uses title when no summary."""
        outcome = TaskOutcome(
            task_id="1.1",
            title="Create utils module",
            status="completed",
        )
        line = outcome.to_context_line()
        assert "Create utils module" in line

    def test_to_context_line_truncates_many_files(self):
        """Test context line truncates when more than 3 files."""
        files = {Path(f"src/file{i}.js"): FileAction.CREATED for i in range(5)}
        outcome = TaskOutcome(
            task_id="1.1",
            title="Create files",
            status="completed",
            files=files,
            summary="Created many files",
        )
        line = outcome.to_context_line()
        assert "(+2 more)" in line

    def test_to_context_line_shows_none_when_no_files(self):
        """Test context line shows 'none' when no files."""
        outcome = TaskOutcome(
            task_id="1.1",
            title="Documentation update",
            status="completed",
        )
        line = outcome.to_context_line()
        assert "Files: none" in line

    def test_to_dict_serializes_correctly(self):
        """Test serialization to dictionary."""
        files = {Path("src/utils.js"): FileAction.CREATED}
        outcome = TaskOutcome(
            task_id="1.1",
            title="Create utils",
            status="completed",
            files=files,
            key_decisions=["Used ES6"],
            summary="Created utils",
            duration_seconds=10.0,
            tokens_used=1500,
        )
        data = outcome.to_dict()

        assert data["task_id"] == "1.1"
        assert data["title"] == "Create utils"
        assert data["status"] == "completed"
        assert data["files"] == {"src/utils.js": "created"}
        assert data["key_decisions"] == ["Used ES6"]
        assert data["summary"] == "Created utils"
        assert data["duration_seconds"] == 10.0
        assert data["tokens_used"] == 1500

    def test_from_dict_deserializes_correctly(self):
        """Test deserialization from dictionary."""
        data = {
            "task_id": "2.1",
            "title": "Add tests",
            "status": "completed",
            "files": {"tests/test_utils.js": "created"},
            "key_decisions": ["Used node:test"],
            "summary": "Added unit tests",
            "duration_seconds": 8.5,
            "tokens_used": 1200,
        }
        outcome = TaskOutcome.from_dict(data)

        assert outcome.task_id == "2.1"
        assert outcome.title == "Add tests"
        assert Path("tests/test_utils.js") in outcome.files
        assert outcome.files[Path("tests/test_utils.js")] == FileAction.CREATED
        assert outcome.key_decisions == ["Used node:test"]
        assert outcome.duration_seconds == 8.5

    def test_roundtrip_serialization(self):
        """Test to_dict -> from_dict roundtrip."""
        original = TaskOutcome(
            task_id="3.1",
            title="Refactor code",
            status="failed",
            files={
                Path("src/old.js"): FileAction.DELETED,
                Path("src/new.js"): FileAction.CREATED,
            },
            key_decisions=["Split into modules"],
            summary="Refactored but tests failed",
            duration_seconds=25.0,
            tokens_used=3000,
        )
        data = original.to_dict()
        restored = TaskOutcome.from_dict(data)

        assert restored.task_id == original.task_id
        assert restored.status == original.status
        assert len(restored.files) == len(original.files)
        assert restored.tokens_used == original.tokens_used


# =============================================================================
# Test Token Counting
# =============================================================================


class TestTokenCounting:
    """Tests for simple token estimation."""

    def test_empty_string(self):
        """Test empty string returns 0 tokens."""
        assert _count_tokens_simple("") == 0

    def test_short_string(self):
        """Test short string token count."""
        # "Hello" = 5 chars, (5 + 3) // 4 = 2 tokens
        assert _count_tokens_simple("Hello") == 2

    def test_medium_string(self):
        """Test medium string token count."""
        # 16 chars, (16 + 3) // 4 = 4 tokens
        text = "Hello, world!!!"  # 16 chars including comma/space
        assert _count_tokens_simple(text) == (len(text) + 3) // 4

    def test_long_string(self):
        """Test long string estimates conservatively."""
        text = "a" * 1000
        tokens = _count_tokens_simple(text)
        # Should be approximately 250 tokens (1000/4)
        assert 250 <= tokens <= 260


# =============================================================================
# Test RunMemory
# =============================================================================


class TestRunMemory:
    """Tests for RunMemory class."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        memory = RunMemory(run_id="run_001")
        assert memory.run_id == "run_001"
        assert memory.outcomes == []
        assert memory.roadmap_path == ""
        assert memory.started_at  # Should be set automatically

    def test_create_with_roadmap(self):
        """Test creating with roadmap path."""
        memory = RunMemory(
            run_id="run_002",
            roadmap_path="/path/to/ROADMAP.md",
        )
        assert memory.roadmap_path == "/path/to/ROADMAP.md"

    def test_add_outcome_appends(self):
        """Test add_outcome appends to list."""
        memory = RunMemory(run_id="run_001")
        outcome1 = TaskOutcome(task_id="1.1", title="Task 1", status="completed")
        outcome2 = TaskOutcome(task_id="1.2", title="Task 2", status="completed")

        memory.add_outcome(outcome1)
        assert len(memory.outcomes) == 1

        memory.add_outcome(outcome2)
        assert len(memory.outcomes) == 2
        assert memory.outcomes[0].task_id == "1.1"
        assert memory.outcomes[1].task_id == "1.2"

    def test_get_outcome_finds_task(self):
        """Test get_outcome returns correct task."""
        memory = RunMemory(run_id="run_001")
        memory.add_outcome(TaskOutcome(task_id="1.1", title="Task 1", status="completed"))
        memory.add_outcome(TaskOutcome(task_id="1.2", title="Task 2", status="failed"))

        outcome = memory.get_outcome("1.2")
        assert outcome is not None
        assert outcome.title == "Task 2"
        assert outcome.status == "failed"

    def test_get_outcome_returns_none_for_missing(self):
        """Test get_outcome returns None for non-existent task."""
        memory = RunMemory(run_id="run_001")
        memory.add_outcome(TaskOutcome(task_id="1.1", title="Task 1", status="completed"))

        assert memory.get_outcome("9.9") is None

    def test_get_completed_files_maps_correctly(self):
        """Test get_completed_files returns correct mapping."""
        memory = RunMemory(run_id="run_001")
        memory.add_outcome(
            TaskOutcome(
                task_id="1.1",
                title="Create utils",
                status="completed",
                files={Path("src/utils.js"): FileAction.CREATED},
            )
        )
        memory.add_outcome(
            TaskOutcome(
                task_id="1.2",
                title="Create greeter",
                status="completed",
                files={
                    Path("src/greeter.js"): FileAction.CREATED,
                    Path("src/utils.js"): FileAction.MODIFIED,  # Modified by later task
                },
            )
        )

        files = memory.get_completed_files()
        assert len(files) == 2
        # utils.js should map to task 1.2 (most recent)
        assert files[Path("src/utils.js")].task_id == "1.2"
        assert files[Path("src/greeter.js")].task_id == "1.2"

    def test_get_files_by_action_filters_correctly(self):
        """Test get_files_by_action returns correct files."""
        memory = RunMemory(run_id="run_001")
        memory.add_outcome(
            TaskOutcome(
                task_id="1.1",
                title="Refactor",
                status="completed",
                files={
                    Path("src/new.js"): FileAction.CREATED,
                    Path("src/old.js"): FileAction.DELETED,
                    Path("src/config.js"): FileAction.MODIFIED,
                },
            )
        )

        created = memory.get_files_by_action(FileAction.CREATED)
        assert len(created) == 1
        assert Path("src/new.js") in created

        deleted = memory.get_files_by_action(FileAction.DELETED)
        assert len(deleted) == 1
        assert Path("src/old.js") in deleted

    def test_get_context_empty_when_no_outcomes(self):
        """Test get_context returns empty string when no outcomes."""
        memory = RunMemory(run_id="run_001")
        assert memory.get_context() == ""

    def test_get_context_includes_all_tasks(self):
        """Test get_context includes all completed tasks."""
        memory = RunMemory(run_id="run_001")
        memory.add_outcome(
            TaskOutcome(
                task_id="1.1",
                title="Create utils",
                status="completed",
                summary="Created formatName function",
            )
        )
        memory.add_outcome(
            TaskOutcome(
                task_id="1.2",
                title="Create greeter",
                status="completed",
                summary="Created greet function",
            )
        )

        context = memory.get_context()
        assert "## Previously Completed Tasks (This Run)" in context
        assert "**1.1**" in context
        assert "**1.2**" in context
        assert "formatName" in context
        assert "greet" in context

    def test_get_context_includes_file_map(self):
        """Test get_context includes file map section."""
        memory = RunMemory(run_id="run_001")
        memory.add_outcome(
            TaskOutcome(
                task_id="1.1",
                title="Create utils",
                status="completed",
                files={Path("src/utils.js"): FileAction.CREATED},
            )
        )

        context = memory.get_context()
        assert "### Files Created/Modified" in context
        assert "`src/utils.js`" in context
        assert "(created" in context

    def test_get_context_respects_token_budget(self):
        """Test get_context truncates when over budget."""
        memory = RunMemory(run_id="run_001")

        # Add many tasks to exceed budget
        for i in range(50):
            memory.add_outcome(
                TaskOutcome(
                    task_id=f"1.{i}",
                    title=f"Task {i} with a very long description that takes up tokens",
                    status="completed",
                    summary=f"Completed task {i} successfully with detailed outcome",
                    files={Path(f"src/file{i}.js"): FileAction.CREATED},
                )
            )

        # Very small budget
        context = memory.get_context(max_tokens=200)

        # Should be truncated
        tokens = _count_tokens_simple(context)
        assert tokens <= 250  # Some buffer for truncation

        # Should have truncation note
        assert "earlier tasks omitted" in context or len(context) < 2000

    def test_get_context_with_custom_max_tokens(self):
        """Test get_context respects custom max_tokens."""
        memory = RunMemory(run_id="run_001")
        memory.add_outcome(
            TaskOutcome(
                task_id="1.1",
                title="Short task",
                status="completed",
            )
        )

        # Large budget - should include everything
        context = memory.get_context(max_tokens=10000)
        assert "**1.1**" in context

    def test_get_stats_counts_correctly(self):
        """Test get_stats returns correct counts."""
        memory = RunMemory(run_id="run_001")
        memory.add_outcome(
            TaskOutcome(
                task_id="1.1",
                title="T1",
                status="completed",
                duration_seconds=10.0,
                tokens_used=1000,
            )
        )
        memory.add_outcome(
            TaskOutcome(
                task_id="1.2",
                title="T2",
                status="completed",
                duration_seconds=15.0,
                tokens_used=1500,
            )
        )
        memory.add_outcome(
            TaskOutcome(
                task_id="1.3",
                title="T3",
                status="failed",
                duration_seconds=5.0,
                tokens_used=500,
            )
        )
        memory.add_outcome(
            TaskOutcome(
                task_id="1.4",
                title="T4",
                status="skipped",
                duration_seconds=0.0,
                tokens_used=0,
            )
        )

        stats = memory.get_stats()
        assert stats["total"] == 4
        assert stats["completed"] == 2
        assert stats["failed"] == 1
        assert stats["skipped"] == 1
        assert stats["total_tokens"] == 3000
        assert stats["total_duration_seconds"] == 30

    def test_to_dict_serializes_correctly(self):
        """Test to_dict produces correct summary."""
        memory = RunMemory(
            run_id="run_001",
            roadmap_path="/path/ROADMAP.md",
        )
        memory.add_outcome(
            TaskOutcome(
                task_id="1.1",
                title="Create file",
                status="completed",
                files={Path("src/file.js"): FileAction.CREATED},
                duration_seconds=10.0,
                tokens_used=1000,
            )
        )

        data = memory.to_dict()
        assert data["run_id"] == "run_001"
        assert data["roadmap_path"] == "/path/ROADMAP.md"
        assert data["task_count"] == 1
        assert data["completed"] == 1
        assert data["total_tokens"] == 1000
        assert "src/file.js" in data["files_created"]

    def test_to_full_dict_includes_outcomes(self):
        """Test to_full_dict includes all outcome details."""
        memory = RunMemory(run_id="run_001")
        memory.add_outcome(
            TaskOutcome(
                task_id="1.1",
                title="Create utils",
                status="completed",
                summary="Created utils module",
            )
        )

        data = memory.to_full_dict()
        assert "outcomes" in data
        assert len(data["outcomes"]) == 1
        assert data["outcomes"][0]["task_id"] == "1.1"
        assert data["outcomes"][0]["summary"] == "Created utils module"

    def test_from_dict_deserializes_correctly(self):
        """Test from_dict reconstructs RunMemory."""
        data = {
            "run_id": "run_002",
            "started_at": "2025-01-07T10:00:00+00:00",
            "roadmap_path": "/path/ROADMAP.md",
            "outcomes": [
                {
                    "task_id": "1.1",
                    "title": "Create utils",
                    "status": "completed",
                    "files": {"src/utils.js": "created"},
                    "key_decisions": ["Used ES6"],
                    "summary": "Created utils",
                    "duration_seconds": 10.0,
                    "tokens_used": 1000,
                }
            ],
        }
        memory = RunMemory.from_dict(data)

        assert memory.run_id == "run_002"
        assert memory.roadmap_path == "/path/ROADMAP.md"
        assert len(memory.outcomes) == 1
        assert memory.outcomes[0].task_id == "1.1"
        assert Path("src/utils.js") in memory.outcomes[0].files

    def test_roundtrip_serialization(self):
        """Test to_full_dict -> from_dict roundtrip."""
        original = RunMemory(
            run_id="run_003",
            roadmap_path="/test/ROADMAP.md",
        )
        original.add_outcome(
            TaskOutcome(
                task_id="2.1",
                title="Add feature",
                status="completed",
                files={
                    Path("src/feature.js"): FileAction.CREATED,
                    Path("src/index.js"): FileAction.MODIFIED,
                },
                key_decisions=["Used factory pattern"],
                summary="Added new feature module",
                duration_seconds=20.0,
                tokens_used=2500,
            )
        )

        data = original.to_full_dict()
        restored = RunMemory.from_dict(data)

        assert restored.run_id == original.run_id
        assert restored.roadmap_path == original.roadmap_path
        assert len(restored.outcomes) == len(original.outcomes)
        assert restored.outcomes[0].task_id == original.outcomes[0].task_id
        assert len(restored.outcomes[0].files) == 2


# =============================================================================
# Test Context Truncation
# =============================================================================


class TestContextTruncation:
    """Tests for context truncation behavior."""

    def test_truncation_keeps_recent_tasks(self):
        """Test truncation prioritizes recent tasks."""
        memory = RunMemory(run_id="run_001")

        # Add many tasks
        for i in range(20):
            memory.add_outcome(
                TaskOutcome(
                    task_id=f"1.{i}",
                    title=f"Task {i}",
                    status="completed",
                    summary=f"Completed task number {i}",
                )
            )

        # Small budget
        context = memory.get_context(max_tokens=300)

        # Recent tasks should be present
        assert "1.19" in context  # Most recent
        # Old tasks may be omitted
        # (exact behavior depends on truncation algorithm)

    def test_truncation_shows_omission_note(self):
        """Test truncation adds note about omitted tasks."""
        memory = RunMemory(run_id="run_001")

        for i in range(30):
            memory.add_outcome(
                TaskOutcome(
                    task_id=f"1.{i}",
                    title=f"Task {i} with detailed description",
                    status="completed",
                    summary=f"Completed task {i} with a longer summary text",
                )
            )

        context = memory.get_context(max_tokens=200)

        # Should mention omitted tasks
        assert "omitted" in context.lower() or "..." in context

    def test_truncation_still_shows_files(self):
        """Test truncation still includes file map."""
        memory = RunMemory(run_id="run_001")

        for i in range(20):
            memory.add_outcome(
                TaskOutcome(
                    task_id=f"1.{i}",
                    title=f"Task {i}",
                    status="completed",
                    files={Path(f"src/file{i}.js"): FileAction.CREATED},
                )
            )

        context = memory.get_context(max_tokens=400)

        # Should still have files section
        assert "Files Created/Modified" in context or "files" in context.lower()


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_task_id(self):
        """Test handling empty task_id."""
        outcome = TaskOutcome(task_id="", title="No ID", status="completed")
        line = outcome.to_context_line()
        assert "****" in line  # Empty ID shows as ****:

    def test_unicode_in_summary(self):
        """Test handling unicode characters."""
        outcome = TaskOutcome(
            task_id="1.1",
            title="Unicode test",
            status="completed",
            summary="Created 文件 with émojis 🎉",
        )
        line = outcome.to_context_line()
        assert "文件" in line
        assert "🎉" in line

    def test_very_long_summary_in_context(self):
        """Test handling very long summary."""
        long_summary = "A" * 1000
        outcome = TaskOutcome(
            task_id="1.1",
            title="Long summary",
            status="completed",
            summary=long_summary,
        )
        # Should not raise
        line = outcome.to_context_line()
        assert "A" * 100 in line  # At least some of it

    def test_path_with_spaces(self):
        """Test handling paths with spaces."""
        outcome = TaskOutcome(
            task_id="1.1",
            title="Create file",
            status="completed",
            files={Path("src/my file.js"): FileAction.CREATED},
        )
        data = outcome.to_dict()
        restored = TaskOutcome.from_dict(data)
        assert Path("src/my file.js") in restored.files

    def test_multiple_file_actions_same_task(self):
        """Test task with multiple different file actions."""
        outcome = TaskOutcome(
            task_id="1.1",
            title="Refactor",
            status="completed",
            files={
                Path("src/new.js"): FileAction.CREATED,
                Path("src/updated.js"): FileAction.MODIFIED,
                Path("src/old.js"): FileAction.DELETED,
            },
        )
        assert len(outcome.files) == 3
        line = outcome.to_context_line()
        # Should show multiple actions
        assert "created" in line.lower() or "modified" in line.lower()
