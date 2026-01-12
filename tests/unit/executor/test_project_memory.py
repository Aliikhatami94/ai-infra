"""Tests for the project_memory module (Phase 5.8.2).

This module tests:
- FileInfo dataclass and serialization
- RunSummary dataclass and serialization
- ProjectMemory class with persistence, context generation, and run updates
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_infra.executor.project_memory import (
    FileInfo,
    ProjectMemory,
    RunSummary,
    _count_tokens_simple,
)
from ai_infra.executor.run_memory import (
    FileAction,
    RunMemory,
    TaskOutcome,
)

# =============================================================================
# Test FileInfo
# =============================================================================


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        info = FileInfo(path="src/utils.py")
        assert info.path == "src/utils.py"
        assert info.purpose == ""
        assert info.created_by_task is None
        assert info.last_modified_by_task is None
        assert info.imports == []
        assert info.exports == []

    def test_create_full(self):
        """Test creating with all args."""
        info = FileInfo(
            path="src/utils.py",
            purpose="Utility functions for formatting",
            created_by_task="1.1",
            last_modified_by_task="2.3",
            imports=["os", "sys"],
            exports=["format_name", "format_date"],
        )
        assert info.path == "src/utils.py"
        assert info.purpose == "Utility functions for formatting"
        assert info.created_by_task == "1.1"
        assert info.last_modified_by_task == "2.3"
        assert info.imports == ["os", "sys"]
        assert info.exports == ["format_name", "format_date"]

    def test_to_dict(self):
        """Test serialization to dictionary."""
        info = FileInfo(
            path="src/main.py",
            purpose="Main entry point",
            created_by_task="1.0",
        )
        data = info.to_dict()
        assert data["path"] == "src/main.py"
        assert data["purpose"] == "Main entry point"
        assert data["created_by_task"] == "1.0"
        assert data["last_modified_by_task"] is None

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "path": "src/helpers.py",
            "purpose": "Helper functions",
            "created_by_task": "2.1",
            "last_modified_by_task": "3.2",
            "imports": ["json"],
            "exports": ["parse_json"],
        }
        info = FileInfo.from_dict(data)
        assert info.path == "src/helpers.py"
        assert info.purpose == "Helper functions"
        assert info.created_by_task == "2.1"
        assert info.last_modified_by_task == "3.2"
        assert info.imports == ["json"]
        assert info.exports == ["parse_json"]

    def test_from_dict_with_missing_optional_fields(self):
        """Test deserialization handles missing optional fields."""
        data = {"path": "src/module.py"}
        info = FileInfo.from_dict(data)
        assert info.path == "src/module.py"
        assert info.purpose == ""
        assert info.created_by_task is None
        assert info.imports == []

    def test_roundtrip_serialization(self):
        """Test to_dict and from_dict are inverses."""
        original = FileInfo(
            path="test.py",
            purpose="Test file",
            created_by_task="1.1",
            last_modified_by_task="1.2",
            imports=["pytest"],
            exports=["test_func"],
        )
        data = original.to_dict()
        restored = FileInfo.from_dict(data)
        assert restored.path == original.path
        assert restored.purpose == original.purpose
        assert restored.created_by_task == original.created_by_task
        assert restored.last_modified_by_task == original.last_modified_by_task


# =============================================================================
# Test RunSummary
# =============================================================================


class TestRunSummary:
    """Tests for RunSummary dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal args."""
        summary = RunSummary(
            run_id="run_123",
            timestamp="2026-01-07T10:00:00Z",
        )
        assert summary.run_id == "run_123"
        assert summary.timestamp == "2026-01-07T10:00:00Z"
        assert summary.roadmap_path == ""
        assert summary.tasks_completed == 0
        assert summary.tasks_failed == 0
        assert summary.key_files_created == []
        assert summary.lessons_learned == []

    def test_create_full(self):
        """Test creating with all args."""
        summary = RunSummary(
            run_id="run_456",
            timestamp="2026-01-07T11:30:00Z",
            roadmap_path="/path/to/ROADMAP.md",
            tasks_completed=5,
            tasks_failed=1,
            key_files_created=["src/main.py", "src/utils.py"],
            lessons_learned=["Import paths must be relative"],
        )
        assert summary.run_id == "run_456"
        assert summary.tasks_completed == 5
        assert summary.tasks_failed == 1
        assert len(summary.key_files_created) == 2
        assert len(summary.lessons_learned) == 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        summary = RunSummary(
            run_id="run_789",
            timestamp="2026-01-07T12:00:00Z",
            tasks_completed=3,
        )
        data = summary.to_dict()
        assert data["run_id"] == "run_789"
        assert data["timestamp"] == "2026-01-07T12:00:00Z"
        assert data["tasks_completed"] == 3

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "run_id": "run_abc",
            "timestamp": "2026-01-07T13:00:00Z",
            "roadmap_path": "/test/ROADMAP.md",
            "tasks_completed": 10,
            "tasks_failed": 2,
            "key_files_created": ["a.py", "b.py"],
            "lessons_learned": ["lesson1"],
        }
        summary = RunSummary.from_dict(data)
        assert summary.run_id == "run_abc"
        assert summary.roadmap_path == "/test/ROADMAP.md"
        assert summary.tasks_completed == 10
        assert summary.tasks_failed == 2

    def test_from_dict_with_missing_fields(self):
        """Test deserialization handles missing optional fields."""
        data = {
            "run_id": "run_minimal",
            "timestamp": "2026-01-07T14:00:00Z",
        }
        summary = RunSummary.from_dict(data)
        assert summary.run_id == "run_minimal"
        assert summary.roadmap_path == ""
        assert summary.tasks_completed == 0
        assert summary.key_files_created == []

    def test_roundtrip_serialization(self):
        """Test to_dict and from_dict are inverses."""
        original = RunSummary(
            run_id="run_test",
            timestamp="2026-01-07T15:00:00Z",
            roadmap_path="/ROADMAP.md",
            tasks_completed=7,
            tasks_failed=0,
            key_files_created=["x.py"],
            lessons_learned=["test lesson"],
        )
        data = original.to_dict()
        restored = RunSummary.from_dict(data)
        assert restored.run_id == original.run_id
        assert restored.tasks_completed == original.tasks_completed
        assert restored.key_files_created == original.key_files_created


# =============================================================================
# Test ProjectMemory - Creation and Loading
# =============================================================================


class TestProjectMemoryCreation:
    """Tests for ProjectMemory creation and loading."""

    def test_create_new(self, tmp_path: Path):
        """Test creating new project memory."""
        memory = ProjectMemory(project_root=tmp_path)
        assert memory.project_root == tmp_path
        assert memory.project_type == "unknown"
        assert memory.detected_frameworks == []
        assert memory.key_files == {}
        assert memory.conventions == []
        assert memory.run_history == []

    def test_load_creates_new_if_missing(self, tmp_path: Path):
        """Test load returns new instance if file doesn't exist."""
        memory = ProjectMemory.load(tmp_path)
        assert memory.project_root == tmp_path
        assert memory.project_type == "unknown"
        assert len(memory.key_files) == 0

    def test_load_reads_existing(self, tmp_path: Path):
        """Test load reads existing memory file."""
        # Create memory file manually
        executor_dir = tmp_path / ".executor"
        executor_dir.mkdir()
        memory_file = executor_dir / "project-memory.json"
        memory_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "project_type": "python",
                    "detected_frameworks": ["fastapi"],
                    "key_files": {
                        "main.py": {
                            "path": "main.py",
                            "purpose": "Entry point",
                        }
                    },
                    "conventions": ["Use type hints"],
                    "run_history": [],
                }
            )
        )

        memory = ProjectMemory.load(tmp_path)
        assert memory.project_type == "python"
        assert memory.detected_frameworks == ["fastapi"]
        assert "main.py" in memory.key_files
        assert memory.conventions == ["Use type hints"]

    def test_load_handles_corrupted_file(self, tmp_path: Path):
        """Test load returns new instance if file is corrupted."""
        executor_dir = tmp_path / ".executor"
        executor_dir.mkdir()
        memory_file = executor_dir / "project-memory.json"
        memory_file.write_text("not valid json {{{")

        memory = ProjectMemory.load(tmp_path)
        # Should return fresh instance
        assert memory.project_type == "unknown"
        assert len(memory.key_files) == 0

    def test_load_handles_invalid_schema(self, tmp_path: Path):
        """Test load returns new instance if schema is wrong."""
        executor_dir = tmp_path / ".executor"
        executor_dir.mkdir()
        memory_file = executor_dir / "project-memory.json"
        memory_file.write_text('{"unexpected": "schema"}')

        memory = ProjectMemory.load(tmp_path)
        # Should return fresh instance (graceful handling)
        assert memory.project_type == "unknown"


# =============================================================================
# Test ProjectMemory - Saving
# =============================================================================


class TestProjectMemorySaving:
    """Tests for ProjectMemory persistence."""

    def test_save_creates_directory(self, tmp_path: Path):
        """Test save creates .executor directory."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.project_type = "node"

        saved_path = memory.save()

        assert saved_path.exists()
        assert saved_path.parent.name == ".executor"

    def test_save_writes_json(self, tmp_path: Path):
        """Test save writes valid JSON."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.project_type = "python"
        memory.detected_frameworks = ["django"]
        memory.add_convention("Use Black formatter")

        memory.save()

        memory_path = tmp_path / ".executor" / "project-memory.json"
        data = json.loads(memory_path.read_text())
        assert data["project_type"] == "python"
        assert data["detected_frameworks"] == ["django"]
        assert "Use Black formatter" in data["conventions"]

    def test_save_includes_version(self, tmp_path: Path):
        """Test saved file includes version number."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.save()

        memory_path = tmp_path / ".executor" / "project-memory.json"
        data = json.loads(memory_path.read_text())
        assert data["version"] == 1

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Test saved data can be loaded correctly."""
        # Create and populate memory
        memory = ProjectMemory(project_root=tmp_path)
        memory.project_type = "rust"
        memory.add_framework("tokio")
        memory.add_convention("Use async/await")
        memory.key_files["src/main.rs"] = FileInfo(
            path="src/main.rs",
            purpose="Main entry point",
            created_by_task="1.0",
        )
        memory.run_history.append(
            RunSummary(
                run_id="run_1",
                timestamp="2026-01-07T10:00:00Z",
                tasks_completed=5,
            )
        )
        memory.save()

        # Load and verify
        loaded = ProjectMemory.load(tmp_path)
        assert loaded.project_type == "rust"
        assert "tokio" in loaded.detected_frameworks
        assert "Use async/await" in loaded.conventions
        assert "src/main.rs" in loaded.key_files
        assert len(loaded.run_history) == 1
        assert loaded.run_history[0].tasks_completed == 5


# =============================================================================
# Test ProjectMemory - Update from Run
# =============================================================================


class TestProjectMemoryUpdateFromRun:
    """Tests for updating project memory from run memory."""

    def _create_run_memory(
        self,
        project_root: Path,
        run_id: str = "test_run",
        tasks: list[tuple[str, str, str, dict]] | None = None,
    ) -> RunMemory:
        """Helper to create a RunMemory with task outcomes.

        Args:
            project_root: Project root for relative paths.
            run_id: Run identifier.
            tasks: List of (task_id, title, status, files_dict) tuples.
        """
        run = RunMemory(run_id=run_id, roadmap_path="/test/ROADMAP.md")

        if tasks:
            for task_id, title, status, files in tasks:
                file_dict = {project_root / p: FileAction(a) for p, a in files.items()}
                run.add_outcome(
                    TaskOutcome(
                        task_id=task_id,
                        title=title,
                        status=status,
                        files=file_dict,
                        summary=f"Summary for {title}",
                    )
                )

        return run

    def test_update_from_run_adds_history(self, tmp_path: Path):
        """Test run is added to history."""
        memory = ProjectMemory(project_root=tmp_path)
        run = self._create_run_memory(tmp_path, "run_1")
        run.add_outcome(
            TaskOutcome(
                task_id="1.1",
                title="Test task",
                status="completed",
            )
        )

        memory.update_from_run(run)

        assert len(memory.run_history) == 1
        assert memory.run_history[0].run_id == "run_1"

    def test_update_from_run_trims_old_history(self, tmp_path: Path):
        """Test old history is trimmed to max."""
        memory = ProjectMemory(project_root=tmp_path)
        memory._max_history = 3

        # Add 5 runs
        for i in range(5):
            run = self._create_run_memory(tmp_path, f"run_{i}")
            run.add_outcome(
                TaskOutcome(
                    task_id="1.1",
                    title="Test",
                    status="completed",
                )
            )
            memory.update_from_run(run)

        assert len(memory.run_history) == 3
        # Should keep most recent
        assert memory.run_history[0].run_id == "run_2"
        assert memory.run_history[2].run_id == "run_4"

    def test_update_from_run_tracks_files(self, tmp_path: Path):
        """Test files are tracked from run."""
        memory = ProjectMemory(project_root=tmp_path)
        run = self._create_run_memory(
            tmp_path,
            "run_1",
            tasks=[
                ("1.1", "Create utils", "completed", {"src/utils.py": "created"}),
                ("1.2", "Create main", "completed", {"src/main.py": "created"}),
            ],
        )

        memory.update_from_run(run)

        assert len(memory.key_files) == 2
        assert "src/utils.py" in memory.key_files
        assert "src/main.py" in memory.key_files

    def test_update_from_run_updates_existing_files(self, tmp_path: Path):
        """Test existing file entries are updated."""
        memory = ProjectMemory(project_root=tmp_path)

        # First run creates file
        run1 = self._create_run_memory(
            tmp_path,
            "run_1",
            tasks=[("1.1", "Create", "completed", {"src/mod.py": "created"})],
        )
        memory.update_from_run(run1)

        # Second run modifies file
        run2 = self._create_run_memory(
            tmp_path,
            "run_2",
            tasks=[("2.1", "Update", "completed", {"src/mod.py": "modified"})],
        )
        memory.update_from_run(run2)

        # File info should be updated
        info = memory.key_files["src/mod.py"]
        assert info.created_by_task == "1.1"  # Original creator
        assert info.last_modified_by_task == "2.1"  # Updated

    def test_update_from_run_records_completed_count(self, tmp_path: Path):
        """Test completed/failed counts are recorded."""
        memory = ProjectMemory(project_root=tmp_path)
        run = RunMemory(run_id="run_1")
        run.add_outcome(TaskOutcome(task_id="1.1", title="A", status="completed"))
        run.add_outcome(TaskOutcome(task_id="1.2", title="B", status="completed"))
        run.add_outcome(TaskOutcome(task_id="1.3", title="C", status="failed"))

        memory.update_from_run(run)

        assert memory.run_history[0].tasks_completed == 2
        assert memory.run_history[0].tasks_failed == 1

    def test_update_from_run_extracts_lessons(self, tmp_path: Path):
        """Test lessons are extracted from failures."""
        memory = ProjectMemory(project_root=tmp_path)
        run = RunMemory(run_id="run_1")
        run.add_outcome(
            TaskOutcome(
                task_id="1.1",
                title="Failed task",
                status="failed",
                key_decisions=["Wrong import path used"],
            )
        )

        memory.update_from_run(run)

        assert len(memory.run_history[0].lessons_learned) == 1
        assert "Wrong import path" in memory.run_history[0].lessons_learned[0]

    def test_update_from_run_saves_automatically(self, tmp_path: Path):
        """Test update_from_run saves changes."""
        memory = ProjectMemory(project_root=tmp_path)
        run = self._create_run_memory(tmp_path, "run_1")
        run.add_outcome(TaskOutcome(task_id="1.1", title="T", status="completed"))

        memory.update_from_run(run)

        # Verify file exists
        memory_path = tmp_path / ".executor" / "project-memory.json"
        assert memory_path.exists()


# =============================================================================
# Test ProjectMemory - Context Generation
# =============================================================================


class TestProjectMemoryContext:
    """Tests for context generation."""

    def test_get_context_empty_when_new(self, tmp_path: Path):
        """Test empty context for new memory."""
        memory = ProjectMemory(project_root=tmp_path)
        context = memory.get_context()
        assert context == ""

    def test_get_context_includes_project_type(self, tmp_path: Path):
        """Test context includes project type."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.project_type = "python"
        memory.add_convention("placeholder")  # Need some data

        context = memory.get_context()
        assert "**Type**: python" in context

    def test_get_context_includes_frameworks(self, tmp_path: Path):
        """Test context includes frameworks."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.project_type = "node"
        memory.add_framework("react")
        memory.add_framework("express")
        memory.add_convention("placeholder")

        context = memory.get_context()
        assert "**Frameworks**: react, express" in context

    def test_get_context_includes_conventions(self, tmp_path: Path):
        """Test context includes conventions."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.add_convention("Use type hints")
        memory.add_convention("Follow PEP 8")

        context = memory.get_context()
        assert "### Conventions" in context
        assert "Use type hints" in context
        assert "Follow PEP 8" in context

    def test_get_context_includes_key_files(self, tmp_path: Path):
        """Test context includes key files."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.key_files["src/main.py"] = FileInfo(
            path="src/main.py",
            purpose="Entry point",
        )
        memory.key_files["src/utils.py"] = FileInfo(
            path="src/utils.py",
            purpose="Utility functions",
        )

        context = memory.get_context()
        assert "### Key Files" in context
        assert "`src/main.py`" in context
        assert "Entry point" in context

    def test_get_context_includes_recent_runs(self, tmp_path: Path):
        """Test context includes recent runs."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.run_history.append(
            RunSummary(
                run_id="run_1",
                timestamp="2026-01-07T10:00:00Z",
                tasks_completed=5,
                tasks_failed=1,
            )
        )

        context = memory.get_context()
        assert "### Recent Runs" in context
        assert "2026-01-07" in context
        assert "5 completed" in context
        assert "1 failed" in context

    def test_get_context_respects_token_budget(self, tmp_path: Path):
        """Test context stays within token budget."""
        memory = ProjectMemory(project_root=tmp_path)

        # Add lots of data
        for i in range(50):
            memory.key_files[f"src/file_{i}.py"] = FileInfo(
                path=f"src/file_{i}.py",
                purpose=f"File number {i} with a longer description to use tokens",
            )
        for i in range(20):
            memory.add_convention(f"Convention {i}: Follow this important rule")

        # Small budget
        context = memory.get_context(max_tokens=500)
        token_count = _count_tokens_simple(context)
        assert token_count <= 500

    def test_get_context_truncates_gracefully(self, tmp_path: Path):
        """Test truncation produces valid context."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.project_type = "python"

        # Add lots of files
        for i in range(30):
            memory.key_files[f"src/module_{i}.py"] = FileInfo(
                path=f"src/module_{i}.py",
                purpose=f"Module {i} description",
            )

        context = memory.get_context(max_tokens=300)

        # Should still have structure
        assert "## Project Context" in context
        assert "**Type**: python" in context
        # Should have truncation indicator
        assert "more" in context


# =============================================================================
# Test ProjectMemory - Utility Methods
# =============================================================================


class TestProjectMemoryUtilities:
    """Tests for utility methods."""

    def test_add_convention(self, tmp_path: Path):
        """Test adding conventions."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.add_convention("Use type hints")
        memory.add_convention("Follow PEP 8")

        assert len(memory.conventions) == 2
        assert "Use type hints" in memory.conventions

    def test_add_convention_no_duplicates(self, tmp_path: Path):
        """Test same convention isn't added twice."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.add_convention("Use type hints")
        memory.add_convention("Use type hints")

        assert len(memory.conventions) == 1

    def test_add_convention_trims_to_max(self, tmp_path: Path):
        """Test conventions are trimmed to max 20."""
        memory = ProjectMemory(project_root=tmp_path)
        for i in range(25):
            memory.add_convention(f"Convention {i}")

        assert len(memory.conventions) == 20
        # Should keep most recent
        assert "Convention 24" in memory.conventions

    def test_set_project_type(self, tmp_path: Path):
        """Test setting project type."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.set_project_type("rust")
        assert memory.project_type == "rust"

    def test_add_framework(self, tmp_path: Path):
        """Test adding frameworks."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.add_framework("fastapi")
        memory.add_framework("sqlalchemy")

        assert len(memory.detected_frameworks) == 2
        assert "fastapi" in memory.detected_frameworks

    def test_add_framework_no_duplicates(self, tmp_path: Path):
        """Test same framework isn't added twice."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.add_framework("react")
        memory.add_framework("react")

        assert len(memory.detected_frameworks) == 1

    def test_get_file_info(self, tmp_path: Path):
        """Test getting file info by path."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.key_files["src/main.py"] = FileInfo(
            path="src/main.py",
            purpose="Entry point",
        )

        info = memory.get_file_info("src/main.py")
        assert info is not None
        assert info.purpose == "Entry point"

        # Non-existent file
        assert memory.get_file_info("not/found.py") is None

    def test_get_files_by_task(self, tmp_path: Path):
        """Test getting files by task ID."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.key_files["a.py"] = FileInfo(
            path="a.py",
            created_by_task="1.1",
        )
        memory.key_files["b.py"] = FileInfo(
            path="b.py",
            created_by_task="1.2",
            last_modified_by_task="1.1",
        )
        memory.key_files["c.py"] = FileInfo(
            path="c.py",
            created_by_task="1.2",
        )

        files = memory.get_files_by_task("1.1")
        assert len(files) == 2  # a.py (created) and b.py (modified)

    def test_get_last_run(self, tmp_path: Path):
        """Test getting last run summary."""
        memory = ProjectMemory(project_root=tmp_path)

        # No runs
        assert memory.get_last_run() is None

        # Add runs
        memory.run_history.append(
            RunSummary(
                run_id="run_1",
                timestamp="2026-01-01T10:00:00Z",
            )
        )
        memory.run_history.append(
            RunSummary(
                run_id="run_2",
                timestamp="2026-01-02T10:00:00Z",
            )
        )

        last = memory.get_last_run()
        assert last is not None
        assert last.run_id == "run_2"

    def test_get_stats(self, tmp_path: Path):
        """Test getting statistics."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.project_type = "python"
        memory.add_framework("fastapi")
        memory.add_convention("Use type hints")
        memory.key_files["a.py"] = FileInfo(path="a.py")
        memory.key_files["b.py"] = FileInfo(path="b.py")
        memory.run_history.append(
            RunSummary(
                run_id="run_1",
                timestamp="2026-01-01T10:00:00Z",
                tasks_completed=5,
                tasks_failed=1,
            )
        )

        stats = memory.get_stats()
        assert stats["run_count"] == 1
        assert stats["key_file_count"] == 2
        assert stats["convention_count"] == 1
        assert stats["framework_count"] == 1
        assert stats["total_tasks_completed"] == 5
        assert stats["total_tasks_failed"] == 1

    def test_clear(self, tmp_path: Path):
        """Test clearing memory."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.project_type = "python"
        memory.add_framework("fastapi")
        memory.add_convention("Test")
        memory.key_files["a.py"] = FileInfo(path="a.py")
        memory.run_history.append(
            RunSummary(
                run_id="run_1",
                timestamp="2026-01-01T10:00:00Z",
            )
        )

        memory.clear()

        assert memory.project_type == "unknown"
        assert memory.detected_frameworks == []
        assert memory.conventions == []
        assert memory.key_files == {}
        assert memory.run_history == []

    def test_delete(self, tmp_path: Path):
        """Test deleting memory file."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.save()

        memory_path = tmp_path / ".executor" / "project-memory.json"
        assert memory_path.exists()

        result = memory.delete()
        assert result is True
        assert not memory_path.exists()

        # Second delete returns False
        assert memory.delete() is False


# =============================================================================
# Test Token Counting
# =============================================================================


class TestTokenCounting:
    """Tests for token counting utility."""

    def test_count_tokens_simple_empty(self):
        """Test counting empty string."""
        assert _count_tokens_simple("") == 0

    def test_count_tokens_simple_short(self):
        """Test counting short string."""
        # "hello" = 5 chars, (5+3)//4 = 2 tokens
        assert _count_tokens_simple("hello") == 2

    def test_count_tokens_simple_longer(self):
        """Test counting longer string."""
        # 100 chars = (100+3)//4 = 25 tokens
        text = "a" * 100
        assert _count_tokens_simple(text) == 25

    def test_count_tokens_conservative(self):
        """Test that counting is conservative (overestimates)."""
        # Real tokenizers typically give fewer tokens
        text = "The quick brown fox jumps over the lazy dog."
        tokens = _count_tokens_simple(text)
        # This 44 char string would be ~9-10 real tokens
        # Our estimate: (44+3)//4 = 11
        assert tokens >= 10


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_memory_with_unicode_content(self, tmp_path: Path):
        """Test memory handles unicode content."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.add_convention("Use émojis 🎉 sparingly")
        memory.key_files["日本語.py"] = FileInfo(
            path="日本語.py",
            purpose="Japanese file name test",
        )

        memory.save()
        loaded = ProjectMemory.load(tmp_path)

        assert "émojis" in loaded.conventions[0]
        assert "日本語.py" in loaded.key_files

    def test_memory_with_long_paths(self, tmp_path: Path):
        """Test memory handles long file paths."""
        memory = ProjectMemory(project_root=tmp_path)
        long_path = "/".join(["dir"] * 20) + "/file.py"
        memory.key_files[long_path] = FileInfo(
            path=long_path,
            purpose="Deeply nested file",
        )

        memory.save()
        loaded = ProjectMemory.load(tmp_path)
        assert long_path in loaded.key_files

    def test_memory_with_special_characters_in_purpose(self, tmp_path: Path):
        """Test memory handles special chars in purpose."""
        memory = ProjectMemory(project_root=tmp_path)
        memory.key_files["test.py"] = FileInfo(
            path="test.py",
            purpose='Contains "quotes" and\nnewlines and\ttabs',
        )

        memory.save()
        loaded = ProjectMemory.load(tmp_path)
        assert "quotes" in loaded.key_files["test.py"].purpose

    def test_concurrent_saves_dont_corrupt(self, tmp_path: Path):
        """Test multiple saves don't corrupt file."""
        memory = ProjectMemory(project_root=tmp_path)

        # Rapid saves
        for i in range(10):
            memory.add_convention(f"Convention {i}")
            memory.save()

        # Load and verify
        loaded = ProjectMemory.load(tmp_path)
        assert len(loaded.conventions) == 10

    def test_key_files_trimmed_to_max(self, tmp_path: Path):
        """Test key files are trimmed to max."""
        memory = ProjectMemory(project_root=tmp_path)
        memory._max_key_files = 5

        # Add via update_from_run to trigger trimming
        run = RunMemory(run_id="run_1")
        for i in range(10):
            run.add_outcome(
                TaskOutcome(
                    task_id=f"1.{i}",
                    title=f"Create file {i}",
                    status="completed",
                    files={tmp_path / f"file_{i}.py": FileAction.CREATED},
                )
            )

        memory.update_from_run(run)

        assert len(memory.key_files) == 5
