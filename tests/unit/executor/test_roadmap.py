"""Unit tests for ROADMAP format data models."""

from __future__ import annotations

import pytest

from ai_infra.executor.models import TaskStatus
from ai_infra.executor.roadmap import (
    ParsedTask,
    Phase,
    Priority,
    Roadmap,
    Section,
    Subtask,
)

# =============================================================================
# TestPriority
# =============================================================================


class TestPriority:
    """Tests for Priority enum."""

    def test_values(self):
        """Test all priority values exist."""
        assert Priority.HIGH.value == "high"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.LOW.value == "low"

    def test_from_string_exact(self):
        """Test parsing exact strings."""
        assert Priority.from_string("high") == Priority.HIGH
        assert Priority.from_string("medium") == Priority.MEDIUM
        assert Priority.from_string("low") == Priority.LOW

    def test_from_string_case_insensitive(self):
        """Test case-insensitive parsing."""
        assert Priority.from_string("HIGH") == Priority.HIGH
        assert Priority.from_string("High") == Priority.HIGH
        assert Priority.from_string("MEDIUM") == Priority.MEDIUM

    def test_from_string_with_whitespace(self):
        """Test parsing with whitespace."""
        assert Priority.from_string("  high  ") == Priority.HIGH
        assert Priority.from_string("\tmedium\n") == Priority.MEDIUM

    def test_from_string_invalid_defaults_medium(self):
        """Test invalid string defaults to MEDIUM."""
        assert Priority.from_string("invalid") == Priority.MEDIUM
        assert Priority.from_string("") == Priority.MEDIUM
        assert Priority.from_string("urgent") == Priority.MEDIUM


# =============================================================================
# TestSubtask
# =============================================================================


class TestSubtask:
    """Tests for Subtask dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal data."""
        subtask = Subtask(id="1.1.1.1", title="Sub-task 1")
        assert subtask.id == "1.1.1.1"
        assert subtask.title == "Sub-task 1"
        assert subtask.completed is False
        assert subtask.line_number is None

    def test_create_completed(self):
        """Test creating a completed subtask."""
        subtask = Subtask(id="1.1.1.1", title="Done", completed=True, line_number=42)
        assert subtask.completed is True
        assert subtask.line_number == 42

    def test_to_dict(self):
        """Test serialization."""
        subtask = Subtask(id="1.1.1.1", title="Test", completed=True, line_number=10)
        data = subtask.to_dict()

        assert data["id"] == "1.1.1.1"
        assert data["title"] == "Test"
        assert data["completed"] is True
        assert data["line_number"] == 10

    def test_from_dict(self):
        """Test deserialization."""
        data = {"id": "1.1.1.2", "title": "Another", "completed": False}
        subtask = Subtask.from_dict(data)

        assert subtask.id == "1.1.1.2"
        assert subtask.title == "Another"
        assert subtask.completed is False

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = Subtask(id="1.1.1.1", title="Test", completed=True, line_number=5)
        restored = Subtask.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.completed == original.completed
        assert restored.line_number == original.line_number


# =============================================================================
# TestParsedTask
# =============================================================================


class TestParsedTask:
    """Tests for ParsedTask dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal data."""
        task = ParsedTask(id="1.1.1", title="Implement feature")
        assert task.id == "1.1.1"
        assert task.title == "Implement feature"
        assert task.status == TaskStatus.PENDING
        assert task.file_hints == []
        assert task.subtasks == []

    def test_create_full(self):
        """Test creating with all fields."""
        subtask = Subtask(id="1.1.1.1", title="Sub 1")
        task = ParsedTask(
            id="1.1.1",
            title="Full task",
            description="Description here",
            status=TaskStatus.COMPLETED,
            file_hints=["src/main.py"],
            code_context=["def foo(): pass"],
            subtasks=[subtask],
            context="Full context text",
            phase_id="1",
            section_id="1.1",
            line_number=50,
            dependencies=["1.0.1"],
        )

        assert task.description == "Description here"
        assert task.is_completed
        assert len(task.file_hints) == 1
        assert len(task.subtasks) == 1

    def test_is_pending(self):
        """Test is_pending property."""
        pending = ParsedTask(id="1", title="T", status=TaskStatus.PENDING)
        completed = ParsedTask(id="2", title="T", status=TaskStatus.COMPLETED)

        assert pending.is_pending is True
        assert completed.is_pending is False

    def test_is_completed(self):
        """Test is_completed property."""
        pending = ParsedTask(id="1", title="T", status=TaskStatus.PENDING)
        completed = ParsedTask(id="2", title="T", status=TaskStatus.COMPLETED)

        assert pending.is_completed is False
        assert completed.is_completed is True

    def test_subtask_count(self):
        """Test subtask counting."""
        task = ParsedTask(
            id="1.1.1",
            title="Task",
            subtasks=[
                Subtask(id="1.1.1.1", title="S1", completed=False),
                Subtask(id="1.1.1.2", title="S2", completed=True),
                Subtask(id="1.1.1.3", title="S3", completed=True),
            ],
        )

        assert task.subtask_count == 3
        assert task.completed_subtask_count == 2

    def test_to_task(self):
        """Test conversion to base Task."""
        parsed = ParsedTask(
            id="1.1.1",
            title="Test task",
            description="Desc",
            file_hints=["file.py"],
            phase_id="1",
            section_id="1.1",
            dependencies=["1.0.1"],
        )
        task = parsed.to_task()

        assert task.id == "1.1.1"
        assert task.title == "Test task"
        assert task.description == "Desc"
        assert task.file_hints == ["file.py"]
        assert task.phase == "1"
        assert task.section == "1.1"
        assert task.dependencies == ["1.0.1"]

    def test_to_dict(self):
        """Test serialization."""
        task = ParsedTask(
            id="1.1.1",
            title="Test",
            status=TaskStatus.PENDING,
            file_hints=["a.py"],
        )
        data = task.to_dict()

        assert data["id"] == "1.1.1"
        assert data["title"] == "Test"
        assert data["status"] == "pending"
        assert data["file_hints"] == ["a.py"]

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "id": "1.1.1",
            "title": "From dict",
            "status": "completed",
            "file_hints": ["b.py"],
            "subtasks": [{"id": "1.1.1.1", "title": "Sub", "completed": True}],
        }
        task = ParsedTask.from_dict(data)

        assert task.id == "1.1.1"
        assert task.title == "From dict"
        assert task.status == TaskStatus.COMPLETED
        assert len(task.subtasks) == 1
        assert task.subtasks[0].completed is True


# =============================================================================
# TestSection
# =============================================================================


class TestSection:
    """Tests for Section dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal data."""
        section = Section(id="1.1", title="Parser Implementation")
        assert section.id == "1.1"
        assert section.title == "Parser Implementation"
        assert section.tasks == []

    def test_create_with_tasks(self):
        """Test creating with tasks."""
        tasks = [
            ParsedTask(id="1.1.1", title="Task 1", status=TaskStatus.COMPLETED),
            ParsedTask(id="1.1.2", title="Task 2", status=TaskStatus.PENDING),
            ParsedTask(id="1.1.3", title="Task 3", status=TaskStatus.PENDING),
        ]
        section = Section(
            id="1.1",
            title="Test Section",
            file_hints=["src/test.py"],
            tasks=tasks,
            phase_id="1",
        )

        assert section.task_count == 3
        assert section.completed_count == 1
        assert section.pending_count == 2

    def test_to_dict(self):
        """Test serialization."""
        section = Section(
            id="1.1",
            title="Test",
            file_hints=["a.py"],
            tasks=[ParsedTask(id="1.1.1", title="T")],
        )
        data = section.to_dict()

        assert data["id"] == "1.1"
        assert data["title"] == "Test"
        assert len(data["tasks"]) == 1

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "id": "1.2",
            "title": "From dict",
            "tasks": [{"id": "1.2.1", "title": "Task", "status": "pending"}],
        }
        section = Section.from_dict(data)

        assert section.id == "1.2"
        assert len(section.tasks) == 1


# =============================================================================
# TestPhase
# =============================================================================


class TestPhase:
    """Tests for Phase dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal data."""
        phase = Phase(id="1", name="Core Features")
        assert phase.id == "1"
        assert phase.name == "Core Features"
        assert phase.priority == Priority.MEDIUM
        assert phase.sections == []

    def test_create_full(self):
        """Test creating with all fields."""
        section = Section(
            id="1.1",
            title="Section",
            tasks=[
                ParsedTask(id="1.1.1", title="T1", status=TaskStatus.COMPLETED),
                ParsedTask(id="1.1.2", title="T2", status=TaskStatus.PENDING),
            ],
        )
        phase = Phase(
            id="1",
            name="Phase One",
            goal="Complete the core",
            priority=Priority.HIGH,
            effort="2 weeks",
            prerequisite="Phase 0",
            description="Main phase",
            sections=[section],
            line_number=10,
        )

        assert phase.goal == "Complete the core"
        assert phase.priority == Priority.HIGH
        assert phase.task_count == 2
        assert phase.completed_count == 1
        assert phase.pending_count == 1

    def test_all_tasks(self):
        """Test iterating over all tasks."""
        phase = Phase(
            id="1",
            name="Test",
            sections=[
                Section(
                    id="1.1",
                    title="S1",
                    tasks=[
                        ParsedTask(id="1.1.1", title="T1"),
                        ParsedTask(id="1.1.2", title="T2"),
                    ],
                ),
                Section(
                    id="1.2",
                    title="S2",
                    tasks=[ParsedTask(id="1.2.1", title="T3")],
                ),
            ],
        )

        tasks = list(phase.all_tasks())
        assert len(tasks) == 3
        assert [t.id for t in tasks] == ["1.1.1", "1.1.2", "1.2.1"]

    def test_to_dict(self):
        """Test serialization."""
        phase = Phase(id="1", name="Test", priority=Priority.HIGH)
        data = phase.to_dict()

        assert data["id"] == "1"
        assert data["name"] == "Test"
        assert data["priority"] == "high"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "id": "2",
            "name": "From dict",
            "priority": "low",
            "goal": "Test goal",
        }
        phase = Phase.from_dict(data)

        assert phase.id == "2"
        assert phase.priority == Priority.LOW
        assert phase.goal == "Test goal"


# =============================================================================
# TestRoadmap
# =============================================================================


class TestRoadmap:
    """Tests for Roadmap dataclass."""

    @pytest.fixture
    def sample_roadmap(self) -> Roadmap:
        """Create a sample roadmap for testing."""
        return Roadmap(
            path="./ROADMAP.md",
            title="Test Project",
            phases=[
                Phase(
                    id="0",
                    name="Foundation",
                    sections=[
                        Section(
                            id="0.1",
                            title="Setup",
                            tasks=[
                                ParsedTask(id="0.1.1", title="T1", status=TaskStatus.COMPLETED),
                                ParsedTask(id="0.1.2", title="T2", status=TaskStatus.COMPLETED),
                            ],
                        ),
                    ],
                ),
                Phase(
                    id="1",
                    name="Core",
                    sections=[
                        Section(
                            id="1.1",
                            title="Parser",
                            tasks=[
                                ParsedTask(id="1.1.1", title="T3", status=TaskStatus.COMPLETED),
                                ParsedTask(id="1.1.2", title="T4", status=TaskStatus.PENDING),
                                ParsedTask(id="1.1.3", title="T5", status=TaskStatus.PENDING),
                            ],
                        ),
                        Section(
                            id="1.2",
                            title="State",
                            tasks=[
                                ParsedTask(id="1.2.1", title="T6", status=TaskStatus.PENDING),
                            ],
                        ),
                    ],
                ),
            ],
        )

    def test_create_minimal(self):
        """Test creating with minimal data."""
        roadmap = Roadmap(path="./ROADMAP.md")
        assert roadmap.path == "./ROADMAP.md"
        assert roadmap.phases == []
        assert roadmap.total_tasks == 0

    def test_total_tasks(self, sample_roadmap: Roadmap):
        """Test total task count."""
        assert sample_roadmap.total_tasks == 6

    def test_pending_count(self, sample_roadmap: Roadmap):
        """Test pending task count."""
        assert sample_roadmap.pending_count == 3

    def test_completed_count(self, sample_roadmap: Roadmap):
        """Test completed task count."""
        assert sample_roadmap.completed_count == 3

    def test_progress(self, sample_roadmap: Roadmap):
        """Test progress calculation."""
        assert sample_roadmap.progress == 0.5  # 3/6

    def test_progress_empty(self):
        """Test progress with no tasks."""
        roadmap = Roadmap(path="./ROADMAP.md")
        assert roadmap.progress == 1.0  # Complete if nothing to do

    def test_phase_count(self, sample_roadmap: Roadmap):
        """Test phase count."""
        assert sample_roadmap.phase_count == 2

    def test_all_tasks(self, sample_roadmap: Roadmap):
        """Test iterating all tasks."""
        tasks = list(sample_roadmap.all_tasks())
        assert len(tasks) == 6

    def test_pending_tasks(self, sample_roadmap: Roadmap):
        """Test iterating pending tasks."""
        pending = list(sample_roadmap.pending_tasks())
        assert len(pending) == 3
        assert all(t.is_pending for t in pending)

    def test_completed_tasks(self, sample_roadmap: Roadmap):
        """Test iterating completed tasks."""
        completed = list(sample_roadmap.completed_tasks())
        assert len(completed) == 3
        assert all(t.is_completed for t in completed)

    def test_next_pending(self, sample_roadmap: Roadmap):
        """Test getting next pending task."""
        task = sample_roadmap.next_pending()
        assert task is not None
        assert task.id == "1.1.2"  # First pending in order

    def test_next_pending_none(self):
        """Test next_pending when all complete."""
        roadmap = Roadmap(
            path="./ROADMAP.md",
            phases=[
                Phase(
                    id="0",
                    name="Done",
                    sections=[
                        Section(
                            id="0.1",
                            title="Done",
                            tasks=[
                                ParsedTask(id="0.1.1", title="T", status=TaskStatus.COMPLETED),
                            ],
                        ),
                    ],
                ),
            ],
        )
        assert roadmap.next_pending() is None

    def test_get_task(self, sample_roadmap: Roadmap):
        """Test getting task by ID."""
        task = sample_roadmap.get_task("1.1.2")
        assert task is not None
        assert task.title == "T4"

    def test_get_task_not_found(self, sample_roadmap: Roadmap):
        """Test getting non-existent task."""
        assert sample_roadmap.get_task("9.9.9") is None

    def test_get_phase(self, sample_roadmap: Roadmap):
        """Test getting phase by ID."""
        phase = sample_roadmap.get_phase("1")
        assert phase is not None
        assert phase.name == "Core"

    def test_get_phase_not_found(self, sample_roadmap: Roadmap):
        """Test getting non-existent phase."""
        assert sample_roadmap.get_phase("99") is None

    def test_get_section(self, sample_roadmap: Roadmap):
        """Test getting section by ID."""
        section = sample_roadmap.get_section("1.2")
        assert section is not None
        assert section.title == "State"

    def test_get_section_not_found(self, sample_roadmap: Roadmap):
        """Test getting non-existent section."""
        assert sample_roadmap.get_section("9.9") is None

    def test_summary(self, sample_roadmap: Roadmap):
        """Test human-readable summary."""
        summary = sample_roadmap.summary()

        assert "./ROADMAP.md" in summary
        assert "Phases: 2" in summary
        assert "6" in summary  # total tasks
        assert "50%" in summary  # progress
        assert "Foundation" in summary
        assert "Core" in summary

    def test_to_dict(self, sample_roadmap: Roadmap):
        """Test serialization."""
        data = sample_roadmap.to_dict()

        assert data["path"] == "./ROADMAP.md"
        assert data["title"] == "Test Project"
        assert len(data["phases"]) == 2

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "path": "./TEST.md",
            "title": "From Dict",
            "phases": [
                {
                    "id": "0",
                    "name": "Test Phase",
                    "sections": [],
                }
            ],
            "parsed_at": "2026-01-06T12:00:00",
        }
        roadmap = Roadmap.from_dict(data)

        assert roadmap.path == "./TEST.md"
        assert roadmap.title == "From Dict"
        assert len(roadmap.phases) == 1

    def test_roundtrip(self, sample_roadmap: Roadmap):
        """Test serialization roundtrip."""
        restored = Roadmap.from_dict(sample_roadmap.to_dict())

        assert restored.path == sample_roadmap.path
        assert restored.total_tasks == sample_roadmap.total_tasks
        assert restored.pending_count == sample_roadmap.pending_count

    def test_parse_errors(self):
        """Test parse errors are captured."""
        roadmap = Roadmap(
            path="./ROADMAP.md",
            parse_errors=["Line 10: Invalid checkbox format", "Line 25: Missing phase header"],
        )

        assert len(roadmap.parse_errors) == 2
        summary = roadmap.summary()
        assert "Parse Errors: 2" in summary
