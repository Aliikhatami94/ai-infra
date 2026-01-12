"""Data models for parsed ROADMAP.md files.

This module defines the structure of parsed roadmaps, phases, sections, and tasks
as specified in docs/executor/roadmap-format.md.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ai_infra.executor.models import Task, TaskStatus


class Priority(str, Enum):
    """Priority levels for phases and tasks."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_string(cls, value: str) -> Priority:
        """Parse priority from string (case-insensitive)."""
        normalized = value.strip().lower()
        for priority in cls:
            if priority.value == normalized:
                return priority
        return cls.MEDIUM  # Default


@dataclass
class Subtask:
    """A subtask nested under a parent task.

    Attributes:
        id: Subtask identifier (e.g., "1.1.1.1")
        title: Subtask title text
        completed: Whether the subtask is marked complete
        line_number: Source line number in ROADMAP
    """

    id: str
    title: str
    completed: bool = False
    line_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "completed": self.completed,
            "line_number": self.line_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Subtask:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            completed=data.get("completed", False),
            line_number=data.get("line_number"),
        )


@dataclass
class ParsedTask:
    """A task parsed from ROADMAP.md.

    Extends the base Task with additional parsing metadata.

    Attributes:
        id: Task identifier (e.g., "1.1.1")
        title: Task title from bold text
        description: Description paragraphs
        status: Current status from checkbox
        file_hints: Files mentioned for this task
        code_context: Code blocks from task context
        subtasks: Nested subtasks
        context: Full context text for agent
        phase_id: Parent phase identifier
        section_id: Parent section identifier
        line_number: Source line number in ROADMAP
        dependencies: Task IDs this depends on
    """

    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    file_hints: list[str] = field(default_factory=list)
    code_context: list[str] = field(default_factory=list)
    subtasks: list[Subtask] = field(default_factory=list)
    context: str = ""
    phase_id: str = ""
    section_id: str = ""
    line_number: int | None = None
    dependencies: list[str] = field(default_factory=list)

    def to_task(self) -> Task:
        """Convert to base Task object for execution."""
        return Task(
            id=self.id,
            title=self.title,
            description=self.description,
            status=self.status,
            file_hints=self.file_hints,
            dependencies=self.dependencies,
            phase=self.phase_id,
            section=self.section_id,
            metadata={
                "code_context": self.code_context,
                "subtasks": [s.to_dict() for s in self.subtasks],
                "line_number": self.line_number,
                "full_context": self.context,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "file_hints": self.file_hints,
            "code_context": self.code_context,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "context": self.context,
            "phase_id": self.phase_id,
            "section_id": self.section_id,
            "line_number": self.line_number,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedTask:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=TaskStatus(data.get("status", "pending")),
            file_hints=data.get("file_hints", []),
            code_context=data.get("code_context", []),
            subtasks=[Subtask.from_dict(s) for s in data.get("subtasks", [])],
            context=data.get("context", ""),
            phase_id=data.get("phase_id", ""),
            section_id=data.get("section_id", ""),
            line_number=data.get("line_number"),
            dependencies=data.get("dependencies", []),
        )

    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status == TaskStatus.COMPLETED

    @property
    def is_pending(self) -> bool:
        """Check if task is pending."""
        return self.status == TaskStatus.PENDING

    @property
    def subtask_count(self) -> int:
        """Get number of subtasks."""
        return len(self.subtasks)

    @property
    def completed_subtask_count(self) -> int:
        """Get number of completed subtasks."""
        return sum(1 for s in self.subtasks if s.completed)


@dataclass
class Section:
    """A section within a phase.

    Attributes:
        id: Section identifier (e.g., "1.1")
        title: Section title
        description: Section description text
        file_hints: Files mentioned for this section
        tasks: Tasks in this section
        phase_id: Parent phase identifier
        line_number: Source line number in ROADMAP
    """

    id: str
    title: str
    description: str = ""
    file_hints: list[str] = field(default_factory=list)
    tasks: list[ParsedTask] = field(default_factory=list)
    phase_id: str = ""
    line_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "file_hints": self.file_hints,
            "tasks": [t.to_dict() for t in self.tasks],
            "phase_id": self.phase_id,
            "line_number": self.line_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Section:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            file_hints=data.get("file_hints", []),
            tasks=[ParsedTask.from_dict(t) for t in data.get("tasks", [])],
            phase_id=data.get("phase_id", ""),
            line_number=data.get("line_number"),
        )

    @property
    def task_count(self) -> int:
        """Get total number of tasks."""
        return len(self.tasks)

    @property
    def pending_count(self) -> int:
        """Get number of pending tasks."""
        return sum(1 for t in self.tasks if t.is_pending)

    @property
    def completed_count(self) -> int:
        """Get number of completed tasks."""
        return sum(1 for t in self.tasks if t.is_completed)


@dataclass
class Phase:
    """A phase in the roadmap.

    Attributes:
        id: Phase identifier (e.g., "1", "0.5")
        name: Phase name
        goal: What this phase achieves
        priority: Priority level
        effort: Time estimate
        prerequisite: Requirements text
        description: Phase description text
        sections: Sections in this phase
        line_number: Source line number in ROADMAP
    """

    id: str
    name: str
    goal: str = ""
    priority: Priority = Priority.MEDIUM
    effort: str = ""
    prerequisite: str = ""
    description: str = ""
    sections: list[Section] = field(default_factory=list)
    line_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "goal": self.goal,
            "priority": self.priority.value,
            "effort": self.effort,
            "prerequisite": self.prerequisite,
            "description": self.description,
            "sections": [s.to_dict() for s in self.sections],
            "line_number": self.line_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Phase:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            goal=data.get("goal", ""),
            priority=Priority.from_string(data.get("priority", "medium")),
            effort=data.get("effort", ""),
            prerequisite=data.get("prerequisite", ""),
            description=data.get("description", ""),
            sections=[Section.from_dict(s) for s in data.get("sections", [])],
            line_number=data.get("line_number"),
        )

    @property
    def task_count(self) -> int:
        """Get total number of tasks in this phase."""
        return sum(s.task_count for s in self.sections)

    @property
    def pending_count(self) -> int:
        """Get number of pending tasks."""
        return sum(s.pending_count for s in self.sections)

    @property
    def completed_count(self) -> int:
        """Get number of completed tasks."""
        return sum(s.completed_count for s in self.sections)

    def all_tasks(self) -> Iterator[ParsedTask]:
        """Iterate over all tasks in this phase."""
        for section in self.sections:
            yield from section.tasks


@dataclass
class Roadmap:
    """A parsed ROADMAP.md file.

    Attributes:
        path: Path to the source ROADMAP.md
        title: Document title (if present)
        description: Document description
        phases: Phases in the roadmap
        parse_errors: Any errors encountered during parsing
        parsed_at: When the roadmap was parsed
    """

    path: str
    title: str = ""
    description: str = ""
    phases: list[Phase] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    parsed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "title": self.title,
            "description": self.description,
            "phases": [p.to_dict() for p in self.phases],
            "parse_errors": self.parse_errors,
            "parsed_at": self.parsed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Roadmap:
        """Create from dictionary."""
        return cls(
            path=data["path"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            phases=[Phase.from_dict(p) for p in data.get("phases", [])],
            parse_errors=data.get("parse_errors", []),
            parsed_at=datetime.fromisoformat(data["parsed_at"])
            if data.get("parsed_at")
            else datetime.utcnow(),
        )

    # =========================================================================
    # Aggregate Properties
    # =========================================================================

    @property
    def total_tasks(self) -> int:
        """Get total number of tasks across all phases."""
        return sum(p.task_count for p in self.phases)

    @property
    def pending_count(self) -> int:
        """Get number of pending tasks."""
        return sum(p.pending_count for p in self.phases)

    @property
    def completed_count(self) -> int:
        """Get number of completed tasks."""
        return sum(p.completed_count for p in self.phases)

    @property
    def progress(self) -> float:
        """Get completion progress (0.0 to 1.0)."""
        if self.total_tasks == 0:
            return 1.0
        return self.completed_count / self.total_tasks

    @property
    def phase_count(self) -> int:
        """Get number of phases."""
        return len(self.phases)

    # =========================================================================
    # Task Access
    # =========================================================================

    def all_tasks(self) -> Iterator[ParsedTask]:
        """Iterate over all tasks in the roadmap."""
        for phase in self.phases:
            yield from phase.all_tasks()

    def pending_tasks(self) -> Iterator[ParsedTask]:
        """Iterate over pending tasks only."""
        for task in self.all_tasks():
            if task.is_pending:
                yield task

    def completed_tasks(self) -> Iterator[ParsedTask]:
        """Iterate over completed tasks only."""
        for task in self.all_tasks():
            if task.is_completed:
                yield task

    def next_pending(self) -> ParsedTask | None:
        """Get the next pending task in order."""
        for task in self.pending_tasks():
            return task
        return None

    def get_task(self, task_id: str) -> ParsedTask | None:
        """Get a task by ID."""
        for task in self.all_tasks():
            if task.id == task_id:
                return task
        return None

    def get_phase(self, phase_id: str) -> Phase | None:
        """Get a phase by ID."""
        for phase in self.phases:
            if phase.id == phase_id:
                return phase
        return None

    def get_section(self, section_id: str) -> Section | None:
        """Get a section by ID."""
        for phase in self.phases:
            for section in phase.sections:
                if section.id == section_id:
                    return section
        return None

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Roadmap: {self.path}",
            f"Phases: {self.phase_count}",
            f"Tasks: {self.total_tasks} ({self.completed_count} completed, {self.pending_count} pending)",
            f"Progress: {self.progress:.0%}",
        ]

        if self.parse_errors:
            lines.append(f"Parse Errors: {len(self.parse_errors)}")

        lines.append("\nPhases:")
        for phase in self.phases:
            status = f"{phase.completed_count}/{phase.task_count}"
            lines.append(f"  {phase.id}: {phase.name} [{status}]")

        return "\n".join(lines)
