"""Tests for RoadmapParser."""

from __future__ import annotations

import tempfile
from io import StringIO
from pathlib import Path

import pytest

from ai_infra.executor import (
    ParsedTask,
    ParseError,
    ParserConfig,
    Phase,
    Priority,
    Roadmap,
    RoadmapParser,
    Section,
    Subtask,
    TaskStatus,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def parser() -> RoadmapParser:
    """Create a default parser."""
    return RoadmapParser()


@pytest.fixture
def strict_parser() -> RoadmapParser:
    """Create a strict parser."""
    return RoadmapParser(ParserConfig(strict=True))


@pytest.fixture
def sample_roadmap() -> str:
    """Sample ROADMAP content."""
    return """\
# Test Project ROADMAP

This document tracks development progress.

---

## Phase 0: Foundation

> **Goal**: Establish core infrastructure
> **Priority**: HIGH
> **Effort**: 1 week

Build the foundational components.

### 0.1 Project Setup

**Files**: `pyproject.toml`, `src/__init__.py`

Initial project configuration.

- [x] **Initialize Poetry project**
  Set up pyproject.toml with dependencies.

- [x] **Configure linting**
  Set up ruff, mypy, and pre-commit hooks.

### 0.2 Core Models

**Files**: `src/models.py`

- [ ] **Define data models**
  Create Pydantic models for core entities.

  ```python
  from pydantic import BaseModel

  class User(BaseModel):
      id: str
      name: str
  ```

  - [ ] Add User model
  - [ ] Add Project model
  - [x] Add Task model

- [ ] **Add serialization helpers**
  JSON/YAML serialization utilities.

---

## Phase 1: Core Features

> **Goal**: Implement main functionality
> **Priority**: HIGH
> **Effort**: 2 weeks
> **Prerequisite**: Phase 0 complete

### 1.1 API Implementation

**Files**: `src/api.py`, `src/routes/`

- [ ] **Create FastAPI app**
  Initialize the application with proper configuration.

- [ ] **Implement CRUD endpoints**
  - [ ] GET /users
  - [ ] POST /users
  - [ ] PUT /users/{id}
  - [ ] DELETE /users/{id}
"""


@pytest.fixture
def minimal_roadmap() -> str:
    """Minimal valid ROADMAP."""
    return """\
## Phase 1: Simple

### 1.1 Tasks

- [ ] **Do something**
  Description here.
"""


@pytest.fixture
def temp_roadmap_file(sample_roadmap: str) -> Path:
    """Create a temporary ROADMAP file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(sample_roadmap)
        return Path(f.name)


# =============================================================================
# Basic Parsing Tests
# =============================================================================


class TestBasicParsing:
    """Test basic parsing functionality."""

    def test_parse_string(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test parsing from string."""
        roadmap = parser.parse_string(sample_roadmap, path="ROADMAP.md")

        assert roadmap.path == "ROADMAP.md"
        assert len(roadmap.phases) == 2

    def test_parse_file(self, parser: RoadmapParser, temp_roadmap_file: Path) -> None:
        """Test parsing from file path."""
        roadmap = parser.parse(temp_roadmap_file)

        assert roadmap.path == str(temp_roadmap_file)
        assert len(roadmap.phases) == 2

    def test_parse_file_object(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test parsing from file object."""
        file = StringIO(sample_roadmap)
        roadmap = parser.parse_file(file, path="test.md")

        assert roadmap.path == "test.md"
        assert len(roadmap.phases) == 2

    def test_parse_nonexistent_file(self, parser: RoadmapParser) -> None:
        """Test error on nonexistent file."""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/ROADMAP.md")

    def test_extract_title(self, parser: RoadmapParser) -> None:
        """Test document title extraction."""
        content = "# My Project\n\n## Phase 1: Test\n"
        roadmap = parser.parse_string(content)

        assert roadmap.title == "My Project"

    def test_no_title(self, parser: RoadmapParser) -> None:
        """Test parsing without document title."""
        content = "## Phase 1: Test\n### 1.1 Section\n- [ ] **Task**\n"
        roadmap = parser.parse_string(content)

        assert roadmap.title == ""


# =============================================================================
# Phase Parsing Tests
# =============================================================================


class TestPhaseParsing:
    """Test phase parsing."""

    def test_parse_phase_header(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test phase header parsing."""
        roadmap = parser.parse_string(sample_roadmap)

        assert roadmap.phases[0].id == "0"
        assert roadmap.phases[0].name == "Foundation"
        assert roadmap.phases[1].id == "1"
        assert roadmap.phases[1].name == "Core Features"

    def test_parse_phase_metadata(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test phase metadata extraction."""
        roadmap = parser.parse_string(sample_roadmap)

        phase0 = roadmap.phases[0]
        assert phase0.goal == "Establish core infrastructure"
        assert phase0.priority == Priority.HIGH
        assert phase0.effort == "1 week"

        phase1 = roadmap.phases[1]
        assert phase1.prerequisite == "Phase 0 complete"

    def test_phase_priorities(self, parser: RoadmapParser) -> None:
        """Test different priority values."""
        content = """\
## Phase 1: High
> **Priority**: HIGH

### 1.1 Tasks
- [ ] **Task**

## Phase 2: Low
> **Priority**: low

### 2.1 Tasks
- [ ] **Task**

## Phase 3: Default

### 3.1 Tasks
- [ ] **Task**
"""
        roadmap = parser.parse_string(content)

        assert roadmap.phases[0].priority == Priority.HIGH
        assert roadmap.phases[1].priority == Priority.LOW
        assert roadmap.phases[2].priority == Priority.MEDIUM

    def test_phase_with_decimal_id(self, parser: RoadmapParser) -> None:
        """Test phase with decimal ID."""
        content = "## Phase 0.5: Bugfix\n### 0.5.1 Fixes\n- [ ] **Fix bug**\n"
        roadmap = parser.parse_string(content)

        assert roadmap.phases[0].id == "0.5"
        assert roadmap.phases[0].name == "Bugfix"

    def test_phase_case_insensitive(self, parser: RoadmapParser) -> None:
        """Test case-insensitive phase header."""
        content = "## Phase 1: Test\n### 1.1 Section\n- [ ] **Task**\n"
        roadmap = parser.parse_string(content)

        assert len(roadmap.phases) == 1
        assert roadmap.phases[0].id == "1"

    def test_phase_uppercase(self, parser: RoadmapParser) -> None:
        """Test uppercase PHASE header."""
        content = "## PHASE 2: Uppercase\n### 2.1 Section\n- [ ] **Task**\n"
        roadmap = parser.parse_string(content)

        assert len(roadmap.phases) == 1
        assert roadmap.phases[0].id == "2"
        assert roadmap.phases[0].name == "Uppercase"


# =============================================================================
# Section Parsing Tests
# =============================================================================


class TestSectionParsing:
    """Test section parsing."""

    def test_parse_sections(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test section extraction."""
        roadmap = parser.parse_string(sample_roadmap)

        phase0 = roadmap.phases[0]
        assert len(phase0.sections) == 2
        assert phase0.sections[0].id == "0.1"
        assert phase0.sections[0].title == "Project Setup"
        assert phase0.sections[1].id == "0.2"
        assert phase0.sections[1].title == "Core Models"

    def test_section_file_hints(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test file hint extraction in sections."""
        roadmap = parser.parse_string(sample_roadmap)

        section = roadmap.phases[0].sections[0]
        assert "pyproject.toml" in section.file_hints
        assert "src/__init__.py" in section.file_hints

    def test_section_description(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test section description extraction."""
        roadmap = parser.parse_string(sample_roadmap)

        section = roadmap.phases[0].sections[0]
        assert "Initial project configuration" in section.description

    def test_multiple_file_formats(self, parser: RoadmapParser) -> None:
        """Test various file hint formats."""
        content = """\
## Phase 1: Test

### 1.1 Section A
**File**: `single.py`
- [ ] **Task A**

### 1.2 Section B
**Files**: `a.py`, `b.py`, `c.py`
- [ ] **Task B**
"""
        roadmap = parser.parse_string(content)

        assert roadmap.phases[0].sections[0].file_hints == ["single.py"]
        assert roadmap.phases[0].sections[1].file_hints == ["a.py", "b.py", "c.py"]


# =============================================================================
# Task Parsing Tests
# =============================================================================


class TestTaskParsing:
    """Test task parsing."""

    def test_parse_tasks(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test task extraction."""
        roadmap = parser.parse_string(sample_roadmap)

        section = roadmap.phases[0].sections[0]
        assert len(section.tasks) == 2
        assert section.tasks[0].title == "Initialize Poetry project"
        assert section.tasks[1].title == "Configure linting"

    def test_task_status(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test task status parsing."""
        roadmap = parser.parse_string(sample_roadmap)

        section = roadmap.phases[0].sections[0]
        assert section.tasks[0].status == TaskStatus.COMPLETED
        assert section.tasks[1].status == TaskStatus.COMPLETED

        section2 = roadmap.phases[0].sections[1]
        assert section2.tasks[0].status == TaskStatus.PENDING

    def test_task_id_generation(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test automatic task ID generation."""
        roadmap = parser.parse_string(sample_roadmap)

        # Phase 0, Section 0.1
        section = roadmap.phases[0].sections[0]
        assert section.tasks[0].id == "0.1.1"
        assert section.tasks[1].id == "0.1.2"

        # Phase 0, Section 0.2
        section2 = roadmap.phases[0].sections[1]
        assert section2.tasks[0].id == "0.2.1"
        assert section2.tasks[1].id == "0.2.2"

    def test_task_description(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test task description extraction."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.phases[0].sections[0].tasks[0]
        assert "Set up pyproject.toml" in task.description

    def test_task_file_hints_inherited(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test that tasks inherit section file hints."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.phases[0].sections[0].tasks[0]
        assert "pyproject.toml" in task.file_hints
        assert "src/__init__.py" in task.file_hints

    def test_task_code_context(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test code block extraction."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.phases[0].sections[1].tasks[0]
        assert len(task.code_context) == 1
        assert "class User(BaseModel)" in task.code_context[0]

    def test_task_context(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test full context string."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.phases[0].sections[0].tasks[0]
        assert task.context.startswith("Task: Initialize Poetry project")

    def test_bold_title_extraction(self, parser: RoadmapParser) -> None:
        """Test extracting title from bold text."""
        content = """\
## Phase 1: Test
### 1.1 Section
- [ ] **This is the title**
  Some description.
"""
        roadmap = parser.parse_string(content)
        task = roadmap.phases[0].sections[0].tasks[0]

        assert task.title == "This is the title"

    def test_non_bold_title(self, parser: RoadmapParser) -> None:
        """Test title without bold formatting."""
        content = """\
## Phase 1: Test
### 1.1 Section
- [ ] Simple task title
  Some description.
"""
        roadmap = parser.parse_string(content)
        task = roadmap.phases[0].sections[0].tasks[0]

        assert task.title == "Simple task title"


# =============================================================================
# Subtask Parsing Tests
# =============================================================================


class TestSubtaskParsing:
    """Test subtask parsing."""

    def test_parse_subtasks(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test subtask extraction."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.phases[0].sections[1].tasks[0]
        assert len(task.subtasks) == 3

    def test_subtask_ids(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test subtask ID generation."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.phases[0].sections[1].tasks[0]
        assert task.subtasks[0].id == "0.2.1.1"
        assert task.subtasks[1].id == "0.2.1.2"
        assert task.subtasks[2].id == "0.2.1.3"

    def test_subtask_titles(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test subtask title extraction."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.phases[0].sections[1].tasks[0]
        assert task.subtasks[0].title == "Add User model"
        assert task.subtasks[1].title == "Add Project model"
        assert task.subtasks[2].title == "Add Task model"

    def test_subtask_status(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test subtask completion status."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.phases[0].sections[1].tasks[0]
        assert task.subtasks[0].completed is False
        assert task.subtasks[1].completed is False
        assert task.subtasks[2].completed is True


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and malformed input."""

    def test_no_phases(self, parser: RoadmapParser) -> None:
        """Test content without phase headers."""
        content = "# Title\n\nJust some text.\n"
        roadmap = parser.parse_string(content)

        # Should create no phases if no tasks
        assert len(roadmap.phases) == 0

    def test_no_sections(self, parser: RoadmapParser) -> None:
        """Test phase without section headers."""
        content = """\
## Phase 1: Test

- [ ] **Direct task**
  No section header above.
"""
        roadmap = parser.parse_string(content)

        # Should create default section
        assert len(roadmap.phases) == 1
        assert len(roadmap.phases[0].sections) == 1
        assert roadmap.phases[0].sections[0].title == "Default"
        assert len(roadmap.phases[0].sections[0].tasks) == 1

    def test_empty_content(self, parser: RoadmapParser) -> None:
        """Test empty content."""
        roadmap = parser.parse_string("")

        assert roadmap.title == ""
        assert len(roadmap.phases) == 0

    def test_mixed_checkbox_styles(self, parser: RoadmapParser) -> None:
        """Test different checkbox markers."""
        content = """\
## Phase 1: Test
### 1.1 Section
- [ ] **Task with dash**
* [ ] **Task with asterisk**
+ [ ] **Task with plus**
"""
        roadmap = parser.parse_string(content)
        section = roadmap.phases[0].sections[0]

        assert len(section.tasks) == 3
        assert section.tasks[0].title == "Task with dash"
        assert section.tasks[1].title == "Task with asterisk"
        assert section.tasks[2].title == "Task with plus"

    def test_uppercase_checkbox(self, parser: RoadmapParser) -> None:
        """Test uppercase X in checkbox."""
        content = """\
## Phase 1: Test
### 1.1 Section
- [X] **Completed with uppercase X**
"""
        roadmap = parser.parse_string(content)
        task = roadmap.phases[0].sections[0].tasks[0]

        assert task.status == TaskStatus.COMPLETED

    def test_deeply_nested_lists(self, parser: RoadmapParser) -> None:
        """Test deeply nested list items."""
        content = """\
## Phase 1: Test
### 1.1 Section
- [ ] **Parent task**
  - [ ] Level 1 subtask
    - Regular list item (not a subtask)
      - Even deeper nesting
"""
        roadmap = parser.parse_string(content)
        task = roadmap.phases[0].sections[0].tasks[0]

        # Should capture as subtask
        assert len(task.subtasks) == 1
        assert task.subtasks[0].title == "Level 1 subtask"

    def test_unclosed_code_block(self, parser: RoadmapParser) -> None:
        """Test handling unclosed code block."""
        content = """\
## Phase 1: Test
### 1.1 Section
- [ ] **Task with broken code**
  ```python
  def broken():
      pass
  # No closing backticks

- [ ] **Next task**
"""
        # Should not crash
        roadmap = parser.parse_string(content)
        assert len(roadmap.phases) == 1

    def test_horizontal_rules_ignored(self, parser: RoadmapParser) -> None:
        """Test that horizontal rules don't break parsing."""
        content = """\
# Title

---

## Phase 1: Test

---

### 1.1 Section

- [ ] **Task**

---
"""
        roadmap = parser.parse_string(content)

        assert len(roadmap.phases) == 1
        assert roadmap.phases[0].sections[0].tasks[0].title == "Task"

    def test_html_comments_in_content(self, parser: RoadmapParser) -> None:
        """Test HTML comments are preserved in context."""
        content = """\
## Phase 1: Test
### 1.1 Section
- [ ] **Task**
  Some text <!-- comment --> more text
"""
        roadmap = parser.parse_string(content)
        task = roadmap.phases[0].sections[0].tasks[0]

        assert "<!-- comment -->" in task.context

    def test_special_characters_in_title(self, parser: RoadmapParser) -> None:
        """Test special characters in task title."""
        content = """\
## Phase 1: Test
### 1.1 Section
- [ ] **Task with `code` and (parentheses) and [brackets]**
"""
        roadmap = parser.parse_string(content)
        task = roadmap.phases[0].sections[0].tasks[0]

        assert "code" in task.title
        assert "parentheses" in task.title


# =============================================================================
# Parser Configuration Tests
# =============================================================================


class TestParserConfig:
    """Test parser configuration options."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ParserConfig()

        assert config.strict is False
        assert config.extract_code_blocks is True
        assert config.default_phase_id == "0"
        assert config.default_section_id == "0.1"

    def test_strict_mode(self) -> None:
        """Test strict mode configuration."""
        config = ParserConfig(strict=True)
        parser = RoadmapParser(config)

        assert parser.config.strict is True

    def test_disable_code_extraction(self) -> None:
        """Test disabling code block extraction."""
        content = """\
## Phase 1: Test
### 1.1 Section
- [ ] **Task**
  ```python
  def example():
      pass
  ```
"""
        config = ParserConfig(extract_code_blocks=False)
        parser = RoadmapParser(config)
        roadmap = parser.parse_string(content)

        task = roadmap.phases[0].sections[0].tasks[0]
        assert len(task.code_context) == 0

    def test_custom_default_ids(self) -> None:
        """Test custom default phase/section IDs."""
        content = "## Phase 1: Test\n- [ ] **Orphan task**\n"
        config = ParserConfig(default_phase_id="X", default_section_id="X.1")
        parser = RoadmapParser(config)
        roadmap = parser.parse_string(content)

        # Phase parsed, but default section used
        assert len(roadmap.phases) == 1
        assert roadmap.phases[0].sections[0].id == "X.1"


# =============================================================================
# Roadmap Object Tests
# =============================================================================


class TestRoadmapObject:
    """Test Roadmap object methods."""

    def test_total_tasks(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test total task count."""
        roadmap = parser.parse_string(sample_roadmap)

        assert roadmap.total_tasks == 6

    def test_pending_count(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test pending task count."""
        roadmap = parser.parse_string(sample_roadmap)

        assert roadmap.pending_count == 4

    def test_completed_count(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test completed task count."""
        roadmap = parser.parse_string(sample_roadmap)

        assert roadmap.completed_count == 2

    def test_progress(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test progress calculation."""
        roadmap = parser.parse_string(sample_roadmap)

        # 2 completed out of 6 = 0.333...
        assert 0.33 <= roadmap.progress <= 0.34

    def test_next_pending(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test getting next pending task."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.next_pending()
        assert task is not None
        assert task.status == TaskStatus.PENDING
        assert task.id == "0.2.1"

    def test_get_task(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test getting task by ID."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.get_task("0.1.1")
        assert task is not None
        assert task.title == "Initialize Poetry project"

        missing = roadmap.get_task("99.99.99")
        assert missing is None

    def test_get_phase(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test getting phase by ID."""
        roadmap = parser.parse_string(sample_roadmap)

        phase = roadmap.get_phase("0")
        assert phase is not None
        assert phase.name == "Foundation"

        missing = roadmap.get_phase("99")
        assert missing is None

    def test_summary(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test summary generation."""
        roadmap = parser.parse_string(sample_roadmap, path="ROADMAP.md")

        summary = roadmap.summary()
        assert "ROADMAP.md" in summary
        assert "Phases: 2" in summary
        assert "Foundation" in summary

    def test_all_tasks_iterator(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test iterating all tasks."""
        roadmap = parser.parse_string(sample_roadmap)

        tasks = list(roadmap.all_tasks())
        assert len(tasks) == 6

    def test_pending_tasks_iterator(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test iterating pending tasks."""
        roadmap = parser.parse_string(sample_roadmap)

        pending = list(roadmap.pending_tasks())
        assert len(pending) == 4
        assert all(t.status == TaskStatus.PENDING for t in pending)


# =============================================================================
# Task Conversion Tests
# =============================================================================


class TestTaskConversion:
    """Test ParsedTask to Task conversion."""

    def test_to_task(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test converting ParsedTask to Task."""
        roadmap = parser.parse_string(sample_roadmap)

        parsed = roadmap.get_task("0.2.1")
        assert parsed is not None

        task = parsed.to_task()
        assert task.id == "0.2.1"
        assert task.title == "Define data models"
        assert task.status == TaskStatus.PENDING
        assert "src/models.py" in task.file_hints
        assert task.phase == "0"
        assert task.section == "0.2"
        assert "subtasks" in task.metadata
        assert "code_context" in task.metadata


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling and reporting."""

    def test_errors_property(self, parser: RoadmapParser) -> None:
        """Test accessing parse errors."""
        content = "## Phase 1: Test\n### 1.1 Section\n- [ ] **Task**\n"
        parser.parse_string(content)

        # Should have no errors for valid content
        assert len(parser.errors) == 0

    def test_parse_error_class(self) -> None:
        """Test ParseError class."""
        error = ParseError(
            line_number=10,
            message="Invalid format",
            context="some context text",
        )

        assert error.line_number == 10
        assert "Line 10" in str(error)
        assert "Invalid format" in str(error)
        assert "some context" in str(error)

    def test_parse_error_without_context(self) -> None:
        """Test ParseError without context."""
        error = ParseError(line_number=5, message="Error message")

        assert str(error) == "Line 5: Error message"


# =============================================================================
# Complex Document Tests
# =============================================================================


class TestComplexDocuments:
    """Test parsing complex ROADMAP documents."""

    def test_multiple_code_blocks(self, parser: RoadmapParser) -> None:
        """Test task with multiple code blocks."""
        content = """\
## Phase 1: Test
### 1.1 Section
- [ ] **Multi-code task**
  First example:
  ```python
  def first():
      pass
  ```

  Second example:
  ```javascript
  function second() {}
  ```
"""
        roadmap = parser.parse_string(content)
        task = roadmap.phases[0].sections[0].tasks[0]

        assert len(task.code_context) == 2
        assert any("python" in c for c in task.code_context)
        assert any("javascript" in c for c in task.code_context)

    def test_large_document(self, parser: RoadmapParser) -> None:
        """Test parsing a large document."""
        phases = []
        for p in range(5):
            sections = []
            for s in range(3):
                tasks = "\n".join(
                    f"- [ ] **Task {p}.{s}.{t}**\n  Description for task {t}." for t in range(1, 6)
                )
                sections.append(f"### {p}.{s + 1} Section {s + 1}\n\n{tasks}")
            phases.append(f"## Phase {p}: Phase {p}\n\n" + "\n\n".join(sections))

        content = "\n\n".join(phases)
        roadmap = parser.parse_string(content)

        assert len(roadmap.phases) == 5
        assert roadmap.total_tasks == 75  # 5 phases * 3 sections * 5 tasks

    def test_real_world_structure(self, parser: RoadmapParser) -> None:
        """Test realistic ROADMAP structure."""
        content = """\
# Project ROADMAP

This document tracks the development progress of the project.

## Phase 0: Foundation

> **Goal**: Establish core infrastructure
> **Priority**: HIGH
> **Effort**: 1 week

### 0.1 Project Setup

**Files**: `pyproject.toml`, `setup.cfg`

- [x] **Initialize repository**
  Create Git repo with proper .gitignore.

- [x] **Set up CI/CD**
  Configure GitHub Actions.

### 0.2 Documentation

**Files**: `docs/`, `README.md`

- [ ] **Write README**
  Project description, installation, usage.

- [ ] **Add contributing guide**
  How to contribute to the project.
  - [ ] Code of conduct
  - [ ] Pull request template
  - [ ] Issue templates

## Phase 1: Core Implementation

> **Goal**: Build main features
> **Priority**: HIGH
> **Prerequisite**: Phase 0 complete

### 1.1 Core Module

**Files**: `src/core/`

- [ ] **Design API surface**
  Define public interfaces.

  ```python
  class CoreService:
      def process(self, data: dict) -> Result:
          pass
  ```

- [ ] **Implement processing pipeline**
  Main data processing logic.
"""
        roadmap = parser.parse_string(content, path="ROADMAP.md")

        assert roadmap.title == "Project ROADMAP"
        assert len(roadmap.phases) == 2
        assert roadmap.total_tasks == 6
        assert roadmap.completed_count == 2
        assert roadmap.pending_count == 4

        # Check specific task
        task = roadmap.get_task("0.2.2")
        assert task is not None
        assert "contributing" in task.title.lower()
        assert len(task.subtasks) == 3


# =============================================================================
# Integration with Roadmap Models
# =============================================================================


class TestModelIntegration:
    """Test integration with roadmap data models."""

    def test_parsed_to_roadmap(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test that parser produces valid Roadmap objects."""
        roadmap = parser.parse_string(sample_roadmap)

        # Verify type
        assert isinstance(roadmap, Roadmap)
        assert all(isinstance(p, Phase) for p in roadmap.phases)
        assert all(isinstance(s, Section) for p in roadmap.phases for s in p.sections)
        assert all(
            isinstance(t, ParsedTask) for p in roadmap.phases for s in p.sections for t in s.tasks
        )

    def test_subtask_type(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test subtask types."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.get_task("0.2.1")
        assert task is not None
        assert all(isinstance(st, Subtask) for st in task.subtasks)

    def test_task_properties(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test ParsedTask property methods."""
        roadmap = parser.parse_string(sample_roadmap)

        task = roadmap.get_task("0.2.1")
        assert task is not None
        assert task.is_pending is True
        assert task.is_completed is False
        assert task.subtask_count == 3
        assert task.completed_subtask_count == 1

    def test_phase_properties(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test Phase property methods."""
        roadmap = parser.parse_string(sample_roadmap)

        phase = roadmap.get_phase("0")
        assert phase is not None
        assert phase.task_count == 4
        assert phase.completed_count == 2
        assert phase.pending_count == 2

    def test_section_properties(self, parser: RoadmapParser, sample_roadmap: str) -> None:
        """Test Section property methods."""
        roadmap = parser.parse_string(sample_roadmap)

        section = roadmap.get_section("0.1")
        assert section is not None
        assert section.task_count == 2
        assert section.completed_count == 2
        assert section.pending_count == 0
