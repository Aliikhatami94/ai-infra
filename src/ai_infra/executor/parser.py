"""ROADMAP.md parser for the Executor module.

This module implements parsing of ROADMAP.md files into structured Roadmap
objects as specified in docs/executor/roadmap-format.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

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
# Regex Patterns
# =============================================================================

# Phase header: ## Phase N: Name (case-insensitive)
PHASE_PATTERN = re.compile(
    r"^##\s+phase\s+([^\s:]+):\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)

# Section header: ### N.M Title
SECTION_PATTERN = re.compile(
    r"^###\s+(\d+(?:\.\d+)*)\s+(.+)$",
    re.MULTILINE,
)

# File hints: **Files**: `file1`, `file2` or **File**: `file`
FILE_HINT_PATTERN = re.compile(
    r"^\*\*Files?\*\*:\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)

# Backtick file extraction
BACKTICK_FILE_PATTERN = re.compile(r"`([^`]+)`")

# Task checkbox: - [ ] or - [x] or * [ ] or * [x] or + [ ] or + [x]
TASK_PATTERN = re.compile(
    r"^[-*+]\s+\[([ xX])\]\s+(.+)$",
    re.MULTILINE,
)

# Bold title extraction: **title**
BOLD_PATTERN = re.compile(r"\*\*([^*]+)\*\*")

# Subtask checkbox (indented): must have leading whitespace
SUBTASK_PATTERN = re.compile(
    r"^\s+[-*+]\s+\[([ xX])\]\s+(.+)$",
    re.MULTILINE,
)

# Code block: ```language ... ```
CODE_BLOCK_PATTERN = re.compile(
    r"```(\w*)\n(.*?)```",
    re.MULTILINE | re.DOTALL,
)

# Phase metadata: > **Key**: Value
METADATA_PATTERN = re.compile(
    r"^>\s*\*\*(\w+)\*\*:\s*(.+)$",
    re.MULTILINE,
)


# =============================================================================
# Parser Configuration
# =============================================================================


@dataclass
class ParserConfig:
    """Configuration for ROADMAP parsing.

    Attributes:
        strict: If True, raise errors on malformed input. If False, gracefully degrade.
        extract_code_blocks: Whether to extract code blocks into code_context.
        default_phase_id: Phase ID for tasks without a phase.
        default_section_id: Section ID for tasks without a section.
    """

    strict: bool = False
    extract_code_blocks: bool = True
    default_phase_id: str = "0"
    default_section_id: str = "0.1"


# =============================================================================
# Parser Errors
# =============================================================================


@dataclass
class ParseError:
    """A parsing error with location information.

    Attributes:
        line_number: Line where the error occurred.
        message: Error description.
        context: Surrounding text for context.
    """

    line_number: int
    message: str
    context: str = ""

    def __str__(self) -> str:
        """Format error for display."""
        if self.context:
            return f"Line {self.line_number}: {self.message} (near: {self.context[:50]})"
        return f"Line {self.line_number}: {self.message}"


# =============================================================================
# RoadmapParser
# =============================================================================


class RoadmapParser:
    """Parser for ROADMAP.md files.

    Parses Markdown files following the ROADMAP format specification into
    structured Roadmap objects with phases, sections, and tasks.

    Example:
        >>> parser = RoadmapParser()
        >>> roadmap = parser.parse("./ROADMAP.md")
        >>> print(f"Found {len(roadmap.phases)} phases")
        >>> print(f"Total tasks: {roadmap.total_tasks}")
        >>> print(f"Pending: {roadmap.pending_count}")
        >>>
        >>> task = roadmap.next_pending()
        >>> print(f"Next: {task.id} - {task.title}")

    Attributes:
        config: Parser configuration.
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        """Initialize the parser.

        Args:
            config: Parser configuration. Uses defaults if not provided.
        """
        self.config = config or ParserConfig()
        self._errors: list[ParseError] = []

    def parse(self, path: str | Path) -> Roadmap:
        """Parse a ROADMAP.md file.

        Args:
            path: Path to the ROADMAP.md file.

        Returns:
            Parsed Roadmap object.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If strict mode and parsing fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ROADMAP file not found: {path}")

        with open(path, encoding="utf-8") as f:
            return self.parse_file(f, str(path))

    def parse_file(self, file: TextIO, path: str = "") -> Roadmap:
        """Parse a ROADMAP from a file object.

        Args:
            file: File-like object with ROADMAP content.
            path: Path for the resulting Roadmap (metadata only).

        Returns:
            Parsed Roadmap object.
        """
        content = file.read()
        return self.parse_string(content, path)

    def parse_string(self, content: str, path: str = "") -> Roadmap:
        """Parse a ROADMAP from a string.

        Args:
            content: ROADMAP content as string.
            path: Path for the resulting Roadmap (metadata only).

        Returns:
            Parsed Roadmap object.
        """
        self._errors = []
        lines = content.split("\n")

        # Extract document title (first # heading)
        title = self._extract_title(lines)

        # Parse into phases
        phases = self._parse_phases(content, lines)

        # Build roadmap
        roadmap = Roadmap(
            path=path,
            title=title,
            phases=phases,
            parse_errors=[str(e) for e in self._errors],
        )

        return roadmap

    # =========================================================================
    # Title Extraction
    # =========================================================================

    def _extract_title(self, lines: list[str]) -> str:
        """Extract document title from first # heading."""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                return stripped[2:].strip()
        return ""

    # =========================================================================
    # Phase Parsing
    # =========================================================================

    def _parse_phases(self, content: str, lines: list[str]) -> list[Phase]:
        """Parse all phases from content."""
        phases: list[Phase] = []

        # Find all phase headers and their positions
        phase_matches = list(PHASE_PATTERN.finditer(content))

        if not phase_matches:
            # No phases found - create default phase with all content
            default_phase = self._create_default_phase(content, lines)
            if default_phase.task_count > 0:
                phases.append(default_phase)
            return phases

        # Process each phase
        for i, match in enumerate(phase_matches):
            phase_id = match.group(1)
            phase_name = match.group(2).strip()
            phase_start = match.end()

            # Determine phase end
            if i + 1 < len(phase_matches):
                phase_end = phase_matches[i + 1].start()
            else:
                phase_end = len(content)

            phase_content = content[phase_start:phase_end]
            phase_line = content[: match.start()].count("\n") + 1

            # Parse phase metadata and sections
            phase = self._parse_phase(
                phase_id=phase_id,
                phase_name=phase_name,
                content=phase_content,
                line_number=phase_line,
            )
            phases.append(phase)

        return phases

    def _parse_phase(
        self,
        phase_id: str,
        phase_name: str,
        content: str,
        line_number: int,
    ) -> Phase:
        """Parse a single phase from its content."""
        # Extract metadata block
        goal, priority, effort, prerequisite = self._extract_phase_metadata(content)

        # Extract description (non-metadata text before first section)
        description = self._extract_phase_description(content)

        # Parse sections
        sections = self._parse_sections(content, phase_id, line_number)

        return Phase(
            id=phase_id,
            name=phase_name,
            goal=goal,
            priority=priority,
            effort=effort,
            prerequisite=prerequisite,
            description=description,
            sections=sections,
            line_number=line_number,
        )

    def _extract_phase_metadata(self, content: str) -> tuple[str, Priority, str, str]:
        """Extract metadata from phase content."""
        goal = ""
        priority = Priority.MEDIUM
        effort = ""
        prerequisite = ""

        for match in METADATA_PATTERN.finditer(content):
            key = match.group(1).lower()
            value = match.group(2).strip()

            if key == "goal":
                goal = value
            elif key == "priority":
                priority = Priority.from_string(value)
            elif key == "effort":
                effort = value
            elif key == "prerequisite":
                prerequisite = value

        return goal, priority, effort, prerequisite

    def _extract_phase_description(self, content: str) -> str:
        """Extract description text before first section."""
        # Find first section header
        section_match = SECTION_PATTERN.search(content)
        if not section_match:
            return ""

        # Get content before section, skip metadata lines
        before_section = content[: section_match.start()]
        lines = before_section.split("\n")
        description_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip metadata lines and empty lines at start
            if stripped.startswith(">"):
                continue
            if stripped:
                description_lines.append(line)

        return "\n".join(description_lines).strip()

    def _create_default_phase(self, content: str, lines: list[str]) -> Phase:
        """Create a default phase for content without phase headers."""
        sections = self._parse_sections(content, self.config.default_phase_id, 1)

        return Phase(
            id=self.config.default_phase_id,
            name="Default",
            sections=sections,
            line_number=1,
        )

    # =========================================================================
    # Section Parsing
    # =========================================================================

    def _parse_sections(self, content: str, phase_id: str, phase_line: int) -> list[Section]:
        """Parse all sections from phase content."""
        sections: list[Section] = []

        # Find all section headers
        section_matches = list(SECTION_PATTERN.finditer(content))

        if not section_matches:
            # No sections - create default section with tasks
            default_section = self._create_default_section(content, phase_id, phase_line)
            if default_section.task_count > 0:
                sections.append(default_section)
            return sections

        # Process each section
        for i, match in enumerate(section_matches):
            section_id = match.group(1)
            section_title = match.group(2).strip()
            section_start = match.end()

            # Determine section end
            if i + 1 < len(section_matches):
                section_end = section_matches[i + 1].start()
            else:
                section_end = len(content)

            section_content = content[section_start:section_end]
            section_line = phase_line + content[: match.start()].count("\n")

            section = self._parse_section(
                section_id=section_id,
                section_title=section_title,
                content=section_content,
                phase_id=phase_id,
                line_number=section_line,
            )
            sections.append(section)

        return sections

    def _parse_section(
        self,
        section_id: str,
        section_title: str,
        content: str,
        phase_id: str,
        line_number: int,
    ) -> Section:
        """Parse a single section from its content."""
        # Extract file hints
        file_hints = self._extract_file_hints(content)

        # Extract description (before first task)
        description = self._extract_section_description(content)

        # Parse tasks
        tasks = self._parse_tasks(content, phase_id, section_id, line_number)

        return Section(
            id=section_id,
            title=section_title,
            description=description,
            file_hints=file_hints,
            tasks=tasks,
            phase_id=phase_id,
            line_number=line_number,
        )

    def _extract_section_description(self, content: str) -> str:
        """Extract description text before first task."""
        # Find first task
        task_match = TASK_PATTERN.search(content)
        if not task_match:
            return ""

        before_task = content[: task_match.start()]
        lines = before_task.split("\n")
        description_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip file hints
            if FILE_HINT_PATTERN.match(stripped):
                continue
            if stripped:
                description_lines.append(line)

        return "\n".join(description_lines).strip()

    def _create_default_section(self, content: str, phase_id: str, line_number: int) -> Section:
        """Create a default section for content without section headers."""
        file_hints = self._extract_file_hints(content)
        tasks = self._parse_tasks(content, phase_id, self.config.default_section_id, line_number)

        return Section(
            id=self.config.default_section_id,
            title="Default",
            file_hints=file_hints,
            tasks=tasks,
            phase_id=phase_id,
            line_number=line_number,
        )

    # =========================================================================
    # File Hint Extraction
    # =========================================================================

    def _extract_file_hints(self, content: str) -> list[str]:
        """Extract file hints from content."""
        files: list[str] = []

        for match in FILE_HINT_PATTERN.finditer(content):
            hint_line = match.group(1)
            # Extract files from backticks
            for file_match in BACKTICK_FILE_PATTERN.finditer(hint_line):
                file_path = file_match.group(1).strip()
                if file_path and file_path not in files:
                    files.append(file_path)

        return files

    # =========================================================================
    # Task Parsing
    # =========================================================================

    def _parse_tasks(
        self,
        content: str,
        phase_id: str,
        section_id: str,
        base_line: int,
    ) -> list[ParsedTask]:
        """Parse all tasks from section content."""
        tasks: list[ParsedTask] = []
        content.split("\n")

        # Find section-level file hints (apply to all tasks)
        section_file_hints = self._extract_file_hints(content)

        # Find all task checkboxes (non-indented)
        task_matches = list(TASK_PATTERN.finditer(content))

        for i, match in enumerate(task_matches):
            # Skip if this is indented (subtask)
            line_start = content.rfind("\n", 0, match.start())
            if line_start == -1:
                line_start = 0
            else:
                line_start += 1

            line_text = content[line_start : match.end()]
            if line_text.startswith((" ", "\t")):
                continue

            task_num = len(tasks) + 1
            task_id = f"{phase_id}.{section_id.split('.')[-1]}.{task_num}"

            checkbox = match.group(1)
            title_raw = match.group(2).strip()
            task_start = match.end()

            # Determine task end
            if i + 1 < len(task_matches):
                # Find next non-indented task
                task_end = self._find_next_task_start(content, i, task_matches)
            else:
                task_end = len(content)

            task_content = content[task_start:task_end]
            task_line = base_line + content[: match.start()].count("\n")

            # Determine status
            status = TaskStatus.COMPLETED if checkbox.lower() == "x" else TaskStatus.PENDING

            # Extract title (from bold if present)
            title = self._extract_title_from_raw(title_raw)

            # Parse task content
            task = self._parse_task_content(
                task_id=task_id,
                title=title,
                status=status,
                content=task_content,
                phase_id=phase_id,
                section_id=section_id,
                line_number=task_line,
                section_file_hints=section_file_hints,
            )
            tasks.append(task)

        return tasks

    def _find_next_task_start(
        self, content: str, current_idx: int, task_matches: list[re.Match]
    ) -> int:
        """Find the start of the next non-indented task."""
        for j in range(current_idx + 1, len(task_matches)):
            match = task_matches[j]
            line_start = content.rfind("\n", 0, match.start())
            if line_start == -1:
                line_start = 0
            else:
                line_start += 1

            line_text = content[line_start : match.end()]
            if not line_text.startswith((" ", "\t")):
                return match.start()

        return len(content)

    def _extract_title_from_raw(self, raw: str) -> str:
        """Extract title from raw text, handling bold formatting."""
        bold_match = BOLD_PATTERN.search(raw)
        if bold_match:
            return bold_match.group(1).strip()
        return raw.strip()

    def _parse_task_content(
        self,
        task_id: str,
        title: str,
        status: TaskStatus,
        content: str,
        phase_id: str,
        section_id: str,
        line_number: int,
        section_file_hints: list[str],
    ) -> ParsedTask:
        """Parse task content to extract description, code, and subtasks."""
        # Extract subtasks
        subtasks = self._extract_subtasks(content, task_id)

        # Extract code blocks
        code_context = self._extract_code_blocks(content)

        # Extract task-specific file hints
        task_file_hints = self._extract_file_hints(content)

        # Combine file hints (section + task specific)
        all_file_hints = section_file_hints.copy()
        for hint in task_file_hints:
            if hint not in all_file_hints:
                all_file_hints.append(hint)

        # Extract description (first paragraph before code/subtasks)
        description = self._extract_task_description(content)

        # Build full context
        context = self._build_task_context(title, content)

        return ParsedTask(
            id=task_id,
            title=title,
            description=description,
            status=status,
            file_hints=all_file_hints,
            code_context=code_context,
            subtasks=subtasks,
            context=context,
            phase_id=phase_id,
            section_id=section_id,
            line_number=line_number,
        )

    def _extract_subtasks(self, content: str, parent_id: str) -> list[Subtask]:
        """Extract subtasks from task content."""
        subtasks: list[Subtask] = []

        for match in SUBTASK_PATTERN.finditer(content):
            subtask_num = len(subtasks) + 1
            subtask_id = f"{parent_id}.{subtask_num}"
            checkbox = match.group(1)
            title = match.group(2).strip()

            # Remove bold if present
            title = self._extract_title_from_raw(title)

            completed = checkbox.lower() == "x"
            line_offset = content[: match.start()].count("\n")

            subtasks.append(
                Subtask(
                    id=subtask_id,
                    title=title,
                    completed=completed,
                    line_number=line_offset,
                )
            )

        return subtasks

    def _extract_code_blocks(self, content: str) -> list[str]:
        """Extract code blocks from content."""
        if not self.config.extract_code_blocks:
            return []

        blocks: list[str] = []
        for match in CODE_BLOCK_PATTERN.finditer(content):
            language = match.group(1)
            code = match.group(2)
            if code.strip():
                if language:
                    blocks.append(f"```{language}\n{code}```")
                else:
                    blocks.append(f"```\n{code}```")

        return blocks

    def _extract_task_description(self, content: str) -> str:
        """Extract description from task content."""
        lines = content.strip().split("\n")
        description_lines = []

        in_code_block = False
        for line in lines:
            stripped = line.strip()

            # Track code blocks
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                break  # Stop at first code block

            if in_code_block:
                continue

            # Stop at subtasks
            if SUBTASK_PATTERN.match(line):
                break

            # Skip file hints
            if FILE_HINT_PATTERN.match(stripped):
                continue

            # Add non-empty lines
            if stripped:
                description_lines.append(stripped)
            elif description_lines:
                # Keep one blank line if we have content
                description_lines.append("")

        # Clean up trailing empty lines
        while description_lines and not description_lines[-1]:
            description_lines.pop()

        return "\n".join(description_lines)

    def _build_task_context(self, title: str, content: str) -> str:
        """Build full context string for agent execution.

        Includes task title and content, but stops at section headers to avoid
        bleeding content from subsequent tasks into this task's context.
        """
        # Include title and clean content
        context_lines = [f"Task: {title}", ""]

        # Add content, but stop at section headers (which belong to next task)
        for line in content.split("\n"):
            stripped = line.strip()
            # Stop at markdown headers (## or ### or ####) which indicate next section/task
            if stripped.startswith("#"):
                break
            context_lines.append(line.rstrip())

        return "\n".join(context_lines).strip()

    # =========================================================================
    # Error Handling
    # =========================================================================

    def _add_error(self, line: int, message: str, context: str = "") -> None:
        """Add a parse error."""
        error = ParseError(line_number=line, message=message, context=context)
        self._errors.append(error)

        if self.config.strict:
            raise ValueError(str(error))

    @property
    def errors(self) -> list[ParseError]:
        """Get parse errors from last parse operation."""
        return self._errors.copy()
