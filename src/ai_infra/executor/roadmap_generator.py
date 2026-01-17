"""Roadmap generator for natural language to ROADMAP.md conversion.

This module generates standard ROADMAP.md files from natural language prompts,
enabling autonomous task execution without manual roadmap authoring.

Phase 3.1.1 of EXECUTOR_1.md - Intelligent Task Handling.

Usage:
    from ai_infra.executor.roadmap_generator import RoadmapGenerator, GeneratedRoadmap

    generator = RoadmapGenerator(agent)
    roadmap = await generator.generate(
        prompt="Add JWT authentication to my FastAPI app",
        workspace=Path("/path/to/project"),
    )
    print(roadmap.content)  # Full ROADMAP.md content

The generated ROADMAP.md follows the standard format that the existing
parser already understands, so no changes to parsing or execution are needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.executor.project_analyzer import ProjectAnalyzer, ProjectInfo

if TYPE_CHECKING:
    from ai_infra.llm.agent import Agent

__all__ = [
    "RoadmapGenerator",
    "GeneratedRoadmap",
    "GenerationStyle",
    "ValidationIssue",
]


# =============================================================================
# Generation Style Enum
# =============================================================================


class GenerationStyle:
    """Roadmap generation styles."""

    MINIMAL = "minimal"  # 3-5 high-level tasks
    STANDARD = "standard"  # 5-10 tasks with some detail
    DETAILED = "detailed"  # 10+ tasks with full breakdown


# =============================================================================
# Validation Issue
# =============================================================================


@dataclass
class ValidationIssue:
    """An issue found during roadmap validation.

    Attributes:
        task_id: ID of the task with the issue.
        issue_type: Type of issue (dependency, file, scope, etc.).
        message: Human-readable description of the issue.
        severity: low, medium, or high.
    """

    task_id: str
    issue_type: str
    message: str
    severity: str = "medium"


# =============================================================================
# Generated Roadmap
# =============================================================================


@dataclass
class GeneratedRoadmap:
    """A generated ROADMAP.md from natural language.

    Attributes:
        content: Full markdown content for ROADMAP.md.
        title: Roadmap title extracted from content.
        task_count: Number of tasks in the roadmap.
        estimated_time: Estimated total time (e.g., "2-3 hours").
        complexity: Overall complexity (low, medium, high).
        confidence: Confidence score (0.0 to 1.0).
        validation_issues: List of validation issues found.
        project_info: Project context used for generation.
    """

    content: str
    title: str = ""
    task_count: int = 0
    estimated_time: str = ""
    complexity: str = "medium"
    confidence: float = 1.0
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    project_info: ProjectInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "task_count": self.task_count,
            "estimated_time": self.estimated_time,
            "complexity": self.complexity,
            "confidence": self.confidence,
            "validation_issues": [
                {
                    "task_id": i.task_id,
                    "issue_type": i.issue_type,
                    "message": i.message,
                    "severity": i.severity,
                }
                for i in self.validation_issues
            ],
            "content_length": len(self.content),
        }

    @property
    def is_valid(self) -> bool:
        """Check if roadmap has no high-severity issues."""
        return not any(i.severity == "high" for i in self.validation_issues)

    @property
    def has_warnings(self) -> bool:
        """Check if roadmap has medium-severity issues."""
        return any(i.severity == "medium" for i in self.validation_issues)


# =============================================================================
# Prompt Templates
# =============================================================================


GENERATION_PROMPT_TEMPLATE = """You are a senior software architect planning development work.

USER REQUEST:
{prompt}

PROJECT CONTEXT:
{project_context}

Generate a ROADMAP.md file with clear, actionable tasks.

STYLE: {style}
- minimal: 3-5 high-level tasks only
- standard: 5-10 tasks with reasonable detail
- detailed: 10+ tasks with full breakdown including subtasks

REQUIREMENTS:
1. Each task must be independently verifiable
2. Tasks should build on each other logically
3. Include testing tasks where appropriate
4. Reference specific files when possible
5. Keep task titles concise but descriptive

OUTPUT FORMAT (use exactly this markdown format):
```markdown
# Roadmap: [Descriptive Title]

## Overview
[2-3 sentence description of what this roadmap accomplishes]

## Estimated Time
[Total estimated time, e.g., "2-4 hours"]

## Complexity
[low/medium/high]

## Tasks

### Phase 1: [Phase Name]

- [ ] **Task 1.1**: [Task title]
  - Description: [What this task accomplishes]
  - Files: [Expected files to create/modify]

- [ ] **Task 1.2**: [Task title]
  - Description: [What this task accomplishes]
  - Files: [Expected files to create/modify]
  - Depends on: Task 1.1

### Phase 2: [Phase Name]

- [ ] **Task 2.1**: [Task title]
  - Description: [What this task accomplishes]
  - Files: [Expected files to create/modify]
  - Depends on: Task 1.2

## Success Criteria
- [Criterion 1]
- [Criterion 2]
```

Generate the ROADMAP.md content now:"""


# =============================================================================
# Roadmap Generator
# =============================================================================


class RoadmapGenerator:
    """Generate ROADMAP.md from natural language prompts.

    Uses an AI agent to analyze the project context and generate
    a structured roadmap that the existing executor can parse and execute.

    Example:
        >>> from ai_infra import Agent
        >>> from ai_infra.executor.roadmap_generator import RoadmapGenerator
        >>>
        >>> agent = Agent(model="claude-sonnet-4-20250514")
        >>> generator = RoadmapGenerator(agent)
        >>>
        >>> roadmap = await generator.generate(
        ...     prompt="Add user authentication with JWT",
        ...     workspace=Path("/path/to/project"),
        ... )
        >>> print(roadmap.content)
    """

    def __init__(
        self,
        agent: Agent,
        project_analyzer: ProjectAnalyzer | None = None,
    ) -> None:
        """Initialize roadmap generator.

        Args:
            agent: AI agent for generation.
            project_analyzer: Optional custom project analyzer.
        """
        self.agent = agent
        self.project_analyzer = project_analyzer or ProjectAnalyzer()

    async def generate(
        self,
        prompt: str,
        workspace: Path,
        style: str = GenerationStyle.STANDARD,
        validate: bool = True,
    ) -> GeneratedRoadmap:
        """Generate ROADMAP.md from natural language prompt.

        Args:
            prompt: Natural language description of the work to do.
            workspace: Path to the project directory.
            style: Generation style (minimal, standard, detailed).
            validate: Whether to validate the generated roadmap.

        Returns:
            GeneratedRoadmap with content and metadata.

        Raises:
            ValueError: If prompt is empty or workspace doesn't exist.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        workspace = Path(workspace).resolve()
        if not workspace.exists():
            raise ValueError(f"Workspace does not exist: {workspace}")

        # Analyze project context
        project_info = await self.project_analyzer.analyze(workspace)

        # Build the generation prompt
        full_prompt = self._build_prompt(prompt, project_info, style)

        # Generate roadmap using agent
        result = await self.agent.ainvoke(full_prompt)

        # Extract content from agent response
        content = self._extract_content(result)

        # Parse metadata from content
        roadmap = self._parse_roadmap(content, project_info)

        # Validate if requested
        if validate:
            roadmap = await self._validate_roadmap(roadmap, workspace)

        return roadmap

    def _build_prompt(
        self,
        user_prompt: str,
        project_info: ProjectInfo,
        style: str,
    ) -> str:
        """Build the full generation prompt with context."""
        project_context = project_info.to_context_string()

        return GENERATION_PROMPT_TEMPLATE.format(
            prompt=user_prompt,
            project_context=project_context,
            style=style,
        )

    def _extract_content(self, result: Any) -> str:
        """Extract markdown content from agent response."""
        # Handle different response types
        if hasattr(result, "content"):
            text = result.content
        elif isinstance(result, str):
            text = result
        elif isinstance(result, dict) and "content" in result:
            text = result["content"]
        else:
            text = str(result)

        # Extract markdown from code blocks if present
        md_match = re.search(r"```markdown\s*(.*?)\s*```", text, re.DOTALL)
        if md_match:
            return md_match.group(1).strip()

        # Try to find content starting with "# Roadmap"
        roadmap_match = re.search(r"(# Roadmap:.*)", text, re.DOTALL)
        if roadmap_match:
            return roadmap_match.group(1).strip()

        # Return as-is
        return text.strip()

    def _parse_roadmap(
        self,
        content: str,
        project_info: ProjectInfo,
    ) -> GeneratedRoadmap:
        """Parse metadata from generated roadmap content."""
        # Extract title
        title_match = re.search(r"^# Roadmap:\s*(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Generated Roadmap"

        # Count tasks (lines starting with "- [ ]")
        task_count = len(re.findall(r"^- \[ \]", content, re.MULTILINE))

        # Extract estimated time
        time_match = re.search(r"## Estimated Time\s*\n+([^\n#]+)", content, re.IGNORECASE)
        estimated_time = time_match.group(1).strip() if time_match else "Unknown"

        # Extract complexity
        complexity_match = re.search(r"## Complexity\s*\n+([^\n#]+)", content, re.IGNORECASE)
        complexity = "medium"
        if complexity_match:
            complexity_text = complexity_match.group(1).strip().lower()
            if "low" in complexity_text:
                complexity = "low"
            elif "high" in complexity_text:
                complexity = "high"

        return GeneratedRoadmap(
            content=content,
            title=title,
            task_count=task_count,
            estimated_time=estimated_time,
            complexity=complexity,
            confidence=1.0,
            project_info=project_info,
        )

    async def _validate_roadmap(
        self,
        roadmap: GeneratedRoadmap,
        workspace: Path,
    ) -> GeneratedRoadmap:
        """Validate the generated roadmap for common issues."""
        issues: list[ValidationIssue] = []

        # Extract all task IDs and dependencies
        task_ids: set[str] = set()
        dependencies: dict[str, list[str]] = {}

        # Parse tasks from content
        task_pattern = re.compile(
            r"- \[ \] \*\*Task ([0-9.]+)\*\*:\s*([^\n]+)"
            r"(?:.*?Depends on:\s*([^\n]+))?",
            re.DOTALL,
        )

        for match in task_pattern.finditer(roadmap.content):
            task_id = match.group(1)
            task_ids.add(task_id)

            if match.group(3):
                deps_text = match.group(3)
                # Extract task IDs from dependencies
                dep_matches = re.findall(r"Task\s*([0-9.]+)", deps_text)
                dependencies[task_id] = dep_matches

        # Check for invalid dependencies
        for task_id, deps in dependencies.items():
            for dep in deps:
                if dep not in task_ids:
                    issues.append(
                        ValidationIssue(
                            task_id=task_id,
                            issue_type="dependency",
                            message=f"References unknown task: Task {dep}",
                            severity="high",
                        )
                    )

        # Check for circular dependencies
        circular = self._find_circular_dependencies(dependencies)
        for cycle in circular:
            issues.append(
                ValidationIssue(
                    task_id=cycle[0],
                    issue_type="circular_dependency",
                    message=f"Circular dependency: {' -> '.join(cycle)}",
                    severity="high",
                )
            )

        # Check task count
        if roadmap.task_count == 0:
            issues.append(
                ValidationIssue(
                    task_id="",
                    issue_type="empty",
                    message="No tasks found in roadmap",
                    severity="high",
                )
            )
        elif roadmap.task_count > 30:
            issues.append(
                ValidationIssue(
                    task_id="",
                    issue_type="scope",
                    message=f"Too many tasks ({roadmap.task_count}), consider breaking into phases",
                    severity="medium",
                )
            )

        # Update confidence based on issues
        confidence = 1.0
        for issue in issues:
            if issue.severity == "high":
                confidence *= 0.5
            elif issue.severity == "medium":
                confidence *= 0.8

        roadmap.validation_issues = issues
        roadmap.confidence = max(0.1, confidence)

        return roadmap

    def _find_circular_dependencies(
        self,
        dependencies: dict[str, list[str]],
    ) -> list[list[str]]:
        """Find circular dependencies in task graph."""
        cycles: list[list[str]] = []
        visited: set[str] = set()
        path: list[str] = []
        path_set: set[str] = set()

        def dfs(node: str) -> None:
            if node in path_set:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            path.append(node)
            path_set.add(node)

            for dep in dependencies.get(node, []):
                dfs(dep)

            path.pop()
            path_set.remove(node)

        for node in dependencies:
            if node not in visited:
                dfs(node)

        return cycles

    async def generate_and_save(
        self,
        prompt: str,
        workspace: Path,
        output: Path | str = "ROADMAP.md",
        style: str = GenerationStyle.STANDARD,
        validate: bool = True,
    ) -> GeneratedRoadmap:
        """Generate and save ROADMAP.md to file.

        Args:
            prompt: Natural language description of the work.
            workspace: Path to the project directory.
            output: Output file path (relative to workspace or absolute).
            style: Generation style.
            validate: Whether to validate.

        Returns:
            GeneratedRoadmap with content and metadata.
        """
        roadmap = await self.generate(
            prompt=prompt,
            workspace=workspace,
            style=style,
            validate=validate,
        )

        # Resolve output path
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = workspace / output_path

        # Write content
        output_path.write_text(roadmap.content)

        return roadmap
