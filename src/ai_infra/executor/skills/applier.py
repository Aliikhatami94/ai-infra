"""Skill application for task execution.

Phase 5.1.5 of EXECUTOR_2.md: SkillApplier to inject skills into prompts.

This module provides:
- SkillApplier: Find and format relevant skills for prompts
- SkillInjectionResult: Result of skill injection

Example:
    ```python
    from ai_infra.executor.skills.applier import SkillApplier
    from ai_infra.executor.skills.database import SkillsDatabase

    # Create applier
    db = SkillsDatabase()
    applier = SkillApplier(db=db)

    # Build context from task
    context = applier.build_context(task, project_context)

    # Get relevant skills
    skills = applier.get_relevant_skills(context)

    # Format for prompt
    skills_prompt = applier.format_skills_for_prompt(skills)
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ai_infra.executor.skills.database import SkillsDatabase
from ai_infra.executor.skills.models import Skill, SkillContext, SkillType

if TYPE_CHECKING:
    from ai_infra.executor.models import ProjectContext, Task

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ApplierConfig:
    """Configuration for skill application.

    Attributes:
        max_skills_per_prompt: Maximum skills to include in a prompt.
        min_confidence: Minimum confidence to include a skill.
        min_match_score: Minimum match score to include a skill.
        include_anti_patterns: Whether to include anti-patterns.
        include_recovery_skills: Whether to include recovery skills.
        max_pattern_length: Max characters for pattern in prompt.
    """

    max_skills_per_prompt: int = 5
    min_confidence: float = 0.3
    min_match_score: float = 0.3
    include_anti_patterns: bool = True
    include_recovery_skills: bool = True
    max_pattern_length: int = 500


# =============================================================================
# Result Models
# =============================================================================


@dataclass
class SkillInjectionResult:
    """Result of skill injection into a prompt.

    Attributes:
        skills_applied: Skills that were applied.
        skills_prompt: Formatted prompt section.
        anti_patterns_applied: Anti-patterns that were applied.
        recovery_skills_applied: Recovery skills applied.
    """

    skills_applied: list[Skill] = field(default_factory=list)
    skills_prompt: str = ""
    anti_patterns_applied: list[Skill] = field(default_factory=list)
    recovery_skills_applied: list[Skill] = field(default_factory=list)

    @property
    def total_skills(self) -> int:
        """Total number of skills applied."""
        return (
            len(self.skills_applied)
            + len(self.anti_patterns_applied)
            + len(self.recovery_skills_applied)
        )

    def get_skill_ids(self) -> list[str]:
        """Get IDs of all applied skills."""
        ids = []
        ids.extend(s.id for s in self.skills_applied)
        ids.extend(s.id for s in self.anti_patterns_applied)
        ids.extend(s.id for s in self.recovery_skills_applied)
        return ids


# =============================================================================
# Skill Applier
# =============================================================================


class SkillApplier:
    """Apply skills to task execution prompts.

    Phase 5.1.5: Finds relevant skills and formats them for inclusion
    in agent prompts.

    Attributes:
        db: SkillsDatabase for skill lookup.
        config: Application configuration.

    Example:
        ```python
        applier = SkillApplier(db=db)

        # Get skills for a task
        context = applier.build_context(task, project)
        result = applier.inject_skills(context)

        # Use in prompt
        full_prompt = f"{base_prompt}\\n\\n{result.skills_prompt}"
        ```
    """

    def __init__(
        self,
        db: SkillsDatabase,
        config: ApplierConfig | None = None,
    ) -> None:
        """Initialize the skill applier.

        Args:
            db: Skills database for lookup.
            config: Application configuration.
        """
        self.db = db
        self.config = config or ApplierConfig()

    def build_context(
        self,
        task: Task,
        project_context: ProjectContext | None = None,
    ) -> SkillContext:
        """Build a SkillContext from task and project.

        Args:
            task: The task being executed.
            project_context: Optional project context.

        Returns:
            SkillContext for skill matching.
        """
        # Extract keywords from task title and description
        keywords = self._extract_keywords(f"{task.title} {task.description or ''}")

        # Get language from project context if available
        language = ""
        framework = None
        file_hints = None

        if project_context:
            language = getattr(project_context, "language", "") or ""
            framework = getattr(project_context, "framework", None)
            file_hints = getattr(project_context, "relevant_files", None)

        context = SkillContext(
            language=language,
            framework=framework,
            task_title=task.title,
            task_description=task.description or "",
            task_keywords=keywords,
            file_hints=file_hints or [],
        )

        return context

    def get_relevant_skills(
        self,
        context: SkillContext,
        limit: int | None = None,
    ) -> list[Skill]:
        """Get skills relevant to the given context.

        Args:
            context: Execution context.
            limit: Maximum skills to return.

        Returns:
            List of relevant skills, sorted by score.
        """
        limit = limit or self.config.max_skills_per_prompt

        # Get matching skills (excludes anti-patterns and recovery)
        skills = self.db.find_matching(
            context=context,
            limit=limit,
            min_score=self.config.min_match_score,
        )

        # Filter by confidence
        skills = [
            s
            for s in skills
            if s.confidence >= self.config.min_confidence
            and s.type not in (SkillType.ANTI_PATTERN, SkillType.RECOVERY)
        ]

        return skills

    def get_anti_patterns(
        self,
        context: SkillContext,
        limit: int = 3,
    ) -> list[Skill]:
        """Get anti-patterns relevant to the context.

        Args:
            context: Execution context.
            limit: Maximum anti-patterns to return.

        Returns:
            List of relevant anti-patterns.
        """
        if not self.config.include_anti_patterns:
            return []

        # find_anti_patterns takes context and limit
        anti_patterns = self.db.find_anti_patterns(context, limit=limit)

        # Filter by confidence
        result = [ap for ap in anti_patterns if ap.confidence >= self.config.min_confidence]

        return result

    def get_recovery_skills(
        self,
        error_type: str | None = None,
        context: SkillContext | None = None,
    ) -> list[Skill]:
        """Get recovery skills for an error.

        Args:
            error_type: Type of error to recover from.
            context: Optional context for filtering.

        Returns:
            List of relevant recovery skills.
        """
        if not self.config.include_recovery_skills:
            return []

        if not error_type:
            return []

        recovery_skills = self.db.find_recovery_skills(
            error_type=error_type,
            context=context,
        )

        return recovery_skills[:3]

    def inject_skills(
        self,
        context: SkillContext,
    ) -> SkillInjectionResult:
        """Inject all relevant skills for a context.

        Gets skills, anti-patterns, and formats them for prompts.

        Args:
            context: Execution context.

        Returns:
            SkillInjectionResult with formatted prompt.
        """
        # Get skills of each type
        skills = self.get_relevant_skills(context)
        anti_patterns = self.get_anti_patterns(context)

        # Format the prompt section
        prompt_parts = []

        if skills:
            prompt_parts.append(self._format_skills_section(skills))

        if anti_patterns:
            prompt_parts.append(self._format_anti_patterns_section(anti_patterns))

        skills_prompt = "\n\n".join(prompt_parts) if prompt_parts else ""

        return SkillInjectionResult(
            skills_applied=skills,
            skills_prompt=skills_prompt,
            anti_patterns_applied=anti_patterns,
        )

    def format_skills_for_prompt(
        self,
        skills: list[Skill],
    ) -> str:
        """Format skills for inclusion in a prompt.

        Args:
            skills: Skills to format.

        Returns:
            Formatted prompt section.
        """
        if not skills:
            return ""

        return self._format_skills_section(skills)

    def record_skill_usage(
        self,
        skill_ids: list[str],
        success: bool,
    ) -> None:
        """Record whether skills were helpful.

        Args:
            skill_ids: IDs of skills that were used.
            success: Whether the task succeeded.
        """
        for skill_id in skill_ids:
            if success:
                self.db.update_success(skill_id)
            else:
                self.db.update_failure(skill_id)

    # =========================================================================
    # Formatting Methods
    # =========================================================================

    def _format_skills_section(self, skills: list[Skill]) -> str:
        """Format skills section for prompt."""
        lines = ["## Relevant Skills from Past Experience", ""]

        for i, skill in enumerate(skills, 1):
            lines.append(f"### {i}. {skill.title}")
            lines.append(skill.description)

            if skill.pattern:
                pattern = skill.pattern[: self.config.max_pattern_length]
                lines.append("")
                lines.append("**Pattern:**")
                lines.append(f"```\n{pattern}\n```")

            if skill.rationale:
                lines.append("")
                lines.append(f"**Why:** {skill.rationale}")

            lines.append("")

        return "\n".join(lines)

    def _format_anti_patterns_section(self, anti_patterns: list[Skill]) -> str:
        """Format anti-patterns section for prompt."""
        lines = ["## Things to Avoid (from Past Failures)", ""]

        for i, ap in enumerate(anti_patterns, 1):
            lines.append(f"### {i}. AVOID: {ap.title}")
            lines.append(ap.description)

            if ap.anti_example:
                example = ap.anti_example[: self.config.max_pattern_length]
                lines.append("")
                lines.append("**Do NOT do this:**")
                lines.append(f"```\n{example}\n```")

            if ap.pattern:
                better = ap.pattern[: self.config.max_pattern_length]
                lines.append("")
                lines.append("**Instead, do this:**")
                lines.append(f"```\n{better}\n```")

            lines.append("")

        return "\n".join(lines)

    def _format_recovery_section(self, recovery_skills: list[Skill]) -> str:
        """Format recovery skills section for prompt."""
        lines = ["## Recovery Approaches (from Past Fixes)", ""]

        for i, skill in enumerate(recovery_skills, 1):
            lines.append(f"### {i}. {skill.title}")
            lines.append(skill.description)

            if skill.pattern:
                pattern = skill.pattern[: self.config.max_pattern_length]
                lines.append("")
                lines.append("**Fix approach:**")
                lines.append(f"```\n{pattern}\n```")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        import re

        # Remove common words
        stop_words = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "add",
            "create",
            "implement",
            "fix",
            "update",
            "modify",
            "change",
            "new",
            "use",
            "using",
            "should",
            "must",
            "will",
            "can",
            "could",
            "would",
            "be",
            "is",
            "are",
            "was",
            "were",
            "been",
            "being",
            "have",
            "has",
            "had",
        }

        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        keywords = [w for w in words if w not in stop_words]

        # Return unique keywords
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique[:10]


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "ApplierConfig",
    "SkillApplier",
    "SkillInjectionResult",
]
