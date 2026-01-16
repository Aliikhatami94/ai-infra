"""Skills database for persistent storage.

Phase 5.1.3 of EXECUTOR_2.md: SkillsDatabase for persistence.

This module provides:
- SkillsDatabase: Persistent storage for learned skills

Example:
    ```python
    from ai_infra.executor.skills.database import SkillsDatabase
    from ai_infra.executor.skills.models import Skill, SkillType, SkillContext

    # Create database (default location: ~/.ai-infra/skills.json)
    db = SkillsDatabase()

    # Add a skill
    skill = Skill.create(
        type=SkillType.PATTERN,
        title="Use async context managers",
        description="Always use async with for database connections",
        languages=["python"],
    )
    db.add(skill)

    # Find matching skills
    context = SkillContext(language="python", task_keywords=["database"])
    matches = db.find_matching(context, limit=5)

    # Update skill usage
    db.update_success(skill.id)
    ```
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_infra.executor.skills.models import Skill, SkillContext, SkillType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SkillsDatabase:
    """Persistent storage for learned skills.

    Phase 5.1.3: Stores skills to JSON file for persistence across sessions.

    Attributes:
        path: Path to the skills JSON file.
        skills: List of all skills.

    Example:
        ```python
        db = SkillsDatabase()

        # Add skill
        skill = Skill.create(
            type=SkillType.PATTERN,
            title="Error handling pattern",
            description="Use try/except with specific exceptions",
        )
        db.add(skill)

        # Find relevant skills
        context = SkillContext(language="python", task_keywords=["error"])
        matches = db.find_matching(context)
        ```
    """

    DEFAULT_PATH = Path.home() / ".ai-infra" / "skills.json"

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        auto_save: bool = True,
    ) -> None:
        """Initialize the skills database.

        Args:
            path: Path to skills file (None for default ~/.ai-infra/skills.json).
            auto_save: Whether to auto-save after modifications.
        """
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.auto_save = auto_save
        self._skills: dict[str, Skill] = {}
        self._loaded = False

        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing skills
        self._load()

    def _load(self) -> None:
        """Load skills from disk."""
        if not self.path.exists():
            self._skills = {}
            self._loaded = True
            return

        try:
            with open(self.path) as f:
                data = json.load(f)

            skills_data = data.get("skills", [])
            self._skills = {}

            for skill_data in skills_data:
                try:
                    skill = Skill.from_dict(skill_data)
                    self._skills[skill.id] = skill
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to load skill: {e}")

            self._loaded = True
            logger.debug(f"Loaded {len(self._skills)} skills from {self.path}")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load skills database: {e}")
            self._skills = {}
            self._loaded = True

    def save(self) -> None:
        """Save skills to disk."""
        data = {
            "version": "1.0",
            "updated_at": datetime.now(UTC).isoformat(),
            "skills": [skill.to_dict() for skill in self._skills.values()],
        }

        try:
            with open(self.path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved {len(self._skills)} skills to {self.path}")
        except OSError as e:
            logger.error(f"Failed to save skills database: {e}")

    def _auto_save(self) -> None:
        """Auto-save if enabled."""
        if self.auto_save:
            self.save()

    @property
    def skills(self) -> list[Skill]:
        """Get all skills as a list."""
        return list(self._skills.values())

    def __len__(self) -> int:
        """Get number of skills."""
        return len(self._skills)

    def __contains__(self, skill_id: str) -> bool:
        """Check if skill exists."""
        return skill_id in self._skills

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def add(self, skill: Skill) -> None:
        """Add a new skill.

        Args:
            skill: The skill to add.
        """
        self._skills[skill.id] = skill
        logger.debug(f"Added skill: {skill.title}")
        self._auto_save()

    def get(self, skill_id: str) -> Skill | None:
        """Get a skill by ID.

        Args:
            skill_id: ID of the skill.

        Returns:
            The skill or None if not found.
        """
        return self._skills.get(skill_id)

    def update(self, skill: Skill) -> None:
        """Update an existing skill.

        Args:
            skill: The skill to update.
        """
        if skill.id in self._skills:
            skill.updated_at = datetime.now(UTC)
            self._skills[skill.id] = skill
            self._auto_save()

    def delete(self, skill_id: str) -> bool:
        """Delete a skill.

        Args:
            skill_id: ID of the skill to delete.

        Returns:
            True if deleted, False if not found.
        """
        if skill_id in self._skills:
            del self._skills[skill_id]
            self._auto_save()
            return True
        return False

    # =========================================================================
    # Query Operations
    # =========================================================================

    def find_matching(
        self,
        context: SkillContext,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> list[Skill]:
        """Find skills matching a context.

        Args:
            context: The context to match against.
            limit: Maximum number of skills to return.
            min_score: Minimum match score (0-1).

        Returns:
            List of matching skills, sorted by score.
        """
        scored = []

        for skill in self._skills.values():
            score = skill.matches(context)
            if score >= min_score:
                # Boost score by confidence
                adjusted_score = score * (0.5 + 0.5 * skill.confidence)
                scored.append((skill, adjusted_score))

        # Sort by adjusted score
        scored.sort(key=lambda x: x[1], reverse=True)

        return [skill for skill, _ in scored[:limit]]

    def find_by_type(self, skill_type: SkillType) -> list[Skill]:
        """Find all skills of a given type.

        Args:
            skill_type: Type of skills to find.

        Returns:
            List of matching skills.
        """
        return [s for s in self._skills.values() if s.type == skill_type]

    def find_by_language(self, language: str) -> list[Skill]:
        """Find skills for a programming language.

        Args:
            language: Programming language.

        Returns:
            List of matching skills.
        """
        language = language.lower()
        return [
            s for s in self._skills.values() if language in [lang.lower() for lang in s.languages]
        ]

    def find_by_framework(self, framework: str) -> list[Skill]:
        """Find skills for a framework.

        Args:
            framework: Framework name.

        Returns:
            List of matching skills.
        """
        framework = framework.lower()
        return [
            s for s in self._skills.values() if framework in [fw.lower() for fw in s.frameworks]
        ]

    def find_by_keywords(self, keywords: list[str]) -> list[Skill]:
        """Find skills matching keywords.

        Args:
            keywords: Keywords to search for.

        Returns:
            List of matching skills.
        """
        keywords = [kw.lower() for kw in keywords]
        results = []

        for skill in self._skills.values():
            skill_keywords = [kw.lower() for kw in skill.task_keywords]
            if any(kw in skill_keywords for kw in keywords):
                results.append(skill)

        return results

    def find_anti_patterns(
        self,
        context: SkillContext,
        limit: int = 3,
    ) -> list[Skill]:
        """Find anti-patterns to avoid for a context.

        Args:
            context: The context to match against.
            limit: Maximum number to return.

        Returns:
            List of anti-pattern skills.
        """
        anti_patterns = [s for s in self._skills.values() if s.type == SkillType.ANTI_PATTERN]

        scored = []
        for skill in anti_patterns:
            score = skill.matches(context)
            if score > 0.2:
                scored.append((skill, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in scored[:limit]]

    def find_recovery_skills(
        self,
        error_type: str,
        context: SkillContext | None = None,
    ) -> list[Skill]:
        """Find recovery skills for an error type.

        Args:
            error_type: Type of error (e.g., "ImportError", "SyntaxError").
            context: Optional context for better matching.

        Returns:
            List of recovery skills.
        """
        recovery_skills = [s for s in self._skills.values() if s.type == SkillType.RECOVERY]

        # Filter by error type
        error_type_lower = error_type.lower()
        matches = [
            s
            for s in recovery_skills
            if error_type_lower in s.description.lower() or error_type_lower in s.title.lower()
        ]

        # Further filter by context if provided
        if context:
            context_with_error = SkillContext(
                language=context.language,
                framework=context.framework,
                task_keywords=context.task_keywords,
                task_title=context.task_title,
                task_description=context.task_description,
                error_type=error_type,
            )
            scored = [(s, s.matches(context_with_error)) for s in matches]
            scored.sort(key=lambda x: x[1], reverse=True)
            matches = [s for s, _ in scored]

        return matches

    # =========================================================================
    # Usage Tracking
    # =========================================================================

    def update_success(self, skill_id: str) -> None:
        """Mark a skill as successfully applied.

        Args:
            skill_id: ID of the skill.
        """
        skill = self._skills.get(skill_id)
        if skill:
            skill.record_success()
            self._auto_save()
            logger.debug(f"Recorded success for skill: {skill.title}")

    def update_failure(self, skill_id: str) -> None:
        """Mark a skill as failed when applied.

        Args:
            skill_id: ID of the skill.
        """
        skill = self._skills.get(skill_id)
        if skill:
            skill.record_failure()
            self._auto_save()
            logger.debug(f"Recorded failure for skill: {skill.title}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with statistics.
        """
        skills = list(self._skills.values())

        if not skills:
            return {
                "total": 0,
                "by_type": {},
                "by_language": {},
                "avg_confidence": 0.0,
                "total_uses": 0,
            }

        # Count by type
        by_type = {}
        for skill_type in SkillType:
            count = sum(1 for s in skills if s.type == skill_type)
            if count > 0:
                by_type[skill_type.value] = count

        # Count by language
        by_language: dict[str, int] = {}
        for skill in skills:
            for lang in skill.languages:
                by_language[lang] = by_language.get(lang, 0) + 1

        # Calculate averages
        avg_confidence = sum(s.confidence for s in skills) / len(skills)
        total_uses = sum(s.total_uses for s in skills)

        return {
            "total": len(skills),
            "by_type": by_type,
            "by_language": by_language,
            "avg_confidence": avg_confidence,
            "total_uses": total_uses,
        }

    def get_top_skills(self, limit: int = 10) -> list[Skill]:
        """Get most successful skills.

        Args:
            limit: Maximum number to return.

        Returns:
            List of skills sorted by success count.
        """
        skills = sorted(
            self._skills.values(),
            key=lambda s: s.success_count,
            reverse=True,
        )
        return skills[:limit]

    def prune_low_confidence(self, threshold: float = 0.3) -> int:
        """Remove skills with low confidence that have been tried.

        Args:
            threshold: Minimum confidence to keep.

        Returns:
            Number of skills removed.
        """
        to_remove = [
            skill_id
            for skill_id, skill in self._skills.items()
            if skill.total_uses >= 3 and skill.confidence < threshold
        ]

        for skill_id in to_remove:
            del self._skills[skill_id]

        if to_remove:
            self._auto_save()
            logger.info(f"Pruned {len(to_remove)} low-confidence skills")

        return len(to_remove)

    def clear(self) -> None:
        """Clear all skills."""
        self._skills.clear()
        self._auto_save()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "SkillsDatabase",
]
