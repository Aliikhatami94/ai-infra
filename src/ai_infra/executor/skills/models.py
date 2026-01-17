"""Data models for the skills system.

Phase 5.1.2 of EXECUTOR_2.md: Unified Skill data model.

This module provides:
- SkillType: Enum of skill categories
- Skill: A learned skill from past execution
- SkillContext: Context for skill matching

Example:
    ```python
    from ai_infra.executor.skills.models import Skill, SkillType, SkillContext

    # Create a skill
    skill = Skill(
        id="skill-001",
        type=SkillType.PATTERN,
        title="FastAPI route with error handling",
        description="Use try/except with HTTPException",
        languages=["python"],
        frameworks=["fastapi"],
        task_keywords=["api", "endpoint", "route"],
        pattern=\"\"\"
        @app.get("/items/{item_id}")
        async def get_item(item_id: int):
            try:
                item = await db.get_item(item_id)
                if not item:
                    raise HTTPException(status_code=404)
                return item
            except DBError as e:
                raise HTTPException(status_code=500, detail=str(e))
        \"\"\",
        rationale="Consistent error handling prevents unhandled exceptions",
    )

    # Match against context
    context = SkillContext(
        language="python",
        framework="fastapi",
        task_keywords=["api", "endpoint"],
        task_title="Add user endpoint",
        task_description="Create GET /users/{id} endpoint",
    )

    score = skill.matches(context)
    print(f"Match score: {score:.0%}")
    ```
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class SkillType(Enum):
    """Type of learned skill.

    Attributes:
        PATTERN: Successful code pattern to follow.
        ANTI_PATTERN: Approach to avoid.
        APPROACH: Strategy for a task type.
        TOOL_USAGE: How to use a specific tool.
        RECOVERY: How to recover from an error.
    """

    PATTERN = "pattern"
    """Successful code pattern to follow."""

    ANTI_PATTERN = "anti_pattern"
    """Approach that failed - avoid this."""

    APPROACH = "approach"
    """General strategy for a task type."""

    TOOL_USAGE = "tool_usage"
    """How to use a specific tool effectively."""

    RECOVERY = "recovery"
    """How to recover from a specific error."""


@dataclass
class SkillContext:
    """Context for skill matching.

    Used to find skills relevant to the current task.

    Attributes:
        language: Programming language (e.g., "python", "typescript").
        framework: Framework in use (e.g., "fastapi", "react").
        task_keywords: Keywords from the task.
        task_title: Title of the task.
        task_description: Full task description.
        file_hints: Files related to the task.
        error_type: Type of error (for recovery skills).
    """

    language: str
    framework: str | None = None
    task_keywords: list[str] = field(default_factory=list)
    task_title: str = ""
    task_description: str = ""
    file_hints: list[str] = field(default_factory=list)
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "language": self.language,
            "framework": self.framework,
            "task_keywords": self.task_keywords,
            "task_title": self.task_title,
            "task_description": self.task_description,
            "file_hints": self.file_hints,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillContext:
        """Create from dictionary."""
        return cls(
            language=data.get("language", ""),
            framework=data.get("framework"),
            task_keywords=data.get("task_keywords", []),
            task_title=data.get("task_title", ""),
            task_description=data.get("task_description", ""),
            file_hints=data.get("file_hints", []),
            error_type=data.get("error_type"),
        )


@dataclass
class Skill:
    """A learned skill from past execution.

    Skills are extracted from successful task completions and failures,
    and are used to inform future task execution.

    Attributes:
        id: Unique identifier for this skill.
        type: Type of skill (pattern, anti-pattern, approach, etc.).
        title: Short descriptive title.
        description: What this skill teaches.
        languages: Programming languages this applies to.
        frameworks: Frameworks this applies to.
        task_keywords: Keywords that indicate relevance.
        pattern: The reusable pattern or code example.
        rationale: Why this works (or doesn't).
        anti_example: What not to do (for anti-patterns).
        created_at: When this skill was first created.
        updated_at: When this skill was last updated.
        success_count: Times this skill led to success.
        failure_count: Times this skill led to failure.
        source_task_id: ID of the task this was extracted from.
        metadata: Additional metadata.

    Example:
        ```python
        skill = Skill(
            id="skill-001",
            type=SkillType.PATTERN,
            title="Async context manager for DB",
            description="Use async with for database connections",
            languages=["python"],
            frameworks=["sqlalchemy"],
            pattern="async with engine.begin() as conn: ...",
            rationale="Ensures proper cleanup on exceptions",
        )
        ```
    """

    id: str
    type: SkillType
    title: str
    description: str
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    task_keywords: list[str] = field(default_factory=list)
    pattern: str | None = None
    rationale: str | None = None
    anti_example: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    success_count: int = 0
    failure_count: int = 0
    source_task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        type: SkillType,
        title: str,
        description: str,
        **kwargs: Any,
    ) -> Skill:
        """Create a new skill with auto-generated ID.

        Args:
            type: Type of skill.
            title: Short title.
            description: What this skill teaches.
            **kwargs: Additional skill attributes.

        Returns:
            New Skill instance.
        """
        return cls(
            id=f"skill-{uuid.uuid4().hex[:12]}",
            type=type,
            title=title,
            description=description,
            **kwargs,
        )

    @property
    def confidence(self) -> float:
        """Calculate confidence in this skill (0-1).

        Based on success/failure ratio with a prior.

        Returns:
            Confidence score between 0 and 1.
        """
        # Use Laplace smoothing (add 1 to each)
        total = self.success_count + self.failure_count + 2
        successes = self.success_count + 1  # Prior: 1 success
        return successes / total

    @property
    def total_uses(self) -> int:
        """Get total number of times this skill was applied."""
        return self.success_count + self.failure_count

    def matches(self, context: SkillContext) -> float:
        """Score how well this skill matches a context.

        Args:
            context: The context to match against.

        Returns:
            Match score between 0 and 1.
        """
        score = 0.0
        weights_used = 0.0

        # Language match (weight: 0.3)
        if self.languages:
            weights_used += 0.3
            if context.language and context.language.lower() in [
                lang.lower() for lang in self.languages
            ]:
                score += 0.3

        # Framework match (weight: 0.25)
        if self.frameworks:
            weights_used += 0.25
            if context.framework and context.framework.lower() in [
                fw.lower() for fw in self.frameworks
            ]:
                score += 0.25

        # Keyword match (weight: 0.35)
        if self.task_keywords and context.task_keywords:
            weights_used += 0.35
            context_keywords = set(kw.lower() for kw in context.task_keywords)
            skill_keywords = set(kw.lower() for kw in self.task_keywords)
            overlap = context_keywords & skill_keywords
            if skill_keywords:
                keyword_score = len(overlap) / len(skill_keywords)
                score += 0.35 * keyword_score

        # Error type match for recovery skills (weight: 0.1)
        if self.type == SkillType.RECOVERY and context.error_type:
            weights_used += 0.1
            if context.error_type.lower() in self.description.lower():
                score += 0.1

        # Normalize if we used weights
        if weights_used > 0:
            # Scale to full 0-1 range based on weights used
            max_possible = weights_used
            score = score / max_possible if max_possible > 0 else 0

        return score

    def record_success(self) -> None:
        """Record a successful application of this skill."""
        self.success_count += 1
        self.updated_at = datetime.now(UTC)

    def record_failure(self) -> None:
        """Record a failed application of this skill."""
        self.failure_count += 1
        self.updated_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "task_keywords": self.task_keywords,
            "pattern": self.pattern,
            "rationale": self.rationale,
            "anti_example": self.anti_example,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "source_task_id": self.source_task_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=SkillType(data["type"]),
            title=data["title"],
            description=data["description"],
            languages=data.get("languages", []),
            frameworks=data.get("frameworks", []),
            task_keywords=data.get("task_keywords", []),
            pattern=data.get("pattern"),
            rationale=data.get("rationale"),
            anti_example=data.get("anti_example"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(UTC),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.now(UTC),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            source_task_id=data.get("source_task_id"),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """Get string representation."""
        return f"Skill({self.type.value}: {self.title})"

    def __repr__(self) -> str:
        """Get detailed representation."""
        return (
            f"Skill(id={self.id!r}, type={self.type.value!r}, "
            f"title={self.title!r}, confidence={self.confidence:.0%})"
        )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "Skill",
    "SkillContext",
    "SkillType",
]
