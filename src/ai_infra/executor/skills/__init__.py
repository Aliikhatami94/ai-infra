"""Skills system for executor learning and adaptation.

Phase 5.1 of EXECUTOR_2.md: Skills System.

This package provides:
- Skill: Learned patterns from past executions
- SkillsDatabase: Persistent storage for skills
- SkillExtractor: Learn skills from successes and failures
- SkillApplier: Inject relevant skills into task context

Example:
    ```python
    from ai_infra.executor.skills import (
        Skill,
        SkillType,
        SkillContext,
        SkillsDatabase,
        SkillApplier,
        SkillExtractor,
    )

    # Create database
    db = SkillsDatabase()

    # Find relevant skills for a task
    applier = SkillApplier(db)
    skills = applier.get_relevant_skills(task, project_context)

    # Format for prompt
    skills_prompt = applier.format_skills_for_prompt(skills)
    ```
"""

from ai_infra.executor.skills.applier import (
    ApplierConfig,
    SkillApplier,
    SkillInjectionResult,
)
from ai_infra.executor.skills.database import (
    SkillsDatabase,
)
from ai_infra.executor.skills.extractor import (
    ExtractionConfig,
    SkillExtractor,
    TaskResult,
)
from ai_infra.executor.skills.models import (
    Skill,
    SkillContext,
    SkillType,
)

__all__ = [
    # Models
    "Skill",
    "SkillContext",
    "SkillType",
    # Database
    "SkillsDatabase",
    # Applier
    "ApplierConfig",
    "SkillApplier",
    "SkillInjectionResult",
    # Extractor
    "ExtractionConfig",
    "SkillExtractor",
    "TaskResult",
]
