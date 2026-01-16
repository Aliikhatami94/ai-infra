"""Skill extraction from task executions.

Phase 5.1.4 of EXECUTOR_2.md: SkillExtractor to learn from executions.

This module provides:
- SkillExtractor: Extract skills from successes and failures
- TaskResult: Result of a task execution (for extraction)
- ExtractionConfig: Configuration for extraction

Example:
    ```python
    from ai_infra.executor.skills.extractor import SkillExtractor
    from ai_infra.executor.skills.database import SkillsDatabase

    # Create extractor
    db = SkillsDatabase()
    extractor = SkillExtractor(db=db, llm=my_llm)

    # Extract from success
    skill = await extractor.extract_from_success(task, result, context)
    if skill:
        print(f"Learned: {skill.title}")

    # Extract from failure
    skill = await extractor.extract_from_failure(task, error, context)
    if skill:
        print(f"Anti-pattern: {skill.title}")
    ```
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ai_infra.executor.skills.database import SkillsDatabase
from ai_infra.executor.skills.models import Skill, SkillContext, SkillType

if TYPE_CHECKING:
    from ai_infra.executor.models import Task
    from ai_infra.llm import LLM

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class TaskResult:
    """Result of a task execution for skill extraction.

    Attributes:
        success: Whether the task succeeded.
        actions_summary: Summary of actions taken.
        diff_summary: Summary of code changes.
        files_modified: Files that were modified.
        files_created: Files that were created.
        execution_time: Time taken in seconds.
        tokens_used: Tokens used for execution.
        error_message: Error message if failed.
    """

    success: bool
    actions_summary: str = ""
    diff_summary: str = ""
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    tokens_used: int = 0
    error_message: str | None = None


@dataclass
class ExtractionConfig:
    """Configuration for skill extraction.

    Attributes:
        min_confidence_to_save: Minimum confidence to save a skill.
        extract_on_success: Whether to extract from successes.
        extract_on_failure: Whether to extract from failures.
        max_pattern_length: Maximum length for code patterns.
        require_rationale: Require rationale in extracted skills.
    """

    min_confidence_to_save: float = 0.5
    extract_on_success: bool = True
    extract_on_failure: bool = True
    max_pattern_length: int = 2000
    require_rationale: bool = True


# =============================================================================
# Skill Extractor
# =============================================================================


class SkillExtractor:
    """Extract skills from task executions.

    Phase 5.1.4: Uses LLM to analyze executions and extract reusable skills.

    Attributes:
        db: SkillsDatabase for storing extracted skills.
        llm: LLM for analyzing executions.
        config: Extraction configuration.

    Example:
        ```python
        extractor = SkillExtractor(db=db, llm=my_llm)

        # After successful task
        skill = await extractor.extract_from_success(task, result, context)

        # After failed task
        anti_pattern = await extractor.extract_from_failure(task, error, context)
        ```
    """

    def __init__(
        self,
        db: SkillsDatabase,
        llm: LLM | None = None,
        config: ExtractionConfig | None = None,
    ) -> None:
        """Initialize the skill extractor.

        Args:
            db: Skills database for storage.
            llm: LLM for analysis (optional for heuristic extraction).
            config: Extraction configuration.
        """
        self.db = db
        self.llm = llm
        self.config = config or ExtractionConfig()

    async def extract_from_success(
        self,
        task: Task,
        result: TaskResult,
        context: SkillContext,
    ) -> Skill | None:
        """Extract a skill from a successful task.

        Analyzes what worked and creates a reusable pattern.

        Args:
            task: The task that succeeded.
            result: Execution result.
            context: Execution context.

        Returns:
            Extracted Skill or None if nothing worth extracting.
        """
        if not self.config.extract_on_success:
            return None

        if self.llm:
            return await self._extract_with_llm(task, result, context)
        else:
            return self._extract_heuristic(task, result, context)

    async def extract_from_failure(
        self,
        task: Task,
        error_message: str,
        context: SkillContext,
    ) -> Skill | None:
        """Extract an anti-pattern from a failed task.

        Analyzes what went wrong and creates a pattern to avoid.

        Args:
            task: The task that failed.
            error_message: Error message from the failure.
            context: Execution context.

        Returns:
            Extracted anti-pattern Skill or None.
        """
        if not self.config.extract_on_failure:
            return None

        if self.llm:
            return await self._extract_anti_pattern_with_llm(task, error_message, context)
        else:
            return self._extract_anti_pattern_heuristic(task, error_message, context)

    async def extract_recovery_skill(
        self,
        error_message: str,
        fix_actions: str,
        context: SkillContext,
    ) -> Skill | None:
        """Extract a recovery skill from a successful fix.

        Args:
            error_message: The error that was fixed.
            fix_actions: Description of the fix.
            context: Execution context.

        Returns:
            Extracted recovery Skill or None.
        """
        if self.llm:
            return await self._extract_recovery_with_llm(error_message, fix_actions, context)
        else:
            return self._extract_recovery_heuristic(error_message, fix_actions, context)

    # =========================================================================
    # LLM-based Extraction
    # =========================================================================

    async def _extract_with_llm(
        self,
        task: Task,
        result: TaskResult,
        context: SkillContext,
    ) -> Skill | None:
        """Extract skill using LLM analysis."""
        if not self.llm:
            return None

        prompt = f"""Analyze this successful task completion and extract a reusable skill.

TASK: {task.title}
DESCRIPTION: {task.description or "N/A"}

ACTIONS TAKEN:
{result.actions_summary or "N/A"}

CODE CHANGES:
{result.diff_summary[:1500] if result.diff_summary else "N/A"}

FILES MODIFIED: {", ".join(result.files_modified) or "None"}
FILES CREATED: {", ".join(result.files_created) or "None"}

CONTEXT:
- Language: {context.language}
- Framework: {context.framework or "N/A"}

Extract a skill if there's a reusable pattern worth remembering.
Only extract skills that would help with similar future tasks.

Output ONLY valid JSON (no markdown, no explanation):
{{
    "should_extract": true/false,
    "skill_type": "pattern" | "approach" | "tool_usage",
    "title": "Concise skill name (max 60 chars)",
    "description": "What this skill teaches (1-2 sentences)",
    "pattern": "The reusable code pattern or approach",
    "rationale": "Why this works well",
    "keywords": ["relevant", "keywords", "max", "5"]
}}
"""

        try:
            response = await self.llm.agenerate(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            data = self._parse_json_response(content)

            if not data.get("should_extract"):
                return None

            skill = Skill.create(
                type=SkillType(data.get("skill_type", "pattern")),
                title=data["title"][:100],
                description=data["description"],
                languages=[context.language] if context.language else [],
                frameworks=[context.framework] if context.framework else [],
                task_keywords=data.get("keywords", [])[:5],
                pattern=data.get("pattern", "")[: self.config.max_pattern_length],
                rationale=data.get("rationale"),
                source_task_id=task.id,
                success_count=1,
            )

            self.db.add(skill)
            logger.info(f"Extracted skill: {skill.title}")
            return skill

        except Exception as e:
            logger.warning(f"Failed to extract skill with LLM: {e}")
            return None

    async def _extract_anti_pattern_with_llm(
        self,
        task: Task,
        error_message: str,
        context: SkillContext,
    ) -> Skill | None:
        """Extract anti-pattern using LLM analysis."""
        if not self.llm:
            return None

        prompt = f"""Analyze this task failure and extract an anti-pattern to avoid.

TASK: {task.title}
DESCRIPTION: {task.description or "N/A"}

ERROR: {error_message[:500]}

CONTEXT:
- Language: {context.language}
- Framework: {context.framework or "N/A"}

Is there a pattern to avoid in the future? Only extract if it's a recurring mistake.

Output ONLY valid JSON (no markdown, no explanation):
{{
    "should_extract": true/false,
    "title": "What to avoid (max 60 chars)",
    "description": "Why this fails",
    "anti_example": "The approach that failed (code or description)",
    "better_approach": "What to do instead",
    "keywords": ["relevant", "keywords"]
}}
"""

        try:
            response = await self.llm.agenerate(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            data = self._parse_json_response(content)

            if not data.get("should_extract"):
                return None

            skill = Skill.create(
                type=SkillType.ANTI_PATTERN,
                title=data["title"][:100],
                description=data["description"],
                languages=[context.language] if context.language else [],
                frameworks=[context.framework] if context.framework else [],
                task_keywords=data.get("keywords", [])[:5],
                anti_example=data.get("anti_example", "")[: self.config.max_pattern_length],
                pattern=data.get("better_approach"),
                source_task_id=task.id,
                failure_count=1,
            )

            self.db.add(skill)
            logger.info(f"Extracted anti-pattern: {skill.title}")
            return skill

        except Exception as e:
            logger.warning(f"Failed to extract anti-pattern with LLM: {e}")
            return None

    async def _extract_recovery_with_llm(
        self,
        error_message: str,
        fix_actions: str,
        context: SkillContext,
    ) -> Skill | None:
        """Extract recovery skill using LLM analysis."""
        if not self.llm:
            return None

        prompt = f"""Analyze this error recovery and extract a reusable fix pattern.

ERROR: {error_message[:500]}

FIX APPLIED:
{fix_actions[:1000]}

CONTEXT:
- Language: {context.language}
- Framework: {context.framework or "N/A"}

Extract a recovery skill if this fix is reusable for similar errors.

Output ONLY valid JSON (no markdown, no explanation):
{{
    "should_extract": true/false,
    "title": "How to fix [error type] (max 60 chars)",
    "description": "When and why this fix works",
    "pattern": "The fix approach or code",
    "error_pattern": "Type of error this fixes",
    "keywords": ["relevant", "keywords"]
}}
"""

        try:
            response = await self.llm.agenerate(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            data = self._parse_json_response(content)

            if not data.get("should_extract"):
                return None

            skill = Skill.create(
                type=SkillType.RECOVERY,
                title=data["title"][:100],
                description=data["description"],
                languages=[context.language] if context.language else [],
                frameworks=[context.framework] if context.framework else [],
                task_keywords=data.get("keywords", [])[:5],
                pattern=data.get("pattern", "")[: self.config.max_pattern_length],
                success_count=1,
                metadata={"error_pattern": data.get("error_pattern", "")},
            )

            self.db.add(skill)
            logger.info(f"Extracted recovery skill: {skill.title}")
            return skill

        except Exception as e:
            logger.warning(f"Failed to extract recovery skill with LLM: {e}")
            return None

    # =========================================================================
    # Heuristic Extraction (no LLM)
    # =========================================================================

    def _extract_heuristic(
        self,
        task: Task,
        result: TaskResult,
        context: SkillContext,
    ) -> Skill | None:
        """Extract skill using heuristics (no LLM)."""
        # Only extract if there are code changes
        if not result.diff_summary and not result.files_created:
            return None

        # Extract keywords from task title
        keywords = self._extract_keywords(task.title)
        if not keywords:
            return None

        skill = Skill.create(
            type=SkillType.APPROACH,
            title=f"Approach: {task.title[:50]}",
            description=f"Successfully completed: {task.title}",
            languages=[context.language] if context.language else [],
            frameworks=[context.framework] if context.framework else [],
            task_keywords=keywords,
            rationale=f"Completed in {result.execution_time:.1f}s",
            source_task_id=task.id,
            success_count=1,
        )

        self.db.add(skill)
        return skill

    def _extract_anti_pattern_heuristic(
        self,
        task: Task,
        error_message: str,
        context: SkillContext,
    ) -> Skill | None:
        """Extract anti-pattern using heuristics."""
        # Extract error type
        error_type = self._extract_error_type(error_message)
        if not error_type:
            return None

        keywords = self._extract_keywords(task.title)

        skill = Skill.create(
            type=SkillType.ANTI_PATTERN,
            title=f"Avoid: {error_type} in {context.language}",
            description=f"Task '{task.title[:30]}' failed with {error_type}",
            languages=[context.language] if context.language else [],
            frameworks=[context.framework] if context.framework else [],
            task_keywords=keywords,
            anti_example=error_message[:500],
            source_task_id=task.id,
            failure_count=1,
        )

        self.db.add(skill)
        return skill

    def _extract_recovery_heuristic(
        self,
        error_message: str,
        fix_actions: str,
        context: SkillContext,
    ) -> Skill | None:
        """Extract recovery skill using heuristics."""
        error_type = self._extract_error_type(error_message)
        if not error_type:
            return None

        skill = Skill.create(
            type=SkillType.RECOVERY,
            title=f"Fix: {error_type}",
            description=f"Recovery approach for {error_type}",
            languages=[context.language] if context.language else [],
            frameworks=[context.framework] if context.framework else [],
            pattern=fix_actions[:1000],
            success_count=1,
            metadata={"error_type": error_type},
        )

        self.db.add(skill)
        return skill

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM response.

        Args:
            content: Raw LLM response content.

        Returns:
            Parsed JSON dictionary.
        """
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in content
        brace_match = re.search(r"\{.*\}", content, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return {"should_extract": False}

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text.

        Args:
            text: Text to extract from.

        Returns:
            List of keywords.
        """
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

        return unique[:5]

    def _extract_error_type(self, error_message: str) -> str | None:
        """Extract error type from error message.

        Args:
            error_message: Error message string.

        Returns:
            Error type or None.
        """
        # Common error patterns
        patterns = [
            r"(\w+Error):",
            r"(\w+Exception):",
            r"E\s+(\w+Error)",
            r"raise (\w+Error)",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                return match.group(1)

        return None


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "ExtractionConfig",
    "SkillExtractor",
    "TaskResult",
]
