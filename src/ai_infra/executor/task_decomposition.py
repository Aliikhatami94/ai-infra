"""Task decomposition for intelligent task handling.

This module provides task complexity estimation and automatic decomposition
for complex tasks that should be broken into smaller, manageable pieces.

Phase 3.2 of EXECUTOR_1.md - Intelligent Task Handling.

Usage:
    from ai_infra.executor.task_decomposition import (
        ComplexityEstimator,
        TaskComplexity,
        TaskDecomposer,
    )

    # Estimate complexity
    estimator = ComplexityEstimator()
    complexity = estimator.estimate(task)

    if complexity.recommended_action == "decompose":
        # Decompose the task
        decomposer = TaskDecomposer(agent)
        subtasks = await decomposer.decompose(task, context)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoItem
    from ai_infra.llm.agent import Agent

__all__ = [
    "ComplexityEstimator",
    "ComplexityLevel",
    "DecomposedTask",
    "RecommendedAction",
    "TaskComplexity",
    "TaskDecomposer",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Complexity Levels and Actions
# =============================================================================


class ComplexityLevel(str, Enum):
    """Task complexity level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RecommendedAction(str, Enum):
    """Recommended action based on complexity analysis."""

    EXECUTE = "execute"
    """Execute the task as-is."""

    DECOMPOSE = "decompose"
    """Decompose into smaller tasks before execution."""

    CLARIFY = "clarify"
    """Request clarification from user before execution."""


# =============================================================================
# Complexity Keywords
# =============================================================================

# Keywords that increase complexity score (weight 2)
HIGH_COMPLEXITY_KEYWORDS: tuple[str, ...] = (
    "refactor",
    "architect",
    "redesign",
    "migrate",
    "rewrite",
    "restructure",
    "overhaul",
    "rearchitect",
    "implement entire",
    "build complete",
    "full implementation",
    "comprehensive",
    "end-to-end",
    "from scratch",
)

# Keywords that moderately increase complexity (weight 1)
MEDIUM_COMPLEXITY_KEYWORDS: tuple[str, ...] = (
    "integrate",
    "optimize",
    "performance",
    "security",
    "concurrent",
    "parallel",
    "async",
    "database",
    "authentication",
    "authorization",
    "caching",
    "distributed",
)

# Keywords that decrease complexity (weight -1)
SIMPLE_KEYWORDS: tuple[str, ...] = (
    "fix typo",
    "rename",
    "format",
    "update version",
    "add comment",
    "docstring",
    "import",
    "bump",
    "delete",
    "remove unused",
)


# =============================================================================
# Task Complexity
# =============================================================================


@dataclass
class TaskComplexity:
    """Analysis of task complexity.

    Attributes:
        score: Numeric complexity score (0-20+).
        level: Categorical complexity level.
        factors: List of factors contributing to complexity.
        estimated_time: Estimated time to complete (e.g., "5 min", "2 hours").
        recommended_action: Suggested action (execute, decompose, clarify).
        confidence: Confidence in the analysis (0.0 to 1.0).
    """

    score: int
    level: ComplexityLevel
    factors: list[str] = field(default_factory=list)
    estimated_time: str = ""
    recommended_action: RecommendedAction = RecommendedAction.EXECUTE
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "level": self.level.value,
            "factors": self.factors,
            "estimated_time": self.estimated_time,
            "recommended_action": self.recommended_action.value,
            "confidence": self.confidence,
        }

    @property
    def should_decompose(self) -> bool:
        """Check if task should be decomposed."""
        return self.recommended_action == RecommendedAction.DECOMPOSE

    @property
    def is_complex(self) -> bool:
        """Check if task is considered complex."""
        return self.level in (ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH)


# =============================================================================
# Decomposed Task
# =============================================================================


@dataclass
class DecomposedTask:
    """A subtask created from decomposition.

    Attributes:
        title: The subtask title.
        description: Detailed description of what to do.
        depends_on: List of subtask indices this depends on (0-indexed).
        estimated_complexity: Estimated complexity of this subtask.
        files_hint: Files expected to be modified.
    """

    title: str
    description: str = ""
    depends_on: list[int] = field(default_factory=list)
    estimated_complexity: str = "low"
    files_hint: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "depends_on": self.depends_on,
            "estimated_complexity": self.estimated_complexity,
            "files_hint": self.files_hint,
        }


# =============================================================================
# Complexity Estimator
# =============================================================================


@dataclass
class EstimatorConfig:
    """Configuration for complexity estimation.

    Attributes:
        decompose_threshold: Score above which decomposition is recommended.
        clarify_threshold: Score above which clarification is recommended.
        title_length_penalty: Extra score per 100 chars of title.
        max_dependencies_simple: Max dependencies for simple tasks.
    """

    decompose_threshold: int = 8
    clarify_threshold: int = 15
    title_length_penalty: int = 1
    max_dependencies_simple: int = 2


class ComplexityEstimator:
    """Estimate task complexity to determine execution strategy.

    Analyzes task characteristics to produce a complexity score and
    recommended action. Used to decide whether to execute a task
    directly or decompose it into smaller pieces.

    Example:
        >>> estimator = ComplexityEstimator()
        >>> task = TodoItem(id=1, title="Refactor the entire auth system")
        >>> complexity = estimator.estimate(task)
        >>> print(complexity.level)
        ComplexityLevel.HIGH
        >>> print(complexity.recommended_action)
        RecommendedAction.DECOMPOSE
    """

    def __init__(self, config: EstimatorConfig | None = None) -> None:
        """Initialize the complexity estimator.

        Args:
            config: Configuration for estimation thresholds.
        """
        self.config = config or EstimatorConfig()

    def estimate(
        self,
        task: TodoItem | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> TaskComplexity:
        """Estimate complexity of a task.

        Args:
            task: The task to analyze (TodoItem or dict with title/description).
            context: Optional execution context (dependency count, etc.).

        Returns:
            TaskComplexity with score, level, and recommendation.
        """
        score = 0
        factors: list[str] = []

        # Extract task info
        title, description = self._extract_task_info(task)
        text = f"{title} {description}".lower()

        # 1. Title length analysis
        title_len = len(title)
        if title_len > 200:
            score += 3
            factors.append(f"Very long title ({title_len} chars)")
        elif title_len > 100:
            score += 2
            factors.append(f"Long title ({title_len} chars)")
        elif title_len > 50:
            score += 1
            factors.append(f"Moderate title length ({title_len} chars)")

        # 2. Multiple objectives detection
        conjunctions = text.count(" and ") + text.count(" then ") + text.count(", and ")
        if conjunctions >= 3:
            score += 3
            factors.append(f"Multiple objectives ({conjunctions} conjunctions)")
        elif conjunctions >= 1:
            score += 1
            factors.append("Contains multiple objectives")

        # 3. High complexity keywords
        for keyword in HIGH_COMPLEXITY_KEYWORDS:
            if keyword in text:
                score += 2
                factors.append(f"Complex keyword: '{keyword}'")

        # 4. Medium complexity keywords
        for keyword in MEDIUM_COMPLEXITY_KEYWORDS:
            if keyword in text:
                score += 1
                if len(factors) < 10:  # Limit factor list
                    factors.append(f"Technical keyword: '{keyword}'")

        # 5. Simple keywords (reduce score)
        for keyword in SIMPLE_KEYWORDS:
            if keyword in text:
                score -= 1
                factors.append(f"Simple task indicator: '{keyword}'")

        # 6. Context-based scoring
        if context:
            # Dependency count
            dep_count = context.get("dependency_count", 0)
            if dep_count > 5:
                score += 3
                factors.append(f"Many dependencies ({dep_count})")
            elif dep_count > self.config.max_dependencies_simple:
                score += 1
                factors.append(f"{dep_count} dependencies")

            # File scope estimate
            file_count = context.get("file_count_estimate", 1)
            if file_count > 10:
                score += 3
                factors.append(f"Large file scope ({file_count} files)")
            elif file_count > 5:
                score += 2
                factors.append(f"Multiple files ({file_count})")
            elif file_count > 2:
                score += 1

            # Previous failures
            failures = context.get("previous_failures", 0)
            if failures > 0:
                score += min(failures * 2, 6)
                factors.append(f"Previous failures: {failures}")

        # 7. Vagueness indicators
        vague_patterns = [
            r"\bsomething\b",
            r"\bwhatever\b",
            r"\betc\.?\b",
            r"\bstuff\b",
            r"\bthings?\b",
            r"\bsome\b",
            r"\bmaybe\b",
            r"\bprobably\b",
        ]
        vague_count = sum(1 for pattern in vague_patterns if re.search(pattern, text))
        if vague_count >= 2:
            score += 2
            factors.append("Contains vague language")
        elif vague_count == 1:
            score += 1

        # Ensure non-negative
        score = max(0, score)

        # Determine level
        if score <= 3:
            level = ComplexityLevel.LOW
            estimated_time = "5-15 min"
        elif score <= 6:
            level = ComplexityLevel.MEDIUM
            estimated_time = "15-45 min"
        elif score <= 10:
            level = ComplexityLevel.HIGH
            estimated_time = "1-2 hours"
        else:
            level = ComplexityLevel.VERY_HIGH
            estimated_time = "2+ hours"

        # Determine recommended action
        if score >= self.config.clarify_threshold:
            recommended = RecommendedAction.CLARIFY
        elif score >= self.config.decompose_threshold:
            recommended = RecommendedAction.DECOMPOSE
        else:
            recommended = RecommendedAction.EXECUTE

        # Calculate confidence (lower if factors are conflicting)
        confidence = 1.0 - (0.1 * min(len(factors), 5))

        logger.debug(
            f"Complexity estimate: score={score}, level={level.value}, "
            f"action={recommended.value}, factors={len(factors)}"
        )

        return TaskComplexity(
            score=score,
            level=level,
            factors=factors[:10],  # Limit factors
            estimated_time=estimated_time,
            recommended_action=recommended,
            confidence=confidence,
        )

    def _extract_task_info(self, task: TodoItem | dict[str, Any]) -> tuple[str, str]:
        """Extract title and description from task.

        Args:
            task: TodoItem or dict.

        Returns:
            Tuple of (title, description).
        """
        if hasattr(task, "title"):
            title = getattr(task, "title", "") or ""
            description = getattr(task, "description", "") or ""
        elif isinstance(task, dict):
            title = task.get("title", "")
            description = task.get("description", "")
        else:
            title = str(task)
            description = ""

        return title, description


# =============================================================================
# Task Decomposer
# =============================================================================


DECOMPOSE_PROMPT_TEMPLATE = """You are a senior software architect breaking down a complex task.

TASK TO DECOMPOSE:
Title: {title}
Description: {description}

PROJECT CONTEXT:
{project_context}

COMPLEXITY ANALYSIS:
- Score: {complexity_score}/20
- Level: {complexity_level}
- Factors: {complexity_factors}

INSTRUCTIONS:
1. Break this task into 2-5 smaller, independent subtasks
2. Each subtask should be achievable in a single focused work session
3. Order subtasks by dependency (what must be done first)
4. Each subtask should have clear, testable output
5. Keep the original intent - don't add or remove scope

OUTPUT FORMAT (JSON array):
```json
[
  {{
    "title": "Subtask 1: Create base structure",
    "description": "Create the initial file structure and basic imports",
    "depends_on": [],
    "files_hint": ["src/new_file.py"]
  }},
  {{
    "title": "Subtask 2: Implement core logic",
    "description": "Add the main functionality",
    "depends_on": [0],
    "files_hint": ["src/new_file.py"]
  }}
]
```

SUBTASKS:"""


class TaskDecomposer:
    """Decompose complex tasks into smaller subtasks.

    Uses an AI agent to intelligently break down tasks that are
    too complex to execute in a single pass.

    Example:
        >>> decomposer = TaskDecomposer(agent)
        >>> subtasks = await decomposer.decompose(
        ...     task=complex_task,
        ...     complexity=complexity_analysis,
        ... )
        >>> for st in subtasks:
        ...     print(f"- {st.title}")
    """

    def __init__(
        self,
        agent: Agent,
        min_subtasks: int = 2,
        max_subtasks: int = 5,
    ) -> None:
        """Initialize the task decomposer.

        Args:
            agent: AI agent for decomposition.
            min_subtasks: Minimum number of subtasks to produce.
            max_subtasks: Maximum number of subtasks to produce.
        """
        self.agent = agent
        self.min_subtasks = min_subtasks
        self.max_subtasks = max_subtasks

    async def decompose(
        self,
        task: TodoItem | dict[str, Any],
        complexity: TaskComplexity | None = None,
        project_context: str = "",
    ) -> list[DecomposedTask]:
        """Decompose a complex task into subtasks.

        Args:
            task: The task to decompose.
            complexity: Pre-computed complexity analysis (optional).
            project_context: Additional project context for the LLM.

        Returns:
            List of DecomposedTask subtasks.

        Raises:
            ValueError: If decomposition fails or produces invalid results.
        """
        # Extract task info
        if hasattr(task, "title"):
            title = getattr(task, "title", "") or ""
            description = getattr(task, "description", "") or ""
        elif isinstance(task, dict):
            title = task.get("title", "")
            description = task.get("description", "")
        else:
            title = str(task)
            description = ""

        # Compute complexity if not provided
        if complexity is None:
            estimator = ComplexityEstimator()
            complexity = estimator.estimate(task)

        # Build prompt
        prompt = DECOMPOSE_PROMPT_TEMPLATE.format(
            title=title,
            description=description or "(no additional description)",
            project_context=project_context or "No specific context provided",
            complexity_score=complexity.score,
            complexity_level=complexity.level.value,
            complexity_factors=", ".join(complexity.factors) or "None identified",
        )

        # Call agent
        try:
            result = await self.agent.ainvoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            logger.error(f"Decomposition LLM call failed: {e}")
            raise ValueError(f"Failed to decompose task: {e}") from e

        # Parse response
        subtasks = self._parse_subtasks(content)

        if len(subtasks) < self.min_subtasks:
            logger.warning(
                f"Decomposition produced only {len(subtasks)} subtasks, "
                f"expected at least {self.min_subtasks}"
            )
            # Still return what we got if > 0
            if len(subtasks) == 0:
                raise ValueError("Decomposition produced no valid subtasks")

        if len(subtasks) > self.max_subtasks:
            logger.warning(
                f"Decomposition produced {len(subtasks)} subtasks, "
                f"truncating to {self.max_subtasks}"
            )
            subtasks = subtasks[: self.max_subtasks]

        logger.info(f"Decomposed '{title[:50]}...' into {len(subtasks)} subtasks")

        return subtasks

    def _parse_subtasks(self, content: str) -> list[DecomposedTask]:
        """Parse subtasks from LLM response.

        Args:
            content: Raw LLM response.

        Returns:
            List of DecomposedTask objects.
        """
        import json

        subtasks: list[DecomposedTask] = []

        # Try to extract JSON array
        json_match = re.search(r"\[[\s\S]*\]", content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "title" in item:
                            subtasks.append(
                                DecomposedTask(
                                    title=item.get("title", ""),
                                    description=item.get("description", ""),
                                    depends_on=item.get("depends_on", []),
                                    estimated_complexity=item.get("estimated_complexity", "low"),
                                    files_hint=item.get("files_hint", []),
                                )
                            )
            except json.JSONDecodeError:
                pass

        # Fallback: parse numbered list
        if not subtasks:
            subtasks = self._parse_numbered_list(content)

        return subtasks

    def _parse_numbered_list(self, content: str) -> list[DecomposedTask]:
        """Parse numbered list format as fallback.

        Args:
            content: Raw content.

        Returns:
            List of DecomposedTask objects.
        """
        subtasks: list[DecomposedTask] = []

        # Match patterns like "1. Task title" or "- Task title"
        pattern = r"(?:^|\n)\s*(?:\d+[\.\)]\s*|\-\s*|\*\s*)(.+?)(?=\n\s*(?:\d+[\.\)]|\-|\*)|$)"
        matches = re.findall(pattern, content, re.MULTILINE)

        for match in matches:
            title = match.strip()
            if title and len(title) > 3:  # Skip very short matches
                # Clean up markdown formatting
                title = re.sub(r"\*\*(.+?)\*\*", r"\1", title)
                title = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", title)
                subtasks.append(DecomposedTask(title=title))

        return subtasks


# =============================================================================
# Convenience Functions
# =============================================================================


def should_decompose_task(
    task: TodoItem | dict[str, Any],
    threshold: int = 8,
) -> tuple[bool, TaskComplexity]:
    """Quick check if a task should be decomposed.

    Args:
        task: The task to check.
        threshold: Complexity score threshold for decomposition.

    Returns:
        Tuple of (should_decompose, complexity_analysis).
    """
    estimator = ComplexityEstimator(EstimatorConfig(decompose_threshold=threshold))
    complexity = estimator.estimate(task)
    return complexity.should_decompose, complexity


async def auto_decompose_if_needed(
    task: TodoItem | dict[str, Any],
    agent: Agent,
    threshold: int = 8,
    project_context: str = "",
) -> tuple[bool, list[DecomposedTask]]:
    """Automatically decompose a task if it exceeds complexity threshold.

    Args:
        task: The task to potentially decompose.
        agent: Agent for decomposition.
        threshold: Complexity score threshold.
        project_context: Additional context for decomposition.

    Returns:
        Tuple of (was_decomposed, subtasks).
        If not decomposed, subtasks will be empty.
    """
    should_decompose, complexity = should_decompose_task(task, threshold)

    if not should_decompose:
        return False, []

    decomposer = TaskDecomposer(agent)
    try:
        subtasks = await decomposer.decompose(
            task=task,
            complexity=complexity,
            project_context=project_context,
        )
        return True, subtasks
    except ValueError as e:
        logger.warning(f"Auto-decompose failed: {e}")
        return False, []
