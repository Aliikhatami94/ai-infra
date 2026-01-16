"""Model routing for the Executor.

Phase 1.1: Routes tasks to appropriate models based on complexity.

This module enables intelligent model selection to optimize for:
- Latency: Simple tasks use fast models (200ms vs 2s)
- Cost: 5-50x cheaper for simple tasks
- Quality: Complex tasks still use powerful models

Usage:
    from ai_infra.executor.routing import ModelRouter, ModelTier

    router = ModelRouter()
    config = router.select_model(task, context)
    agent = Agent(model=config.model_name, max_tokens=config.max_tokens)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoItem

logger = get_logger("executor.routing")


# =============================================================================
# Model Tier Enum
# =============================================================================


class ModelTier(str, Enum):
    """Model capability tiers for task routing.

    FAST: Low-latency models for simple tasks (typos, renames, format).
    BALANCED: Medium models for standard coding tasks.
    POWERFUL: Full-capability models for complex architecture/refactoring.
    """

    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"


# =============================================================================
# Model Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a model tier.

    Attributes:
        tier: The model tier.
        model_name: The model identifier (e.g., "claude-sonnet-4-20250514").
        max_tokens: Maximum output tokens for this tier.
        expected_latency_ms: Expected response latency in milliseconds.
        cost_per_1k_tokens: Cost per 1k output tokens in USD.
    """

    tier: ModelTier
    model_name: str
    max_tokens: int
    expected_latency_ms: int
    cost_per_1k_tokens: float


# Default model configurations per tier
MODEL_CONFIGS: dict[ModelTier, ModelConfig] = {
    ModelTier.FAST: ModelConfig(
        tier=ModelTier.FAST,
        model_name="gpt-4.1-nano",
        max_tokens=4096,
        expected_latency_ms=200,
        cost_per_1k_tokens=0.0001,
    ),
    ModelTier.BALANCED: ModelConfig(
        tier=ModelTier.BALANCED,
        model_name="claude-sonnet-4-20250514",
        max_tokens=8192,
        expected_latency_ms=500,
        cost_per_1k_tokens=0.003,
    ),
    ModelTier.POWERFUL: ModelConfig(
        tier=ModelTier.POWERFUL,
        model_name="claude-opus-4-20250514",
        max_tokens=16384,
        expected_latency_ms=2000,
        cost_per_1k_tokens=0.015,
    ),
}


# =============================================================================
# Complexity Keywords
# =============================================================================

# Keywords indicating high complexity (architecture, design)
COMPLEX_KEYWORDS: tuple[str, ...] = (
    "refactor",
    "architect",
    "redesign",
    "migrate",
    "optimize",
    "rewrite",
    "restructure",
    "overhaul",
    "rearchitect",
    "performance",
    "security",
    "concurrent",
    "parallel",
    "distributed",
)

# Keywords indicating low complexity (simple changes)
SIMPLE_KEYWORDS: tuple[str, ...] = (
    "fix",
    "typo",
    "rename",
    "format",
    "update",
    "add",
    "remove",
    "delete",
    "comment",
    "docstring",
    "import",
    "version",
    "bump",
)


# =============================================================================
# Routing Configuration
# =============================================================================


@dataclass
class RoutingConfig:
    """Configuration for model routing behavior.

    Attributes:
        enabled: Whether routing is enabled (False = use default model).
        force_tier: Force a specific tier (overrides routing).
        force_model: Force a specific model (overrides everything).
        default_tier: Tier to use when routing is disabled.
        complexity_threshold_fast: Max score for FAST tier.
        complexity_threshold_balanced: Max score for BALANCED tier.
    """

    enabled: bool = True
    force_tier: ModelTier | None = None
    force_model: str | None = None
    default_tier: ModelTier = ModelTier.BALANCED
    complexity_threshold_fast: int = 2
    complexity_threshold_balanced: int = 5


# =============================================================================
# Task Context
# =============================================================================


@dataclass
class TaskContext:
    """Context for routing decisions.

    Attributes:
        previous_failures: Number of previous failures on this task.
        similar_task_failed: Whether a similar task failed recently.
        file_count_estimate: Estimated number of files to modify.
        has_dependencies: Whether task has dependencies on other tasks.
        dependency_count: Number of task dependencies.
        complexity_from_plan: Complexity from execution plan (low/medium/high).
    """

    previous_failures: int = 0
    similar_task_failed: bool = False
    file_count_estimate: int = 1
    has_dependencies: bool = False
    dependency_count: int = 0
    complexity_from_plan: str | None = None


# =============================================================================
# Model Router
# =============================================================================


class ModelRouter:
    """Routes tasks to appropriate models based on complexity.

    The router analyzes task characteristics to select the optimal model tier:
    - FAST: Simple tasks like typos, renames, format fixes (200ms, $0.0001/1k)
    - BALANCED: Standard coding tasks (500ms, $0.003/1k)
    - POWERFUL: Complex refactoring, architecture (2s, $0.015/1k)

    Usage:
        router = ModelRouter()
        config = router.select_model(task, context)
        agent = Agent(model=config.model_name)

    With configuration:
        router = ModelRouter(RoutingConfig(force_tier=ModelTier.POWERFUL))
        config = router.select_model(task)  # Always returns POWERFUL
    """

    def __init__(
        self,
        config: RoutingConfig | None = None,
        model_configs: dict[ModelTier, ModelConfig] | None = None,
    ):
        """Initialize the ModelRouter.

        Args:
            config: Routing configuration.
            model_configs: Override default model configurations.
        """
        self.config = config or RoutingConfig()
        self.model_configs = model_configs or MODEL_CONFIGS

    def select_model(
        self,
        task: TodoItem | dict[str, Any] | None = None,
        context: TaskContext | None = None,
    ) -> ModelConfig:
        """Select the appropriate model for a task.

        Args:
            task: The task to route (TodoItem or dict with title/description).
            context: Optional context for routing decisions.

        Returns:
            ModelConfig with the selected model settings.
        """
        # Force model override (highest priority)
        if self.config.force_model:
            logger.debug(f"Using forced model: {self.config.force_model}")
            return ModelConfig(
                tier=ModelTier.BALANCED,
                model_name=self.config.force_model,
                max_tokens=8192,
                expected_latency_ms=1000,
                cost_per_1k_tokens=0.01,
            )

        # Force tier override
        if self.config.force_tier:
            logger.debug(f"Using forced tier: {self.config.force_tier}")
            return self.model_configs[self.config.force_tier]

        # Routing disabled - use default
        if not self.config.enabled:
            logger.debug(f"Routing disabled, using default tier: {self.config.default_tier}")
            return self.model_configs[self.config.default_tier]

        # Compute complexity and select tier
        tier = self._classify_task(task, context)
        config = self.model_configs[tier]

        logger.info(
            f"Routed task to {tier.value} tier: "
            f"model={config.model_name}, "
            f"expected_latency={config.expected_latency_ms}ms"
        )

        return config

    def _classify_task(
        self,
        task: TodoItem | dict[str, Any] | None,
        context: TaskContext | None,
    ) -> ModelTier:
        """Classify task into a model tier based on complexity.

        Args:
            task: The task to classify.
            context: Optional context for classification.

        Returns:
            The appropriate ModelTier.
        """
        score = self._compute_complexity_score(task, context)

        if score <= self.config.complexity_threshold_fast:
            return ModelTier.FAST
        if score <= self.config.complexity_threshold_balanced:
            return ModelTier.BALANCED
        return ModelTier.POWERFUL

    def _compute_complexity_score(
        self,
        task: TodoItem | dict[str, Any] | None,
        context: TaskContext | None,
    ) -> int:
        """Compute a complexity score for the task.

        Higher scores indicate more complex tasks. Score components:
        - Title length: +1 for >100 chars, +1 for >200 chars
        - Complex keywords: +2 each (refactor, architect, etc.)
        - Simple keywords: -1 each (fix, typo, etc.)
        - Dependencies: +1 for >2, +2 for >5
        - File count: +1 for >3 files, +2 for >7 files
        - Previous failures: +2 if similar task failed
        - Plan complexity: +0/+1/+3 for low/medium/high

        Args:
            task: The task to score.
            context: Optional context for scoring.

        Returns:
            Non-negative complexity score.
        """
        score = 0

        # Extract task info
        title = ""
        description = ""

        if task is not None:
            if hasattr(task, "title"):
                title = task.title or ""
                description = getattr(task, "description", "") or ""
            elif isinstance(task, dict):
                title = task.get("title", "")
                description = task.get("description", "")

        text = f"{title} {description}".lower()

        # Title length scoring
        if len(title) > 100:
            score += 1
        if len(title) > 200:
            score += 1

        # Keyword analysis
        for keyword in COMPLEX_KEYWORDS:
            if keyword in text:
                score += 2

        for keyword in SIMPLE_KEYWORDS:
            if keyword in text:
                score -= 1

        # Context-based scoring
        if context:
            # Dependencies
            if context.dependency_count > 2:
                score += 1
            if context.dependency_count > 5:
                score += 2

            # File scope
            if context.file_count_estimate > 3:
                score += 1
            if context.file_count_estimate > 7:
                score += 2

            # Previous failures
            if context.similar_task_failed or context.previous_failures > 0:
                score += 2

            # Plan complexity (from planning node)
            if context.complexity_from_plan:
                complexity = context.complexity_from_plan.lower()
                if complexity == "high":
                    score += 3
                elif complexity == "medium":
                    score += 1
                # low adds nothing

        logger.debug(f"Complexity score for '{title[:50]}...': {max(0, score)}")
        return max(0, score)  # Ensure non-negative


# =============================================================================
# Routing Metrics
# =============================================================================


@dataclass
class RoutingDecision:
    """Record of a routing decision for metrics tracking.

    Attributes:
        task_id: ID of the task.
        task_title: Title of the task.
        selected_tier: The tier selected by routing.
        complexity_score: The computed complexity score.
        model_name: The model selected.
        expected_latency_ms: Expected latency at routing time.
        actual_latency_ms: Actual latency after execution (set later).
        success: Whether the task succeeded (set later).
    """

    task_id: str
    task_title: str
    selected_tier: ModelTier
    complexity_score: int
    model_name: str
    expected_latency_ms: int
    actual_latency_ms: float | None = None
    success: bool | None = None


class RoutingMetrics:
    """Tracks routing decisions and outcomes for analysis.

    Usage:
        metrics = RoutingMetrics()
        decision = metrics.record_decision(task, config, score)
        # After execution
        metrics.record_outcome(decision, actual_latency_ms=450, success=True)
    """

    def __init__(self) -> None:
        """Initialize the metrics tracker."""
        self.decisions: list[RoutingDecision] = []

    def record_decision(
        self,
        task: TodoItem | dict[str, Any],
        config: ModelConfig,
        complexity_score: int,
    ) -> RoutingDecision:
        """Record a routing decision.

        Args:
            task: The task being routed.
            config: The selected model configuration.
            complexity_score: The computed complexity score.

        Returns:
            The recorded RoutingDecision.
        """
        # Extract task info
        if hasattr(task, "id"):
            task_id = str(task.id)
            task_title = task.title or ""
        elif isinstance(task, dict):
            task_id = str(task.get("id", "unknown"))
            task_title = task.get("title", "")
        else:
            task_id = "unknown"
            task_title = ""

        decision = RoutingDecision(
            task_id=task_id,
            task_title=task_title[:100],  # Truncate for storage
            selected_tier=config.tier,
            complexity_score=complexity_score,
            model_name=config.model_name,
            expected_latency_ms=config.expected_latency_ms,
        )
        self.decisions.append(decision)

        logger.debug(
            f"Recorded routing decision: task={task_id}, "
            f"tier={config.tier.value}, score={complexity_score}"
        )

        return decision

    def record_outcome(
        self,
        decision: RoutingDecision,
        actual_latency_ms: float,
        success: bool,
    ) -> None:
        """Record the outcome of a routing decision.

        Args:
            decision: The routing decision to update.
            actual_latency_ms: Actual execution latency.
            success: Whether the task succeeded.
        """
        decision.actual_latency_ms = actual_latency_ms
        decision.success = success

        logger.debug(
            f"Recorded routing outcome: task={decision.task_id}, "
            f"latency={actual_latency_ms:.1f}ms, success={success}"
        )

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of routing metrics.

        Returns:
            Dictionary with routing statistics.
        """
        if not self.decisions:
            return {"total": 0, "by_tier": {}, "avg_latency": {}}

        by_tier: dict[str, int] = {}
        latencies: dict[str, list[float]] = {}
        success_counts: dict[str, tuple[int, int]] = {}

        for decision in self.decisions:
            tier = decision.selected_tier.value
            by_tier[tier] = by_tier.get(tier, 0) + 1

            if decision.actual_latency_ms is not None:
                if tier not in latencies:
                    latencies[tier] = []
                latencies[tier].append(decision.actual_latency_ms)

            if decision.success is not None:
                if tier not in success_counts:
                    success_counts[tier] = (0, 0)
                successes, total = success_counts[tier]
                success_counts[tier] = (
                    successes + (1 if decision.success else 0),
                    total + 1,
                )

        return {
            "total": len(self.decisions),
            "by_tier": by_tier,
            "avg_latency": {
                tier: sum(lats) / len(lats) for tier, lats in latencies.items() if lats
            },
            "success_rate": {
                tier: successes / total if total > 0 else 0.0
                for tier, (successes, total) in success_counts.items()
            },
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_router(
    *,
    enabled: bool = True,
    force_tier: str | ModelTier | None = None,
    force_model: str | None = None,
) -> ModelRouter:
    """Create a ModelRouter with common configurations.

    Args:
        enabled: Whether routing is enabled.
        force_tier: Force a specific tier ("fast", "balanced", "powerful").
        force_model: Force a specific model name.

    Returns:
        Configured ModelRouter.
    """
    tier = None
    if force_tier:
        if isinstance(force_tier, str):
            tier = ModelTier(force_tier.lower())
        else:
            tier = force_tier

    config = RoutingConfig(
        enabled=enabled,
        force_tier=tier,
        force_model=force_model,
    )

    return ModelRouter(config=config)
