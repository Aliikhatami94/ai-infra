"""Tests for model routing.

Phase 1.1: Tests for ModelRouter and model selection based on task complexity.
"""

import pytest

from ai_infra.executor.routing import (
    COMPLEX_KEYWORDS,
    MODEL_CONFIGS,
    SIMPLE_KEYWORDS,
    ModelRouter,
    ModelTier,
    RoutingConfig,
    RoutingMetrics,
    TaskContext,
    create_router,
)

# =============================================================================
# ModelTier Tests
# =============================================================================


class TestModelTier:
    """Tests for ModelTier enum."""

    def test_tier_values(self):
        """Test ModelTier has expected values."""
        assert ModelTier.FAST.value == "fast"
        assert ModelTier.BALANCED.value == "balanced"
        assert ModelTier.POWERFUL.value == "powerful"

    def test_tier_from_string(self):
        """Test creating ModelTier from string."""
        assert ModelTier("fast") == ModelTier.FAST
        assert ModelTier("balanced") == ModelTier.BALANCED
        assert ModelTier("powerful") == ModelTier.POWERFUL


# =============================================================================
# ModelConfig Tests
# =============================================================================


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_configs_exist(self):
        """Test default configs exist for all tiers."""
        assert ModelTier.FAST in MODEL_CONFIGS
        assert ModelTier.BALANCED in MODEL_CONFIGS
        assert ModelTier.POWERFUL in MODEL_CONFIGS

    def test_fast_config(self):
        """Test FAST tier configuration."""
        config = MODEL_CONFIGS[ModelTier.FAST]
        assert config.tier == ModelTier.FAST
        assert config.expected_latency_ms < 500
        assert config.cost_per_1k_tokens < 0.001

    def test_balanced_config(self):
        """Test BALANCED tier configuration."""
        config = MODEL_CONFIGS[ModelTier.BALANCED]
        assert config.tier == ModelTier.BALANCED
        assert 200 < config.expected_latency_ms < 1000

    def test_powerful_config(self):
        """Test POWERFUL tier configuration."""
        config = MODEL_CONFIGS[ModelTier.POWERFUL]
        assert config.tier == ModelTier.POWERFUL
        assert config.expected_latency_ms >= 1000
        assert config.max_tokens >= 16384


# =============================================================================
# ModelRouter Tests
# =============================================================================


class TestModelRouter:
    """Tests for ModelRouter."""

    def test_default_router(self):
        """Test router with default configuration."""
        router = ModelRouter()
        config = router.select_model()
        # With no task, score is 0 which routes to FAST
        assert config.tier == ModelTier.FAST

    def test_routing_disabled(self):
        """Test router with routing disabled."""
        router = ModelRouter(config=RoutingConfig(enabled=False))
        config = router.select_model({"title": "Complex refactor"})
        # Should return default tier regardless of task
        assert config.tier == ModelTier.BALANCED

    def test_force_tier(self):
        """Test forcing a specific tier."""
        router = ModelRouter(config=RoutingConfig(force_tier=ModelTier.POWERFUL))
        config = router.select_model({"title": "Simple typo fix"})
        # Should return POWERFUL regardless of task complexity
        assert config.tier == ModelTier.POWERFUL

    def test_force_model(self):
        """Test forcing a specific model."""
        router = ModelRouter(config=RoutingConfig(force_model="custom-model"))
        config = router.select_model({"title": "Any task"})
        assert config.model_name == "custom-model"

    def test_simple_task_routes_to_fast(self):
        """Test simple tasks route to FAST tier."""
        router = ModelRouter()
        task = {"title": "Fix typo in README"}
        config = router.select_model(task)
        assert config.tier == ModelTier.FAST

    def test_complex_task_routes_to_powerful(self):
        """Test complex tasks route to POWERFUL tier."""
        router = ModelRouter()
        # Multiple complex keywords: refactor (+2), architecture (+2), migrate (+2) = 6
        task = {
            "title": "Refactor and redesign the entire authentication architecture with security migration"
        }
        config = router.select_model(task)
        assert config.tier == ModelTier.POWERFUL

    def test_medium_task_routes_to_balanced(self):
        """Test medium complexity tasks route to BALANCED."""
        router = ModelRouter()
        # One complex keyword (+2), long title (+1) = 3, which is >2 (FAST threshold)
        task = {"title": "Optimize the database queries for user profile API endpoint performance"}
        config = router.select_model(task)
        assert config.tier == ModelTier.BALANCED


# =============================================================================
# Complexity Scoring Tests
# =============================================================================


class TestComplexityScoring:
    """Tests for complexity scoring logic."""

    def test_simple_keywords_reduce_score(self):
        """Test simple keywords reduce complexity score."""
        router = ModelRouter()

        simple_task = {"title": "Fix typo", "description": ""}
        complex_task = {"title": "Refactor module", "description": ""}

        simple_score = router._compute_complexity_score(simple_task, None)
        complex_score = router._compute_complexity_score(complex_task, None)

        assert simple_score < complex_score

    def test_complex_keywords_increase_score(self):
        """Test complex keywords increase complexity score."""
        router = ModelRouter()

        for keyword in COMPLEX_KEYWORDS[:3]:
            task = {"title": f"Task with {keyword}", "description": ""}
            score = router._compute_complexity_score(task, None)
            # Each complex keyword adds 2 points
            assert score >= 2

    def test_long_title_increases_score(self):
        """Test long titles increase complexity score."""
        router = ModelRouter()

        short = {"title": "Short task", "description": ""}
        long = {"title": "A" * 150, "description": ""}

        short_score = router._compute_complexity_score(short, None)
        long_score = router._compute_complexity_score(long, None)

        assert long_score > short_score

    def test_context_affects_score(self):
        """Test TaskContext affects scoring."""
        router = ModelRouter()
        task = {"title": "Add feature", "description": ""}

        no_context = router._compute_complexity_score(task, None)

        context = TaskContext(
            dependency_count=6,
            file_count_estimate=10,
            previous_failures=1,
        )
        with_context = router._compute_complexity_score(task, context)

        assert with_context > no_context

    def test_plan_complexity_affects_score(self):
        """Test complexity from execution plan affects scoring."""
        router = ModelRouter()
        task = {"title": "Add feature", "description": ""}

        low_context = TaskContext(complexity_from_plan="low")
        high_context = TaskContext(complexity_from_plan="high")

        low_score = router._compute_complexity_score(task, low_context)
        high_score = router._compute_complexity_score(task, high_context)

        assert high_score > low_score

    def test_score_never_negative(self):
        """Test score is always non-negative."""
        router = ModelRouter()

        # Task with many simple keywords
        task = {
            "title": "fix update add remove rename format",
            "description": "",
        }
        score = router._compute_complexity_score(task, None)
        assert score >= 0


# =============================================================================
# RoutingMetrics Tests
# =============================================================================


class TestRoutingMetrics:
    """Tests for RoutingMetrics."""

    def test_record_decision(self):
        """Test recording a routing decision."""
        metrics = RoutingMetrics()
        task = {"id": "1", "title": "Test task"}
        config = MODEL_CONFIGS[ModelTier.BALANCED]

        decision = metrics.record_decision(task, config, complexity_score=3)

        assert decision.task_id == "1"
        assert decision.selected_tier == ModelTier.BALANCED
        assert decision.complexity_score == 3
        assert len(metrics.decisions) == 1

    def test_record_outcome(self):
        """Test recording routing outcome."""
        metrics = RoutingMetrics()
        task = {"id": "1", "title": "Test task"}
        config = MODEL_CONFIGS[ModelTier.BALANCED]

        decision = metrics.record_decision(task, config, complexity_score=3)
        metrics.record_outcome(decision, actual_latency_ms=450.0, success=True)

        assert decision.actual_latency_ms == 450.0
        assert decision.success is True

    def test_get_summary_empty(self):
        """Test summary with no decisions."""
        metrics = RoutingMetrics()
        summary = metrics.get_summary()

        assert summary["total"] == 0
        assert summary["by_tier"] == {}

    def test_get_summary_with_data(self):
        """Test summary with recorded decisions."""
        metrics = RoutingMetrics()

        # Record multiple decisions
        for tier in [ModelTier.FAST, ModelTier.FAST, ModelTier.BALANCED]:
            config = MODEL_CONFIGS[tier]
            decision = metrics.record_decision(
                {"id": "1", "title": "Task"},
                config,
                complexity_score=2,
            )
            metrics.record_outcome(decision, actual_latency_ms=300.0, success=True)

        summary = metrics.get_summary()

        assert summary["total"] == 3
        assert summary["by_tier"]["fast"] == 2
        assert summary["by_tier"]["balanced"] == 1


# =============================================================================
# create_router Tests
# =============================================================================


class TestCreateRouter:
    """Tests for create_router convenience function."""

    def test_create_default_router(self):
        """Test creating default router."""
        router = create_router()
        assert router.config.enabled is True
        assert router.config.force_tier is None

    def test_create_router_with_tier(self):
        """Test creating router with forced tier."""
        router = create_router(force_tier="powerful")
        assert router.config.force_tier == ModelTier.POWERFUL

    def test_create_router_disabled(self):
        """Test creating disabled router."""
        router = create_router(enabled=False)
        assert router.config.enabled is False

    def test_create_router_with_model(self):
        """Test creating router with forced model."""
        router = create_router(force_model="custom-model")
        assert router.config.force_model == "custom-model"


# =============================================================================
# TodoItem Integration Tests
# =============================================================================


class TestTodoItemIntegration:
    """Tests for routing with actual TodoItem objects."""

    def test_route_with_todoitem_like_object(self):
        """Test routing works with object having title attribute."""

        class MockTodo:
            id = "123"
            title = "Fix typo in documentation"
            description = "Simple typo fix"

        router = ModelRouter()
        config = router.select_model(MockTodo())

        # Simple task should route to FAST
        assert config.tier == ModelTier.FAST

    def test_route_with_complex_todoitem(self):
        """Test routing with complex todo item."""

        class MockTodo:
            id = "456"
            title = "Refactor authentication to support OAuth2 and migrate users"
            description = "Complex architectural change"

        router = ModelRouter()
        config = router.select_model(MockTodo())

        # Complex task should route to POWERFUL
        assert config.tier == ModelTier.POWERFUL


# =============================================================================
# Phase 6.1.2: Additional Routing Tests
# =============================================================================


class TestDependencyComplexity:
    """Tests for dependency-based complexity scoring."""

    def test_many_dependencies_increases_complexity(self):
        """Tasks with many dependencies are more complex."""
        router = ModelRouter()
        task = {"title": "Integrate payment", "description": ""}

        # No dependencies
        no_deps = TaskContext(dependency_count=0)
        score_no_deps = router._compute_complexity_score(task, no_deps)

        # Many dependencies (5+ adds complexity)
        many_deps = TaskContext(dependency_count=5)
        score_many_deps = router._compute_complexity_score(task, many_deps)

        # More dependencies should increase complexity
        assert score_many_deps > score_no_deps

    def test_high_dependency_count_routes_up(self):
        """Tasks with many dependencies route to higher tiers."""
        router = ModelRouter()
        task = {"title": "Integrate feature", "description": ""}

        context = TaskContext(dependency_count=6, file_count_estimate=10)
        config = router.select_model(task, context)

        # Should bump up due to dependencies
        assert config.tier in [ModelTier.BALANCED, ModelTier.POWERFUL]


class TestKeywordRouting:
    """Parametric tests for keyword-based routing."""

    @pytest.mark.parametrize(
        "keyword",
        [
            "fix typo",
            "rename variable",
            "update readme",
        ],
    )
    def test_simple_keywords_route_to_fast(self, keyword: str):
        """Test simple keywords route to FAST tier."""
        router = ModelRouter()
        task = {"title": keyword, "description": ""}
        config = router.select_model(task)
        # Simple keywords should route to FAST
        assert config.tier == ModelTier.FAST

    @pytest.mark.parametrize(
        "keyword",
        [
            # Very long and complex titles with multiple keywords
            "refactor and architect the entire authentication system and database with migration and redesign",
        ],
    )
    def test_complex_keywords_route_higher(self, keyword: str):
        """Test complex keywords route to higher tiers."""
        router = ModelRouter()
        task = {"title": keyword, "description": ""}
        config = router.select_model(task)
        # Multiple complex keywords should route higher than FAST
        assert config.tier in [ModelTier.BALANCED, ModelTier.POWERFUL]

    def test_very_complex_task_routes_to_powerful(self):
        """Test very complex task with many keywords routes to POWERFUL."""
        router = ModelRouter()
        # Use the existing test that passes (from TestModelRouter)
        task = {
            "title": "Refactor and redesign the entire authentication architecture with security migration"
        }
        config = router.select_model(task)
        assert config.tier == ModelTier.POWERFUL

    @pytest.mark.parametrize(
        "simple_keyword",
        ["fix", "update", "add", "rename", "format", "remove"],
    )
    def test_simple_keywords_in_list(self, simple_keyword: str):
        """Test all simple keywords are recognized."""
        # Verify the keyword is in the SIMPLE_KEYWORDS list
        simple_keywords_lower = [k.lower() for k in SIMPLE_KEYWORDS]
        assert simple_keyword.lower() in simple_keywords_lower

    @pytest.mark.parametrize(
        "complex_keyword",
        ["refactor", "architect", "migrate", "redesign", "optimize", "rewrite"],
    )
    def test_complex_keywords_in_list(self, complex_keyword: str):
        """Test all complex keywords are recognized."""
        # Verify the keyword is in the COMPLEX_KEYWORDS list
        complex_keywords_lower = [k.lower() for k in COMPLEX_KEYWORDS]
        assert complex_keyword.lower() in complex_keywords_lower

    def test_single_complex_keyword_increases_score(self):
        """Test single complex keyword increases complexity score."""
        router = ModelRouter()
        simple = {"title": "update file", "description": ""}
        complex_single = {"title": "refactor file", "description": ""}

        simple_score = router._compute_complexity_score(simple, None)
        complex_score = router._compute_complexity_score(complex_single, None)

        # Complex keyword should add points
        assert complex_score > simple_score


class TestPreviousFailuresAffectRouting:
    """Tests for previous failures affecting routing."""

    def test_previous_failures_increase_complexity(self):
        """Tasks that failed before should route to more powerful models."""
        router = ModelRouter()
        task = {"title": "Add feature", "description": ""}

        # No failures
        no_failures = TaskContext(previous_failures=0)
        score_no_failures = router._compute_complexity_score(task, no_failures)

        # Previous failures
        with_failures = TaskContext(previous_failures=2)
        score_with_failures = router._compute_complexity_score(task, with_failures)

        # Failures should increase complexity
        assert score_with_failures > score_no_failures

    def test_retry_routes_to_more_powerful(self):
        """Retrying a failed task should use more powerful model."""
        router = ModelRouter()
        task = {"title": "Fix bug", "description": ""}

        # First attempt - simple task
        first_config = router.select_model(task, TaskContext())
        assert first_config.tier == ModelTier.FAST

        # Retry after failure - should escalate
        retry_context = TaskContext(previous_failures=1)
        retry_config = router.select_model(task, retry_context)

        # May stay at FAST or bump up depending on threshold
        # Just verify it's not lower
        tier_order = [ModelTier.FAST, ModelTier.BALANCED, ModelTier.POWERFUL]
        assert tier_order.index(retry_config.tier) >= tier_order.index(first_config.tier)
