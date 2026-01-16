"""Tests for Phase 10.1: Cost Tracking Accuracy.

This module tests:
- 10.1.1: ModelPricing and MODEL_PRICING database
- 10.1.2: TokenMetrics with cost calculation
- 10.1.3: Summary output with cost display
- 10.1.4: Cost matches expected
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from ai_infra.executor.metrics import TokenMetrics, format_run_summary_with_nodes
from ai_infra.executor.pricing import (
    MODEL_PRICING,
    ModelPricing,
    get_model_aliases,
    get_pricing,
    list_supported_models,
)

# =============================================================================
# Test ModelPricing (10.1.1)
# =============================================================================


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_create_pricing(self) -> None:
        """Test creating a ModelPricing instance."""
        pricing = ModelPricing(
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        assert pricing.input_per_million == Decimal("3.00")
        assert pricing.output_per_million == Decimal("15.00")
        assert pricing.cached_input_per_million is None

    def test_create_pricing_with_cache(self) -> None:
        """Test creating a ModelPricing with cache pricing."""
        pricing = ModelPricing(
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
            cached_input_per_million=Decimal("0.30"),
        )
        assert pricing.cached_input_per_million == Decimal("0.30")

    def test_pricing_is_frozen(self) -> None:
        """Test that ModelPricing is immutable."""
        pricing = ModelPricing(
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        with pytest.raises(AttributeError):
            pricing.input_per_million = Decimal("5.00")  # type: ignore[misc]

    def test_calculate_cost_basic(self) -> None:
        """Test basic cost calculation."""
        pricing = ModelPricing(
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        # 1M input tokens + 1M output tokens
        cost = pricing.calculate_cost(input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == Decimal("18.00")  # 3 + 15

    def test_calculate_cost_with_fractions(self) -> None:
        """Test cost calculation with fractional token counts."""
        pricing = ModelPricing(
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        # 1000 input + 500 output
        cost = pricing.calculate_cost(input_tokens=1000, output_tokens=500)
        expected = Decimal("1000") / Decimal("1000000") * Decimal("3.00") + Decimal(
            "500"
        ) / Decimal("1000000") * Decimal("15.00")
        assert cost == expected

    def test_calculate_cost_with_cache(self) -> None:
        """Test cost calculation with cached tokens."""
        pricing = ModelPricing(
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
            cached_input_per_million=Decimal("0.30"),
        )
        # 1000 input + 500 output + 2000 cached
        cost = pricing.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=2000,
        )
        expected_input = Decimal("1000") / Decimal("1000000") * Decimal("3.00")
        expected_output = Decimal("500") / Decimal("1000000") * Decimal("15.00")
        expected_cached = Decimal("2000") / Decimal("1000000") * Decimal("0.30")
        assert cost == expected_input + expected_output + expected_cached

    def test_calculate_cost_zero_tokens(self) -> None:
        """Test cost calculation with zero tokens."""
        pricing = ModelPricing(
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        cost = pricing.calculate_cost(input_tokens=0, output_tokens=0)
        assert cost == Decimal("0")

    def test_calculate_cost_cache_ignored_if_no_pricing(self) -> None:
        """Test that cached tokens are ignored if no cache pricing."""
        pricing = ModelPricing(
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
            cached_input_per_million=None,
        )
        cost_with_cache = pricing.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=2000,
        )
        cost_without_cache = pricing.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=0,
        )
        assert cost_with_cache == cost_without_cache


# =============================================================================
# Test MODEL_PRICING Database (10.1.1)
# =============================================================================


class TestModelPricingDatabase:
    """Tests for MODEL_PRICING database."""

    def test_database_not_empty(self) -> None:
        """Test that pricing database is not empty."""
        assert len(MODEL_PRICING) > 0

    def test_common_models_have_pricing(self) -> None:
        """Test that common models have pricing entries."""
        expected_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250514",
            "claude-opus-4-20250514",
        ]
        for model in expected_models:
            assert model in MODEL_PRICING, f"Missing pricing for {model}"

    def test_all_pricing_has_required_fields(self) -> None:
        """Test all pricing entries have required fields."""
        for model, pricing in MODEL_PRICING.items():
            assert pricing.input_per_million > 0, f"{model} has zero input price"
            assert pricing.output_per_million > 0, f"{model} has zero output price"

    def test_output_more_expensive_than_input(self) -> None:
        """Test that output is typically more expensive than input."""
        for model, pricing in MODEL_PRICING.items():
            assert pricing.output_per_million >= pricing.input_per_million, (
                f"{model} has output cheaper than input"
            )

    def test_cached_cheaper_than_regular(self) -> None:
        """Test that cached input is cheaper than regular input."""
        for model, pricing in MODEL_PRICING.items():
            if pricing.cached_input_per_million is not None:
                assert pricing.cached_input_per_million < pricing.input_per_million, (
                    f"{model} has cached more expensive than regular"
                )


# =============================================================================
# Test get_pricing Function (10.1.1)
# =============================================================================


class TestGetPricing:
    """Tests for get_pricing function."""

    def test_get_known_model(self) -> None:
        """Test getting pricing for a known model."""
        pricing = get_pricing("claude-sonnet-4-20250514")
        assert pricing.input_per_million == Decimal("3.00")
        assert pricing.output_per_million == Decimal("15.00")

    def test_get_model_with_alias(self) -> None:
        """Test getting pricing using model alias."""
        pricing = get_pricing("sonnet")
        assert pricing.input_per_million == Decimal("3.00")

    def test_get_unknown_model_returns_default(self) -> None:
        """Test that unknown models get default pricing."""
        pricing = get_pricing("unknown-model-xyz")
        assert pricing.input_per_million > 0
        assert pricing.output_per_million > 0

    def test_case_insensitive(self) -> None:
        """Test that model lookup is case-insensitive."""
        pricing1 = get_pricing("gpt-4o")
        pricing2 = get_pricing("GPT-4O")
        assert pricing1 == pricing2

    def test_whitespace_handling(self) -> None:
        """Test that whitespace is handled."""
        pricing = get_pricing("  gpt-4o  ")
        assert pricing.input_per_million == Decimal("2.50")


class TestListSupportedModels:
    """Tests for list_supported_models function."""

    def test_returns_list(self) -> None:
        """Test that it returns a list."""
        models = list_supported_models()
        assert isinstance(models, list)

    def test_contains_common_models(self) -> None:
        """Test that common models are in the list."""
        models = list_supported_models()
        assert "gpt-4o" in models
        assert "claude-sonnet-4-20250514" in models

    def test_list_is_sorted(self) -> None:
        """Test that the list is sorted."""
        models = list_supported_models()
        assert models == sorted(models)


class TestGetModelAliases:
    """Tests for get_model_aliases function."""

    def test_returns_dict(self) -> None:
        """Test that it returns a dictionary."""
        aliases = get_model_aliases()
        assert isinstance(aliases, dict)

    def test_contains_common_aliases(self) -> None:
        """Test that common aliases are present."""
        aliases = get_model_aliases()
        assert "sonnet" in aliases
        assert "haiku" in aliases
        assert "opus" in aliases


# =============================================================================
# Test TokenMetrics (10.1.2)
# =============================================================================


class TestTokenMetrics:
    """Tests for TokenMetrics dataclass."""

    def test_create_default(self) -> None:
        """Test creating TokenMetrics with defaults."""
        metrics = TokenMetrics()
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.cached_tokens == 0
        assert metrics.model == ""

    def test_create_with_values(self) -> None:
        """Test creating TokenMetrics with values."""
        metrics = TokenMetrics(
            input_tokens=1500,
            output_tokens=500,
            cached_tokens=1000,
            model="gpt-4o",
        )
        assert metrics.input_tokens == 1500
        assert metrics.output_tokens == 500
        assert metrics.cached_tokens == 1000
        assert metrics.model == "gpt-4o"

    def test_total_property(self) -> None:
        """Test total tokens property."""
        metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=200,
        )
        assert metrics.total == 1700

    def test_total_input_property(self) -> None:
        """Test total_input property."""
        metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=200,
        )
        assert metrics.total_input == 1200


class TestTokenMetricsCostCalculation:
    """Tests for TokenMetrics cost calculation (10.1.2)."""

    def test_calculate_cost_with_model(self) -> None:
        """Test cost calculation with specified model."""
        metrics = TokenMetrics(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="claude-sonnet-4-20250514",
        )
        cost = metrics.calculate_cost()
        # 3.00 input + 15.00 output = 18.00
        assert cost == Decimal("18.00")

    def test_calculate_cost_override_model(self) -> None:
        """Test cost calculation with model override."""
        metrics = TokenMetrics(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="gpt-4o-mini",
        )
        # Use different model for calculation
        cost = metrics.calculate_cost(model="claude-sonnet-4-20250514")
        assert cost == Decimal("18.00")

    def test_calculate_cost_float(self) -> None:
        """Test calculate_cost_float returns float."""
        metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4o-mini",
        )
        cost = metrics.calculate_cost_float()
        assert isinstance(cost, float)

    def test_calculate_cost_with_cached(self) -> None:
        """Test cost calculation includes cached tokens."""
        metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=2000,
            model="claude-sonnet-4-20250514",
        )
        cost = metrics.calculate_cost()
        # Should include cached token cost
        assert cost > Decimal("0")

    def test_calculate_cost_default_model(self) -> None:
        """Test cost calculation with no model uses default."""
        metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
        )
        # Should not raise
        cost = metrics.calculate_cost()
        assert cost > Decimal("0")


class TestTokenMetricsMerge:
    """Tests for TokenMetrics merge functionality."""

    def test_merge_basic(self) -> None:
        """Test basic merge of two TokenMetrics."""
        m1 = TokenMetrics(input_tokens=1000, output_tokens=500)
        m2 = TokenMetrics(input_tokens=2000, output_tokens=800)
        merged = m1.merge(m2)
        assert merged.input_tokens == 3000
        assert merged.output_tokens == 1300

    def test_merge_with_cached(self) -> None:
        """Test merge includes cached tokens."""
        m1 = TokenMetrics(cached_tokens=500)
        m2 = TokenMetrics(cached_tokens=300)
        merged = m1.merge(m2)
        assert merged.cached_tokens == 800

    def test_merge_preserves_first_model(self) -> None:
        """Test merge keeps first non-empty model."""
        m1 = TokenMetrics(model="gpt-4o")
        m2 = TokenMetrics(model="claude-sonnet")
        merged = m1.merge(m2)
        assert merged.model == "gpt-4o"

    def test_merge_uses_second_if_first_empty(self) -> None:
        """Test merge uses second model if first is empty."""
        m1 = TokenMetrics(model="")
        m2 = TokenMetrics(model="claude-sonnet")
        merged = m1.merge(m2)
        assert merged.model == "claude-sonnet"


class TestTokenMetricsSerialization:
    """Tests for TokenMetrics serialization."""

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=200,
            model="gpt-4o",
        )
        data = metrics.to_dict()
        assert data["input_tokens"] == 1000
        assert data["output_tokens"] == 500
        assert data["cached_tokens"] == 200
        assert data["total"] == 1700
        assert data["model"] == "gpt-4o"

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cached_tokens": 200,
            "model": "gpt-4o",
        }
        metrics = TokenMetrics.from_dict(data)
        assert metrics.input_tokens == 1000
        assert metrics.output_tokens == 500
        assert metrics.cached_tokens == 200
        assert metrics.model == "gpt-4o"

    def test_from_dict_handles_missing(self) -> None:
        """Test from_dict handles missing fields."""
        data: dict = {}
        metrics = TokenMetrics.from_dict(data)
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.cached_tokens == 0
        assert metrics.model == ""


class TestTokenMetricsFormatSummary:
    """Tests for TokenMetrics format_summary."""

    def test_format_basic(self) -> None:
        """Test basic formatting."""
        metrics = TokenMetrics(
            input_tokens=1500,
            output_tokens=500,
        )
        summary = metrics.format_summary(include_cost=False)
        assert "1,500 in" in summary
        assert "500 out" in summary

    def test_format_with_cached(self) -> None:
        """Test formatting includes cached when present."""
        metrics = TokenMetrics(
            input_tokens=1500,
            output_tokens=500,
            cached_tokens=1000,
        )
        summary = metrics.format_summary(include_cost=False)
        assert "1,000 cached" in summary

    def test_format_with_cost(self) -> None:
        """Test formatting includes cost."""
        metrics = TokenMetrics(
            input_tokens=1500,
            output_tokens=500,
            model="gpt-4o-mini",
        )
        summary = metrics.format_summary(include_cost=True)
        assert "$" in summary


# =============================================================================
# Integration Tests
# =============================================================================


class TestPricingIntegration:
    """Integration tests for pricing system."""

    def test_realistic_request_cost(self) -> None:
        """Test cost calculation for realistic request."""
        # Typical Claude Sonnet request: 2000 input, 800 output
        metrics = TokenMetrics(
            input_tokens=2000,
            output_tokens=800,
            model="claude-sonnet-4-20250514",
        )
        cost = metrics.calculate_cost_float()

        # Expected: (2000/1M * 3) + (800/1M * 15) = 0.006 + 0.012 = 0.018
        assert 0.01 < cost < 0.03

    def test_cached_request_cost_reduction(self) -> None:
        """Test that cached tokens reduce cost."""
        # Without cache
        metrics_no_cache = TokenMetrics(
            input_tokens=10000,
            output_tokens=1000,
            model="claude-sonnet-4-20250514",
        )
        cost_no_cache = metrics_no_cache.calculate_cost()

        # With cache (8000 of input is cached)
        metrics_cached = TokenMetrics(
            input_tokens=2000,
            output_tokens=1000,
            cached_tokens=8000,
            model="claude-sonnet-4-20250514",
        )
        cost_cached = metrics_cached.calculate_cost()

        # Cached version should be cheaper
        assert cost_cached < cost_no_cache

    def test_model_comparison(self) -> None:
        """Test comparing costs across models."""
        metrics = TokenMetrics(
            input_tokens=10000,
            output_tokens=5000,
        )

        cost_haiku = metrics.calculate_cost(model="claude-haiku-4-20250514")
        cost_sonnet = metrics.calculate_cost(model="claude-sonnet-4-20250514")
        cost_opus = metrics.calculate_cost(model="claude-opus-4-20250514")

        # Haiku < Sonnet < Opus
        assert cost_haiku < cost_sonnet < cost_opus


# =============================================================================
# Test 10.1.4: Cost Matches Expected
# =============================================================================


class TestCostMatchesExpected:
    """Tests for Phase 10.1.4: Verify cost calculation accuracy.

    These tests ensure that cost calculations match expected values
    based on known model pricing.
    """

    def test_cost_calculation_gpt5_mini(self) -> None:
        """Test cost calculation matches expected for gpt-5-mini.

        Phase 10.1.4 acceptance test:
        - 120k input tokens @ $0.25/M = $0.03
        - 50k output tokens @ $2.00/M = $0.10
        - Total expected: $0.13
        """
        metrics = TokenMetrics(input_tokens=120000, output_tokens=50000)
        cost = metrics.calculate_cost("gpt-5-mini")

        # Expected: (120k * 0.25 + 50k * 2.00) / 1M = $0.03 + $0.10 = $0.13
        assert Decimal("0.12") <= cost <= Decimal("0.14")

    def test_cost_calculation_exact_gpt5_mini(self) -> None:
        """Test exact cost calculation for gpt-5-mini."""
        metrics = TokenMetrics(input_tokens=120000, output_tokens=50000)
        cost = metrics.calculate_cost("gpt-5-mini")

        # Exact calculation:
        # Input: 120000 / 1_000_000 * 0.25 = 0.030
        # Output: 50000 / 1_000_000 * 2.00 = 0.100
        # Total: 0.130
        assert cost == Decimal("0.130")

    def test_cost_calculation_gpt4o(self) -> None:
        """Test cost calculation for gpt-4o."""
        metrics = TokenMetrics(input_tokens=10000, output_tokens=5000)
        cost = metrics.calculate_cost("gpt-4o")

        # gpt-4o pricing: $2.50/M input, $10.00/M output
        # Input: 10000 / 1_000_000 * 2.50 = 0.025
        # Output: 5000 / 1_000_000 * 10.00 = 0.050
        # Total: 0.075
        assert cost == Decimal("0.075")

    def test_cost_calculation_claude_sonnet(self) -> None:
        """Test cost calculation for claude-sonnet-4."""
        metrics = TokenMetrics(input_tokens=50000, output_tokens=20000)
        cost = metrics.calculate_cost("claude-sonnet-4-20250514")

        # claude-sonnet-4 pricing: $3.00/M input, $15.00/M output
        # Input: 50000 / 1_000_000 * 3.00 = 0.15
        # Output: 20000 / 1_000_000 * 15.00 = 0.30
        # Total: 0.45
        assert cost == Decimal("0.45")

    def test_cost_calculation_with_cached_tokens(self) -> None:
        """Test cost calculation with cached tokens."""
        metrics = TokenMetrics(
            input_tokens=50000,
            output_tokens=10000,
            cached_tokens=100000,  # 100k cached
        )
        cost = metrics.calculate_cost("claude-sonnet-4-20250514")

        # claude-sonnet-4 pricing: $3.00/M input, $15.00/M output, $0.30/M cached
        # Input: 50000 / 1_000_000 * 3.00 = 0.15
        # Output: 10000 / 1_000_000 * 15.00 = 0.15
        # Cached: 100000 / 1_000_000 * 0.30 = 0.03
        # Total: 0.33
        assert cost == Decimal("0.33")

    def test_cost_within_10_percent_of_actual(self) -> None:
        """Test that calculated cost is within 10% of expected (Phase 10 success metric)."""
        # Simulate a realistic conversation
        metrics = TokenMetrics(
            input_tokens=8000,  # System prompt + context
            output_tokens=2000,  # Model response
        )
        cost = metrics.calculate_cost("claude-sonnet-4-20250514")

        # Expected cost:
        # Input: 8000 / 1_000_000 * 3.00 = 0.024
        # Output: 2000 / 1_000_000 * 15.00 = 0.030
        # Total: 0.054
        expected = Decimal("0.054")

        # Verify within 10%
        lower_bound = expected * Decimal("0.90")
        upper_bound = expected * Decimal("1.10")
        assert lower_bound <= cost <= upper_bound


# =============================================================================
# Test 10.1.3: Summary Output with Cost Display
# =============================================================================


class TestSummaryOutputWithCost:
    """Tests for Phase 10.1.3: Summary output shows accurate model-specific cost.

    Verifies that format_run_summary_with_nodes displays token breakdown
    and cost when provided with TokenMetrics.
    """

    def test_summary_includes_cost_when_token_metrics_provided(self) -> None:
        """Test that summary includes cost when TokenMetrics is provided."""
        token_metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4o-mini",
        )

        output = format_run_summary_with_nodes(
            total_duration_ms=5000,
            total_tokens=1500,
            node_metrics=None,
            token_metrics=token_metrics,
            model="gpt-4o-mini",
        )

        assert "Cost:" in output
        assert "gpt-4o-mini" in output
        assert "$" in output

    def test_summary_shows_token_breakdown(self) -> None:
        """Test that summary shows input/output token breakdown."""
        token_metrics = TokenMetrics(
            input_tokens=8000,
            output_tokens=2000,
            model="claude-sonnet-4-20250514",
        )

        output = format_run_summary_with_nodes(
            total_duration_ms=10000,
            total_tokens=10000,
            node_metrics=None,
            token_metrics=token_metrics,
        )

        assert "input: 8,000" in output
        assert "output: 2,000" in output

    def test_summary_shows_cached_tokens_when_present(self) -> None:
        """Test that summary shows cached token count when present."""
        token_metrics = TokenMetrics(
            input_tokens=5000,
            output_tokens=1000,
            cached_tokens=3000,
            model="claude-sonnet-4-20250514",
        )

        output = format_run_summary_with_nodes(
            total_duration_ms=5000,
            total_tokens=9000,
            node_metrics=None,
            token_metrics=token_metrics,
        )

        assert "cached: 3,000" in output

    def test_summary_omits_cached_when_zero(self) -> None:
        """Test that summary omits cached count when zero."""
        token_metrics = TokenMetrics(
            input_tokens=5000,
            output_tokens=1000,
            cached_tokens=0,
            model="gpt-4o",
        )

        output = format_run_summary_with_nodes(
            total_duration_ms=5000,
            total_tokens=6000,
            node_metrics=None,
            token_metrics=token_metrics,
        )

        assert "cached" not in output

    def test_summary_uses_token_metrics_model_as_fallback(self) -> None:
        """Test that summary uses TokenMetrics.model when model param is None."""
        token_metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-5-mini",
        )

        output = format_run_summary_with_nodes(
            total_duration_ms=5000,
            total_tokens=1500,
            node_metrics=None,
            token_metrics=token_metrics,
            model=None,  # No explicit model
        )

        assert "gpt-5-mini" in output

    def test_summary_shows_unknown_when_no_model(self) -> None:
        """Test that summary shows 'unknown' when no model provided."""
        token_metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
            model="",  # No model set
        )

        output = format_run_summary_with_nodes(
            total_duration_ms=5000,
            total_tokens=1500,
            node_metrics=None,
            token_metrics=token_metrics,
            model=None,
        )

        assert "unknown" in output

    def test_summary_backward_compatible_without_token_metrics(self) -> None:
        """Test that summary works without token_metrics (backward compat)."""
        output = format_run_summary_with_nodes(
            total_duration_ms=5000,
            total_tokens=1500,
            node_metrics=None,
        )

        # Should still work without cost info
        assert "Run Summary" in output
        assert "5,000ms" in output
        assert "1,500" in output
        # Cost should NOT appear without token_metrics
        assert "Cost:" not in output

    def test_summary_cost_format_four_decimals(self) -> None:
        """Test that cost is formatted with 4 decimal places."""
        token_metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4o-mini",
        )

        output = format_run_summary_with_nodes(
            total_duration_ms=5000,
            total_tokens=1500,
            node_metrics=None,
            token_metrics=token_metrics,
        )

        # Look for cost with 4 decimals format: $X.XXXX
        import re

        cost_match = re.search(r"\$\d+\.\d{4}", output)
        assert cost_match is not None, f"Expected 4-decimal cost format in: {output}"

    def test_summary_with_realistic_workload(self) -> None:
        """Test summary with realistic executor workload."""
        # Simulate a multi-task executor run
        token_metrics = TokenMetrics(
            input_tokens=120000,  # Large context from multiple tasks
            output_tokens=50000,  # Responses
            cached_tokens=30000,  # Some cached
            model="gpt-5-mini",
        )

        output = format_run_summary_with_nodes(
            total_duration_ms=45000,  # 45 seconds
            total_tokens=200000,
            node_metrics={
                "execute_task": {"tokens_in": 100000, "tokens_out": 40000, "duration_ms": 30000},
                "context": {"tokens_in": 20000, "tokens_out": 10000, "duration_ms": 15000},
            },
            token_metrics=token_metrics,
        )

        # Check all parts are present
        assert "Run Summary" in output
        assert "45,000ms" in output
        assert "input: 120,000" in output
        assert "output: 50,000" in output
        assert "cached: 30,000" in output
        assert "Cost:" in output
        assert "gpt-5-mini" in output
        assert "Per-Node Breakdown" in output
