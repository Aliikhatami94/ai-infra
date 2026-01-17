"""Model-specific pricing for cost tracking (Phase 10.1.1).

This module provides accurate pricing information for LLM API calls,
enabling precise cost calculation based on actual API billing.

The pricing database includes:
- OpenAI models (GPT-4o, GPT-4o-mini, GPT-5-mini, etc.)
- Anthropic models (Claude Sonnet 4, Claude Haiku 4, Claude Opus 4)
- Support for cached input tokens (prompt caching)

Example:
    ```python
    from ai_infra.executor.pricing import get_pricing, ModelPricing

    # Get pricing for a model
    pricing = get_pricing("claude-sonnet-4-20250514")
    print(f"Input: ${pricing.input_per_million}/1M tokens")
    print(f"Output: ${pricing.output_per_million}/1M tokens")

    # Calculate cost for a request
    input_tokens = 1500
    output_tokens = 500
    input_cost = (input_tokens / 1_000_000) * float(pricing.input_per_million)
    output_cost = (output_tokens / 1_000_000) * float(pricing.output_per_million)
    total_cost = input_cost + output_cost
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPricing:
    """Pricing per 1M tokens for a model.

    All prices are in USD per 1 million tokens.

    Attributes:
        input_per_million: Cost per 1M input tokens.
        output_per_million: Cost per 1M output tokens.
        cached_input_per_million: Cost per 1M cached input tokens (prompt caching).
            If None, caching is not supported or priced same as regular input.
    """

    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal | None = None

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> Decimal:
        """Calculate total cost for a request.

        Args:
            input_tokens: Number of input tokens (non-cached).
            output_tokens: Number of output tokens.
            cached_tokens: Number of cached input tokens.

        Returns:
            Total cost in USD as Decimal.
        """
        input_cost = (Decimal(input_tokens) / Decimal(1_000_000)) * self.input_per_million

        output_cost = (Decimal(output_tokens) / Decimal(1_000_000)) * self.output_per_million

        cached_cost = Decimal(0)
        if cached_tokens > 0 and self.cached_input_per_million is not None:
            cached_cost = (
                Decimal(cached_tokens) / Decimal(1_000_000)
            ) * self.cached_input_per_million

        return input_cost + output_cost + cached_cost


# =============================================================================
# Model Pricing Database
# =============================================================================

# Current pricing as of January 2026
# Sources:
# - OpenAI: https://openai.com/pricing
# - Anthropic: https://anthropic.com/pricing

MODEL_PRICING: dict[str, ModelPricing] = {
    # =========================================================================
    # OpenAI Models
    # =========================================================================
    "gpt-5-mini": ModelPricing(
        input_per_million=Decimal("0.25"),
        output_per_million=Decimal("2.00"),
        cached_input_per_million=Decimal("0.025"),
    ),
    "gpt-4o": ModelPricing(
        input_per_million=Decimal("2.50"),
        output_per_million=Decimal("10.00"),
        cached_input_per_million=Decimal("1.25"),
    ),
    "gpt-4o-mini": ModelPricing(
        input_per_million=Decimal("0.15"),
        output_per_million=Decimal("0.60"),
        cached_input_per_million=Decimal("0.075"),
    ),
    "gpt-4.1": ModelPricing(
        input_per_million=Decimal("2.00"),
        output_per_million=Decimal("8.00"),
        cached_input_per_million=Decimal("0.50"),
    ),
    "gpt-4.1-mini": ModelPricing(
        input_per_million=Decimal("0.40"),
        output_per_million=Decimal("1.60"),
        cached_input_per_million=Decimal("0.10"),
    ),
    "gpt-4.1-nano": ModelPricing(
        input_per_million=Decimal("0.10"),
        output_per_million=Decimal("0.40"),
        cached_input_per_million=Decimal("0.025"),
    ),
    "o1": ModelPricing(
        input_per_million=Decimal("15.00"),
        output_per_million=Decimal("60.00"),
        cached_input_per_million=Decimal("7.50"),
    ),
    "o1-mini": ModelPricing(
        input_per_million=Decimal("3.00"),
        output_per_million=Decimal("12.00"),
        cached_input_per_million=Decimal("1.50"),
    ),
    "o3-mini": ModelPricing(
        input_per_million=Decimal("1.10"),
        output_per_million=Decimal("4.40"),
        cached_input_per_million=Decimal("0.55"),
    ),
    # =========================================================================
    # Anthropic Models
    # =========================================================================
    "claude-sonnet-4-20250514": ModelPricing(
        input_per_million=Decimal("3.00"),
        output_per_million=Decimal("15.00"),
        cached_input_per_million=Decimal("0.30"),
    ),
    "claude-haiku-4-20250514": ModelPricing(
        input_per_million=Decimal("0.25"),
        output_per_million=Decimal("1.25"),
        cached_input_per_million=Decimal("0.025"),
    ),
    "claude-opus-4-20250514": ModelPricing(
        input_per_million=Decimal("15.00"),
        output_per_million=Decimal("75.00"),
        cached_input_per_million=Decimal("1.50"),
    ),
    # Claude 3.5 models (still commonly used)
    "claude-3-5-sonnet-20241022": ModelPricing(
        input_per_million=Decimal("3.00"),
        output_per_million=Decimal("15.00"),
        cached_input_per_million=Decimal("0.30"),
    ),
    "claude-3-5-haiku-20241022": ModelPricing(
        input_per_million=Decimal("1.00"),
        output_per_million=Decimal("5.00"),
        cached_input_per_million=Decimal("0.10"),
    ),
    # =========================================================================
    # Google Models
    # =========================================================================
    "gemini-2.0-flash": ModelPricing(
        input_per_million=Decimal("0.10"),
        output_per_million=Decimal("0.40"),
        cached_input_per_million=Decimal("0.025"),
    ),
    "gemini-2.0-pro": ModelPricing(
        input_per_million=Decimal("1.25"),
        output_per_million=Decimal("5.00"),
        cached_input_per_million=Decimal("0.3125"),
    ),
    # =========================================================================
    # Model Aliases (common shorthand names)
    # =========================================================================
}

# Add common aliases
_MODEL_ALIASES: dict[str, str] = {
    # OpenAI aliases
    "gpt-4o-latest": "gpt-4o",
    "gpt4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4o",  # Deprecated, redirect to 4o
    # Anthropic aliases
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-4-sonnet": "claude-sonnet-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-haiku-4-20250514",
    "claude-4-haiku": "claude-haiku-4-20250514",
    "haiku": "claude-haiku-4-20250514",
    "claude-opus": "claude-opus-4-20250514",
    "claude-4-opus": "claude-opus-4-20250514",
    "opus": "claude-opus-4-20250514",
    # Claude 3.5 aliases
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    # Google aliases
    "gemini-flash": "gemini-2.0-flash",
    "gemini-pro": "gemini-2.0-pro",
}

# Default fallback pricing (conservative estimate)
_DEFAULT_PRICING = ModelPricing(
    input_per_million=Decimal("3.00"),
    output_per_million=Decimal("15.00"),
    cached_input_per_million=Decimal("0.30"),
)


def get_pricing(model: str) -> ModelPricing:
    """Get pricing for a model with fallback to default.

    Handles model aliases and provides conservative fallback pricing
    for unknown models.

    Args:
        model: Model name or alias (e.g., "claude-sonnet-4-20250514", "sonnet").

    Returns:
        ModelPricing for the model.

    Example:
        ```python
        pricing = get_pricing("claude-sonnet-4-20250514")
        cost = pricing.calculate_cost(input_tokens=1000, output_tokens=500)
        ```
    """
    # Normalize model name
    model_normalized = model.lower().strip()

    # Direct lookup
    if model_normalized in MODEL_PRICING:
        return MODEL_PRICING[model_normalized]

    # Check aliases
    if model_normalized in _MODEL_ALIASES:
        canonical = _MODEL_ALIASES[model_normalized]
        return MODEL_PRICING[canonical]

    # Try partial matching for versioned models
    for known_model in MODEL_PRICING:
        if model_normalized.startswith(known_model.split("-")[0]):
            # Match by provider prefix (e.g., "claude-" or "gpt-")
            if _models_same_family(model_normalized, known_model):
                logger.debug(f"Using pricing for {known_model} as fallback for {model}")
                return MODEL_PRICING[known_model]

    # Fallback to conservative estimate
    logger.warning(
        f"Unknown model '{model}', using default pricing "
        f"(${_DEFAULT_PRICING.input_per_million}/1M in, "
        f"${_DEFAULT_PRICING.output_per_million}/1M out)"
    )
    return _DEFAULT_PRICING


def _models_same_family(model1: str, model2: str) -> bool:
    """Check if two models are in the same family.

    Models are considered same family if they share the same base name
    (e.g., "claude-sonnet" family, "gpt-4o" family).

    Args:
        model1: First model name.
        model2: Second model name.

    Returns:
        True if models are in the same family.
    """

    # Extract base names (before version/date suffix)
    def get_base(m: str) -> str:
        parts = m.split("-")
        # Keep provider and model type, drop version/date
        if len(parts) >= 2:
            # Handle claude-sonnet-4-xxx, gpt-4o-xxx patterns
            if parts[0] in ("claude", "gpt", "gemini", "o1", "o3"):
                return "-".join(parts[:2])
        return parts[0]

    return get_base(model1) == get_base(model2)


def list_supported_models() -> list[str]:
    """List all models with known pricing.

    Returns:
        List of model names (excluding aliases).
    """
    return sorted(MODEL_PRICING.keys())


def get_model_aliases() -> dict[str, str]:
    """Get mapping of model aliases to canonical names.

    Returns:
        Dictionary mapping alias to canonical model name.
    """
    return dict(_MODEL_ALIASES)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "ModelPricing",
    "MODEL_PRICING",
    "get_pricing",
    "list_supported_models",
    "get_model_aliases",
]
