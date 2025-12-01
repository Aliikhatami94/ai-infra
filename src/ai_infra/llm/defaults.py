"""Default models and providers for LLM.

This module provides default model settings. These are now sourced from
the centralized provider registry at `ai_infra.providers`.
"""

from ai_infra.llm.providers.providers import Providers
from ai_infra.providers import ProviderCapability, ProviderRegistry


def get_default_model(provider: str) -> str:
    """Get the default model for a provider from the registry.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")

    Returns:
        Default model name for CHAT capability.
    """
    model = ProviderRegistry.get_default_model(provider, ProviderCapability.CHAT)
    if model:
        return model
    # Fallback to hardcoded defaults for backwards compat
    return DEFAULT_MODELS.get(provider, "gpt-4o-mini")


# Default models per provider - used when model_name is None
# These are sensible, cost-effective defaults that work well for most use cases
# NOTE: These are now sourced from the registry but kept here for backwards compat
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "google_genai": "gemini-2.0-flash",
    "xai": "grok-3-mini",
}

# Legacy defaults (for backward compat)
MODEL = DEFAULT_MODELS["openai"]
PROVIDER = str(Providers.openai)
