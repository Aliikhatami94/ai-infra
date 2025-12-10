"""Anthropic provider configuration.

Anthropic supports:
- Chat: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3.5 Haiku

Note: Anthropic recommends Voyage AI for embeddings.
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

ANTHROPIC = ProviderConfig(
    name="anthropic",
    display_name="Anthropic",
    env_var="ANTHROPIC_API_KEY",
    capabilities={
        ProviderCapability.CHAT: CapabilityConfig(
            models=[
                "claude-sonnet-4-20250514",
                "claude-3-5-sonnet-latest",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-latest",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-latest",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307",
            ],
            default_model="claude-sonnet-4-5",
            features=["streaming", "function_calling", "vision", "extended_thinking"],
            extra={
                "max_tokens_default": 4096,
                "context_window": {
                    "claude-sonnet-4-20250514": 200000,
                    "claude-3-5-sonnet-latest": 200000,
                    "claude-3-opus-latest": 200000,
                },
            },
        ),
        # Anthropic doesn't have native embeddings - recommends Voyage AI
        # See voyage.py for embeddings
    },
)

# Register with the central registry
ProviderRegistry.register(ANTHROPIC)
