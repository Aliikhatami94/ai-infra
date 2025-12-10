"""xAI (Grok) provider configuration.

xAI supports:
- Chat: Grok-3, Grok-2, Grok-beta models

xAI uses an OpenAI-compatible API.
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

XAI = ProviderConfig(
    name="xai",
    display_name="xAI (Grok)",
    env_var="XAI_API_KEY",
    base_url="https://api.x.ai/v1",
    capabilities={
        ProviderCapability.CHAT: CapabilityConfig(
            models=[
                "grok-3",
                "grok-3-mini",
                "grok-2",
                "grok-2-mini",
                "grok-beta",
            ],
            default_model="grok-code-fast-1",
            features=["streaming", "function_calling"],
            extra={
                "openai_compatible": True,
            },
        ),
    },
)

# Register with the central registry
ProviderRegistry.register(XAI)
