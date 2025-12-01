"""Voyage AI provider configuration.

Voyage AI supports:
- Embeddings: High-quality embeddings optimized for RAG

Recommended by Anthropic for use with Claude models.
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

VOYAGE = ProviderConfig(
    name="voyage",
    display_name="Voyage AI",
    env_var="VOYAGE_API_KEY",
    capabilities={
        ProviderCapability.EMBEDDINGS: CapabilityConfig(
            models=[
                "voyage-3",
                "voyage-3-lite",
                "voyage-code-3",
                "voyage-finance-2",
                "voyage-law-2",
                "voyage-large-2",
                "voyage-2",
            ],
            default_model="voyage-3",
            features=["domain_specific", "batch"],
            extra={
                "dimensions": {
                    "voyage-3": 1024,
                    "voyage-3-lite": 512,
                    "voyage-code-3": 1024,
                    "voyage-finance-2": 1024,
                    "voyage-law-2": 1024,
                    "voyage-large-2": 1536,
                    "voyage-2": 1024,
                },
                "max_tokens": {
                    "voyage-3": 32000,
                    "voyage-3-lite": 32000,
                    "voyage-code-3": 16000,
                },
            },
        ),
    },
)

# Register with the central registry
ProviderRegistry.register(VOYAGE)
