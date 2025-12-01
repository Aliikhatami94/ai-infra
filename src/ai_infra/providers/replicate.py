"""Replicate provider configuration.

Replicate supports:
- ImageGen: Various community models including FLUX, SDXL

Known for hosting open-source models with pay-per-use pricing.
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

REPLICATE = ProviderConfig(
    name="replicate",
    display_name="Replicate",
    env_var="REPLICATE_API_TOKEN",
    capabilities={
        ProviderCapability.IMAGEGEN: CapabilityConfig(
            models=[
                # FLUX models (Black Forest Labs)
                "black-forest-labs/flux-schnell",
                "black-forest-labs/flux-dev",
                "black-forest-labs/flux-pro",
                "black-forest-labs/flux-1.1-pro",
                # Stability AI models
                "stability-ai/sdxl",
                "stability-ai/stable-diffusion-3",
                # Other popular models
                "bytedance/sdxl-lightning-4step",
                "lucataco/sdxl-lcm",
            ],
            default_model="black-forest-labs/flux-schnell",
            features=["community_models", "webhooks", "streaming"],
            extra={
                "model_versions": {
                    "black-forest-labs/flux-schnell": "latest",
                    "black-forest-labs/flux-dev": "latest",
                },
            },
        ),
    },
)

# Register with the central registry
ProviderRegistry.register(REPLICATE)
