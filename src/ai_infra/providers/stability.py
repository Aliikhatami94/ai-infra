"""Stability AI provider configuration.

Stability AI supports:
- ImageGen: Stable Diffusion XL, Stable Image Core/Ultra

Known for open-source image generation models.
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

STABILITY = ProviderConfig(
    name="stability",
    display_name="Stability AI",
    env_var="STABILITY_API_KEY",
    capabilities={
        ProviderCapability.IMAGEGEN: CapabilityConfig(
            models=[
                "stable-diffusion-xl-1024-v1-0",
                "stable-diffusion-v1-6",
                "stable-image-ultra",
                "stable-image-core",
                "sd3-large",
                "sd3-large-turbo",
                "sd3-medium",
            ],
            default_model="stable-diffusion-xl-1024-v1-0",
            features=["img2img", "inpainting", "upscaling", "outpainting"],
            extra={
                "sizes": {
                    "stable-diffusion-xl-1024-v1-0": [
                        "1024x1024",
                        "1152x896",
                        "896x1152",
                    ],
                },
                "style_presets": [
                    "3d-model",
                    "analog-film",
                    "anime",
                    "cinematic",
                    "comic-book",
                    "digital-art",
                    "enhance",
                    "fantasy-art",
                    "isometric",
                    "line-art",
                    "low-poly",
                    "neon-punk",
                    "origami",
                    "photographic",
                    "pixel-art",
                    "tile-texture",
                ],
            },
        ),
    },
)

# Register with the central registry
ProviderRegistry.register(STABILITY)
