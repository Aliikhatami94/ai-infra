"""Image generation module with provider-agnostic API.

Supports multiple providers:
- OpenAI (DALL-E 2, DALL-E 3)
- Google (Gemini 2.5 Flash Image, Imagen 3, Imagen 4 / Nano Banana Pro)
- Stability AI (Stable Diffusion)
- Replicate (SDXL, Flux, etc.)

Example:
    ```python
    from ai_infra import ImageGen

    # Zero-config: auto-detects provider from env vars
    gen = ImageGen()

    # Generate an image
    images = gen.generate("A sunset over mountains")

    # With specific provider (default is gemini-2.5-flash-image)
    gen = ImageGen(provider="google")
    images = gen.generate("A futuristic city", size="1024x1024", n=2)

    # List available models from API
    from ai_infra.imagegen import list_available_models
    models = list_available_models("google")
    ```
"""

from ai_infra.imagegen.discovery import (
    PROVIDER_ENV_VARS,
    SUPPORTED_PROVIDERS,
    clear_cache,
    get_api_key,
    is_provider_configured,
    list_all_available_models,
    list_available_models,
    list_configured_providers,
    list_models,
    list_providers,
)
from ai_infra.imagegen.imagegen import ImageGen
from ai_infra.imagegen.models import GeneratedImage, ImageGenProvider

__all__ = [
    "ImageGen",
    "GeneratedImage",
    "ImageGenProvider",
    # Discovery
    "list_providers",
    "list_configured_providers",
    "list_models",
    "list_available_models",
    "list_all_available_models",
    "is_provider_configured",
    "get_api_key",
    "clear_cache",
    "SUPPORTED_PROVIDERS",
    "PROVIDER_ENV_VARS",
]
