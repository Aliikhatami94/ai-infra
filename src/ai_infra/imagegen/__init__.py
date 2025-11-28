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
    ```
"""

from ai_infra.imagegen.imagegen import ImageGen
from ai_infra.imagegen.models import GeneratedImage, ImageGenProvider

__all__ = [
    "ImageGen",
    "GeneratedImage",
    "ImageGenProvider",
]
