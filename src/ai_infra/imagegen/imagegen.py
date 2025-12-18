"""Main ImageGen class with provider-agnostic API."""

from __future__ import annotations

import base64
import io
from typing import Any, BinaryIO, Literal

from ai_infra.imagegen.models import (
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
    GeneratedImage,
    ImageGenProvider,
)
from ai_infra.providers import ProviderCapability, ProviderRegistry

# Provider aliases for backwards compatibility
_PROVIDER_ALIASES = {"google": "google_genai"}
_REVERSE_ALIASES = {"google_genai": "google"}


class ImageGen:
    """Provider-agnostic image generation.

    Supports OpenAI (DALL-E), Google (Imagen), Stability AI, and Replicate.
    Auto-detects provider from environment variables if not specified.

    Example:
        ```python
        # Zero-config: auto-detects from env vars
        gen = ImageGen()
        images = gen.generate("A sunset over mountains")

        # Explicit provider and model
        gen = ImageGen(provider="google", model="imagen-4.0-generate-001")
        images = gen.generate("A futuristic city", n=2)

        # Save generated image
        images[0].save("output.png")
        ```

    Environment Variables:
        - OPENAI_API_KEY: For OpenAI DALL-E
        - GOOGLE_API_KEY or GEMINI_API_KEY: For Google Imagen
        - STABILITY_API_KEY: For Stability AI
        - REPLICATE_API_TOKEN: For Replicate
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ImageGen.

        Args:
            provider: Provider name ("openai", "google", "stability", "replicate").
                     Auto-detected from env vars if not specified.
            model: Model name. Uses provider default if not specified.
            api_key: API key. Uses env var if not specified.
            **kwargs: Additional provider-specific options.
        """
        self._provider, self._api_key = self._resolve_provider_and_key(provider, api_key)
        self._model = model or self._get_default_model(self._provider)
        self._kwargs = kwargs
        self._client: Any = None

    @property
    def provider(self) -> ImageGenProvider:
        """Get the current provider."""
        return self._provider

    @property
    def model(self) -> str | None:
        """Get the current model."""
        return self._model

    def _get_default_model(self, provider: ImageGenProvider) -> str | None:
        """Get default model for provider from registry."""
        # Map provider enum to registry name
        registry_name = provider.value
        if registry_name == "google":
            registry_name = "google_genai"

        config = ProviderRegistry.get(registry_name)
        if config:
            cap = config.get_capability(ProviderCapability.IMAGEGEN)
            if cap and cap.default_model:
                return cap.default_model

        # Fallback to legacy constant
        return DEFAULT_MODELS.get(provider)

    def _resolve_provider_and_key(
        self,
        provider: str | None,
        api_key: str | None,
    ) -> tuple[ImageGenProvider, str]:
        """Resolve provider and API key from args or environment."""

        if provider is not None:
            # Explicit provider
            p = ImageGenProvider(provider.lower())
            key = api_key or self._get_env_key(p)
            if not key:
                raise ValueError(f"No API key found for provider '{provider}'")
            return p, key

        # Auto-detect from registry
        # Provider priority order
        priority = ["openai", "google_genai", "stability", "replicate"]
        for name in priority:
            if ProviderRegistry.is_configured(name):
                # Map to ImageGenProvider enum
                enum_name = _REVERSE_ALIASES.get(name, name)
                key = ProviderRegistry.get_api_key(name)
                if key:
                    return ImageGenProvider(enum_name), key

        raise ValueError(
            "No API key found. Set one of: OPENAI_API_KEY, GOOGLE_API_KEY, "
            "STABILITY_API_KEY, or REPLICATE_API_TOKEN"
        )

    def _get_env_key(self, provider: ImageGenProvider) -> str | None:
        """Get the environment variable key for a provider."""
        # Map to registry name
        registry_name = provider.value
        if registry_name == "google":
            registry_name = "google_genai"

        return ProviderRegistry.get_api_key(registry_name)

    def generate(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["vivid", "natural"] | None = None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the image to generate.
            size: Image size (e.g., "1024x1024", "1792x1024").
            n: Number of images to generate.
            quality: Image quality ("standard" or "hd"). OpenAI only.
            style: Image style ("vivid" or "natural"). OpenAI DALL-E 3 only.
            **kwargs: Additional provider-specific options.

        Returns:
            List of GeneratedImage objects.

        Example:
            ```python
            images = gen.generate(
                "A cat wearing a hat",
                size="1024x1024",
                n=2,
                quality="hd",
            )
            for img in images:
                print(img.url)
            ```
        """
        if self._provider == ImageGenProvider.OPENAI:
            return self._generate_openai(
                prompt, size=size, n=n, quality=quality, style=style, **kwargs
            )
        elif self._provider == ImageGenProvider.GOOGLE:
            return self._generate_google(prompt, size=size, n=n, **kwargs)
        elif self._provider == ImageGenProvider.STABILITY:
            return self._generate_stability(prompt, size=size, n=n, **kwargs)
        elif self._provider == ImageGenProvider.REPLICATE:
            return self._generate_replicate(prompt, size=size, n=n, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    async def agenerate(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["vivid", "natural"] | None = None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async version of generate().

        Args:
            prompt: Text description of the image to generate.
            size: Image size (e.g., "1024x1024", "1792x1024").
            n: Number of images to generate.
            quality: Image quality ("standard" or "hd"). OpenAI only.
            style: Image style ("vivid" or "natural"). OpenAI DALL-E 3 only.
            **kwargs: Additional provider-specific options.

        Returns:
            List of GeneratedImage objects.
        """
        if self._provider == ImageGenProvider.OPENAI:
            return await self._agenerate_openai(
                prompt, size=size, n=n, quality=quality, style=style, **kwargs
            )
        elif self._provider == ImageGenProvider.GOOGLE:
            return await self._agenerate_google(prompt, size=size, n=n, **kwargs)
        elif self._provider == ImageGenProvider.STABILITY:
            return await self._agenerate_stability(prompt, size=size, n=n, **kwargs)
        elif self._provider == ImageGenProvider.REPLICATE:
            return await self._agenerate_replicate(prompt, size=size, n=n, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    # -------------------------------------------------------------------------
    # OpenAI Implementation
    # -------------------------------------------------------------------------

    def _get_openai_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _generate_openai(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        quality: str,
        style: str | None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using OpenAI DALL-E."""
        client = self._get_openai_client()

        params: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "size": size,
            "n": n,
            "response_format": "url",
        }

        # DALL-E 3 specific options
        if self._model == "dall-e-3":
            params["quality"] = quality
            if style:
                params["style"] = style

        params.update(kwargs)

        response = client.images.generate(**params)

        return [
            GeneratedImage(
                url=img.url,
                revised_prompt=getattr(img, "revised_prompt", None),
                model=self._model,
                provider=ImageGenProvider.OPENAI,
            )
            for img in response.data
        ]

    async def _agenerate_openai(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        quality: str,
        style: str | None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using OpenAI DALL-E."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key)

        params: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "size": size,
            "n": n,
            "response_format": "url",
        }

        if self._model == "dall-e-3":
            params["quality"] = quality
            if style:
                params["style"] = style

        params.update(kwargs)

        response = await client.images.generate(**params)

        return [
            GeneratedImage(
                url=img.url,
                revised_prompt=getattr(img, "revised_prompt", None),
                model=self._model,
                provider=ImageGenProvider.OPENAI,
            )
            for img in response.data
        ]

    # -------------------------------------------------------------------------
    # Google Implementation
    # -------------------------------------------------------------------------

    def _get_google_client(self) -> Any:
        """Get or create Google GenAI client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _is_gemini_model(self) -> bool:
        """Check if current model is a Gemini multimodal model."""
        from .models import GEMINI_IMAGE_MODELS

        model = self._model or ""
        return model in GEMINI_IMAGE_MODELS or model.startswith("gemini-")

    def _generate_google(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using Google (Gemini or Imagen)."""
        client = self._get_google_client()

        if self._is_gemini_model():
            # Use generate_content API for Gemini models
            return self._generate_google_gemini(client, prompt, n=n, **kwargs)
        else:
            # Use generate_images API for Imagen models
            return self._generate_google_imagen(client, prompt, size=size, n=n, **kwargs)

    def _generate_google_gemini(
        self,
        client: Any,
        prompt: str,
        *,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using Gemini multimodal API."""
        from google.genai import types

        results = []
        for _ in range(n):
            response = client.models.generate_content(
                model=self._model,
                contents=f"Generate an image: {prompt}",
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    results.append(
                        GeneratedImage(
                            data=part.inline_data.data,
                            model=self._model,
                            provider=ImageGenProvider.GOOGLE,
                        )
                    )
                    break

        return results

    def _generate_google_imagen(
        self,
        client: Any,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using Google Imagen API."""
        response = client.models.generate_images(
            model=self._model,
            prompt=prompt,
            config={
                "numberOfImages": n,
                "outputMimeType": "image/png",
                **kwargs,
            },
        )

        return [
            GeneratedImage(
                data=base64.b64decode(img.image.image_bytes) if hasattr(img, "image") else None,
                model=self._model,
                provider=ImageGenProvider.GOOGLE,
            )
            for img in response.generated_images
        ]

    async def _agenerate_google(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using Google (Gemini or Imagen)."""
        from google import genai

        client = genai.Client(api_key=self._api_key)

        if self._is_gemini_model():
            # Use generate_content API for Gemini models
            return await self._agenerate_google_gemini(client, prompt, n=n, **kwargs)
        else:
            # Use generate_images API for Imagen models
            return await self._agenerate_google_imagen(client, prompt, size=size, n=n, **kwargs)

    async def _agenerate_google_gemini(
        self,
        client: Any,
        prompt: str,
        *,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using Gemini multimodal API."""
        from google.genai import types

        results = []
        for _ in range(n):
            response = await client.aio.models.generate_content(
                model=self._model,
                contents=f"Generate an image: {prompt}",
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    results.append(
                        GeneratedImage(
                            data=part.inline_data.data,
                            model=self._model,
                            provider=ImageGenProvider.GOOGLE,
                        )
                    )
                    break

        return results

    async def _agenerate_google_imagen(
        self,
        client: Any,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using Google Imagen API."""
        response = await client.aio.models.generate_images(
            model=self._model,
            prompt=prompt,
            config={
                "numberOfImages": n,
                "outputMimeType": "image/png",
                **kwargs,
            },
        )

        return [
            GeneratedImage(
                data=base64.b64decode(img.image.image_bytes) if hasattr(img, "image") else None,
                model=self._model,
                provider=ImageGenProvider.GOOGLE,
            )
            for img in response.generated_images
        ]

    # -------------------------------------------------------------------------
    # Stability AI Implementation
    # -------------------------------------------------------------------------

    def _generate_stability(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using Stability AI."""
        import httpx

        width, height = map(int, size.split("x"))

        response = httpx.post(
            f"https://api.stability.ai/v1/generation/{self._model}/text-to-image",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "text_prompts": [{"text": prompt, "weight": 1.0}],
                "width": width,
                "height": height,
                "samples": n,
                **kwargs,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()

        return [
            GeneratedImage(
                data=base64.b64decode(artifact["base64"]),
                model=self._model,
                provider=ImageGenProvider.STABILITY,
                metadata={"seed": artifact.get("seed")},
            )
            for artifact in data.get("artifacts", [])
        ]

    async def _agenerate_stability(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using Stability AI."""
        import httpx

        width, height = map(int, size.split("x"))

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.stability.ai/v1/generation/{self._model}/text-to-image",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "text_prompts": [{"text": prompt, "weight": 1.0}],
                    "width": width,
                    "height": height,
                    "samples": n,
                    **kwargs,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        return [
            GeneratedImage(
                data=base64.b64decode(artifact["base64"]),
                model=self._model,
                provider=ImageGenProvider.STABILITY,
                metadata={"seed": artifact.get("seed")},
            )
            for artifact in data.get("artifacts", [])
        ]

    # -------------------------------------------------------------------------
    # Replicate Implementation
    # -------------------------------------------------------------------------

    def _generate_replicate(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using Replicate."""
        import replicate

        width, height = map(int, size.split("x"))

        # Run the model
        output = replicate.run(
            self._model,
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_outputs": n,
                **kwargs,
            },
        )

        # Output is typically a list of URLs
        if isinstance(output, list):
            return [
                GeneratedImage(
                    url=str(url),
                    model=self._model,
                    provider=ImageGenProvider.REPLICATE,
                )
                for url in output
            ]
        else:
            return [
                GeneratedImage(
                    url=str(output),
                    model=self._model,
                    provider=ImageGenProvider.REPLICATE,
                )
            ]

    async def _agenerate_replicate(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using Replicate."""
        import replicate

        width, height = map(int, size.split("x"))

        # Replicate's async API
        output = await replicate.async_run(
            self._model,
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_outputs": n,
                **kwargs,
            },
        )

        if isinstance(output, list):
            return [
                GeneratedImage(
                    url=str(url),
                    model=self._model,
                    provider=ImageGenProvider.REPLICATE,
                )
                for url in output
            ]
        else:
            return [
                GeneratedImage(
                    url=str(output),
                    model=self._model,
                    provider=ImageGenProvider.REPLICATE,
                )
            ]

    # -------------------------------------------------------------------------
    # Edit and Variations (OpenAI-specific)
    # -------------------------------------------------------------------------

    def edit(
        self,
        image: str | bytes,
        prompt: str,
        *,
        mask: str | bytes | None = None,
        size: str = "1024x1024",
        n: int = 1,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Edit an existing image based on a prompt.

        Note: Currently only supported by OpenAI (DALL-E 2).

        Args:
            image: Path to image file or image bytes.
            prompt: Description of the edit to make.
            mask: Optional mask image (transparent areas will be edited).
            size: Output image size.
            n: Number of variations to generate.
            **kwargs: Additional options.

        Returns:
            List of GeneratedImage objects.
        """
        if self._provider != ImageGenProvider.OPENAI:
            raise NotImplementedError(f"edit() not supported for {self._provider}")

        client = self._get_openai_client()

        # Handle image input
        image_file: BinaryIO
        if isinstance(image, str):
            image_file = open(image, "rb")
        else:
            image_file = io.BytesIO(image)
            image_file.name = "image.png"

        # Handle mask input
        mask_file: BinaryIO | None = None
        if mask is not None:
            if isinstance(mask, str):
                mask_file = open(mask, "rb")
            else:
                mask_file = io.BytesIO(mask)
                mask_file.name = "mask.png"

        try:
            params: dict[str, Any] = {
                "model": "dall-e-2",  # Only DALL-E 2 supports edits
                "image": image_file,
                "prompt": prompt,
                "size": size,
                "n": n,
            }
            if mask_file:
                params["mask"] = mask_file
            params.update(kwargs)

            response = client.images.edit(**params)

            return [
                GeneratedImage(
                    url=img.url,
                    model="dall-e-2",
                    provider=ImageGenProvider.OPENAI,
                )
                for img in response.data
            ]
        finally:
            if isinstance(image, str):
                image_file.close()
            if mask is not None and isinstance(mask, str):
                if mask_file is not None:
                    mask_file.close()

    def variations(
        self,
        image: str | bytes,
        *,
        size: str = "1024x1024",
        n: int = 1,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate variations of an existing image.

        Note: Currently only supported by OpenAI (DALL-E 2).

        Args:
            image: Path to image file or image bytes.
            size: Output image size.
            n: Number of variations to generate.
            **kwargs: Additional options.

        Returns:
            List of GeneratedImage objects.
        """
        if self._provider != ImageGenProvider.OPENAI:
            raise NotImplementedError(f"variations() not supported for {self._provider}")

        client = self._get_openai_client()

        # Handle image input
        image_file: BinaryIO
        if isinstance(image, str):
            image_file = open(image, "rb")
        else:
            image_file = io.BytesIO(image)
            image_file.name = "image.png"

        try:
            response = client.images.create_variation(
                model="dall-e-2",  # Only DALL-E 2 supports variations
                image=image_file,
                size=size,
                n=n,
                **kwargs,
            )

            return [
                GeneratedImage(
                    url=img.url,
                    model="dall-e-2",
                    provider=ImageGenProvider.OPENAI,
                )
                for img in response.data
            ]
        finally:
            if isinstance(image, str):
                image_file.close()

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def list_providers() -> list[str]:
        """List all available providers."""
        return [p.value for p in ImageGenProvider]

    @staticmethod
    def list_models(provider: str) -> list[str]:
        """List available models for a provider.

        Args:
            provider: Provider name.

        Returns:
            List of model names.
        """
        p = ImageGenProvider(provider.lower())
        return AVAILABLE_MODELS.get(p, [])
