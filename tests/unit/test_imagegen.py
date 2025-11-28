"""Unit tests for the ImageGen module.

Tests cover:
- Provider detection and initialization
- Generate method (mocked)
- Edit and variations methods
- Async methods
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.imagegen import GeneratedImage, ImageGen, ImageGenProvider
from ai_infra.imagegen.models import AVAILABLE_MODELS, DEFAULT_MODELS

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_env_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only OpenAI key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)


@pytest.fixture
def mock_env_google(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only Google key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)


@pytest.fixture
def mock_env_stability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only Stability key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("STABILITY_API_KEY", "test-stability-key")
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)


@pytest.fixture
def mock_env_replicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only Replicate key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    monkeypatch.setenv("REPLICATE_API_TOKEN", "test-replicate-token")


@pytest.fixture
def mock_env_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with no API keys."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)


# =============================================================================
# Provider Detection Tests
# =============================================================================


class TestProviderDetection:
    """Tests for automatic provider detection."""

    def test_detects_openai(self, mock_env_openai: None) -> None:
        """Test that OpenAI is detected from env var."""
        gen = ImageGen()
        assert gen.provider == ImageGenProvider.OPENAI
        assert gen.model == DEFAULT_MODELS[ImageGenProvider.OPENAI]

    def test_detects_google(self, mock_env_google: None) -> None:
        """Test that Google is detected from env var."""
        gen = ImageGen()
        assert gen.provider == ImageGenProvider.GOOGLE
        assert gen.model == DEFAULT_MODELS[ImageGenProvider.GOOGLE]

    def test_detects_stability(self, mock_env_stability: None) -> None:
        """Test that Stability is detected from env var."""
        gen = ImageGen()
        assert gen.provider == ImageGenProvider.STABILITY
        assert gen.model == DEFAULT_MODELS[ImageGenProvider.STABILITY]

    def test_detects_replicate(self, mock_env_replicate: None) -> None:
        """Test that Replicate is detected from env var."""
        gen = ImageGen()
        assert gen.provider == ImageGenProvider.REPLICATE
        assert gen.model == DEFAULT_MODELS[ImageGenProvider.REPLICATE]

    def test_raises_without_api_key(self, mock_env_none: None) -> None:
        """Test that error is raised when no API key is found."""
        with pytest.raises(ValueError, match="No API key found"):
            ImageGen()

    def test_explicit_provider(self, mock_env_openai: None) -> None:
        """Test explicit provider selection."""
        gen = ImageGen(provider="openai")
        assert gen.provider == ImageGenProvider.OPENAI

    def test_explicit_provider_with_api_key(self, mock_env_none: None) -> None:
        """Test explicit provider with explicit API key."""
        gen = ImageGen(provider="openai", api_key="sk-explicit-key")
        assert gen.provider == ImageGenProvider.OPENAI

    def test_explicit_model(self, mock_env_openai: None) -> None:
        """Test explicit model selection."""
        gen = ImageGen(model="dall-e-2")
        assert gen.model == "dall-e-2"


# =============================================================================
# Generate Tests (Mocked)
# =============================================================================


class TestGenerateOpenAI:
    """Tests for OpenAI image generation."""

    def test_generate_returns_images(self, mock_env_openai: None) -> None:
        """Test that generate returns GeneratedImage objects."""
        gen = ImageGen()

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(url="https://example.com/image1.png", revised_prompt="A cat"),
            MagicMock(url="https://example.com/image2.png", revised_prompt="A cat"),
        ]

        with patch.object(gen, "_get_openai_client") as mock_client:
            mock_client.return_value.images.generate.return_value = mock_response

            images = gen.generate("A cat wearing a hat", n=2)

            assert len(images) == 2
            assert all(isinstance(img, GeneratedImage) for img in images)
            assert images[0].url == "https://example.com/image1.png"
            assert images[0].provider == ImageGenProvider.OPENAI

    def test_generate_passes_correct_params(self, mock_env_openai: None) -> None:
        """Test that generate passes correct parameters to OpenAI."""
        gen = ImageGen(model="dall-e-3")

        mock_response = MagicMock()
        mock_response.data = [MagicMock(url="https://example.com/image.png")]

        with patch.object(gen, "_get_openai_client") as mock_client:
            mock_client.return_value.images.generate.return_value = mock_response

            gen.generate(
                "A sunset",
                size="1792x1024",
                n=1,
                quality="hd",
                style="vivid",
            )

            mock_client.return_value.images.generate.assert_called_once()
            call_kwargs = mock_client.return_value.images.generate.call_args[1]
            assert call_kwargs["model"] == "dall-e-3"
            assert call_kwargs["prompt"] == "A sunset"
            assert call_kwargs["size"] == "1792x1024"
            assert call_kwargs["n"] == 1
            assert call_kwargs["quality"] == "hd"
            assert call_kwargs["style"] == "vivid"


class TestGenerateGoogle:
    """Tests for Google Imagen generation."""

    def test_generate_returns_images(self, mock_env_google: None) -> None:
        """Test that Google generate returns images."""
        gen = ImageGen()

        mock_img = MagicMock()
        mock_img.image.image_bytes = "aW1hZ2VkYXRh"  # base64 encoded "imagedata"

        mock_response = MagicMock()
        mock_response.generated_images = [mock_img]

        with patch.object(gen, "_get_google_client") as mock_client:
            mock_client.return_value.models.generate_images.return_value = mock_response

            images = gen.generate("A mountain landscape")

            assert len(images) == 1
            assert images[0].provider == ImageGenProvider.GOOGLE


# =============================================================================
# GeneratedImage Tests
# =============================================================================


class TestGeneratedImage:
    """Tests for GeneratedImage dataclass."""

    def test_save_requires_data(self, tmp_path: Any) -> None:
        """Test that save raises error without data."""
        img = GeneratedImage(url="https://example.com/image.png")

        with pytest.raises(ValueError, match="No image data available"):
            img.save(str(tmp_path / "output.png"))

    def test_save_writes_data(self, tmp_path: Any) -> None:
        """Test that save writes image data to file."""
        img = GeneratedImage(data=b"fake image data")
        output_path = tmp_path / "output.png"

        img.save(str(output_path))

        assert output_path.read_bytes() == b"fake image data"

    @pytest.mark.asyncio
    async def test_fetch_from_url(self) -> None:
        """Test fetching image data from URL."""
        img = GeneratedImage(url="https://example.com/image.png")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.content = b"downloaded image data"
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            data = await img.fetch()

            assert data == b"downloaded image data"
            assert img.data == b"downloaded image data"

    @pytest.mark.asyncio
    async def test_fetch_returns_cached_data(self) -> None:
        """Test that fetch returns cached data if already loaded."""
        img = GeneratedImage(data=b"cached data", url="https://example.com/image.png")

        data = await img.fetch()

        assert data == b"cached data"

    @pytest.mark.asyncio
    async def test_fetch_raises_without_url(self) -> None:
        """Test that fetch raises error without URL."""
        img = GeneratedImage()

        with pytest.raises(ValueError, match="No URL available"):
            await img.fetch()


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_list_providers(self) -> None:
        """Test listing available providers."""
        providers = ImageGen.list_providers()

        assert "openai" in providers
        assert "google" in providers
        assert "stability" in providers
        assert "replicate" in providers

    def test_list_models(self) -> None:
        """Test listing models for a provider."""
        openai_models = ImageGen.list_models("openai")

        assert "dall-e-2" in openai_models
        assert "dall-e-3" in openai_models

    def test_list_models_google(self) -> None:
        """Test listing Google models."""
        google_models = ImageGen.list_models("google")

        assert "imagen-3.0-generate-002" in google_models
        assert "imagen-4.0-generate-001" in google_models


# =============================================================================
# Edit and Variations Tests
# =============================================================================


class TestEditAndVariations:
    """Tests for edit and variations methods."""

    def test_edit_only_openai(self, mock_env_google: None) -> None:
        """Test that edit raises error for non-OpenAI providers."""
        gen = ImageGen()

        with pytest.raises(NotImplementedError, match="not supported"):
            gen.edit(b"image data", "Make it blue")

    def test_variations_only_openai(self, mock_env_google: None) -> None:
        """Test that variations raises error for non-OpenAI providers."""
        gen = ImageGen()

        with pytest.raises(NotImplementedError, match="not supported"):
            gen.variations(b"image data")


# =============================================================================
# Model Constants Tests
# =============================================================================


class TestModelConstants:
    """Tests for model constants."""

    def test_all_providers_have_defaults(self) -> None:
        """Test that all providers have default models."""
        for provider in ImageGenProvider:
            assert provider in DEFAULT_MODELS
            assert DEFAULT_MODELS[provider] is not None

    def test_all_providers_have_available_models(self) -> None:
        """Test that all providers have available models list."""
        for provider in ImageGenProvider:
            assert provider in AVAILABLE_MODELS
            assert len(AVAILABLE_MODELS[provider]) > 0

    def test_default_model_in_available(self) -> None:
        """Test that default model is in available models."""
        for provider in ImageGenProvider:
            default = DEFAULT_MODELS[provider]
            available = AVAILABLE_MODELS[provider]
            assert default in available
