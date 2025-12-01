"""Tests for STT (Speech-to-Text) functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_infra.llm.multimodal.discovery import (
    get_default_stt_provider,
    get_stt_provider_info,
    is_stt_configured,
    list_stt_models,
    list_stt_providers,
)
from ai_infra.llm.multimodal.models import TranscriptionResult, TranscriptionSegment


class TestSTTDiscovery:
    """Tests for STT discovery functions."""

    def test_list_providers(self):
        """Test listing STT providers."""
        providers = list_stt_providers()

        assert isinstance(providers, list)
        assert "openai" in providers
        assert "deepgram" in providers
        # Google may be listed as google_genai in registry
        assert "google" in providers or "google_genai" in providers

    def test_list_openai_models(self):
        """Test listing OpenAI STT models."""
        models = list_stt_models("openai")

        assert isinstance(models, list)
        assert "whisper-1" in models

    def test_list_deepgram_models(self):
        """Test listing Deepgram STT models."""
        models = list_stt_models("deepgram")

        assert isinstance(models, list)
        assert "nova-2" in models

    def test_list_models_unknown_provider(self):
        """Test listing models for unknown provider."""
        with pytest.raises(ValueError, match="Unknown STT provider"):
            list_stt_models("unknown_provider")

    def test_get_provider_info(self):
        """Test getting provider info."""
        info = get_stt_provider_info("openai")

        # Registry uses display_name (e.g., "OpenAI") not legacy names like "OpenAI Whisper"
        assert "OpenAI" in info["name"]
        assert "whisper-1" in info["models"]
        assert info["env_var"] == "OPENAI_API_KEY"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_is_stt_configured_with_key(self):
        """Test STT configuration check with key."""
        assert is_stt_configured("openai") is True

    @patch.dict("os.environ", {}, clear=True)
    def test_is_stt_configured_without_key(self):
        """Test STT configuration check without key."""
        import os

        original = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        try:
            assert is_stt_configured("openai") is False
        finally:
            if original:
                os.environ["OPENAI_API_KEY"] = original


class TestTranscriptionResult:
    """Tests for TranscriptionResult model."""

    def test_basic_result(self):
        """Test basic transcription result."""
        result = TranscriptionResult(
            text="Hello, world!",
            language="en",
        )

        assert result.text == "Hello, world!"
        assert result.language == "en"
        assert result.segments == []

    def test_result_with_segments(self):
        """Test transcription result with segments."""
        segments = [
            TranscriptionSegment(text="Hello,", start=0.0, end=0.5),
            TranscriptionSegment(text="world!", start=0.5, end=1.0),
        ]
        result = TranscriptionResult(
            text="Hello, world!",
            language="en",
            segments=segments,
        )

        assert len(result.segments) == 2
        assert result.segments[0].start == 0.0
        assert result.segments[1].end == 1.0


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment model."""

    def test_segment_creation(self):
        """Test creating a segment."""
        segment = TranscriptionSegment(
            text="Hello",
            start=0.0,
            end=0.5,
        )

        assert segment.text == "Hello"
        assert segment.start == 0.0
        assert segment.end == 0.5
