"""Tests for TTS (Text-to-Speech) functionality."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_infra.llm.multimodal.discovery import (
    is_tts_configured,
    list_tts_models,
    list_tts_providers,
    list_tts_voices,
)
from ai_infra.llm.multimodal.models import AudioFormat, Voice


class TestTTSDiscovery:
    """Tests for TTS discovery functions."""

    def test_list_providers(self):
        """Test listing TTS providers."""
        providers = list_tts_providers()

        assert isinstance(providers, list)
        assert "openai" in providers
        assert "elevenlabs" in providers
        # Google may be listed as google_genai in registry
        assert "google" in providers or "google_genai" in providers

    def test_list_openai_voices(self):
        """Test listing OpenAI voices."""
        voices = list_tts_voices("openai")

        assert isinstance(voices, list)
        assert "alloy" in voices
        assert "nova" in voices
        assert "shimmer" in voices

    def test_list_openai_models(self):
        """Test listing OpenAI TTS models."""
        models = list_tts_models("openai")

        assert isinstance(models, list)
        assert "tts-1" in models
        assert "tts-1-hd" in models

    def test_list_voices_unknown_provider(self):
        """Test listing voices for unknown provider."""
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            list_tts_voices("unknown_provider")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_is_tts_configured_with_key(self):
        """Test TTS configuration check with key."""
        assert is_tts_configured("openai") is True

    @patch.dict("os.environ", {}, clear=True)
    def test_is_tts_configured_without_key(self):
        """Test TTS configuration check without key."""
        # Clear environment first
        import os

        original = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        try:
            assert is_tts_configured("openai") is False
        finally:
            if original:
                os.environ["OPENAI_API_KEY"] = original


class TestAudioFormat:
    """Tests for AudioFormat enum."""

    def test_format_values(self):
        """Test AudioFormat enum values."""
        assert AudioFormat.MP3.value == "mp3"
        assert AudioFormat.WAV.value == "wav"
        assert AudioFormat.OPUS.value == "opus"


class TestVoice:
    """Tests for Voice model."""

    def test_voice_creation(self):
        """Test creating a Voice."""
        voice = Voice(
            id="alloy",
            name="Alloy",
            provider="openai",
        )

        assert voice.id == "alloy"
        assert voice.name == "Alloy"
        assert voice.provider == "openai"
