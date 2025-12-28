"""Comprehensive tests for TTS (Text-to-Speech) class.

Tests cover:
- TTS initialization (default provider, explicit provider, voice selection)
- speak() method (text to audio, different voices, output formats)
- aspeak() async method
- speak_to_file() method
- stream() and astream() methods
- list_voices() and list_providers() methods
- Error handling (unsupported provider, import errors)
- Provider-specific implementations (OpenAI, ElevenLabs, Google)
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.llm.multimodal.models import AudioFormat, TTSProvider, Voice
from ai_infra.llm.multimodal.tts import (
    TTS,
    _detect_tts_provider,
    _get_default_model,
    _get_default_voice,
)

# =============================================================================
# Test Helper Functions
# =============================================================================


class TestDetectTTSProvider:
    """Tests for _detect_tts_provider function."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_detect_openai_provider(self):
        """Test detecting OpenAI provider from environment."""
        provider = _detect_tts_provider()
        assert provider == "openai"

    @patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}, clear=True)
    def test_detect_elevenlabs_provider(self):
        """Test detecting ElevenLabs provider from environment."""
        # ProviderRegistry uses ELEVENLABS_API_KEY
        provider = _detect_tts_provider()
        assert provider == "elevenlabs"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_detect_google_provider(self):
        """Test detecting Google provider from environment."""
        # ProviderRegistry uses GOOGLE_API_KEY (or GOOGLE_APPLICATION_CREDENTIALS)
        provider = _detect_tts_provider()
        assert provider == "google"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "key1", "ELEVENLABS_API_KEY": "key2"}, clear=True)
    def test_provider_priority_openai_first(self):
        """Test that OpenAI has priority when multiple providers available."""
        provider = _detect_tts_provider()
        assert provider == "openai"

    @patch.dict("os.environ", {}, clear=True)
    def test_no_provider_configured_raises(self):
        """Test that ValueError is raised when no provider configured."""
        with pytest.raises(ValueError, match="No TTS provider configured"):
            _detect_tts_provider()


class TestGetDefaultVoice:
    """Tests for _get_default_voice function."""

    def test_openai_default_voice(self):
        """Test default voice for OpenAI."""
        voice = _get_default_voice("openai")
        assert voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def test_elevenlabs_default_voice(self):
        """Test default voice for ElevenLabs."""
        voice = _get_default_voice("elevenlabs")
        # Could be from registry or fallback
        assert isinstance(voice, str)

    def test_google_default_voice(self):
        """Test default voice for Google."""
        voice = _get_default_voice("google")
        assert "en" in voice.lower() or voice == "default"

    def test_unknown_provider_default_voice(self):
        """Test default voice for unknown provider."""
        voice = _get_default_voice("unknown")
        assert voice == "default"


class TestGetDefaultModel:
    """Tests for _get_default_model function."""

    def test_openai_default_model(self):
        """Test default model for OpenAI."""
        model = _get_default_model("openai")
        assert model in ["tts-1", "tts-1-hd"]

    def test_elevenlabs_default_model(self):
        """Test default model for ElevenLabs."""
        model = _get_default_model("elevenlabs")
        assert isinstance(model, str)

    def test_google_default_model(self):
        """Test default model for Google."""
        model = _get_default_model("google")
        assert isinstance(model, str)

    def test_unknown_provider_default_model(self):
        """Test default model for unknown provider."""
        model = _get_default_model("unknown")
        assert model == "default"


# =============================================================================
# Test TTS Initialization
# =============================================================================


class TestTTSInitialization:
    """Tests for TTS class initialization."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_init_auto_detect_provider(self):
        """Test TTS initialization with auto-detected provider."""
        tts = TTS()
        assert tts.provider == "openai"

    def test_init_explicit_provider(self):
        """Test TTS initialization with explicit provider."""
        tts = TTS(provider="elevenlabs")
        assert tts.provider == "elevenlabs"

    def test_init_explicit_voice(self):
        """Test TTS initialization with explicit voice."""
        tts = TTS(provider="openai", voice="nova")
        assert tts.voice == "nova"

    def test_init_explicit_model(self):
        """Test TTS initialization with explicit model."""
        tts = TTS(provider="openai", model="tts-1-hd")
        assert tts.model == "tts-1-hd"

    def test_init_with_api_key(self):
        """Test TTS initialization with explicit API key."""
        tts = TTS(provider="openai", api_key="custom-key")
        assert tts._api_key == "custom-key"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_default_voice_used(self):
        """Test that default voice is used when not specified."""
        tts = TTS()
        assert tts.voice is not None
        assert isinstance(tts.voice, str)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_default_model_used(self):
        """Test that default model is used when not specified."""
        tts = TTS()
        assert tts.model is not None
        assert isinstance(tts.model, str)


# =============================================================================
# Test TTS Properties
# =============================================================================


class TestTTSProperties:
    """Tests for TTS properties."""

    def test_provider_property(self):
        """Test provider property returns correct value."""
        tts = TTS(provider="openai")
        assert tts.provider == "openai"

    def test_voice_property(self):
        """Test voice property returns correct value."""
        tts = TTS(provider="openai", voice="shimmer")
        assert tts.voice == "shimmer"

    def test_model_property(self):
        """Test model property returns correct value."""
        tts = TTS(provider="openai", model="tts-1-hd")
        assert tts.model == "tts-1-hd"


# =============================================================================
# Test TTS speak() Method - OpenAI
# =============================================================================


class TestTTSSpeakOpenAI:
    """Tests for TTS.speak() with OpenAI provider."""

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_returns_bytes(self, mock_get_client):
        """Test speak() returns audio bytes."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"fake audio data"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        result = tts.speak("Hello, world!")

        assert isinstance(result, bytes)
        assert result == b"fake audio data"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_with_custom_voice(self, mock_get_client):
        """Test speak() with custom voice parameter."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai", voice="alloy")
        tts.speak("Test", voice="nova")

        mock_client.audio.speech.create.assert_called_once()
        call_kwargs = mock_client.audio.speech.create.call_args[1]
        assert call_kwargs["voice"] == "nova"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_with_custom_model(self, mock_get_client):
        """Test speak() with custom model parameter."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai", model="tts-1")
        tts.speak("Test", model="tts-1-hd")

        call_kwargs = mock_client.audio.speech.create.call_args[1]
        assert call_kwargs["model"] == "tts-1-hd"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_mp3_format(self, mock_get_client):
        """Test speak() with MP3 output format."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"mp3 audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        tts.speak("Test", output_format=AudioFormat.MP3)

        call_kwargs = mock_client.audio.speech.create.call_args[1]
        assert call_kwargs["response_format"] == "mp3"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_wav_format(self, mock_get_client):
        """Test speak() with WAV output format."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"wav audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        tts.speak("Test", output_format=AudioFormat.WAV)

        call_kwargs = mock_client.audio.speech.create.call_args[1]
        assert call_kwargs["response_format"] == "wav"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_flac_format(self, mock_get_client):
        """Test speak() with FLAC output format."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"flac audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        tts.speak("Test", output_format=AudioFormat.FLAC)

        call_kwargs = mock_client.audio.speech.create.call_args[1]
        assert call_kwargs["response_format"] == "flac"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_ogg_maps_to_opus(self, mock_get_client):
        """Test speak() with OGG format maps to opus for OpenAI."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"ogg audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        tts.speak("Test", output_format=AudioFormat.OGG)

        call_kwargs = mock_client.audio.speech.create.call_args[1]
        assert call_kwargs["response_format"] == "opus"


# =============================================================================
# Test TTS speak() Method - ElevenLabs
# =============================================================================


class TestTTSSpeakElevenLabs:
    """Tests for TTS.speak() with ElevenLabs provider."""

    def test_speak_returns_bytes(self):
        """Test speak() returns audio bytes with ElevenLabs."""
        # Mock the import inside the function
        mock_client = MagicMock()
        mock_client.generate.return_value = b"elevenlabs audio"

        mock_client_class = MagicMock(return_value=mock_client)
        mock_elevenlabs_module = MagicMock()
        mock_elevenlabs_module.client.ElevenLabs = mock_client_class

        with patch.dict(
            "sys.modules",
            {
                "elevenlabs": mock_elevenlabs_module,
                "elevenlabs.client": mock_elevenlabs_module.client,
            },
        ):
            with patch(
                "ai_infra.llm.multimodal.tts.TTS._get_elevenlabs_api_key", return_value="test-key"
            ):
                tts = TTS(provider="elevenlabs")
                result = tts.speak("Hello!")

                assert isinstance(result, bytes)
                assert result == b"elevenlabs audio"

    def test_speak_with_iterator_response(self):
        """Test speak() handles iterator response from ElevenLabs."""
        mock_client = MagicMock()
        # ElevenLabs can return an iterator of chunks
        mock_client.generate.return_value = iter([b"chunk1", b"chunk2", b"chunk3"])

        mock_client_class = MagicMock(return_value=mock_client)
        mock_elevenlabs_module = MagicMock()
        mock_elevenlabs_module.client.ElevenLabs = mock_client_class

        with patch.dict(
            "sys.modules",
            {
                "elevenlabs": mock_elevenlabs_module,
                "elevenlabs.client": mock_elevenlabs_module.client,
            },
        ):
            with patch(
                "ai_infra.llm.multimodal.tts.TTS._get_elevenlabs_api_key", return_value="test-key"
            ):
                tts = TTS(provider="elevenlabs")
                result = tts.speak("Hello!")

                assert result == b"chunk1chunk2chunk3"

    def test_speak_mp3_format(self):
        """Test speak() with MP3 format for ElevenLabs."""
        mock_client = MagicMock()
        mock_client.generate.return_value = b"audio"

        mock_client_class = MagicMock(return_value=mock_client)
        mock_elevenlabs_module = MagicMock()
        mock_elevenlabs_module.client.ElevenLabs = mock_client_class

        with patch.dict(
            "sys.modules",
            {
                "elevenlabs": mock_elevenlabs_module,
                "elevenlabs.client": mock_elevenlabs_module.client,
            },
        ):
            with patch(
                "ai_infra.llm.multimodal.tts.TTS._get_elevenlabs_api_key", return_value="test-key"
            ):
                tts = TTS(provider="elevenlabs")
                tts.speak("Test", output_format=AudioFormat.MP3)

                call_kwargs = mock_client.generate.call_args[1]
                assert call_kwargs["output_format"] == "mp3_44100_128"

    def test_speak_wav_format(self):
        """Test speak() with WAV format for ElevenLabs."""
        mock_client = MagicMock()
        mock_client.generate.return_value = b"audio"

        mock_client_class = MagicMock(return_value=mock_client)
        mock_elevenlabs_module = MagicMock()
        mock_elevenlabs_module.client.ElevenLabs = mock_client_class

        with patch.dict(
            "sys.modules",
            {
                "elevenlabs": mock_elevenlabs_module,
                "elevenlabs.client": mock_elevenlabs_module.client,
            },
        ):
            with patch(
                "ai_infra.llm.multimodal.tts.TTS._get_elevenlabs_api_key", return_value="test-key"
            ):
                tts = TTS(provider="elevenlabs")
                tts.speak("Test", output_format=AudioFormat.WAV)

                call_kwargs = mock_client.generate.call_args[1]
                assert call_kwargs["output_format"] == "pcm_44100"


# =============================================================================
# Test TTS speak() Method - Google
# =============================================================================


class TestTTSSpeakGoogle:
    """Tests for TTS.speak() with Google provider."""

    def test_speak_returns_bytes(self):
        """Test speak() returns audio bytes with Google."""
        # Create comprehensive mock for google.cloud.texttospeech
        mock_response = MagicMock()
        mock_response.audio_content = b"google audio"

        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value = mock_response

        mock_tts = MagicMock()
        mock_tts.TextToSpeechClient.return_value = mock_client
        mock_tts.SynthesisInput = MagicMock()
        mock_tts.VoiceSelectionParams = MagicMock()
        mock_tts.AudioConfig = MagicMock()
        mock_tts.AudioEncoding.MP3 = "MP3"
        mock_tts.AudioEncoding.LINEAR16 = "LINEAR16"
        mock_tts.AudioEncoding.OGG_OPUS = "OGG_OPUS"

        mock_google = MagicMock()
        mock_google.cloud.texttospeech = mock_tts

        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.cloud": mock_google.cloud,
                "google.cloud.texttospeech": mock_tts,
            },
        ):
            tts = TTS(provider="google", voice="en-US-Standard-C")
            result = tts.speak("Hello!")

            assert isinstance(result, bytes)
            assert result == b"google audio"

    def test_speak_parses_voice_language(self):
        """Test speak() correctly parses voice name for language code."""
        mock_response = MagicMock()
        mock_response.audio_content = b"audio"

        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value = mock_response

        mock_voice_params = MagicMock()
        mock_tts = MagicMock()
        mock_tts.TextToSpeechClient.return_value = mock_client
        mock_tts.SynthesisInput = MagicMock()
        mock_tts.VoiceSelectionParams = mock_voice_params
        mock_tts.AudioConfig = MagicMock()
        mock_tts.AudioEncoding.MP3 = "MP3"

        mock_google = MagicMock()
        mock_google.cloud.texttospeech = mock_tts

        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.cloud": mock_google.cloud,
                "google.cloud.texttospeech": mock_tts,
            },
        ):
            tts = TTS(provider="google", voice="en-GB-Wavenet-A")
            tts.speak("Test")

            mock_voice_params.assert_called_once()
            call_kwargs = mock_voice_params.call_args[1]
            assert call_kwargs["language_code"] == "en-GB"
            assert call_kwargs["name"] == "en-GB-Wavenet-A"


# =============================================================================
# Test TTS speak() Method - Unsupported Provider
# =============================================================================


class TestTTSSpeakUnsupported:
    """Tests for TTS.speak() with unsupported provider."""

    def test_speak_unsupported_provider_raises(self):
        """Test speak() raises ValueError for unsupported provider."""
        tts = TTS(provider="unsupported")

        with pytest.raises(ValueError, match="Unsupported TTS provider: unsupported"):
            tts.speak("Hello!")


# =============================================================================
# Test TTS aspeak() Method
# =============================================================================


class TestTTSAspeakOpenAI:
    """Tests for TTS.aspeak() async method with OpenAI."""

    @pytest.mark.asyncio
    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_async_client")
    async def test_aspeak_returns_bytes(self, mock_get_client):
        """Test aspeak() returns audio bytes."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"async audio"
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        result = await tts.aspeak("Hello!")

        assert isinstance(result, bytes)
        assert result == b"async audio"

    @pytest.mark.asyncio
    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_async_client")
    async def test_aspeak_with_custom_voice(self, mock_get_client):
        """Test aspeak() with custom voice parameter."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        await tts.aspeak("Test", voice="shimmer")

        mock_client.audio.speech.create.assert_awaited_once()
        call_kwargs = mock_client.audio.speech.create.call_args[1]
        assert call_kwargs["voice"] == "shimmer"

    @pytest.mark.asyncio
    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_async_client")
    async def test_aspeak_with_custom_format(self, mock_get_client):
        """Test aspeak() with custom output format."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        await tts.aspeak("Test", output_format=AudioFormat.FLAC)

        call_kwargs = mock_client.audio.speech.create.call_args[1]
        assert call_kwargs["response_format"] == "flac"


class TestTTSAspeakElevenLabs:
    """Tests for TTS.aspeak() async method with ElevenLabs."""

    @pytest.mark.asyncio
    async def test_aspeak_returns_bytes(self):
        """Test aspeak() returns audio bytes with ElevenLabs."""
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=b"async elevenlabs audio")

        mock_client_class = MagicMock(return_value=mock_client)
        mock_elevenlabs_module = MagicMock()
        mock_elevenlabs_module.client.AsyncElevenLabs = mock_client_class

        with patch.dict(
            "sys.modules",
            {
                "elevenlabs": mock_elevenlabs_module,
                "elevenlabs.client": mock_elevenlabs_module.client,
            },
        ):
            with patch(
                "ai_infra.llm.multimodal.tts.TTS._get_elevenlabs_api_key", return_value="test-key"
            ):
                tts = TTS(provider="elevenlabs")
                result = await tts.aspeak("Hello!")

                assert isinstance(result, bytes)
                assert result == b"async elevenlabs audio"


class TestTTSAspeakGoogle:
    """Tests for TTS.aspeak() async method with Google."""

    @pytest.mark.asyncio
    async def test_aspeak_uses_executor(self):
        """Test aspeak() runs Google TTS in executor."""
        mock_response = MagicMock()
        mock_response.audio_content = b"google async audio"

        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value = mock_response

        mock_tts = MagicMock()
        mock_tts.TextToSpeechClient.return_value = mock_client
        mock_tts.SynthesisInput = MagicMock()
        mock_tts.VoiceSelectionParams = MagicMock()
        mock_tts.AudioConfig = MagicMock()
        mock_tts.AudioEncoding.MP3 = "MP3"

        mock_google = MagicMock()
        mock_google.cloud.texttospeech = mock_tts

        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.cloud": mock_google.cloud,
                "google.cloud.texttospeech": mock_tts,
            },
        ):
            tts = TTS(provider="google", voice="en-US-Standard-A")
            result = await tts.aspeak("Hello!")

            assert isinstance(result, bytes)
            assert result == b"google async audio"


class TestTTSAspeakUnsupported:
    """Tests for TTS.aspeak() with unsupported provider."""

    @pytest.mark.asyncio
    async def test_aspeak_unsupported_provider_raises(self):
        """Test aspeak() raises ValueError for unsupported provider."""
        tts = TTS(provider="unsupported")

        with pytest.raises(ValueError, match="Unsupported TTS provider: unsupported"):
            await tts.aspeak("Hello!")


# =============================================================================
# Test TTS speak_to_file() Method
# =============================================================================


class TestTTSSpeakToFile:
    """Tests for TTS.speak_to_file() method."""

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_to_file_creates_file(self, mock_get_client):
        """Test speak_to_file() creates audio file."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"audio content"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp3"
            tts.speak_to_file("Hello!", output_path)

            assert output_path.exists()
            assert output_path.read_bytes() == b"audio content"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_to_file_infers_mp3_format(self, mock_get_client):
        """Test speak_to_file() infers MP3 format from extension."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"mp3 audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp3"
            tts.speak_to_file("Test", output_path)

            call_kwargs = mock_client.audio.speech.create.call_args[1]
            assert call_kwargs["response_format"] == "mp3"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_to_file_infers_wav_format(self, mock_get_client):
        """Test speak_to_file() infers WAV format from extension."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"wav audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.wav"
            tts.speak_to_file("Test", output_path)

            call_kwargs = mock_client.audio.speech.create.call_args[1]
            assert call_kwargs["response_format"] == "wav"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_to_file_explicit_format_overrides(self, mock_get_client):
        """Test speak_to_file() explicit format overrides inferred."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"flac audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")

        with tempfile.TemporaryDirectory() as tmpdir:
            # File says .mp3 but we explicitly request FLAC
            output_path = Path(tmpdir) / "output.mp3"
            tts.speak_to_file("Test", output_path, output_format=AudioFormat.FLAC)

            call_kwargs = mock_client.audio.speech.create.call_args[1]
            assert call_kwargs["response_format"] == "flac"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_to_file_unknown_extension_defaults_mp3(self, mock_get_client):
        """Test speak_to_file() uses MP3 for unknown extension."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.xyz"
            tts.speak_to_file("Test", output_path)

            call_kwargs = mock_client.audio.speech.create.call_args[1]
            assert call_kwargs["response_format"] == "mp3"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_speak_to_file_with_string_path(self, mock_get_client):
        """Test speak_to_file() works with string path."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/output.mp3"
            tts.speak_to_file("Test", output_path)

            assert Path(output_path).exists()


# =============================================================================
# Test TTS stream() Method
# =============================================================================


class TestTTSStreamOpenAI:
    """Tests for TTS.stream() with OpenAI provider."""

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_stream_yields_chunks(self, mock_get_client):
        """Test stream() yields audio chunks."""
        mock_client = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_context)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_context.iter_bytes.return_value = [b"chunk1", b"chunk2", b"chunk3"]
        mock_client.audio.speech.with_streaming_response.create.return_value = mock_context
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        chunks = list(tts.stream("Hello, world!"))

        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_stream_returns_iterator(self, mock_get_client):
        """Test stream() returns an iterator."""
        mock_client = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_context)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_context.iter_bytes.return_value = [b"data"]
        mock_client.audio.speech.with_streaming_response.create.return_value = mock_context
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        result = tts.stream("Test")

        assert isinstance(result, Iterator)


class TestTTSStreamElevenLabs:
    """Tests for TTS.stream() with ElevenLabs provider."""

    def test_stream_yields_chunks(self):
        """Test stream() yields chunks from ElevenLabs."""
        mock_client = MagicMock()
        mock_client.generate.return_value = iter([b"el_chunk1", b"el_chunk2"])

        mock_client_class = MagicMock(return_value=mock_client)
        mock_elevenlabs_module = MagicMock()
        mock_elevenlabs_module.client.ElevenLabs = mock_client_class

        with patch.dict(
            "sys.modules",
            {
                "elevenlabs": mock_elevenlabs_module,
                "elevenlabs.client": mock_elevenlabs_module.client,
            },
        ):
            with patch(
                "ai_infra.llm.multimodal.tts.TTS._get_elevenlabs_api_key", return_value="test-key"
            ):
                tts = TTS(provider="elevenlabs")
                chunks = list(tts.stream("Hello!"))

                assert chunks == [b"el_chunk1", b"el_chunk2"]


class TestTTSStreamFallback:
    """Tests for TTS.stream() fallback behavior."""

    def test_stream_fallback_single_chunk(self):
        """Test stream() falls back to single chunk for unsupported providers."""
        mock_response = MagicMock()
        mock_response.audio_content = b"google audio"

        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value = mock_response

        mock_tts = MagicMock()
        mock_tts.TextToSpeechClient.return_value = mock_client
        mock_tts.SynthesisInput = MagicMock()
        mock_tts.VoiceSelectionParams = MagicMock()
        mock_tts.AudioConfig = MagicMock()
        mock_tts.AudioEncoding.MP3 = "MP3"

        mock_google = MagicMock()
        mock_google.cloud.texttospeech = mock_tts

        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.cloud": mock_google.cloud,
                "google.cloud.texttospeech": mock_tts,
            },
        ):
            tts = TTS(provider="google", voice="en-US-Standard-A")
            chunks = list(tts.stream("Hello!"))

            # Google doesn't support streaming, so returns entire audio as one chunk
            assert len(chunks) == 1
            assert chunks[0] == b"google audio"


# =============================================================================
# Test TTS astream() Method
# =============================================================================


class TestTTSAstreamOpenAI:
    """Tests for TTS.astream() with OpenAI provider."""

    @pytest.mark.asyncio
    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_async_client")
    async def test_astream_yields_chunks(self, mock_get_client):
        """Test astream() yields audio chunks."""

        async def mock_iter_bytes(*args, **kwargs):
            for chunk in [b"async_chunk1", b"async_chunk2"]:
                yield chunk

        mock_client = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=False)
        mock_context.iter_bytes = mock_iter_bytes
        mock_client.audio.speech.with_streaming_response.create.return_value = mock_context
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        chunks = []
        async for chunk in tts.astream("Hello!"):
            chunks.append(chunk)

        assert chunks == [b"async_chunk1", b"async_chunk2"]


class TestTTSAstreamFallback:
    """Tests for TTS.astream() fallback behavior."""

    @pytest.mark.asyncio
    async def test_astream_fallback_single_chunk(self):
        """Test astream() falls back to single chunk for unsupported providers."""
        mock_response = MagicMock()
        mock_response.audio_content = b"google async audio"

        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value = mock_response

        mock_tts = MagicMock()
        mock_tts.TextToSpeechClient.return_value = mock_client
        mock_tts.SynthesisInput = MagicMock()
        mock_tts.VoiceSelectionParams = MagicMock()
        mock_tts.AudioConfig = MagicMock()
        mock_tts.AudioEncoding.MP3 = "MP3"

        mock_google = MagicMock()
        mock_google.cloud.texttospeech = mock_tts

        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.cloud": mock_google.cloud,
                "google.cloud.texttospeech": mock_tts,
            },
        ):
            tts = TTS(provider="google", voice="en-US-Standard-A")
            chunks = []
            async for chunk in tts.astream("Hello!"):
                chunks.append(chunk)

            # Google doesn't support async streaming
            assert len(chunks) == 1
            assert chunks[0] == b"google async audio"


# =============================================================================
# Test TTS list_voices() Method
# =============================================================================


class TestTTSListVoices:
    """Tests for TTS.list_voices() method."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_list_openai_voices(self):
        """Test list_voices() returns OpenAI voices when configured."""
        voices = TTS.list_voices("openai")

        assert len(voices) > 0
        assert all(isinstance(v, Voice) for v in voices)
        assert all(v.provider == TTSProvider.OPENAI for v in voices)
        voice_ids = [v.id for v in voices]
        assert "alloy" in voice_ids or "Alloy" in voice_ids

    @patch.dict("os.environ", {"ELEVEN_API_KEY": "test-key"}, clear=True)
    def test_list_elevenlabs_voices(self):
        """Test list_voices() returns ElevenLabs voices when configured."""
        voices = TTS.list_voices("elevenlabs")

        assert len(voices) > 0
        assert all(isinstance(v, Voice) for v in voices)
        assert all(v.provider == TTSProvider.ELEVENLABS for v in voices)

    @patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}, clear=True)
    def test_list_elevenlabs_voices_alt_key(self):
        """Test list_voices() works with alternative ElevenLabs key name."""
        voices = TTS.list_voices("elevenlabs")

        assert len(voices) > 0

    @patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds"}, clear=True)
    def test_list_google_voices(self):
        """Test list_voices() returns Google voices when configured."""
        voices = TTS.list_voices("google")

        assert len(voices) > 0
        assert all(isinstance(v, Voice) for v in voices)
        assert all(v.provider == TTSProvider.GOOGLE for v in voices)

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_list_google_voices_alt_key(self):
        """Test list_voices() works with GOOGLE_API_KEY."""
        voices = TTS.list_voices("google")

        assert len(voices) > 0

    @patch.dict("os.environ", {}, clear=True)
    def test_list_voices_no_config_empty(self):
        """Test list_voices() returns empty for unconfigured provider."""
        voices = TTS.list_voices("openai")

        assert voices == []

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "key1", "ELEVEN_API_KEY": "key2"},
        clear=True,
    )
    def test_list_all_voices_when_no_provider(self):
        """Test list_voices() returns all voices when provider is None."""
        voices = TTS.list_voices()

        # Should have voices from both providers
        providers = {v.provider for v in voices}
        assert TTSProvider.OPENAI in providers
        assert TTSProvider.ELEVENLABS in providers


# =============================================================================
# Test TTS list_providers() Method
# =============================================================================


class TestTTSListProviders:
    """Tests for TTS.list_providers() method."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_list_providers_openai(self):
        """Test list_providers() returns openai when configured."""
        providers = TTS.list_providers()

        assert "openai" in providers

    @patch.dict("os.environ", {"ELEVEN_API_KEY": "test-key"}, clear=True)
    def test_list_providers_elevenlabs(self):
        """Test list_providers() returns elevenlabs when configured."""
        providers = TTS.list_providers()

        assert "elevenlabs" in providers

    @patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}, clear=True)
    def test_list_providers_elevenlabs_alt_key(self):
        """Test list_providers() works with alternative key name."""
        providers = TTS.list_providers()

        assert "elevenlabs" in providers

    @patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds"}, clear=True)
    def test_list_providers_google(self):
        """Test list_providers() returns google when configured."""
        providers = TTS.list_providers()

        assert "google" in providers

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_list_providers_google_alt_key(self):
        """Test list_providers() works with GOOGLE_API_KEY."""
        providers = TTS.list_providers()

        assert "google" in providers

    @patch.dict("os.environ", {}, clear=True)
    def test_list_providers_empty(self):
        """Test list_providers() returns empty when nothing configured."""
        providers = TTS.list_providers()

        assert providers == []

    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "key1",
            "ELEVEN_API_KEY": "key2",
            "GOOGLE_APPLICATION_CREDENTIALS": "/path",
        },
        clear=True,
    )
    def test_list_all_providers(self):
        """Test list_providers() returns all configured providers."""
        providers = TTS.list_providers()

        assert "openai" in providers
        assert "elevenlabs" in providers
        assert "google" in providers


# =============================================================================
# Test TTS Import Errors
# =============================================================================


class TestTTSImportErrors:
    """Tests for TTS import error handling."""

    @patch.dict("sys.modules", {"openai": None})
    def test_openai_import_error(self):
        """Test ImportError when openai package not installed."""
        tts = TTS(provider="openai", api_key="test")

        # Directly test the client getter
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai package required"):
                tts._get_openai_client()

    @patch.dict("sys.modules", {"elevenlabs": None, "elevenlabs.client": None})
    def test_elevenlabs_import_error(self):
        """Test ImportError when elevenlabs package not installed."""
        tts = TTS(provider="elevenlabs", api_key="test")

        with pytest.raises(ImportError, match="elevenlabs package required"):
            tts._speak_elevenlabs("test", "voice", "model", AudioFormat.MP3)

    @patch.dict(
        "sys.modules",
        {"google": None, "google.cloud": None, "google.cloud.texttospeech": None},
    )
    def test_google_import_error(self):
        """Test ImportError when google-cloud-texttospeech package not installed."""
        tts = TTS(provider="google", voice="en-US-Standard-A")

        with pytest.raises(ImportError, match="google-cloud-texttospeech package required"):
            tts._speak_google("test", "en-US-Standard-A", "model", AudioFormat.MP3)


# =============================================================================
# Test TTS Static Methods
# =============================================================================


class TestTTSStaticMethods:
    """Tests for TTS static helper methods."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_detect_provider_static(self):
        """Test static _detect_provider method."""
        provider = TTS._detect_provider()
        assert provider == "openai"

    def test_default_voice_static(self):
        """Test static _default_voice method."""
        voice = TTS._default_voice("openai")
        assert isinstance(voice, str)

    def test_default_model_static(self):
        """Test static _default_model method."""
        model = TTS._default_model("openai")
        assert isinstance(model, str)


# =============================================================================
# Test TTS API Key Handling
# =============================================================================


class TestTTSAPIKeyHandling:
    """Tests for TTS API key handling."""

    def test_elevenlabs_api_key_from_init(self):
        """Test ElevenLabs API key from init."""
        tts = TTS(provider="elevenlabs", api_key="custom-eleven-key")
        key = tts._get_elevenlabs_api_key()
        assert key == "custom-eleven-key"

    @patch.dict("os.environ", {"ELEVEN_API_KEY": "env-eleven-key"}, clear=True)
    def test_elevenlabs_api_key_from_env(self):
        """Test ElevenLabs API key from ELEVEN_API_KEY."""
        tts = TTS(provider="elevenlabs")
        key = tts._get_elevenlabs_api_key()
        assert key == "env-eleven-key"

    @patch.dict("os.environ", {"ELEVENLABS_API_KEY": "alt-eleven-key"}, clear=True)
    def test_elevenlabs_api_key_from_alt_env(self):
        """Test ElevenLabs API key from ELEVENLABS_API_KEY."""
        tts = TTS(provider="elevenlabs")
        key = tts._get_elevenlabs_api_key()
        assert key == "alt-eleven-key"

    @patch.dict("os.environ", {}, clear=True)
    def test_elevenlabs_api_key_empty_when_missing(self):
        """Test ElevenLabs API key is empty when not set."""
        tts = TTS(provider="elevenlabs")
        key = tts._get_elevenlabs_api_key()
        assert key == ""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-openai-key"}, clear=True)
    def test_openai_client_uses_env_key(self):
        """Test OpenAI client uses environment API key."""
        mock_openai = MagicMock()
        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI = mock_openai

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            tts = TTS(provider="openai")
            tts._get_openai_client()

            mock_openai.assert_called_once_with(api_key="env-openai-key")

    def test_openai_client_uses_init_key(self):
        """Test OpenAI client uses init API key."""
        mock_openai = MagicMock()
        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI = mock_openai

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            tts = TTS(provider="openai", api_key="custom-openai-key")
            tts._get_openai_client()

            mock_openai.assert_called_once_with(api_key="custom-openai-key")


# =============================================================================
# Test Audio Format Mapping
# =============================================================================


class TestAudioFormatMapping:
    """Tests for audio format mapping in speak methods."""

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_aac_format_mapping_openai(self, mock_get_client):
        """Test AAC format mapping for OpenAI."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"aac audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        tts.speak("Test", output_format=AudioFormat.AAC)

        call_kwargs = mock_client.audio.speech.create.call_args[1]
        assert call_kwargs["response_format"] == "aac"

    @patch("ai_infra.llm.multimodal.tts.TTS._get_openai_client")
    def test_unknown_format_defaults_mp3_openai(self, mock_get_client):
        """Test unknown format defaults to mp3 for OpenAI."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_client.audio.speech.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tts = TTS(provider="openai")
        # PCM is not in OpenAI's format map, should default to mp3
        tts.speak("Test", output_format=AudioFormat.PCM)

        call_kwargs = mock_client.audio.speech.create.call_args[1]
        assert call_kwargs["response_format"] == "mp3"
