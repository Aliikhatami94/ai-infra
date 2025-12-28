"""Comprehensive tests for STT (Speech-to-Text) class.

Tests cover:
- STT initialization (default provider, explicit provider, language selection)
- transcribe() method (audio bytes, file path, timestamps)
- atranscribe() async method
- transcribe_file() method
- stream_transcribe() method
- list_providers() and list_models() methods
- Error handling (unsupported provider, import errors, file not found)
- Provider-specific implementations (OpenAI, Deepgram, Google)
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.llm.multimodal.models import (
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
)
from ai_infra.llm.multimodal.stt import (
    STT,
    _detect_stt_provider,
    _get_default_model,
)

# =============================================================================
# Test Helper Functions
# =============================================================================


class TestDetectSTTProvider:
    """Tests for _detect_stt_provider function."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_detect_openai_provider(self):
        """Test detecting OpenAI provider from environment."""
        provider = _detect_stt_provider()
        assert provider == "openai"

    @patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}, clear=True)
    def test_detect_deepgram_provider(self):
        """Test detecting Deepgram provider from environment."""
        provider = _detect_stt_provider()
        assert provider == "deepgram"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_detect_google_provider(self):
        """Test detecting Google provider from environment."""
        provider = _detect_stt_provider()
        assert provider == "google"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "key1", "DEEPGRAM_API_KEY": "key2"}, clear=True)
    def test_provider_priority_openai_first(self):
        """Test that OpenAI has priority when multiple providers available."""
        provider = _detect_stt_provider()
        assert provider == "openai"

    @patch.dict("os.environ", {}, clear=True)
    def test_no_provider_configured_raises(self):
        """Test that ValueError is raised when no provider configured."""
        with pytest.raises(ValueError, match="No STT provider configured"):
            _detect_stt_provider()


class TestGetDefaultModel:
    """Tests for _get_default_model function."""

    def test_openai_default_model(self):
        """Test default model for OpenAI."""
        model = _get_default_model("openai")
        assert model == "whisper-1"

    def test_deepgram_default_model(self):
        """Test default model for Deepgram."""
        model = _get_default_model("deepgram")
        assert "nova" in model or model == "default"

    def test_google_default_model(self):
        """Test default model for Google."""
        model = _get_default_model("google")
        assert isinstance(model, str)

    def test_unknown_provider_default_model(self):
        """Test default model for unknown provider."""
        model = _get_default_model("unknown")
        assert model == "default"


# =============================================================================
# Test STT Initialization
# =============================================================================


class TestSTTInitialization:
    """Tests for STT class initialization."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_init_auto_detect_provider(self):
        """Test STT initialization with auto-detected provider."""
        stt = STT()
        assert stt.provider == "openai"

    def test_init_explicit_provider(self):
        """Test STT initialization with explicit provider."""
        stt = STT(provider="deepgram")
        assert stt.provider == "deepgram"

    def test_init_explicit_model(self):
        """Test STT initialization with explicit model."""
        stt = STT(provider="openai", model="whisper-1")
        assert stt.model == "whisper-1"

    def test_init_with_language(self):
        """Test STT initialization with language."""
        stt = STT(provider="openai", language="es")
        assert stt.language == "es"

    def test_init_with_api_key(self):
        """Test STT initialization with explicit API key."""
        stt = STT(provider="openai", api_key="custom-key")
        assert stt._api_key == "custom-key"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_default_model_used(self):
        """Test that default model is used when not specified."""
        stt = STT()
        assert stt.model is not None
        assert isinstance(stt.model, str)


# =============================================================================
# Test STT Properties
# =============================================================================


class TestSTTProperties:
    """Tests for STT properties."""

    def test_provider_property(self):
        """Test provider property returns correct value."""
        stt = STT(provider="openai")
        assert stt.provider == "openai"

    def test_model_property(self):
        """Test model property returns correct value."""
        stt = STT(provider="openai", model="whisper-1")
        assert stt.model == "whisper-1"

    def test_language_property(self):
        """Test language property returns correct value."""
        stt = STT(provider="openai", language="fr")
        assert stt.language == "fr"

    def test_language_property_none(self):
        """Test language property returns None when not set."""
        stt = STT(provider="openai")
        assert stt.language is None


# =============================================================================
# Test STT transcribe() Method - OpenAI
# =============================================================================


class TestSTTTranscribeOpenAI:
    """Tests for STT.transcribe() with OpenAI provider."""

    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_client")
    def test_transcribe_bytes_returns_result(self, mock_get_client):
        """Test transcribe() with bytes returns TranscriptionResult."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello, world!"
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")
        result = stt.transcribe(b"fake audio data")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, world!"
        assert result.provider == STTProvider.OPENAI

    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_client")
    def test_transcribe_with_language(self, mock_get_client):
        """Test transcribe() with language parameter."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hola, mundo!"
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")
        _result = stt.transcribe(b"audio", language="es")

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["language"] == "es"

    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_client")
    def test_transcribe_with_timestamps(self, mock_get_client):
        """Test transcribe() with timestamps option."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello, world!"
        mock_response.segments = [
            {"text": "Hello,", "start": 0.0, "end": 0.5, "confidence": 0.95},
            {"text": "world!", "start": 0.5, "end": 1.0, "confidence": 0.98},
        ]
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")
        result = stt.transcribe(b"audio", timestamps=True)

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["response_format"] == "verbose_json"
        assert len(result.segments) == 2

    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_client")
    def test_transcribe_with_word_timestamps(self, mock_get_client):
        """Test transcribe() with word_timestamps option."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello!"
        mock_response.segments = []
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")
        stt.transcribe(b"audio", word_timestamps=True)

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["timestamp_granularities"] == ["word", "segment"]

    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_client")
    def test_transcribe_with_prompt(self, mock_get_client):
        """Test transcribe() with prompt parameter."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "OpenAI Whisper is great"
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")
        stt.transcribe(b"audio", prompt="This is about AI")

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["prompt"] == "This is about AI"

    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_client")
    def test_transcribe_file_path(self, mock_get_client):
        """Test transcribe() with file path."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Transcribed from file"
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio")
            temp_path = f.name

        try:
            result = stt.transcribe(temp_path)
            assert result.text == "Transcribed from file"
        finally:
            Path(temp_path).unlink()


# =============================================================================
# Test STT transcribe() Method - Deepgram
# =============================================================================


class TestSTTTranscribeDeepgram:
    """Tests for STT.transcribe() with Deepgram provider."""

    def test_transcribe_returns_result(self):
        """Test transcribe() returns TranscriptionResult with Deepgram."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_channel = MagicMock()
        mock_alt = MagicMock()
        mock_alt.transcript = "Hello from Deepgram"
        mock_channel.alternatives = [mock_alt]
        mock_result.channels = [mock_channel]
        mock_result.metadata = MagicMock(duration=5.0)
        mock_response.results = mock_result
        mock_client.listen.prerecorded.v.return_value.transcribe_file.return_value = mock_response

        mock_deepgram_module = MagicMock()
        mock_deepgram_module.DeepgramClient.return_value = mock_client
        mock_deepgram_module.PrerecordedOptions = MagicMock()

        with patch.dict("sys.modules", {"deepgram": mock_deepgram_module}):
            with patch(
                "ai_infra.llm.multimodal.stt.STT._get_deepgram_api_key", return_value="test-key"
            ):
                stt = STT(provider="deepgram")
                result = stt.transcribe(b"audio data")

                assert isinstance(result, TranscriptionResult)
                assert result.text == "Hello from Deepgram"
                assert result.provider == STTProvider.DEEPGRAM

    def test_transcribe_with_timestamps(self):
        """Test transcribe() with timestamps for Deepgram."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_channel = MagicMock()
        mock_alt = MagicMock()
        mock_alt.transcript = "Hello"
        mock_alt.words = [
            MagicMock(word="Hello", start=0.0, end=0.5, confidence=0.95, punctuated_word="Hello."),
        ]
        mock_channel.alternatives = [mock_alt]
        mock_result.channels = [mock_channel]
        mock_result.metadata = MagicMock(duration=1.0)
        mock_response.results = mock_result
        mock_client.listen.prerecorded.v.return_value.transcribe_file.return_value = mock_response

        mock_deepgram_module = MagicMock()
        mock_deepgram_module.DeepgramClient.return_value = mock_client
        mock_deepgram_module.PrerecordedOptions = MagicMock()

        with patch.dict("sys.modules", {"deepgram": mock_deepgram_module}):
            with patch("ai_infra.llm.multimodal.stt.STT._get_deepgram_api_key", return_value="key"):
                stt = STT(provider="deepgram")
                result = stt.transcribe(b"audio", timestamps=True)

                assert len(result.segments) > 0


# =============================================================================
# Test STT transcribe() Method - Google
# =============================================================================


class TestSTTTranscribeGoogle:
    """Tests for STT.transcribe() with Google provider."""

    def test_transcribe_returns_result(self):
        """Test transcribe() returns TranscriptionResult with Google."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_alt = MagicMock()
        mock_alt.transcript = "Hello from Google"
        mock_alt.confidence = 0.95
        mock_result.alternatives = [mock_alt]
        mock_response.results = [mock_result]
        mock_client.recognize.return_value = mock_response

        mock_speech = MagicMock()
        mock_speech.SpeechClient.return_value = mock_client
        mock_speech.RecognitionAudio = MagicMock()
        mock_speech.RecognitionConfig = MagicMock()
        mock_speech.RecognitionConfig.AudioEncoding.MP3 = "MP3"

        mock_google = MagicMock()
        mock_google.cloud.speech = mock_speech

        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.cloud": mock_google.cloud,
                "google.cloud.speech": mock_speech,
            },
        ):
            stt = STT(provider="google")
            result = stt.transcribe(b"audio data")

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello from Google"
            assert result.provider == STTProvider.GOOGLE

    def test_transcribe_with_language(self):
        """Test transcribe() with language for Google."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_alt = MagicMock()
        mock_alt.transcript = "Bonjour"
        mock_result.alternatives = [mock_alt]
        mock_response.results = [mock_result]
        mock_client.recognize.return_value = mock_response

        mock_speech = MagicMock()
        mock_speech.SpeechClient.return_value = mock_client
        mock_speech.RecognitionAudio = MagicMock()
        mock_config_class = MagicMock()
        mock_speech.RecognitionConfig = mock_config_class
        mock_speech.RecognitionConfig.AudioEncoding.MP3 = "MP3"

        mock_google = MagicMock()
        mock_google.cloud.speech = mock_speech

        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.cloud": mock_google.cloud,
                "google.cloud.speech": mock_speech,
            },
        ):
            stt = STT(provider="google")
            _result = stt.transcribe(b"audio", language="fr-FR")

            # Check that config was called with correct language
            mock_config_class.assert_called_once()
            call_kwargs = mock_config_class.call_args[1]
            assert call_kwargs["language_code"] == "fr-FR"


# =============================================================================
# Test STT transcribe() Method - Unsupported Provider
# =============================================================================


class TestSTTTranscribeUnsupported:
    """Tests for STT.transcribe() with unsupported provider."""

    def test_transcribe_unsupported_provider_raises(self):
        """Test transcribe() raises ValueError for unsupported provider."""
        stt = STT(provider="unsupported")

        with pytest.raises(ValueError, match="Unsupported STT provider: unsupported"):
            stt.transcribe(b"audio")


# =============================================================================
# Test STT atranscribe() Method
# =============================================================================


class TestSTTAtranscribeOpenAI:
    """Tests for STT.atranscribe() async method with OpenAI."""

    @pytest.mark.asyncio
    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_async_client")
    async def test_atranscribe_returns_result(self, mock_get_client):
        """Test atranscribe() returns TranscriptionResult."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = "Async transcription"
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")
        result = await stt.atranscribe(b"audio")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Async transcription"

    @pytest.mark.asyncio
    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_async_client")
    async def test_atranscribe_with_timestamps(self, mock_get_client):
        """Test atranscribe() with timestamps."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = "Hello"
        mock_response.segments = [{"text": "Hello", "start": 0.0, "end": 0.5}]
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")
        result = await stt.atranscribe(b"audio", timestamps=True)

        assert len(result.segments) == 1


class TestSTTAtranscribeDeepgram:
    """Tests for STT.atranscribe() async method with Deepgram."""

    @pytest.mark.asyncio
    async def test_atranscribe_returns_result(self):
        """Test atranscribe() returns TranscriptionResult with Deepgram."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_channel = MagicMock()
        mock_alt = MagicMock()
        mock_alt.transcript = "Async Deepgram"
        mock_channel.alternatives = [mock_alt]
        mock_result.channels = [mock_channel]
        mock_result.metadata = MagicMock(duration=2.0)
        mock_response.results = mock_result

        mock_async_client = AsyncMock()
        mock_async_client.transcribe_file = AsyncMock(return_value=mock_response)
        mock_client.listen.asyncprerecorded.v.return_value = mock_async_client

        mock_deepgram_module = MagicMock()
        mock_deepgram_module.DeepgramClient.return_value = mock_client
        mock_deepgram_module.PrerecordedOptions = MagicMock()

        with patch.dict("sys.modules", {"deepgram": mock_deepgram_module}):
            with patch("ai_infra.llm.multimodal.stt.STT._get_deepgram_api_key", return_value="key"):
                stt = STT(provider="deepgram")
                result = await stt.atranscribe(b"audio")

                assert isinstance(result, TranscriptionResult)
                assert result.text == "Async Deepgram"


class TestSTTAtranscribeGoogle:
    """Tests for STT.atranscribe() async method with Google."""

    @pytest.mark.asyncio
    async def test_atranscribe_uses_executor(self):
        """Test atranscribe() runs Google STT in executor."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_alt = MagicMock()
        mock_alt.transcript = "Async Google"
        mock_result.alternatives = [mock_alt]
        mock_response.results = [mock_result]
        mock_client.recognize.return_value = mock_response

        mock_speech = MagicMock()
        mock_speech.SpeechClient.return_value = mock_client
        mock_speech.RecognitionAudio = MagicMock()
        mock_speech.RecognitionConfig = MagicMock()
        mock_speech.RecognitionConfig.AudioEncoding.MP3 = "MP3"

        mock_google = MagicMock()
        mock_google.cloud.speech = mock_speech

        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.cloud": mock_google.cloud,
                "google.cloud.speech": mock_speech,
            },
        ):
            stt = STT(provider="google")
            result = await stt.atranscribe(b"audio")

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Async Google"


class TestSTTAtranscribeUnsupported:
    """Tests for STT.atranscribe() with unsupported provider."""

    @pytest.mark.asyncio
    async def test_atranscribe_unsupported_provider_raises(self):
        """Test atranscribe() raises ValueError for unsupported provider."""
        stt = STT(provider="unsupported")

        with pytest.raises(ValueError, match="Unsupported STT provider: unsupported"):
            await stt.atranscribe(b"audio")


# =============================================================================
# Test STT transcribe_file() Method
# =============================================================================


class TestSTTTranscribeFile:
    """Tests for STT.transcribe_file() method."""

    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_client")
    def test_transcribe_file_success(self, mock_get_client):
        """Test transcribe_file() with valid file."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "File transcription"
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"audio content")
            temp_path = f.name

        try:
            result = stt.transcribe_file(temp_path)
            assert result.text == "File transcription"
        finally:
            Path(temp_path).unlink()

    def test_transcribe_file_not_found(self):
        """Test transcribe_file() raises FileNotFoundError."""
        stt = STT(provider="openai")

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            stt.transcribe_file("/nonexistent/path/audio.mp3")

    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_client")
    def test_transcribe_file_with_path_object(self, mock_get_client):
        """Test transcribe_file() with Path object."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Path object transcription"
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"wav audio")
            temp_path = Path(f.name)

        try:
            result = stt.transcribe_file(temp_path)
            assert result.text == "Path object transcription"
        finally:
            temp_path.unlink()

    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_client")
    def test_transcribe_file_with_options(self, mock_get_client):
        """Test transcribe_file() with all options."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "With options"
        mock_response.segments = [{"text": "With options", "start": 0.0, "end": 1.0}]
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"audio")
            temp_path = f.name

        try:
            result = stt.transcribe_file(
                temp_path,
                language="en",
                timestamps=True,
                word_timestamps=True,
                prompt="Meeting notes",
            )
            assert result.text == "With options"
        finally:
            Path(temp_path).unlink()


# =============================================================================
# Test STT stream_transcribe() Method
# =============================================================================


class TestSTTStreamTranscribe:
    """Tests for STT.stream_transcribe() method."""

    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_client")
    def test_stream_transcribe_fallback(self, mock_get_client):
        """Test stream_transcribe() fallback for non-streaming provider."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Fallback transcription"
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")

        def audio_generator() -> Iterator[bytes]:
            yield b"chunk1"
            yield b"chunk2"

        results = list(stt.stream_transcribe(audio_generator()))

        assert len(results) == 1
        assert results[0] == "Fallback transcription"

    def test_stream_transcribe_deepgram(self):
        """Test stream_transcribe() with Deepgram (native streaming)."""
        mock_connection = MagicMock()
        mock_client = MagicMock()
        mock_client.listen.live.v.return_value.start.return_value = mock_connection

        mock_deepgram_module = MagicMock()
        mock_deepgram_module.DeepgramClient.return_value = mock_client
        mock_deepgram_module.LiveOptions = MagicMock()
        mock_deepgram_module.LiveTranscriptionEvents = MagicMock()
        mock_deepgram_module.LiveTranscriptionEvents.Transcript = "Transcript"

        with patch.dict("sys.modules", {"deepgram": mock_deepgram_module}):
            with patch("ai_infra.llm.multimodal.stt.STT._get_deepgram_api_key", return_value="key"):
                stt = STT(provider="deepgram")

                def audio_generator() -> Iterator[bytes]:
                    yield b"chunk1"
                    yield b"chunk2"

                # Just verify it doesn't crash - actual streaming would need more setup
                result = stt.stream_transcribe(audio_generator())
                assert isinstance(result, Iterator)


# =============================================================================
# Test STT list_providers() Method
# =============================================================================


class TestSTTListProviders:
    """Tests for STT.list_providers() method."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_list_providers_openai(self):
        """Test list_providers() returns openai when configured."""
        providers = STT.list_providers()
        assert "openai" in providers

    @patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}, clear=True)
    def test_list_providers_deepgram(self):
        """Test list_providers() returns deepgram when configured."""
        providers = STT.list_providers()
        assert "deepgram" in providers

    @patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": "/path/creds"}, clear=True)
    def test_list_providers_google(self):
        """Test list_providers() returns google when configured."""
        providers = STT.list_providers()
        assert "google" in providers

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_list_providers_google_alt_key(self):
        """Test list_providers() works with GOOGLE_API_KEY."""
        providers = STT.list_providers()
        assert "google" in providers

    @patch.dict("os.environ", {}, clear=True)
    def test_list_providers_empty(self):
        """Test list_providers() returns empty when nothing configured."""
        providers = STT.list_providers()
        assert providers == []

    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "key1",
            "DEEPGRAM_API_KEY": "key2",
            "GOOGLE_APPLICATION_CREDENTIALS": "/path",
        },
        clear=True,
    )
    def test_list_all_providers(self):
        """Test list_providers() returns all configured providers."""
        providers = STT.list_providers()
        assert "openai" in providers
        assert "deepgram" in providers
        assert "google" in providers


# =============================================================================
# Test STT list_models() Method
# =============================================================================


class TestSTTListModels:
    """Tests for STT.list_models() method."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_list_openai_models(self):
        """Test list_models() returns OpenAI models when configured."""
        models = STT.list_models("openai")

        assert len(models) > 0
        assert all(m["provider"] == "openai" for m in models)
        assert any(m["model"] == "whisper-1" for m in models)

    @patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}, clear=True)
    def test_list_deepgram_models(self):
        """Test list_models() returns Deepgram models when configured."""
        models = STT.list_models("deepgram")

        assert len(models) > 0
        assert all(m["provider"] == "deepgram" for m in models)
        model_names = [m["model"] for m in models]
        assert "nova-2" in model_names

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_list_google_models(self):
        """Test list_models() returns Google models when configured."""
        models = STT.list_models("google")

        assert len(models) > 0
        assert all(m["provider"] == "google" for m in models)

    @patch.dict("os.environ", {}, clear=True)
    def test_list_models_empty(self):
        """Test list_models() returns empty for unconfigured provider."""
        models = STT.list_models("openai")
        assert models == []

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "key1", "DEEPGRAM_API_KEY": "key2"},
        clear=True,
    )
    def test_list_all_models_when_no_provider(self):
        """Test list_models() returns all models when provider is None."""
        models = STT.list_models()

        providers = {m["provider"] for m in models}
        assert "openai" in providers
        assert "deepgram" in providers


# =============================================================================
# Test STT _load_audio() Method
# =============================================================================


class TestSTTLoadAudio:
    """Tests for STT._load_audio() helper method."""

    def test_load_audio_bytes(self):
        """Test _load_audio() with bytes."""
        stt = STT(provider="openai")
        audio_bytes, filename = stt._load_audio(b"audio data")

        assert audio_bytes == b"audio data"
        assert filename == "audio.mp3"

    def test_load_audio_file_path_string(self):
        """Test _load_audio() with string file path."""
        stt = STT(provider="openai")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"wav content")
            temp_path = f.name

        try:
            audio_bytes, filename = stt._load_audio(temp_path)
            assert audio_bytes == b"wav content"
            assert filename.endswith(".wav")
        finally:
            Path(temp_path).unlink()

    def test_load_audio_file_path_object(self):
        """Test _load_audio() with Path object."""
        stt = STT(provider="openai")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"mp3 content")
            temp_path = Path(f.name)

        try:
            audio_bytes, filename = stt._load_audio(temp_path)
            assert audio_bytes == b"mp3 content"
            assert filename.endswith(".mp3")
        finally:
            temp_path.unlink()

    def test_load_audio_file_not_found(self):
        """Test _load_audio() raises FileNotFoundError for missing file."""
        stt = STT(provider="openai")

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            stt._load_audio("/nonexistent/file.mp3")

    def test_load_audio_unsupported_type(self):
        """Test _load_audio() raises TypeError for unsupported type."""
        stt = STT(provider="openai")

        with pytest.raises(TypeError, match="Unsupported audio type"):
            stt._load_audio(12345)  # type: ignore


# =============================================================================
# Test STT Import Errors
# =============================================================================


class TestSTTImportErrors:
    """Tests for STT import error handling."""

    @patch.dict("sys.modules", {"openai": None})
    def test_openai_import_error(self):
        """Test ImportError when openai package not installed."""
        stt = STT(provider="openai", api_key="test")

        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai package required"):
                stt._get_openai_client()

    def test_deepgram_import_error(self):
        """Test ImportError when deepgram-sdk package not installed."""
        stt = STT(provider="deepgram", api_key="test")

        with patch.dict("sys.modules", {"deepgram": None}):
            with pytest.raises(ImportError, match="deepgram-sdk package required"):
                stt._transcribe_deepgram(b"audio", None, False, False)

    def test_google_import_error(self):
        """Test ImportError when google-cloud-speech package not installed."""
        stt = STT(provider="google")

        with patch.dict(
            "sys.modules",
            {
                "google": None,
                "google.cloud": None,
                "google.cloud.speech": None,
            },
        ):
            with pytest.raises(ImportError, match="google-cloud-speech package required"):
                stt._transcribe_google(b"audio", None, False, False)


# =============================================================================
# Test STT Static Methods
# =============================================================================


class TestSTTStaticMethods:
    """Tests for STT static helper methods."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_detect_provider_static(self):
        """Test static _detect_provider method."""
        provider = STT._detect_provider()
        assert provider == "openai"

    def test_default_model_static(self):
        """Test static _default_model method."""
        model = STT._default_model("openai")
        assert model == "whisper-1"


# =============================================================================
# Test STT API Key Handling
# =============================================================================


class TestSTTAPIKeyHandling:
    """Tests for STT API key handling."""

    def test_deepgram_api_key_from_init(self):
        """Test Deepgram API key from init."""
        stt = STT(provider="deepgram", api_key="custom-dg-key")
        key = stt._get_deepgram_api_key()
        assert key == "custom-dg-key"

    @patch.dict("os.environ", {"DEEPGRAM_API_KEY": "env-dg-key"}, clear=True)
    def test_deepgram_api_key_from_env(self):
        """Test Deepgram API key from environment."""
        stt = STT(provider="deepgram")
        key = stt._get_deepgram_api_key()
        assert key == "env-dg-key"

    @patch.dict("os.environ", {}, clear=True)
    def test_deepgram_api_key_empty_when_missing(self):
        """Test Deepgram API key is empty when not set."""
        stt = STT(provider="deepgram")
        key = stt._get_deepgram_api_key()
        assert key == ""

    def test_openai_client_uses_init_key(self):
        """Test OpenAI client uses init API key."""
        mock_openai = MagicMock()
        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI = mock_openai

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            stt = STT(provider="openai", api_key="custom-openai-key")
            stt._get_openai_client()

            mock_openai.assert_called_once_with(api_key="custom-openai-key")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-openai-key"}, clear=True)
    def test_openai_client_uses_env_key(self):
        """Test OpenAI client uses environment API key."""
        mock_openai = MagicMock()
        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI = mock_openai

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            stt = STT(provider="openai")
            stt._get_openai_client()

            mock_openai.assert_called_once_with(api_key="env-openai-key")


# =============================================================================
# Test TranscriptionResult Model Integration
# =============================================================================


class TestTranscriptionResultIntegration:
    """Integration tests for TranscriptionResult with STT."""

    @patch("ai_infra.llm.multimodal.stt.STT._get_openai_client")
    def test_result_has_all_fields(self, mock_get_client):
        """Test that TranscriptionResult has all expected fields."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Complete result"
        mock_response.language = "en"
        mock_response.duration = 5.5
        mock_response.segments = [
            {"text": "Complete", "start": 0.0, "end": 0.5, "confidence": 0.95},
            {"text": "result", "start": 0.5, "end": 1.0, "confidence": 0.98},
        ]
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        stt = STT(provider="openai")
        result = stt.transcribe(b"audio", timestamps=True)

        assert result.text == "Complete result"
        assert result.language == "en"
        assert result.duration == 5.5
        assert result.provider == STTProvider.OPENAI
        assert len(result.segments) == 2
        assert result.segments[0].text == "Complete"
        assert result.segments[0].start == 0.0
        assert result.segments[0].confidence == 0.95

    def test_transcription_segment_fields(self):
        """Test TranscriptionSegment has correct fields."""
        segment = TranscriptionSegment(
            text="Hello",
            start=0.0,
            end=0.5,
            confidence=0.95,
            speaker="speaker_1",
        )

        assert segment.text == "Hello"
        assert segment.start == 0.0
        assert segment.end == 0.5
        assert segment.confidence == 0.95
        assert segment.speaker == "speaker_1"
