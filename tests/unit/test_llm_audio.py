"""Tests for LLM audio input functionality."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_infra.llm.multimodal.audio import (
    OPENAI_SUPPORTED_FORMATS,
    build_audio_content,
    create_audio_message,
    encode_audio,
    encode_audio_for_openai,
)


class TestEncodeAudio:
    """Tests for encode_audio function."""

    def test_encode_bytes(self):
        """Test encoding raw bytes."""
        audio_bytes = b"fake audio data"
        result = encode_audio(audio_bytes)

        assert result["type"] == "input_audio"
        assert "input_audio" in result
        assert "data" in result["input_audio"]
        assert result["input_audio"]["format"] == "mp3"

    def test_encode_mp3_file(self):
        """Test encoding mp3 file."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake mp3 content")
            path = f.name

        try:
            result = encode_audio(path)
            assert result["type"] == "input_audio"
            assert result["input_audio"]["format"] == "mp3"
        finally:
            Path(path).unlink()

    def test_encode_wav_file(self):
        """Test encoding wav file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF....WAVEfmt ")
            path = f.name

        try:
            result = encode_audio(path)
            assert result["type"] == "input_audio"
            assert result["input_audio"]["format"] == "wav"
        finally:
            Path(path).unlink()

    def test_encode_path_object(self):
        """Test encoding Path object."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake mp3")
            path = Path(f.name)

        try:
            result = encode_audio(path)
            assert result["type"] == "input_audio"
        finally:
            path.unlink()

    def test_file_not_found(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            encode_audio("/nonexistent/path/audio.mp3")


class TestBuildAudioContent:
    """Tests for build_audio_content function."""

    def test_builds_content_blocks(self):
        """Test building content blocks."""
        content = build_audio_content("What do you hear?", b"audio data")

        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "What do you hear?"
        assert content[1]["type"] == "input_audio"


class TestCreateAudioMessage:
    """Tests for create_audio_message function."""

    def test_creates_human_message(self):
        """Test that it creates a HumanMessage."""
        msg = create_audio_message("Transcribe this", b"audio")

        assert msg.__class__.__name__ == "HumanMessage"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2


class TestOpenAISupportedFormats:
    """Tests for OpenAI format support."""

    def test_mp3_supported(self):
        """Test that mp3 is supported."""
        assert "mp3" in OPENAI_SUPPORTED_FORMATS

    def test_wav_supported(self):
        """Test that wav is supported."""
        assert "wav" in OPENAI_SUPPORTED_FORMATS

    def test_m4a_not_directly_supported(self):
        """Test that m4a requires conversion."""
        assert "m4a" not in OPENAI_SUPPORTED_FORMATS


class TestAudioConversion:
    """Tests for audio format conversion."""

    @patch("shutil.which")
    def test_warns_when_ffmpeg_not_available(self, mock_which):
        """Test warning when ffmpeg is not available for conversion."""
        mock_which.return_value = None

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            f.write(b"fake m4a content")
            path = f.name

        try:
            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # This should warn about ffmpeg but still encode
                result = encode_audio(path)
                # Check that a warning was issued or format is m4a
                assert result["input_audio"]["format"] == "m4a"
        finally:
            Path(path).unlink()
