"""Audio input support for LLM - simple audio input for any provider.

This module provides a simple, provider-agnostic API for audio input
that works with any LLM provider that supports audio. It uses LangChain's
standardized content block format internally for maximum compatibility.

Supported providers (automatic):
- OpenAI (GPT-4o, GPT-4o-audio-preview)
- Google (Gemini 1.5, Gemini 2.0)
- Anthropic (Claude - via transcription fallback)

Example:
    ```python
    from ai_infra.llm import LLM

    llm = LLM()

    # Simple: Just pass audio file path
    response = llm.chat(
        "What is being said in this audio?",
        audio="recording.mp3"
    )

    # Audio URL
    response = llm.chat(
        "Transcribe and summarize this",
        audio="https://example.com/audio.mp3"
    )

    # Raw bytes (e.g., from microphone)
    audio_bytes = record_audio()
    response = llm.chat("What did I say?", audio=audio_bytes)
    ```
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

# Type alias for audio inputs we accept
AudioInput = str | bytes | Path

# Supported audio MIME types
AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
    ".opus": "audio/opus",
    ".pcm": "audio/pcm",
}

# OpenAI only supports these formats for audio input
OPENAI_SUPPORTED_FORMATS = {"mp3", "wav"}


# =============================================================================
# Public API - Simple and Clean
# =============================================================================


def create_audio_message(
    text: str,
    audio: AudioInput,
) -> HumanMessage:
    """Create a LangChain HumanMessage with text and audio.

    This is the recommended way to create audio messages. It returns
    a LangChain HumanMessage that works with any audio-capable model.

    Args:
        text: The text prompt/question about the audio.
        audio: Audio - can be URL, file path, or raw bytes.

    Returns:
        HumanMessage ready for model.invoke().

    Example:
        ```python
        from ai_infra.llm.multimodal import create_audio_message
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o-audio-preview")
        msg = create_audio_message("What's in this audio?", "audio.mp3")
        response = model.invoke([msg])
        ```
    """
    content = build_audio_content(text, audio)
    # HumanMessage expects list[str | dict[Any, Any]] but we have list[dict[str, Any]]
    # These are effectively compatible - dict[str, Any] is more specific
    return HumanMessage(content=content)  # type: ignore[arg-type]


def build_audio_content(
    text: str,
    audio: AudioInput,
) -> list[dict[str, Any]]:
    """Build content blocks for an audio message.

    Creates a list of content blocks in LangChain's standard format.
    This format is automatically converted to provider-specific formats
    by LangChain's chat model adapters.

    Args:
        text: The text prompt/question.
        audio: Audio (URL, path, or bytes).

    Returns:
        List of content blocks ready for HumanMessage.

    Example:
        ```python
        content = build_audio_content(
            "What's being said?",
            "recording.mp3"
        )
        # [
        #     {"type": "text", "text": "What's being said?"},
        #     {"type": "input_audio", "input_audio": {"data": "...", "format": "mp3"}},
        # ]
        ```
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.append(encode_audio(audio))
    return content


def encode_audio(audio: AudioInput) -> dict[str, Any]:
    """Encode a single audio file to LangChain's standard format.

    Automatically detects the audio type and encodes appropriately:
    - URLs are passed through directly
    - File paths are read and base64 encoded with correct format
    - Raw bytes are base64 encoded

    Args:
        audio: URL string, file path, Path object, or raw bytes.

    Returns:
        Content block dict in input_audio format.

    Example:
        ```python
        # File path
        encode_audio("recording.mp3")
        # {"type": "input_audio", "input_audio": {"data": "...", "format": "mp3"}}

        # Bytes
        encode_audio(audio_bytes)
        # {"type": "input_audio", "input_audio": {"data": "...", "format": "mp3"}}
        ```
    """
    if isinstance(audio, str):
        return _encode_string_audio(audio)
    elif isinstance(audio, bytes):
        return _encode_bytes_audio(audio)
    elif isinstance(audio, Path):
        return _encode_path_audio(audio)
    else:
        raise TypeError(
            f"Unsupported audio type: {type(audio)}. Expected str (URL or path), bytes, or Path."
        )


def encode_audio_for_openai(audio: AudioInput) -> dict[str, Any]:
    """Encode audio specifically for OpenAI's format.

    OpenAI uses:
    - input_audio block with base64 data and format

    Args:
        audio: Audio input (URL, path, or bytes).

    Returns:
        Content block in OpenAI's audio format.
    """
    if isinstance(audio, str) and _is_url(audio):
        # For URLs, we need to download and encode
        # For now, just pass the URL - OpenAI may support this in future
        raise NotImplementedError(
            "OpenAI audio input currently requires file path or bytes, not URLs. "
            "Download the audio first."
        )
    return encode_audio(audio)


def encode_audio_for_google(audio: AudioInput) -> dict[str, Any]:
    """Encode audio specifically for Google's Gemini format.

    Google uses inline_data format for audio.

    Args:
        audio: Audio input (URL, path, or bytes).

    Returns:
        Content block in Google's audio format.
    """
    if isinstance(audio, bytes):
        b64_data = base64.standard_b64encode(audio).decode("utf-8")
        return {
            "type": "media",
            "mime_type": "audio/mp3",  # Default
            "data": b64_data,
        }
    elif isinstance(audio, (str, Path)):
        path = Path(audio) if isinstance(audio, str) else audio
        if path.exists():
            mime_type = _get_audio_mime_type(path)
            with open(path, "rb") as f:
                data = f.read()
            b64_data = base64.standard_b64encode(data).decode("utf-8")
            return {
                "type": "media",
                "mime_type": mime_type,
                "data": b64_data,
            }
        elif _is_url(str(audio)):
            return {
                "type": "media",
                "file_uri": str(audio),
            }
    raise ValueError(f"Cannot encode audio: {audio}")


# =============================================================================
# Internal Helpers
# =============================================================================


def _is_url(s: str) -> bool:
    """Check if string is a URL."""
    return s.startswith(("http://", "https://", "data:"))


def _get_audio_mime_type(path: Path) -> str:
    """Get MIME type from file extension."""
    ext = path.suffix.lower()
    if ext in AUDIO_MIME_TYPES:
        return AUDIO_MIME_TYPES[ext]
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "audio/mpeg"


def _get_audio_format(path: Path) -> str:
    """Get audio format from file extension."""
    ext = path.suffix.lower().lstrip(".")
    return ext if ext else "mp3"


def _encode_string_audio(audio: str) -> dict[str, Any]:
    """Encode a string audio (URL or file path)."""
    if _is_url(audio):
        # For URLs, use file_uri format (some providers support this)
        # Note: OpenAI may not support URLs directly for audio
        return {
            "type": "input_audio",
            "input_audio": {
                "url": audio,
            },
        }
    else:
        # Treat as file path
        return _encode_path_audio(Path(audio))


def _encode_bytes_audio(audio: bytes, audio_format: str = "mp3") -> dict[str, Any]:
    """Encode raw bytes to base64 input_audio format."""
    b64_data = base64.standard_b64encode(audio).decode("utf-8")
    return {
        "type": "input_audio",
        "input_audio": {
            "data": b64_data,
            "format": audio_format,
        },
    }


def _encode_path_audio(path: Path) -> dict[str, Any]:
    """Encode a file path to base64 input_audio format.

    Automatically converts unsupported formats (m4a, aac, ogg, etc.) to mp3
    using ffmpeg if available.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio_format = _get_audio_format(path)

    # Check if format needs conversion for OpenAI compatibility
    if audio_format not in OPENAI_SUPPORTED_FORMATS:
        converted_data = _convert_to_mp3(path)
        if converted_data is not None:
            return _encode_bytes_audio(converted_data, "mp3")
        # If conversion failed, try anyway (might work with other providers)

    with open(path, "rb") as f:
        data = f.read()

    return _encode_bytes_audio(data, audio_format)


def _convert_to_mp3(path: Path) -> bytes | None:
    """Convert audio file to mp3 using ffmpeg.

    Returns mp3 bytes if successful, None if ffmpeg not available or conversion fails.
    """
    import shutil
    import subprocess
    import tempfile

    # Check if ffmpeg is available
    if not shutil.which("ffmpeg"):
        import warnings

        warnings.warn(
            f"Audio format '{path.suffix}' may not be supported. "
            "Install ffmpeg for automatic conversion to mp3."
        )
        return None

    try:
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        # Run ffmpeg conversion
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(path),
                "-y",  # Overwrite output
                "-vn",  # No video
                "-acodec",
                "libmp3lame",
                "-q:a",
                "2",  # High quality
                tmp_path,
            ],
            capture_output=True,
            timeout=60,  # 60 second timeout
        )

        if result.returncode == 0:
            with open(tmp_path, "rb") as f:
                mp3_data = f.read()
            return mp3_data
        else:
            import warnings

            warnings.warn(f"ffmpeg conversion failed: {result.stderr.decode()[:200]}")
            return None
    except subprocess.TimeoutExpired:
        import warnings

        warnings.warn("ffmpeg conversion timed out")
        return None
    except Exception as e:
        import warnings

        warnings.warn(f"Audio conversion failed: {e}")
        return None
    finally:
        # Clean up temp file
        import os

        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# =============================================================================
# Backwards Compatibility (deprecated - use new API)
# =============================================================================

# Alias for backwards compatibility
make_audio_message = create_audio_message
