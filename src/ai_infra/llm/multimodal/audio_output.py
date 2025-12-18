"""Audio output support for LLM - get audio responses from LLMs.

This module provides audio output capabilities from LLMs that support
native audio generation (like GPT-4o-audio-preview).

Example:
    ```python
    from ai_infra.llm import LLM
    from ai_infra.llm.multimodal import AudioOutput

    llm = LLM()

    # Get audio response
    result = llm.chat(
        "Say hello in a friendly voice",
        model_name="gpt-4o-audio-preview",
        audio_output=AudioOutput(voice="alloy")
    )

    # Access audio data
    if result.audio:
        with open("response.mp3", "wb") as f:
            f.write(result.audio.data)
    ```
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
from typing import Any


class AudioVoice(str, Enum):
    """Available voices for audio output."""

    # OpenAI voices
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"

    # Google voices (Gemini)
    PUCK = "Puck"
    CHARON = "Charon"
    KORE = "Kore"
    FENRIR = "Fenrir"
    AOEDE = "Aoede"


class AudioOutputFormat(str, Enum):
    """Audio output formats."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OPUS = "opus"
    PCM16 = "pcm16"


@dataclass
class AudioOutput:
    """Configuration for audio output from LLMs.

    Use this to request audio responses from audio-capable models.

    Args:
        voice: The voice to use (e.g., "alloy", "nova", "shimmer").
        format: Audio format for output (default: "mp3").

    Example:
        ```python
        from ai_infra.llm import LLM
        from ai_infra.llm.multimodal import AudioOutput

        llm = LLM()
        result = llm.chat(
            "Tell me a joke",
            model_name="gpt-4o-audio-preview",
            audio_output=AudioOutput(voice="nova")
        )
        ```
    """

    voice: str = "alloy"
    format: str = "mp3"

    def to_openai_modalities(self) -> dict[str, Any]:
        """Convert to OpenAI modalities format."""
        return {
            "modalities": ["text", "audio"],
            "audio": {
                "voice": self.voice,
                "format": self.format,
            },
        }


@dataclass
class AudioResponse:
    """Audio response data from LLM.

    Contains the audio data and metadata from an audio-capable model response.
    """

    data: bytes
    """Raw audio bytes."""

    format: str = "mp3"
    """Audio format (mp3, wav, etc.)."""

    transcript: str | None = None
    """Text transcript of the audio (if available)."""

    id: str | None = None
    """Audio response ID from the API."""

    expires_at: int | None = None
    """Unix timestamp when this audio expires (for OpenAI)."""

    @classmethod
    def from_openai_response(cls, audio_data: dict[str, Any]) -> AudioResponse:
        """Create AudioResponse from OpenAI API response.

        Args:
            audio_data: The 'audio' field from OpenAI's response.

        Returns:
            AudioResponse with decoded audio data.
        """
        # Decode base64 audio data
        raw_data = base64.b64decode(audio_data.get("data", ""))

        return cls(
            data=raw_data,
            format=audio_data.get("format", "mp3"),
            transcript=audio_data.get("transcript"),
            id=audio_data.get("id"),
            expires_at=audio_data.get("expires_at"),
        )

    def save(self, path: str) -> None:
        """Save audio to a file.

        Args:
            path: File path to save audio to.
        """
        with open(path, "wb") as f:
            f.write(self.data)


def parse_audio_response(response: Any) -> AudioResponse | None:
    """Parse audio from LLM response if present.

    Args:
        response: The LLM response object.

    Returns:
        AudioResponse if audio is present, None otherwise.
    """
    # Check for OpenAI-style response with audio
    if hasattr(response, "additional_kwargs"):
        audio_data = response.additional_kwargs.get("audio")
        if audio_data and isinstance(audio_data, dict):
            return AudioResponse.from_openai_response(audio_data)

    # Check for raw response dict
    if isinstance(response, dict):
        audio_data = response.get("audio")
        if audio_data and isinstance(audio_data, dict):
            return AudioResponse.from_openai_response(audio_data)

    return None


# Available voices by provider
VOICES_BY_PROVIDER = {
    "openai": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "google": ["Puck", "Charon", "Kore", "Fenrir", "Aoede"],
}


def list_audio_voices(provider: str = "openai") -> list[str]:
    """List available voices for a provider.

    Args:
        provider: The provider name (openai, google).

    Returns:
        List of available voice names.
    """
    return VOICES_BY_PROVIDER.get(provider, [])


# Audio-capable models by provider
AUDIO_MODELS = {
    "openai": ["gpt-4o-audio-preview", "gpt-4o-realtime-preview"],
    "google": ["gemini-2.0-flash-exp"],  # Gemini with audio output
}


def get_audio_model(provider: str = "openai") -> str:
    """Get the default audio-capable model for a provider.

    Args:
        provider: The provider name.

    Returns:
        Model name that supports audio output.

    Raises:
        ValueError: If provider doesn't support audio output.
    """
    models = AUDIO_MODELS.get(provider)
    if not models:
        raise ValueError(
            f"Provider '{provider}' doesn't have audio output models. "
            f"Supported: {list(AUDIO_MODELS.keys())}"
        )
    return models[0]


def is_audio_model(model_name: str, provider: str = "openai") -> bool:
    """Check if a model supports audio output.

    Args:
        model_name: The model name to check.
        provider: The provider name.

    Returns:
        True if the model supports audio output.
    """
    provider_models = AUDIO_MODELS.get(provider, [])
    return model_name in provider_models
