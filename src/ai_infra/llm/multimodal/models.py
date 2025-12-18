"""Models and types for multimodal support."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class AudioFormat(StrEnum):
    """Supported audio formats."""

    MP3 = "mp3"
    WAV = "wav"
    OPUS = "opus"
    FLAC = "flac"
    PCM = "pcm"
    AAC = "aac"
    OGG = "ogg"
    WEBM = "webm"


class TTSProvider(StrEnum):
    """Supported TTS providers."""

    OPENAI = "openai"
    GOOGLE = "google"
    ELEVENLABS = "elevenlabs"


class STTProvider(StrEnum):
    """Supported STT providers."""

    OPENAI = "openai"
    GOOGLE = "google"
    DEEPGRAM = "deepgram"


@dataclass
class Voice:
    """Represents a TTS voice.

    Attributes:
        id: Voice identifier (provider-specific).
        name: Human-readable voice name.
        provider: The provider this voice belongs to.
        language: Language code (e.g., 'en-US').
        gender: Voice gender ('male', 'female', 'neutral').
        description: Optional description of the voice.
        preview_url: Optional URL to preview the voice.
    """

    id: str
    name: str
    provider: TTSProvider
    language: str = "en-US"
    gender: str = "neutral"
    description: str | None = None
    preview_url: str | None = None


@dataclass
class AudioSegment:
    """Represents audio data.

    Attributes:
        data: Raw audio bytes.
        format: Audio format (mp3, wav, etc.).
        sample_rate: Sample rate in Hz.
        duration: Duration in seconds (if known).
        metadata: Additional metadata.
    """

    data: bytes
    format: AudioFormat = AudioFormat.MP3
    sample_rate: int = 24000
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save audio to a file.

        Args:
            path: File path to save to.
        """
        path = Path(path)
        with open(path, "wb") as f:
            f.write(self.data)

    @classmethod
    def from_file(cls, path: str | Path, format: AudioFormat | None = None) -> AudioSegment:
        """Load audio from a file.

        Args:
            path: File path to load from.
            format: Audio format (auto-detected from extension if not provided).

        Returns:
            AudioSegment with the file data.
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = f.read()

        if format is None:
            ext = path.suffix.lower().lstrip(".")
            try:
                format = AudioFormat(ext)
            except ValueError:
                format = AudioFormat.MP3  # Default

        return cls(data=data, format=format)


@dataclass
class TranscriptionWord:
    """A single word in a transcription with timing.

    Attributes:
        word: The transcribed word.
        start: Start time in seconds.
        end: End time in seconds.
        confidence: Confidence score (0-1).
    """

    word: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class TranscriptionSegment:
    """A segment of transcription (sentence/phrase).

    Attributes:
        text: The transcribed text.
        start: Start time in seconds.
        end: End time in seconds.
        words: Word-level timing (if available).
        confidence: Confidence score (0-1).
        speaker: Speaker ID (if diarization enabled).
    """

    text: str
    start: float
    end: float
    words: list[TranscriptionWord] = field(default_factory=list)
    confidence: float | None = None
    speaker: str | None = None


@dataclass
class TranscriptionResult:
    """Result of a speech-to-text transcription.

    Attributes:
        text: The full transcribed text.
        segments: Timed segments (if available).
        language: Detected language code.
        duration: Audio duration in seconds.
        model: Model used for transcription.
        provider: Provider used.
        metadata: Additional provider-specific metadata.
    """

    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None
    model: str | None = None
    provider: STTProvider | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def words(self) -> list[TranscriptionWord]:
        """Get all words from all segments."""
        if not self.segments:
            return []
        result = []
        for segment in self.segments:
            result.extend(segment.words)
        return result


# Default models per provider
TTS_DEFAULT_MODELS = {
    TTSProvider.OPENAI: "tts-1",
    TTSProvider.GOOGLE: "en-US-Neural2-A",
    TTSProvider.ELEVENLABS: "eleven_monolingual_v1",
}

STT_DEFAULT_MODELS = {
    STTProvider.OPENAI: "whisper-1",
    STTProvider.GOOGLE: "latest_long",
    STTProvider.DEEPGRAM: "nova-2",
}

# Available voices per provider
OPENAI_VOICES = [
    Voice(id="alloy", name="Alloy", provider=TTSProvider.OPENAI, gender="neutral"),
    Voice(id="echo", name="Echo", provider=TTSProvider.OPENAI, gender="male"),
    Voice(id="fable", name="Fable", provider=TTSProvider.OPENAI, gender="neutral"),
    Voice(id="onyx", name="Onyx", provider=TTSProvider.OPENAI, gender="male"),
    Voice(id="nova", name="Nova", provider=TTSProvider.OPENAI, gender="female"),
    Voice(id="shimmer", name="Shimmer", provider=TTSProvider.OPENAI, gender="female"),
]

# Environment variables for each provider
TTS_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
}

STT_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
}
