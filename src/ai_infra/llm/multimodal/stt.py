"""Speech-to-Text (STT) support for ai-infra.

This module provides a unified API for speech-to-text across multiple providers:
- OpenAI Whisper (whisper-1)
- Google Cloud Speech-to-Text
- Deepgram

Example:
    ```python
    from ai_infra.llm.multimodal import STT

    # Basic usage - auto-detect provider
    stt = STT()
    result = stt.transcribe("audio.mp3")
    print(result.text)

    # Transcribe with timestamps
    result = stt.transcribe("audio.mp3", timestamps=True)
    for segment in result.segments:
        print(f"{segment.start:.2f}s: {segment.text}")

    # With specific provider
    stt = STT(provider="openai", language="en")
    result = stt.transcribe("audio.mp3")

    # Async usage
    result = await stt.atranscribe("audio.mp3")
    ```
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ai_infra.llm.multimodal.models import (
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
)
from ai_infra.providers import ProviderCapability, ProviderRegistry

# Provider aliases for backwards compatibility
_PROVIDER_ALIASES = {"google": "google_genai"}


def _get_default_model(provider: str) -> str:
    """Get default model for provider from registry."""
    # Resolve alias
    name = _PROVIDER_ALIASES.get(provider, provider)
    config = ProviderRegistry.get(name)
    if config:
        cap = config.get_capability(ProviderCapability.STT)
        if cap and cap.default_model:
            return cap.default_model
    # Fallback defaults
    return {"openai": "whisper-1", "deepgram": "nova-2", "google": "default"}.get(
        provider, "default"
    )


def _detect_stt_provider() -> str:
    """Detect available STT provider from environment using registry."""
    # Priority order for STT providers
    priority = ["openai", "deepgram", "google_genai"]
    for name in priority:
        if ProviderRegistry.is_configured(name):
            # Return user-facing name (without _genai suffix)
            return "google" if name == "google_genai" else name
    raise ValueError(
        "No STT provider configured. Set OPENAI_API_KEY, DEEPGRAM_API_KEY, "
        "or GOOGLE_APPLICATION_CREDENTIALS."
    )


# Provider priority for auto-detection (legacy constant)
STT_PROVIDER_PRIORITY = ["openai", "deepgram", "google"]


class STT:
    """Speech-to-Text with provider-agnostic API.

    Supports OpenAI Whisper, Google Cloud Speech-to-Text, and Deepgram.
    Auto-detects provider based on available API keys if not specified.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        language: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize STT.

        Args:
            provider: STT provider ("openai", "google", "deepgram").
                     Auto-detected if None.
            model: Model name (provider-specific). Uses default if None.
            language: Language code (e.g., "en", "es"). Auto-detect if None.
            api_key: API key (uses environment variable if None).
        """
        self._provider = provider or _detect_stt_provider()
        self._model = model or _get_default_model(self._provider)
        self._language = language
        self._api_key = api_key

    @staticmethod
    def _detect_provider() -> str:
        """Detect available STT provider from environment."""
        return _detect_stt_provider()

    @staticmethod
    def _default_model(provider: str) -> str:
        """Get default model for provider."""
        return _get_default_model(provider)

    @property
    def provider(self) -> str:
        """Get the current provider name."""
        return self._provider

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    @property
    def language(self) -> str | None:
        """Get the language setting."""
        return self._language

    def transcribe(
        self,
        audio: bytes | str | Path,
        *,
        language: str | None = None,
        timestamps: bool = False,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio data as bytes, file path, or URL.
            language: Override language setting.
            timestamps: Include segment timestamps.
            word_timestamps: Include word-level timestamps (if supported).
            prompt: Optional prompt to guide transcription.

        Returns:
            TranscriptionResult with text and optional segments.
        """
        language = language or self._language

        if self._provider == "openai":
            return self._transcribe_openai(audio, language, timestamps, word_timestamps, prompt)
        elif self._provider == "deepgram":
            return self._transcribe_deepgram(audio, language, timestamps, word_timestamps)
        elif self._provider == "google":
            return self._transcribe_google(audio, language, timestamps, word_timestamps)
        else:
            raise ValueError(f"Unsupported STT provider: {self._provider}")

    async def atranscribe(
        self,
        audio: bytes | str | Path,
        *,
        language: str | None = None,
        timestamps: bool = False,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Async version of transcribe().

        Args:
            audio: Audio data as bytes, file path, or URL.
            language: Override language setting.
            timestamps: Include segment timestamps.
            word_timestamps: Include word-level timestamps (if supported).
            prompt: Optional prompt to guide transcription.

        Returns:
            TranscriptionResult with text and optional segments.
        """
        language = language or self._language

        if self._provider == "openai":
            return await self._atranscribe_openai(
                audio, language, timestamps, word_timestamps, prompt
            )
        elif self._provider == "deepgram":
            return await self._atranscribe_deepgram(audio, language, timestamps, word_timestamps)
        elif self._provider == "google":
            return await self._atranscribe_google(audio, language, timestamps, word_timestamps)
        else:
            raise ValueError(f"Unsupported STT provider: {self._provider}")

    def transcribe_file(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        timestamps: bool = False,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            path: Path to audio file.
            language: Override language setting.
            timestamps: Include segment timestamps.
            word_timestamps: Include word-level timestamps.
            prompt: Optional prompt to guide transcription.

        Returns:
            TranscriptionResult with text and optional segments.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        return self.transcribe(
            path,
            language=language,
            timestamps=timestamps,
            word_timestamps=word_timestamps,
            prompt=prompt,
        )

    def stream_transcribe(
        self,
        audio_stream: Iterator[bytes],
        *,
        language: str | None = None,
    ) -> Iterator[str]:
        """Real-time transcription from audio stream.

        Note: Not all providers support streaming. Falls back to batched
        transcription if streaming is not available.

        Args:
            audio_stream: Iterator yielding audio chunks.
            language: Override language setting.

        Yields:
            Transcribed text segments as they become available.
        """
        language = language or self._language

        if self._provider == "deepgram":
            yield from self._stream_deepgram(audio_stream, language)
        else:
            # Fallback: accumulate all audio and transcribe at once
            audio_data = b"".join(audio_stream)
            result = self.transcribe(audio_data, language=language)
            yield result.text

    @staticmethod
    def list_providers() -> list[str]:
        """List configured STT providers.

        Returns:
            List of provider names that have API keys configured.
        """
        providers = []
        if os.environ.get("OPENAI_API_KEY"):
            providers.append("openai")
        if os.environ.get("DEEPGRAM_API_KEY"):
            providers.append("deepgram")
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_API_KEY"):
            providers.append("google")
        return providers

    @staticmethod
    def list_models(provider: str | None = None) -> list[dict[str, Any]]:
        """List available models for a provider.

        Args:
            provider: Provider name. If None, lists for all configured providers.

        Returns:
            List of model info dicts.
        """
        models: list[dict[str, Any]] = []

        if provider is None or provider == "openai":
            if os.environ.get("OPENAI_API_KEY"):
                models.append(
                    {
                        "provider": "openai",
                        "model": "whisper-1",
                        "description": "OpenAI Whisper - General-purpose speech recognition",
                        "languages": "99+ languages",
                    }
                )

        if provider is None or provider == "deepgram":
            if os.environ.get("DEEPGRAM_API_KEY"):
                deepgram_models = [
                    ("nova-2", "Nova-2 - Most accurate, latest generation"),
                    ("nova", "Nova - Previous generation"),
                    ("enhanced", "Enhanced - High accuracy"),
                    ("base", "Base - Fast, efficient"),
                ]
                for model, desc in deepgram_models:
                    models.append(
                        {
                            "provider": "deepgram",
                            "model": model,
                            "description": desc,
                            "languages": "30+ languages",
                        }
                    )

        if provider is None or provider == "google":
            if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_API_KEY"):
                models.append(
                    {
                        "provider": "google",
                        "model": "default",
                        "description": "Google Cloud Speech-to-Text",
                        "languages": "125+ languages",
                    }
                )

        return models

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _load_audio(self, audio: bytes | str | Path) -> tuple[bytes, str]:
        """Load audio data and determine format.

        Returns:
            Tuple of (audio_bytes, filename).
        """
        if isinstance(audio, bytes):
            return audio, "audio.mp3"
        elif isinstance(audio, (str, Path)):
            path = Path(audio)
            if path.exists():
                return path.read_bytes(), path.name
            # Could be a URL, but we don't handle that yet
            raise FileNotFoundError(f"Audio file not found: {path}")
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

    # =========================================================================
    # OpenAI Whisper Implementation
    # =========================================================================

    def _get_openai_client(self):
        """Get OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for OpenAI STT: pip install openai")

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAI(api_key=api_key)

    def _get_openai_async_client(self):
        """Get async OpenAI client."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required for OpenAI STT: pip install openai")

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        return AsyncOpenAI(api_key=api_key)

    def _transcribe_openai(
        self,
        audio: bytes | str | Path,
        language: str | None,
        timestamps: bool,
        word_timestamps: bool,
        prompt: str | None,
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper."""
        client = self._get_openai_client()
        audio_bytes, filename = self._load_audio(audio)

        # Build kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "file": (filename, audio_bytes),
        }

        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        # Request verbose output for timestamps
        if timestamps or word_timestamps:
            kwargs["response_format"] = "verbose_json"
            if word_timestamps:
                kwargs["timestamp_granularities"] = ["word", "segment"]
            else:
                kwargs["timestamp_granularities"] = ["segment"]

        response = client.audio.transcriptions.create(**kwargs)

        # Parse response based on format
        if timestamps or word_timestamps:
            # verbose_json response has segments
            segments = []
            if hasattr(response, "segments"):
                for seg in response.segments:
                    segments.append(
                        TranscriptionSegment(
                            text=seg.get("text", ""),
                            start=seg.get("start", 0.0),
                            end=seg.get("end", 0.0),
                            confidence=seg.get("confidence"),
                        )
                    )

            return TranscriptionResult(
                text=response.text,
                segments=segments,
                language=getattr(response, "language", language),
                duration=getattr(response, "duration", None),
                provider=STTProvider.OPENAI,
            )
        else:
            return TranscriptionResult(
                text=response.text,
                language=language,
                provider=STTProvider.OPENAI,
            )

    async def _atranscribe_openai(
        self,
        audio: bytes | str | Path,
        language: str | None,
        timestamps: bool,
        word_timestamps: bool,
        prompt: str | None,
    ) -> TranscriptionResult:
        """Async transcribe using OpenAI Whisper."""
        client = self._get_openai_async_client()
        audio_bytes, filename = self._load_audio(audio)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "file": (filename, audio_bytes),
        }

        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        if timestamps or word_timestamps:
            kwargs["response_format"] = "verbose_json"
            if word_timestamps:
                kwargs["timestamp_granularities"] = ["word", "segment"]
            else:
                kwargs["timestamp_granularities"] = ["segment"]

        response = await client.audio.transcriptions.create(**kwargs)

        if timestamps or word_timestamps:
            segments = []
            if hasattr(response, "segments"):
                for seg in response.segments:
                    segments.append(
                        TranscriptionSegment(
                            text=seg.get("text", ""),
                            start=seg.get("start", 0.0),
                            end=seg.get("end", 0.0),
                            confidence=seg.get("confidence"),
                        )
                    )

            return TranscriptionResult(
                text=response.text,
                segments=segments,
                language=getattr(response, "language", language),
                duration=getattr(response, "duration", None),
                provider=STTProvider.OPENAI,
            )
        else:
            return TranscriptionResult(
                text=response.text,
                language=language,
                provider=STTProvider.OPENAI,
            )

    # =========================================================================
    # Deepgram Implementation
    # =========================================================================

    def _get_deepgram_api_key(self) -> str:
        """Get Deepgram API key."""
        return self._api_key or os.environ.get("DEEPGRAM_API_KEY") or ""

    def _transcribe_deepgram(
        self,
        audio: bytes | str | Path,
        language: str | None,
        timestamps: bool,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        """Transcribe using Deepgram."""
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
        except ImportError:
            raise ImportError(
                "deepgram-sdk package required for Deepgram STT: pip install deepgram-sdk"
            )

        client = DeepgramClient(self._get_deepgram_api_key())
        audio_bytes, _ = self._load_audio(audio)

        options = PrerecordedOptions(
            model=self._model,
            smart_format=True,
            punctuate=True,
        )

        if language:
            options.language = language

        if timestamps or word_timestamps:
            options.utterances = True

        source = {"buffer": audio_bytes}
        response = client.listen.prerecorded.v("1").transcribe_file(source, options)

        # Parse Deepgram response
        result = response.results
        channels = result.channels if result else []
        text_parts = []
        segments = []

        for channel in channels:
            for alt in channel.alternatives:
                text_parts.append(alt.transcript)

                if timestamps and hasattr(alt, "words"):
                    # Create segments from utterances or words
                    current_segment_text = []
                    segment_start = None
                    segment_end = None

                    for word in alt.words:
                        if segment_start is None:
                            segment_start = word.start
                        segment_end = word.end
                        current_segment_text.append(word.word)

                        # Create segment on punctuation or at regular intervals
                        if (word.punctuated_word and word.punctuated_word[-1] in ".!?") or len(
                            current_segment_text
                        ) >= 20:
                            segments.append(
                                TranscriptionSegment(
                                    text=" ".join(current_segment_text),
                                    start=segment_start,
                                    end=segment_end,
                                    confidence=word.confidence,
                                )
                            )
                            current_segment_text = []
                            segment_start = None

                    # Add remaining text
                    if current_segment_text:
                        segments.append(
                            TranscriptionSegment(
                                text=" ".join(current_segment_text),
                                start=segment_start or 0.0,
                                end=segment_end or 0.0,
                            )
                        )

        full_text = " ".join(text_parts)

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=language,
            duration=result.metadata.duration if result and result.metadata else None,
            provider=STTProvider.DEEPGRAM,
        )

    async def _atranscribe_deepgram(
        self,
        audio: bytes | str | Path,
        language: str | None,
        timestamps: bool,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        """Async transcribe using Deepgram."""
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
        except ImportError:
            raise ImportError(
                "deepgram-sdk package required for Deepgram STT: pip install deepgram-sdk"
            )

        client = DeepgramClient(self._get_deepgram_api_key())
        audio_bytes, _ = self._load_audio(audio)

        options = PrerecordedOptions(
            model=self._model,
            smart_format=True,
            punctuate=True,
        )

        if language:
            options.language = language

        if timestamps or word_timestamps:
            options.utterances = True

        source = {"buffer": audio_bytes}
        response = await client.listen.asyncprerecorded.v("1").transcribe_file(source, options)

        # Parse response same as sync version
        result = response.results
        channels = result.channels if result else []
        text_parts = []
        segments = []

        for channel in channels:
            for alt in channel.alternatives:
                text_parts.append(alt.transcript)

                if timestamps and hasattr(alt, "words"):
                    current_segment_text = []
                    segment_start = None
                    segment_end = None

                    for word in alt.words:
                        if segment_start is None:
                            segment_start = word.start
                        segment_end = word.end
                        current_segment_text.append(word.word)

                        if (word.punctuated_word and word.punctuated_word[-1] in ".!?") or len(
                            current_segment_text
                        ) >= 20:
                            segments.append(
                                TranscriptionSegment(
                                    text=" ".join(current_segment_text),
                                    start=segment_start,
                                    end=segment_end,
                                    confidence=word.confidence,
                                )
                            )
                            current_segment_text = []
                            segment_start = None

                    if current_segment_text:
                        segments.append(
                            TranscriptionSegment(
                                text=" ".join(current_segment_text),
                                start=segment_start or 0.0,
                                end=segment_end or 0.0,
                            )
                        )

        full_text = " ".join(text_parts)

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=language,
            duration=result.metadata.duration if result and result.metadata else None,
            provider=STTProvider.DEEPGRAM,
        )

    def _stream_deepgram(
        self,
        audio_stream: Iterator[bytes],
        language: str | None,
    ) -> Iterator[str]:
        """Stream transcription using Deepgram."""
        try:
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
        except ImportError:
            raise ImportError(
                "deepgram-sdk package required for Deepgram STT: pip install deepgram-sdk"
            )

        client = DeepgramClient(self._get_deepgram_api_key())

        options = LiveOptions(
            model=self._model,
            punctuate=True,
            interim_results=True,
        )

        if language:
            options.language = language

        connection = client.listen.live.v("1").start(options)

        transcripts: list[str] = []

        def on_message(self_ref: Any, result: Any, **kwargs: Any) -> None:
            transcript = result.channel.alternatives[0].transcript
            if transcript:
                transcripts.append(transcript)

        connection.on(LiveTranscriptionEvents.Transcript, on_message)

        try:
            for chunk in audio_stream:
                connection.send(chunk)
                # Yield any new transcripts
                while transcripts:
                    yield transcripts.pop(0)
        finally:
            connection.finish()

        # Yield remaining transcripts
        while transcripts:
            yield transcripts.pop(0)

    # =========================================================================
    # Google Cloud Speech-to-Text Implementation
    # =========================================================================

    def _transcribe_google(
        self,
        audio: bytes | str | Path,
        language: str | None,
        timestamps: bool,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        """Transcribe using Google Cloud Speech-to-Text."""
        try:
            from google.cloud import speech
        except ImportError:
            raise ImportError(
                "google-cloud-speech package required for Google STT: "
                "pip install google-cloud-speech"
            )

        client = speech.SpeechClient()
        audio_bytes, _ = self._load_audio(audio)

        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            language_code=language or "en-US",
            enable_word_time_offsets=word_timestamps,
            enable_automatic_punctuation=True,
        )

        response = client.recognize(config=config, audio=audio)

        text_parts = []
        segments = []

        for result in response.results:
            alt = result.alternatives[0] if result.alternatives else None
            if alt:
                text_parts.append(alt.transcript)

                if timestamps or word_timestamps:
                    if hasattr(alt, "words") and alt.words:
                        for word_info in alt.words:
                            segments.append(
                                TranscriptionSegment(
                                    text=word_info.word,
                                    start=word_info.start_time.total_seconds(),
                                    end=word_info.end_time.total_seconds(),
                                    confidence=alt.confidence,
                                )
                            )
                    else:
                        # Create a single segment for the result
                        segments.append(
                            TranscriptionSegment(
                                text=alt.transcript,
                                start=0.0,
                                end=0.0,
                                confidence=alt.confidence,
                            )
                        )

        full_text = " ".join(text_parts)

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=language or "en-US",
            provider=STTProvider.GOOGLE,
        )

    async def _atranscribe_google(
        self,
        audio: bytes | str | Path,
        language: str | None,
        timestamps: bool,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        """Async transcribe using Google Cloud Speech-to-Text."""
        # Google Cloud Speech doesn't have native async, run in executor
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._transcribe_google, audio, language, timestamps, word_timestamps
        )
