"""Text-to-Speech (TTS) support for ai-infra.

This module provides a unified API for text-to-speech across multiple providers:
- OpenAI (tts-1, tts-1-hd)
- Google Cloud TTS
- ElevenLabs

Example:
    ```python
    from ai_infra.llm.multimodal import TTS

    # Basic usage - auto-detect provider
    tts = TTS()
    audio = tts.speak("Hello, world!")

    # Save to file
    tts.speak_to_file("Hello!", "greeting.mp3")

    # With specific provider and voice
    tts = TTS(provider="openai", voice="nova")
    audio = tts.speak("Using Nova voice")

    # Async usage
    audio = await tts.aspeak("Async TTS")

    # Streaming for real-time playback
    for chunk in tts.stream("Long text to speak..."):
        play_audio(chunk)
    ```
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

from ai_infra.llm.multimodal.models import AudioFormat, TTSProvider, Voice
from ai_infra.providers import ProviderCapability, ProviderRegistry

# Provider aliases for backwards compatibility
_PROVIDER_ALIASES = {"google": "google_genai"}


def _get_default_voice(provider: str) -> str:
    """Get default voice for provider from registry."""
    # Resolve alias
    name = _PROVIDER_ALIASES.get(provider, provider)
    config = ProviderRegistry.get(name)
    if config:
        cap = config.get_capability(ProviderCapability.TTS)
        if cap and cap.default_voice:
            return cap.default_voice
    # Fallback defaults
    return {
        "openai": "alloy",
        "elevenlabs": "Rachel",
        "google": "en-US-Standard-C",
    }.get(provider, "default")


def _get_default_model(provider: str) -> str:
    """Get default model for provider from registry."""
    # Resolve alias
    name = _PROVIDER_ALIASES.get(provider, provider)
    config = ProviderRegistry.get(name)
    if config:
        cap = config.get_capability(ProviderCapability.TTS)
        if cap and cap.default_model:
            return cap.default_model
    # Fallback defaults
    return {
        "openai": "tts-1",
        "elevenlabs": "eleven_monolingual_v1",
        "google": "standard",
    }.get(provider, "default")


def _detect_tts_provider() -> str:
    """Detect available TTS provider from environment using registry."""
    # Priority order for TTS providers
    priority = ["openai", "elevenlabs", "google_genai"]
    for name in priority:
        if ProviderRegistry.is_configured(name):
            # Return user-facing name (without _genai suffix)
            return "google" if name == "google_genai" else name
    raise ValueError(
        "No TTS provider configured. Set OPENAI_API_KEY, ELEVEN_API_KEY, "
        "or GOOGLE_APPLICATION_CREDENTIALS."
    )


# Legacy constants (for backwards compatibility)
TTS_PROVIDER_PRIORITY = ["openai", "elevenlabs", "google"]
_openai_config = ProviderRegistry.get("openai")
if _openai_config:
    _cap = _openai_config.get_capability(ProviderCapability.TTS)
    OPENAI_VOICES = _cap.voices if _cap else ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
else:
    OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
OPENAI_MODELS = ["tts-1", "tts-1-hd"]

# ElevenLabs default voices (can be extended via API)
_elevenlabs_config = ProviderRegistry.get("elevenlabs")
if _elevenlabs_config:
    _cap = _elevenlabs_config.get_capability(ProviderCapability.TTS)
    ELEVENLABS_DEFAULT_VOICES = _cap.voices if _cap and _cap.voices else []
else:
    ELEVENLABS_DEFAULT_VOICES = ["Rachel", "Drew", "Clyde", "Paul", "Domi", "Dave"]


class TTS:
    """Text-to-Speech with provider-agnostic API.

    Supports OpenAI, Google Cloud TTS, and ElevenLabs.
    Auto-detects provider based on available API keys if not specified.
    """

    def __init__(
        self,
        provider: str | None = None,
        voice: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize TTS.

        Args:
            provider: TTS provider ("openai", "google", "elevenlabs").
                     Auto-detected if None.
            voice: Voice name (provider-specific). Uses default if None.
            model: Model name (provider-specific). Uses default if None.
            api_key: API key (uses environment variable if None).
        """
        self._provider = provider or _detect_tts_provider()
        self._voice = voice or _get_default_voice(self._provider)
        self._model = model or _get_default_model(self._provider)
        self._api_key = api_key

    @staticmethod
    def _detect_provider() -> str:
        """Detect available TTS provider from environment."""
        return _detect_tts_provider()

    @staticmethod
    def _default_voice(provider: str) -> str:
        """Get default voice for provider."""
        return _get_default_voice(provider)

    @staticmethod
    def _default_model(provider: str) -> str:
        """Get default model for provider."""
        return _get_default_model(provider)

    @property
    def provider(self) -> str:
        """Get the current provider name."""
        return self._provider

    @property
    def voice(self) -> str:
        """Get the current voice name."""
        return self._voice

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    def speak(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        output_format: AudioFormat = AudioFormat.MP3,
    ) -> bytes:
        """Convert text to audio bytes.

        Args:
            text: Text to convert to speech.
            voice: Override default voice.
            model: Override default model.
            output_format: Audio format (mp3, wav, etc.).

        Returns:
            Audio data as bytes.
        """
        voice = voice or self._voice
        model = model or self._model

        if self._provider == "openai":
            return self._speak_openai(text, voice, model, output_format)
        elif self._provider == "elevenlabs":
            return self._speak_elevenlabs(text, voice, model, output_format)
        elif self._provider == "google":
            return self._speak_google(text, voice, model, output_format)
        else:
            raise ValueError(f"Unsupported TTS provider: {self._provider}")

    async def aspeak(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        output_format: AudioFormat = AudioFormat.MP3,
    ) -> bytes:
        """Async version of speak().

        Args:
            text: Text to convert to speech.
            voice: Override default voice.
            model: Override default model.
            output_format: Audio format (mp3, wav, etc.).

        Returns:
            Audio data as bytes.
        """
        voice = voice or self._voice
        model = model or self._model

        if self._provider == "openai":
            return await self._aspeak_openai(text, voice, model, output_format)
        elif self._provider == "elevenlabs":
            return await self._aspeak_elevenlabs(text, voice, model, output_format)
        elif self._provider == "google":
            return await self._aspeak_google(text, voice, model, output_format)
        else:
            raise ValueError(f"Unsupported TTS provider: {self._provider}")

    def speak_to_file(
        self,
        text: str,
        path: str | Path,
        *,
        voice: str | None = None,
        model: str | None = None,
        output_format: AudioFormat | None = None,
    ) -> None:
        """Convert text to speech and save to file.

        Args:
            text: Text to convert to speech.
            path: Output file path.
            voice: Override default voice.
            model: Override default model.
            output_format: Audio format (inferred from path if None).
        """
        path = Path(path)

        # Infer format from extension if not specified
        if output_format is None:
            ext = path.suffix.lower().lstrip(".")
            try:
                output_format = AudioFormat(ext)
            except ValueError:
                output_format = AudioFormat.MP3

        audio = self.speak(text, voice=voice, model=model, output_format=output_format)
        path.write_bytes(audio)

    def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        output_format: AudioFormat = AudioFormat.MP3,
    ) -> Iterator[bytes]:
        """Stream audio chunks for real-time playback.

        Args:
            text: Text to convert to speech.
            voice: Override default voice.
            model: Override default model.
            output_format: Audio format.

        Yields:
            Audio chunks as bytes.
        """
        voice = voice or self._voice
        model = model or self._model

        if self._provider == "openai":
            yield from self._stream_openai(text, voice, model, output_format)
        elif self._provider == "elevenlabs":
            yield from self._stream_elevenlabs(text, voice, model, output_format)
        else:
            # Fallback: return entire audio as single chunk
            yield self.speak(text, voice=voice, model=model, output_format=output_format)

    async def astream(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        output_format: AudioFormat = AudioFormat.MP3,
    ) -> AsyncIterator[bytes]:
        """Async stream audio chunks for real-time playback.

        Args:
            text: Text to convert to speech.
            voice: Override default voice.
            model: Override default model.
            output_format: Audio format.

        Yields:
            Audio chunks as bytes.
        """
        voice = voice or self._voice
        model = model or self._model

        if self._provider == "openai":
            async for chunk in self._astream_openai(text, voice, model, output_format):
                yield chunk
        elif self._provider == "elevenlabs":
            async for chunk in self._astream_elevenlabs(text, voice, model, output_format):
                yield chunk
        else:
            # Fallback: return entire audio as single chunk
            yield await self.aspeak(text, voice=voice, model=model, output_format=output_format)

    @staticmethod
    def list_voices(provider: str | None = None) -> list[Voice]:
        """List available voices for a provider.

        Args:
            provider: Provider name. If None, lists for all configured providers.

        Returns:
            List of available Voice objects.
        """
        voices: list[Voice] = []

        if provider is None or provider == "openai":
            if os.environ.get("OPENAI_API_KEY"):
                for v in OPENAI_VOICES:
                    voices.append(
                        Voice(
                            id=v,
                            name=v.title(),
                            provider=TTSProvider.OPENAI,
                            language="en",
                            description=f"OpenAI {v} voice",
                        )
                    )

        if provider is None or provider == "elevenlabs":
            if os.environ.get("ELEVEN_API_KEY") or os.environ.get("ELEVENLABS_API_KEY"):
                for v in ELEVENLABS_DEFAULT_VOICES:
                    voices.append(
                        Voice(
                            id=v.lower(),
                            name=v,
                            provider=TTSProvider.ELEVENLABS,
                            language="en",
                            description=f"ElevenLabs {v} voice",
                        )
                    )

        if provider is None or provider == "google":
            if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_API_KEY"):
                # Google has many voices, just show some common ones
                google_voices = [
                    ("en-US-Standard-A", "English US Standard A"),
                    ("en-US-Standard-B", "English US Standard B"),
                    ("en-US-Standard-C", "English US Standard C"),
                    ("en-US-Standard-D", "English US Standard D"),
                    ("en-US-Wavenet-A", "English US WaveNet A"),
                    ("en-US-Wavenet-B", "English US WaveNet B"),
                    ("en-GB-Standard-A", "English UK Standard A"),
                    ("en-GB-Wavenet-A", "English UK WaveNet A"),
                ]
                for v_id, v_name in google_voices:
                    voices.append(
                        Voice(
                            id=v_id,
                            name=v_name,
                            provider=TTSProvider.GOOGLE,
                            language=v_id.split("-")[0] + "-" + v_id.split("-")[1],
                            description=f"Google {v_name}",
                        )
                    )

        return voices

    @staticmethod
    def list_providers() -> list[str]:
        """List configured TTS providers.

        Returns:
            List of provider names that have API keys configured.
        """
        providers = []
        if os.environ.get("OPENAI_API_KEY"):
            providers.append("openai")
        if os.environ.get("ELEVEN_API_KEY") or os.environ.get("ELEVENLABS_API_KEY"):
            providers.append("elevenlabs")
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_API_KEY"):
            providers.append("google")
        return providers

    # =========================================================================
    # OpenAI TTS Implementation
    # =========================================================================

    def _get_openai_client(self):
        """Get OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for OpenAI TTS: pip install openai")

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAI(api_key=api_key)

    def _get_openai_async_client(self):
        """Get async OpenAI client."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required for OpenAI TTS: pip install openai")

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        return AsyncOpenAI(api_key=api_key)

    def _speak_openai(self, text: str, voice: str, model: str, output_format: AudioFormat) -> bytes:
        """Generate speech using OpenAI TTS."""
        client = self._get_openai_client()

        # Map AudioFormat to OpenAI response format
        format_map = {
            AudioFormat.MP3: "mp3",
            AudioFormat.WAV: "wav",
            AudioFormat.OGG: "opus",
            AudioFormat.FLAC: "flac",
            AudioFormat.AAC: "aac",
        }
        response_format = format_map.get(output_format, "mp3")

        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
        )
        from typing import cast

        return cast("bytes", response.content)

    async def _aspeak_openai(
        self, text: str, voice: str, model: str, output_format: AudioFormat
    ) -> bytes:
        """Async generate speech using OpenAI TTS."""
        client = self._get_openai_async_client()

        format_map = {
            AudioFormat.MP3: "mp3",
            AudioFormat.WAV: "wav",
            AudioFormat.OGG: "opus",
            AudioFormat.FLAC: "flac",
            AudioFormat.AAC: "aac",
        }
        response_format = format_map.get(output_format, "mp3")

        response = await client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
        )
        from typing import cast

        return cast("bytes", response.content)

    def _stream_openai(
        self, text: str, voice: str, model: str, output_format: AudioFormat
    ) -> Iterator[bytes]:
        """Stream audio from OpenAI TTS."""
        client = self._get_openai_client()

        format_map = {
            AudioFormat.MP3: "mp3",
            AudioFormat.WAV: "wav",
            AudioFormat.OGG: "opus",
            AudioFormat.FLAC: "flac",
            AudioFormat.AAC: "aac",
        }
        response_format = format_map.get(output_format, "mp3")

        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
        ) as response:
            for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk

    async def _astream_openai(
        self, text: str, voice: str, model: str, output_format: AudioFormat
    ) -> AsyncIterator[bytes]:
        """Async stream audio from OpenAI TTS."""
        client = self._get_openai_async_client()

        format_map = {
            AudioFormat.MP3: "mp3",
            AudioFormat.WAV: "wav",
            AudioFormat.OGG: "opus",
            AudioFormat.FLAC: "flac",
            AudioFormat.AAC: "aac",
        }
        response_format = format_map.get(output_format, "mp3")

        async with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk

    # =========================================================================
    # ElevenLabs TTS Implementation
    # =========================================================================

    def _get_elevenlabs_api_key(self) -> str:
        """Get ElevenLabs API key."""
        return (
            self._api_key
            or os.environ.get("ELEVEN_API_KEY")
            or os.environ.get("ELEVENLABS_API_KEY")
            or ""
        )

    def _speak_elevenlabs(
        self, text: str, voice: str, model: str, output_format: AudioFormat
    ) -> bytes:
        """Generate speech using ElevenLabs."""
        try:
            from elevenlabs.client import ElevenLabs as ElevenLabsClient
        except ImportError:
            raise ImportError(
                "elevenlabs package required for ElevenLabs TTS: pip install elevenlabs"
            )

        client = ElevenLabsClient(api_key=self._get_elevenlabs_api_key())

        # Map AudioFormat to ElevenLabs output format
        format_map = {
            AudioFormat.MP3: "mp3_44100_128",
            AudioFormat.WAV: "pcm_44100",
            AudioFormat.OGG: "mp3_44100_128",  # ElevenLabs doesn't support OGG directly
        }
        el_format = format_map.get(output_format, "mp3_44100_128")

        audio = client.generate(
            text=text,
            voice=voice,
            model=model,
            output_format=el_format,
        )

        # elevenlabs.generate returns an iterator, collect all bytes
        if hasattr(audio, "__iter__") and not isinstance(audio, bytes):
            return b"".join(audio)
        from typing import cast

        return cast("bytes", audio)

    async def _aspeak_elevenlabs(
        self, text: str, voice: str, model: str, output_format: AudioFormat
    ) -> bytes:
        """Async generate speech using ElevenLabs."""
        try:
            from elevenlabs.client import AsyncElevenLabs
        except ImportError:
            raise ImportError(
                "elevenlabs package required for ElevenLabs TTS: pip install elevenlabs"
            )

        client = AsyncElevenLabs(api_key=self._get_elevenlabs_api_key())

        format_map = {
            AudioFormat.MP3: "mp3_44100_128",
            AudioFormat.WAV: "pcm_44100",
            AudioFormat.OGG: "mp3_44100_128",
        }
        el_format = format_map.get(output_format, "mp3_44100_128")

        audio = await client.generate(
            text=text,
            voice=voice,
            model=model,
            output_format=el_format,
        )

        if hasattr(audio, "__iter__") and not isinstance(audio, bytes):
            chunks = []
            async for chunk in audio:
                chunks.append(chunk)
            return b"".join(chunks)
        from typing import cast

        return cast("bytes", audio)

    def _stream_elevenlabs(
        self, text: str, voice: str, model: str, output_format: AudioFormat
    ) -> Iterator[bytes]:
        """Stream audio from ElevenLabs."""
        try:
            from elevenlabs.client import ElevenLabs as ElevenLabsClient
        except ImportError:
            raise ImportError(
                "elevenlabs package required for ElevenLabs TTS: pip install elevenlabs"
            )

        client = ElevenLabsClient(api_key=self._get_elevenlabs_api_key())

        format_map = {
            AudioFormat.MP3: "mp3_44100_128",
            AudioFormat.WAV: "pcm_44100",
            AudioFormat.OGG: "mp3_44100_128",
        }
        el_format = format_map.get(output_format, "mp3_44100_128")

        audio = client.generate(
            text=text,
            voice=voice,
            model=model,
            output_format=el_format,
        )

        # elevenlabs.generate returns an iterator
        if hasattr(audio, "__iter__") and not isinstance(audio, bytes):
            yield from audio
        else:
            from typing import cast

            yield cast("bytes", audio)

    async def _astream_elevenlabs(
        self, text: str, voice: str, model: str, output_format: AudioFormat
    ) -> AsyncIterator[bytes]:
        """Async stream audio from ElevenLabs."""
        try:
            from elevenlabs.client import AsyncElevenLabs
        except ImportError:
            raise ImportError(
                "elevenlabs package required for ElevenLabs TTS: pip install elevenlabs"
            )

        client = AsyncElevenLabs(api_key=self._get_elevenlabs_api_key())

        format_map = {
            AudioFormat.MP3: "mp3_44100_128",
            AudioFormat.WAV: "pcm_44100",
            AudioFormat.OGG: "mp3_44100_128",
        }
        el_format = format_map.get(output_format, "mp3_44100_128")

        audio = await client.generate(
            text=text,
            voice=voice,
            model=model,
            output_format=el_format,
        )

        if hasattr(audio, "__iter__") and not isinstance(audio, bytes):
            async for chunk in audio:
                yield chunk
        else:
            from typing import cast

            yield cast("bytes", audio)

    # =========================================================================
    # Google TTS Implementation
    # =========================================================================

    def _speak_google(self, text: str, voice: str, model: str, output_format: AudioFormat) -> bytes:
        """Generate speech using Google Cloud TTS."""
        try:
            from google.cloud import texttospeech
        except ImportError:
            raise ImportError(
                "google-cloud-texttospeech package required for Google TTS: "
                "pip install google-cloud-texttospeech"
            )

        client = texttospeech.TextToSpeechClient()

        # Parse voice name for language code
        # e.g., "en-US-Standard-C" -> language_code="en-US"
        parts = voice.split("-")
        language_code = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else "en-US"

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice,
        )

        # Map AudioFormat to Google encoding
        encoding_map = {
            AudioFormat.MP3: texttospeech.AudioEncoding.MP3,
            AudioFormat.WAV: texttospeech.AudioEncoding.LINEAR16,
            AudioFormat.OGG: texttospeech.AudioEncoding.OGG_OPUS,
        }
        audio_encoding = encoding_map.get(output_format, texttospeech.AudioEncoding.MP3)

        audio_config = texttospeech.AudioConfig(audio_encoding=audio_encoding)

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )

        from typing import cast

        return cast("bytes", response.audio_content)

    async def _aspeak_google(
        self, text: str, voice: str, model: str, output_format: AudioFormat
    ) -> bytes:
        """Async generate speech using Google Cloud TTS."""
        # Google Cloud TTS doesn't have native async, run in executor
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._speak_google, text, voice, model, output_format
        )
