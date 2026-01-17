"""Unified voice chat interface.

This module provides the high-level VoiceChat class that combines
Microphone, STT, TTS, and AudioPlayer for seamless voice interaction.

Example:
    ```python
    from ai_infra.llm.multimodal import VoiceChat
    from ai_infra.llm import LLM

    voice = VoiceChat()
    llm = LLM()

    while True:
        user_text = voice.listen()
        if not user_text or user_text.lower() in ["exit", "quit"]:
            break

        print(f"You: {user_text}")
        response = llm.chat(user_text)
        print(f"AI: {response}")

        voice.speak(response)
    ```
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

from ai_infra.llm.multimodal.voice.playback import AudioPlayer
from ai_infra.llm.multimodal.voice.recording import Microphone

# Sentence boundary pattern - matches ., !, ? followed by space or end
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


@dataclass
class VoiceChatConfig:
    """Configuration for voice chat sessions.

    Attributes:
        stt_provider: STT provider for transcription.
        tts_provider: TTS provider for speech synthesis.
        tts_voice: Voice to use for TTS.
        sample_rate: Recording sample rate (16000 for most STT).
        max_recording: Maximum recording duration in seconds.
    """

    stt_provider: str | None = None
    tts_provider: str | None = None
    tts_voice: str | None = None
    sample_rate: int = 16000
    max_recording: float = 300.0


class VoiceChat:
    """Unified voice chat interface combining Microphone, STT, TTS, and AudioPlayer.

    Provides a high-level API for voice-based interaction:
    - Record user speech
    - Transcribe to text
    - (Process with LLM - external)
    - Speak response
    """

    def __init__(
        self,
        stt_provider: str | None = None,
        tts_provider: str | None = None,
        tts_voice: str | None = None,
        sample_rate: int = 16000,
    ):
        """Initialize voice chat components.

        Args:
            stt_provider: STT provider ("openai", "google", "deepgram").
            tts_provider: TTS provider ("openai", "google", "elevenlabs").
            tts_voice: Voice for TTS (provider-specific).
            sample_rate: Recording sample rate (16000 optimal for STT).
        """
        from ai_infra.llm.multimodal.stt import STT
        from ai_infra.llm.multimodal.tts import TTS

        self._mic = Microphone(sample_rate=sample_rate)
        self._player = AudioPlayer()
        self._stt = STT(provider=stt_provider)
        self._tts = TTS(provider=tts_provider, voice=tts_voice)

    @property
    def stt(self):
        """Get the STT instance."""
        return self._stt

    @property
    def tts(self):
        """Get the TTS instance."""
        return self._tts

    @property
    def mic(self) -> Microphone:
        """Get the Microphone instance."""
        return self._mic

    @property
    def player(self) -> AudioPlayer:
        """Get the AudioPlayer instance."""
        return self._player

    @staticmethod
    def is_available() -> tuple[bool, list[str]]:
        """Check if voice chat is available.

        Returns:
            Tuple of (is_available, list of missing components).
        """
        missing = []

        if not Microphone.is_available():
            missing.append("microphone (install sounddevice)")

        if not AudioPlayer.is_available():
            missing.append("audio playback")

        try:
            from ai_infra.llm.multimodal.discovery import (
                get_default_stt_provider,
                get_default_tts_provider,
            )

            if not get_default_stt_provider():
                missing.append("STT provider (set OPENAI_API_KEY or similar)")
            if not get_default_tts_provider():
                missing.append("TTS provider (set OPENAI_API_KEY or similar)")
        except Exception:
            missing.append("STT/TTS providers")

        return len(missing) == 0, missing

    def listen(self, prompt: str | None = None) -> str:
        """Record speech and transcribe to text.

        Args:
            prompt: Optional prompt to guide STT transcription.

        Returns:
            Transcribed text from speech.
        """
        audio = self._mic.record_until_enter()
        result = self._stt.transcribe(audio, prompt=prompt)
        return result.text

    async def alisten(self, prompt: str | None = None) -> str:
        """Async version of listen().

        Args:
            prompt: Optional prompt to guide STT transcription.

        Returns:
            Transcribed text from speech.
        """
        audio = self._mic.record_until_enter()
        result = await self._stt.atranscribe(audio, prompt=prompt)
        return result.text

    def speak(
        self,
        text: str,
        *,
        blocking: bool = True,
    ) -> None:
        """Convert text to speech and play it.

        Args:
            text: Text to speak.
            blocking: If True, wait for playback to complete.
        """
        audio = self._tts.speak(text)
        self._player.play(audio, blocking=blocking)

    async def aspeak(
        self,
        text: str,
        *,
        blocking: bool = True,
    ) -> None:
        """Async version of speak().

        Args:
            text: Text to speak.
            blocking: If True, wait for playback to complete.
        """
        audio = await self._tts.aspeak(text)
        self._player.play(audio, blocking=blocking)

    def speak_stream(
        self,
        text: str,
    ) -> None:
        """Stream text to speech for lower latency.

        Uses TTS streaming if available for faster first-byte response.

        Args:
            text: Text to speak.
        """
        try:
            for chunk in self._tts.stream(text):
                self._player.play(chunk, blocking=True)
        except NotImplementedError:
            self.speak(text)

    def speak_chunks(
        self,
        chunks: Iterator[str],
        min_chars: int = 50,
    ) -> str:
        """Speak text chunks as they arrive, buffering by sentence.

        Optimizes voice latency by speaking complete sentences as soon
        as they're available, rather than waiting for the full response.

        Args:
            chunks: Iterator of text chunks (e.g., from LLM streaming).
            min_chars: Minimum characters before speaking (avoids tiny chunks).

        Returns:
            The complete accumulated text.

        Example:
            ```python
            # Stream LLM response with real-time TTS
            def llm_stream():
                for chunk in llm.stream("Tell me a story"):
                    yield chunk.content

            full_text = voice.speak_chunks(llm_stream())
            ```
        """
        buffer = ""
        full_text = ""

        for chunk in chunks:
            buffer += chunk
            full_text += chunk

            # Check for sentence boundaries
            if len(buffer) >= min_chars and _SENTENCE_END.search(buffer):
                # Find the last sentence boundary
                match = None
                for m in _SENTENCE_END.finditer(buffer):
                    match = m

                if match:
                    # Speak up to and including the sentence end
                    speak_text = buffer[: match.end()].strip()
                    buffer = buffer[match.end() :]

                    if speak_text:
                        audio = self._tts.speak(speak_text)
                        self._player.play(audio, blocking=True)

        # Speak any remaining text
        if buffer.strip():
            audio = self._tts.speak(buffer.strip())
            self._player.play(audio, blocking=True)

        return full_text

    async def aspeak_chunks(
        self,
        chunks: AsyncIterator[str],
        min_chars: int = 50,
    ) -> str:
        """Async version of speak_chunks for async LLM streaming.

        Args:
            chunks: Async iterator of text chunks.
            min_chars: Minimum characters before speaking.

        Returns:
            The complete accumulated text.

        Example:
            ```python
            async def llm_stream():
                async for chunk in llm.astream("Tell me a story"):
                    yield chunk.content

            full_text = await voice.aspeak_chunks(llm_stream())
            ```
        """
        buffer = ""
        full_text = ""

        async for chunk in chunks:
            buffer += chunk
            full_text += chunk

            if len(buffer) >= min_chars and _SENTENCE_END.search(buffer):
                match = None
                for m in _SENTENCE_END.finditer(buffer):
                    match = m

                if match:
                    speak_text = buffer[: match.end()].strip()
                    buffer = buffer[match.end() :]

                    if speak_text:
                        audio = await self._tts.aspeak(speak_text)
                        self._player.play(audio, blocking=True)

        if buffer.strip():
            audio = await self._tts.aspeak(buffer.strip())
            self._player.play(audio, blocking=True)

        return full_text
