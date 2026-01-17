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

from dataclasses import dataclass

from ai_infra.llm.multimodal.voice.playback import AudioPlayer
from ai_infra.llm.multimodal.voice.recording import Microphone


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
