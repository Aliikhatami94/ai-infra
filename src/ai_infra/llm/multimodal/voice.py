"""Voice input/output support for ai-infra.

This module provides microphone recording and audio playback for voice chat:
- Microphone: Record audio from the default input device
- AudioPlayer: Play audio through the default output device

Example - Voice Chat:
    ```python
    from ai_infra.llm.multimodal import Microphone, AudioPlayer, STT, TTS
    from ai_infra.llm import LLM

    mic = Microphone()
    player = AudioPlayer()
    stt = STT()
    tts = TTS()
    llm = LLM()

    # Record user speech
    print("Recording... Press Enter to stop")
    audio = mic.record_until_enter()

    # Transcribe
    result = stt.transcribe(audio)
    print(f"You said: {result.text}")

    # Get LLM response
    response = llm.chat(result.text)
    print(f"AI: {response}")

    # Speak response
    audio = tts.speak(response)
    player.play(audio)
    ```

Example - Simple Recording:
    ```python
    from ai_infra.llm.multimodal import Microphone

    mic = Microphone()

    # Record for fixed duration
    audio = mic.record(duration=5.0)
    mic.save(audio, "recording.wav")

    # Record until Enter is pressed
    audio = mic.record_until_enter()
    ```

Example - Audio Playback:
    ```python
    from ai_infra.llm.multimodal import AudioPlayer

    player = AudioPlayer()

    # Play from bytes
    player.play(audio_bytes)

    # Play from file
    player.play_file("response.mp3")

    # Non-blocking playback
    player.play(audio_bytes, blocking=False)
    player.wait()  # Wait for completion
    player.stop()  # Or stop early
    ```

Requirements:
    Install with voice extras: `pip install ai-infra[voice]`
    This installs: sounddevice, soundfile
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ai_infra.llm.multimodal.models import AudioFormat, AudioSegment

# =============================================================================
# Microphone Recording
# =============================================================================


@dataclass
class RecordingConfig:
    """Configuration for audio recording.

    Attributes:
        sample_rate: Sample rate in Hz (default: 16000 for STT compatibility).
        channels: Number of audio channels (1=mono, 2=stereo).
        dtype: Sample data type ('int16', 'float32').
        format: Output audio format.
        device: Input device index or name (None for default).
    """

    sample_rate: int = 16000
    channels: int = 1
    dtype: Literal["int16", "float32"] = "int16"
    format: AudioFormat = AudioFormat.WAV
    device: int | str | None = None


class Microphone:
    """Record audio from the system microphone.

    Uses sounddevice for cross-platform audio recording.
    Optimized for speech recognition with default 16kHz mono recording.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
    ):
        """Initialize the microphone.

        Args:
            sample_rate: Sample rate in Hz. 16000 is optimal for most STT.
            channels: Number of channels (1 for mono, 2 for stereo).
            device: Input device index or name. None for system default.
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._device = device
        self._recording = False
        self._frames: list[Any] = []
        self._lock = threading.Lock()

    @staticmethod
    def is_available() -> bool:
        """Check if microphone recording is available.

        Returns:
            True if sounddevice is installed and a mic is accessible.
        """
        try:
            import sounddevice as sd
        except ImportError:
            return False

        try:
            # Try to query default input device
            sd.query_devices(kind="input")
            return True
        except sd.PortAudioError:
            return False

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        """List available audio input devices.

        Returns:
            List of device info dictionaries with 'index', 'name', 'channels'.
        """
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            input_devices = []
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    input_devices.append(
                        {
                            "index": i,
                            "name": d["name"],
                            "channels": d["max_input_channels"],
                            "sample_rate": d["default_samplerate"],
                        }
                    )
            return input_devices
        except ImportError:
            return []

    @property
    def sample_rate(self) -> int:
        """Get the sample rate."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Get the number of channels."""
        return self._channels

    def record(
        self,
        duration: float,
        *,
        format: AudioFormat = AudioFormat.WAV,
    ) -> bytes:
        """Record audio for a fixed duration.

        Args:
            duration: Recording duration in seconds.
            format: Output audio format (WAV recommended for STT).

        Returns:
            Audio data as bytes.

        Raises:
            ImportError: If sounddevice is not installed.
            RuntimeError: If recording fails.
        """
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "sounddevice is required for microphone recording. "
                "Install with: pip install ai-infra[voice]"
            ) from e

        try:
            # Calculate number of frames
            frames = int(duration * self._sample_rate)

            # Record audio
            audio = sd.rec(
                frames,
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                device=self._device,
            )
            sd.wait()

            return self._encode_audio(audio, format)
        except Exception as e:
            raise RuntimeError(f"Recording failed: {e}") from e

    def record_until_enter(
        self,
        *,
        format: AudioFormat = AudioFormat.WAV,
        max_duration: float = 300.0,
    ) -> bytes:
        """Record audio until Enter is pressed.

        Args:
            format: Output audio format (WAV recommended for STT).
            max_duration: Maximum recording duration in seconds (safety limit).

        Returns:
            Audio data as bytes.

        Raises:
            ImportError: If sounddevice is not installed.
            RuntimeError: If recording fails.
        """
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "sounddevice is required for microphone recording. "
                "Install with: pip install ai-infra[voice]"
            ) from e

        try:
            import numpy as np

            frames: list[Any] = []

            def callback(indata, frame_count, time_info, status):
                if status:
                    pass  # Ignore status messages
                frames.append(indata.copy())

            # Start recording in callback mode
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                device=self._device,
                callback=callback,
            ):
                # Wait for Enter in a separate thread
                input()

            # Combine all frames
            if not frames:
                raise RuntimeError("No audio recorded")

            audio = np.concatenate(frames, axis=0)

            # Limit to max_duration
            max_frames = int(max_duration * self._sample_rate)
            if len(audio) > max_frames:
                audio = audio[:max_frames]

            return self._encode_audio(audio, format)
        except Exception as e:
            raise RuntimeError(f"Recording failed: {e}") from e

    def record_stream(
        self,
        chunk_duration: float = 0.1,
    ) -> Iterator[bytes]:
        """Stream audio chunks for real-time processing.

        Args:
            chunk_duration: Duration of each chunk in seconds.

        Yields:
            Raw PCM audio chunks (int16).

        Example:
            ```python
            mic = Microphone()
            for chunk in mic.record_stream():
                # Process chunk in real-time
                process(chunk)
                if should_stop():
                    break
            ```
        """
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "sounddevice is required for microphone recording. "
                "Install with: pip install ai-infra[voice]"
            ) from e

        import queue

        q: queue.Queue[bytes | None] = queue.Queue()
        chunk_size = int(chunk_duration * self._sample_rate)

        def callback(indata, frame_count, time_info, status):
            q.put(bytes(indata))

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            device=self._device,
            blocksize=chunk_size,
            callback=callback,
        ):
            while True:
                try:
                    chunk = q.get(timeout=1.0)
                    if chunk is None:
                        break
                    yield chunk
                except queue.Empty:
                    continue

    def _encode_audio(self, audio: Any, format: AudioFormat) -> bytes:
        """Encode numpy audio array to bytes.

        Args:
            audio: Numpy array of audio samples.
            format: Target audio format.

        Returns:
            Encoded audio bytes.
        """
        try:
            import soundfile as sf
        except ImportError:
            # Fallback to raw WAV if soundfile not available
            return self._encode_wav_manual(audio)

        buffer = io.BytesIO()

        # Map AudioFormat to soundfile format
        format_map = {
            AudioFormat.WAV: "WAV",
            AudioFormat.FLAC: "FLAC",
            AudioFormat.OGG: "OGG",
        }
        sf_format = format_map.get(format, "WAV")

        # Use appropriate subtype based on format
        subtype = "PCM_16" if format in (AudioFormat.WAV, AudioFormat.FLAC) else None

        sf.write(
            buffer,
            audio,
            self._sample_rate,
            format=sf_format,
            subtype=subtype,
        )

        return buffer.getvalue()

    def _encode_wav_manual(self, audio: Any) -> bytes:
        """Manually encode WAV without soundfile.

        Args:
            audio: Numpy array of int16 audio samples.

        Returns:
            WAV audio bytes.
        """
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self._channels)
            wav.setsampwidth(2)  # 16-bit = 2 bytes
            wav.setframerate(self._sample_rate)
            wav.writeframes(audio.tobytes())

        return buffer.getvalue()

    def save(
        self,
        audio: bytes,
        path: str | Path,
    ) -> None:
        """Save recorded audio to a file.

        Args:
            audio: Audio data from record() or record_until_enter().
            path: File path to save to.
        """
        path = Path(path)
        with open(path, "wb") as f:
            f.write(audio)


# =============================================================================
# Audio Playback
# =============================================================================


class AudioPlayer:
    """Play audio through the system speakers.

    Supports multiple playback methods with automatic fallback:
    1. sounddevice (cross-platform, recommended)
    2. afplay (macOS)
    3. aplay (Linux)
    4. System default (Windows)
    """

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int | None = None,
    ):
        """Initialize the audio player.

        Args:
            device: Output device index or name. None for system default.
            sample_rate: Sample rate override. None to use audio's native rate.
        """
        self._device = device
        self._sample_rate = sample_rate
        self._current_process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @staticmethod
    def is_available() -> bool:
        """Check if audio playback is available.

        Returns:
            True if any playback method is available.
        """
        # Check sounddevice
        try:
            import sounddevice as sd

            sd.query_devices(kind="output")
            return True
        except (ImportError, Exception):
            pass

        # Check system commands
        if sys.platform == "darwin":
            return os.path.exists("/usr/bin/afplay")
        elif sys.platform.startswith("linux"):
            import shutil

            return shutil.which("aplay") is not None or shutil.which("paplay") is not None
        elif sys.platform == "win32":
            return True  # Windows has built-in playback

        return False

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        """List available audio output devices.

        Returns:
            List of device info dictionaries.
        """
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            output_devices = []
            for i, d in enumerate(devices):
                if d["max_output_channels"] > 0:
                    output_devices.append(
                        {
                            "index": i,
                            "name": d["name"],
                            "channels": d["max_output_channels"],
                            "sample_rate": d["default_samplerate"],
                        }
                    )
            return output_devices
        except ImportError:
            return []

    def play(
        self,
        audio: bytes | AudioSegment,
        *,
        blocking: bool = True,
    ) -> None:
        """Play audio data.

        Args:
            audio: Audio data as bytes or AudioSegment.
            blocking: If True, wait for playback to complete.
        """
        if isinstance(audio, AudioSegment):
            audio_bytes = audio.data
            audio_format = audio.format
        else:
            audio_bytes = audio
            audio_format = self._detect_format(audio_bytes)

        # Try sounddevice first
        if self._try_sounddevice_play(audio_bytes, audio_format, blocking):
            return

        # Fallback to system command
        self._play_system(audio_bytes, audio_format, blocking)

    def play_file(
        self,
        path: str | Path,
        *,
        blocking: bool = True,
    ) -> None:
        """Play audio from a file.

        Args:
            path: Path to the audio file.
            blocking: If True, wait for playback to complete.
        """
        path = Path(path)
        with open(path, "rb") as f:
            audio = f.read()
        self.play(audio, blocking=blocking)

    def stop(self) -> None:
        """Stop current playback."""
        with self._lock:
            if self._current_process and self._current_process.poll() is None:
                self._current_process.terminate()
                self._current_process = None

    def wait(self) -> None:
        """Wait for current playback to complete."""
        with self._lock:
            if self._current_process:
                self._current_process.wait()

    def _detect_format(self, audio: bytes) -> AudioFormat:
        """Detect audio format from bytes."""
        # Check magic bytes
        if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
            return AudioFormat.WAV
        elif audio[:4] == b"fLaC":
            return AudioFormat.FLAC
        elif audio[:4] == b"OggS":
            return AudioFormat.OGG
        elif audio[:3] == b"ID3" or (audio[0:2] == b"\xff\xfb"):
            return AudioFormat.MP3
        else:
            # Default to MP3 (most common from TTS)
            return AudioFormat.MP3

    def _try_sounddevice_play(
        self,
        audio: bytes,
        format: AudioFormat,
        blocking: bool,
    ) -> bool:
        """Try to play using sounddevice.

        Returns True if successful, False to try fallback.
        """
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            return False

        try:
            buffer = io.BytesIO(audio)
            data, sample_rate = sf.read(buffer)

            if self._sample_rate:
                sample_rate = self._sample_rate

            if blocking:
                sd.play(data, sample_rate, device=self._device)
                sd.wait()
            else:
                sd.play(data, sample_rate, device=self._device)

            return True
        except Exception:
            return False

    def _play_system(
        self,
        audio: bytes,
        format: AudioFormat,
        blocking: bool,
    ) -> None:
        """Play using system command."""
        # Write to temp file
        suffix = f".{format.value}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio)
            temp_path = f.name

        try:
            if sys.platform == "darwin":
                cmd = ["afplay", temp_path]
            elif sys.platform.startswith("linux"):
                # Try paplay (PulseAudio) first, then aplay (ALSA)
                import shutil

                if shutil.which("paplay"):
                    cmd = ["paplay", temp_path]
                else:
                    cmd = ["aplay", temp_path]
            elif sys.platform == "win32":
                # Use PowerShell for Windows
                cmd = [
                    "powershell",
                    "-c",
                    f'(New-Object Media.SoundPlayer "{temp_path}").PlaySync()',
                ]
            else:
                raise RuntimeError(f"Unsupported platform: {sys.platform}")

            with self._lock:
                self._current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            if blocking:
                self._current_process.wait()
        finally:
            if blocking:
                # Clean up temp file after playback
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


# =============================================================================
# Voice Chat Session
# =============================================================================


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

    Example:
        ```python
        from ai_infra.llm.multimodal import VoiceChat
        from ai_infra.llm import LLM

        voice = VoiceChat()
        llm = LLM()

        while True:
            # Record and transcribe user input
            user_text = voice.listen()
            if not user_text or user_text.lower() in ["exit", "quit"]:
                break

            print(f"You: {user_text}")

            # Get LLM response
            response = llm.chat(user_text)
            print(f"AI: {response}")

            # Speak the response
            voice.speak(response)
        ```
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
        # Lazy imports to avoid circular dependencies
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

        # Check STT/TTS providers
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
        # Recording is sync (hardware), but transcription can be async
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
        # Try streaming if provider supports it
        try:
            for chunk in self._tts.stream(text):
                self._player.play(chunk, blocking=True)
        except NotImplementedError:
            # Fallback to non-streaming
            self.speak(text)
