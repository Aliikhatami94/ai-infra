"""Microphone recording support for voice chat.

This module provides microphone recording functionality:
- Microphone: Record audio from the default input device
- RecordingConfig: Configuration for audio recording

Example:
    ```python
    from ai_infra.llm.multimodal.voice import Microphone

    mic = Microphone()

    # Record for fixed duration
    audio = mic.record(duration=5.0)
    mic.save(audio, "recording.wav")

    # Record until Enter is pressed
    audio = mic.record_until_enter()
    ```

Requirements:
    Install with voice extras: `pip install ai-infra[voice]`
"""

from __future__ import annotations

import io
import threading
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ai_infra.llm.multimodal.models import AudioFormat


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
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError:
            return False

        try:
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
            import sounddevice as sd  # type: ignore[import-not-found]

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
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "sounddevice is required for microphone recording. "
                "Install with: pip install ai-infra[voice]"
            ) from e

        try:
            frames = int(duration * self._sample_rate)
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
            import sounddevice as sd  # type: ignore[import-not-found]
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

            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                device=self._device,
                callback=callback,
            ):
                input()

            if not frames:
                raise RuntimeError("No audio recorded")

            audio = np.concatenate(frames, axis=0)
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
        """
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
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
        """Encode numpy audio array to bytes."""
        try:
            import soundfile as sf  # type: ignore[import-not-found]
        except ImportError:
            return self._encode_wav_manual(audio)

        buffer = io.BytesIO()
        format_map = {
            AudioFormat.WAV: "WAV",
            AudioFormat.FLAC: "FLAC",
            AudioFormat.OGG: "OGG",
        }
        sf_format = format_map.get(format, "WAV")
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
        """Manually encode WAV without soundfile."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self._channels)
            wav.setsampwidth(2)
            wav.setframerate(self._sample_rate)
            wav.writeframes(audio.tobytes())
        return buffer.getvalue()

    def save(self, audio: bytes, path: str | Path) -> None:
        """Save recorded audio to a file."""
        path = Path(path)
        with open(path, "wb") as f:
            f.write(audio)
