"""Audio playback support for voice chat.

This module provides audio playback functionality:
- AudioPlayer: Play audio through the default output device

Example:
    ```python
    from ai_infra.llm.multimodal.voice import AudioPlayer

    player = AudioPlayer()

    # Play from bytes
    player.play(audio_bytes)

    # Play from file
    player.play_file("response.mp3")

    # Non-blocking playback
    player.play(audio_bytes, blocking=False)
    player.wait()
    player.stop()
    ```

Requirements:
    Install with voice extras: `pip install ai-infra[voice]`
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from ai_infra.llm.multimodal.models import AudioFormat, AudioSegment


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
        try:
            import sounddevice as sd  # type: ignore[import-not-found]

            sd.query_devices(kind="output")
            return True
        except (ImportError, Exception):
            pass

        if sys.platform == "darwin":
            return os.path.exists("/usr/bin/afplay")
        elif sys.platform.startswith("linux"):
            import shutil

            return shutil.which("aplay") is not None or shutil.which("paplay") is not None
        elif sys.platform == "win32":
            return True

        return False

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        """List available audio output devices."""
        try:
            import sounddevice as sd  # type: ignore[import-not-found]

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

        if self._try_sounddevice_play(audio_bytes, audio_format, blocking):
            return

        self._play_system(audio_bytes, audio_format, blocking)

    def play_file(
        self,
        path: str | Path,
        *,
        blocking: bool = True,
    ) -> None:
        """Play audio from a file."""
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
        if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
            return AudioFormat.WAV
        elif audio[:4] == b"fLaC":
            return AudioFormat.FLAC
        elif audio[:4] == b"OggS":
            return AudioFormat.OGG
        elif audio[:3] == b"ID3" or (audio[0:2] == b"\xff\xfb"):
            return AudioFormat.MP3
        else:
            return AudioFormat.MP3

    def _try_sounddevice_play(
        self,
        audio: bytes,
        format: AudioFormat,
        blocking: bool,
    ) -> bool:
        """Try to play using sounddevice. Returns True if successful."""
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
            import soundfile as sf  # type: ignore[import-not-found]
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
        suffix = f".{format.value}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio)
            temp_path = f.name

        try:
            if sys.platform == "darwin":
                cmd = ["afplay", temp_path]
            elif sys.platform.startswith("linux"):
                import shutil

                if shutil.which("paplay"):
                    cmd = ["paplay", temp_path]
                else:
                    cmd = ["aplay", temp_path]
            elif sys.platform == "win32":
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
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
