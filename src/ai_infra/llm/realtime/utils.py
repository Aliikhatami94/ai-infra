"""Audio utilities for Realtime Voice API.

This module provides helper functions for working with audio data
in realtime voice applications:

- Sample rate conversion between providers (16kHz ↔ 24kHz)
- Audio chunking for streaming
- Format conversion utilities

Example:
    >>> from ai_infra.llm.realtime.utils import resample_pcm16, chunk_audio
    >>>
    >>> # Convert 16kHz audio to 24kHz
    >>> audio_24k = resample_pcm16(audio_16k, from_rate=16000, to_rate=24000)
    >>>
    >>> # Split audio into streaming chunks
    >>> for chunk in chunk_audio(audio_data, chunk_size=4800):
    ...     await session.send_audio(chunk)
"""

from __future__ import annotations

import struct
from collections.abc import Iterator


def resample_pcm16(audio: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM16 audio between sample rates.

    Uses linear interpolation for simple, efficient resampling.
    For production use with high quality requirements, consider
    using a dedicated audio library like scipy or librosa.

    Args:
        audio: Raw PCM16 audio bytes (signed 16-bit little-endian).
        from_rate: Source sample rate in Hz (e.g., 16000).
        to_rate: Target sample rate in Hz (e.g., 24000).

    Returns:
        Resampled PCM16 audio bytes.

    Example:
        >>> # Convert Gemini 16kHz input to OpenAI 24kHz
        >>> audio_24k = resample_pcm16(audio_16k, 16000, 24000)
        >>>
        >>> # Convert OpenAI 24kHz output to 16kHz for playback
        >>> audio_16k = resample_pcm16(audio_24k, 24000, 16000)
    """
    if from_rate == to_rate:
        return audio

    if len(audio) == 0:
        return audio

    # Unpack PCM16 samples (signed 16-bit little-endian)
    num_samples = len(audio) // 2
    samples = struct.unpack(f"<{num_samples}h", audio)

    # Calculate resampling ratio
    ratio = to_rate / from_rate
    new_length = int(len(samples) * ratio)

    if new_length == 0:
        return b""

    # Linear interpolation resampling
    resampled = []
    for i in range(new_length):
        src_idx = i / ratio
        idx = int(src_idx)
        frac = src_idx - idx

        if idx + 1 < len(samples):
            # Interpolate between adjacent samples
            sample = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
        else:
            # Use last sample
            sample = samples[idx] if idx < len(samples) else 0

        # Clamp to valid PCM16 range
        sample = max(-32768, min(32767, sample))
        resampled.append(sample)

    return struct.pack(f"<{len(resampled)}h", *resampled)


def chunk_audio(audio: bytes, chunk_size: int = 4800) -> Iterator[bytes]:
    """Split audio into chunks for streaming.

    Splits audio data into fixed-size chunks suitable for streaming
    over WebSocket connections. The last chunk may be smaller than
    the specified chunk size.

    Args:
        audio: Raw audio bytes to split.
        chunk_size: Size of each chunk in bytes.
                   Default is 4800 bytes (100ms at 24kHz PCM16).

    Yields:
        Audio chunks of the specified size.

    Example:
        >>> for chunk in chunk_audio(audio_data):
        ...     await session.send_audio(chunk)
        >>>
        >>> # Custom chunk size for lower latency
        >>> for chunk in chunk_audio(audio_data, chunk_size=2400):
        ...     await session.send_audio(chunk)
    """
    for i in range(0, len(audio), chunk_size):
        yield audio[i : i + chunk_size]


def pcm16_to_float32(audio: bytes) -> list[float]:
    """Convert PCM16 audio to normalized float32 samples.

    Args:
        audio: Raw PCM16 audio bytes.

    Returns:
        List of float samples in range [-1.0, 1.0].

    Example:
        >>> samples = pcm16_to_float32(audio_bytes)
        >>> # Now samples can be used with numpy, scipy, etc.
    """
    num_samples = len(audio) // 2
    samples = struct.unpack(f"<{num_samples}h", audio)
    return [s / 32768.0 for s in samples]


def float32_to_pcm16(samples: list[float]) -> bytes:
    """Convert normalized float32 samples to PCM16 audio.

    Args:
        samples: List of float samples in range [-1.0, 1.0].

    Returns:
        Raw PCM16 audio bytes.

    Example:
        >>> audio_bytes = float32_to_pcm16(processed_samples)
    """
    pcm_samples = []
    for s in samples:
        # Clamp to valid range
        s = max(-1.0, min(1.0, s))
        # Convert to 16-bit signed integer
        pcm_samples.append(int(s * 32767))
    return struct.pack(f"<{len(pcm_samples)}h", *pcm_samples)


def calculate_duration_ms(audio: bytes, sample_rate: int = 24000) -> float:
    """Calculate duration of PCM16 audio in milliseconds.

    Args:
        audio: Raw PCM16 audio bytes.
        sample_rate: Sample rate in Hz (default 24000).

    Returns:
        Duration in milliseconds.

    Example:
        >>> duration = calculate_duration_ms(audio_chunk)
        >>> print(f"Chunk is {duration:.1f}ms")
    """
    num_samples = len(audio) // 2  # 2 bytes per sample for PCM16
    return (num_samples / sample_rate) * 1000


def silence_pcm16(duration_ms: float, sample_rate: int = 24000) -> bytes:
    """Generate silent PCM16 audio of specified duration.

    Args:
        duration_ms: Duration of silence in milliseconds.
        sample_rate: Sample rate in Hz (default 24000).

    Returns:
        Raw PCM16 audio bytes containing silence.

    Example:
        >>> # Generate 500ms of silence
        >>> silence = silence_pcm16(500)
    """
    num_samples = int((duration_ms / 1000) * sample_rate)
    return b"\x00\x00" * num_samples  # Each sample is 2 bytes of zeros
