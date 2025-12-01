"""Unit tests for realtime audio utilities."""

import pytest

from ai_infra.llm.realtime.utils import (
    calculate_duration_ms,
    chunk_audio,
    float32_to_pcm16,
    pcm16_to_float32,
    resample_pcm16,
    silence_pcm16,
)


class TestResamplePcm16:
    """Test resample_pcm16 function."""

    def test_same_rate_returns_unchanged(self):
        """Test that same rate returns original audio."""
        audio = b"\x00\x10\x00\x20\x00\x30"  # 3 samples
        result = resample_pcm16(audio, 24000, 24000)
        assert result == audio

    def test_empty_audio_returns_empty(self):
        """Test that empty audio returns empty."""
        result = resample_pcm16(b"", 16000, 24000)
        assert result == b""

    def test_upsample_16k_to_24k(self):
        """Test upsampling from 16kHz to 24kHz."""
        # 4 samples at 16kHz
        audio = b"\x00\x10\x00\x20\x00\x30\x00\x40"
        result = resample_pcm16(audio, 16000, 24000)

        # Should be longer (1.5x samples)
        assert len(result) > len(audio)
        # 4 samples * 1.5 = 6 samples = 12 bytes
        assert len(result) == 12

    def test_downsample_24k_to_16k(self):
        """Test downsampling from 24kHz to 16kHz."""
        # 6 samples at 24kHz
        audio = b"\x00\x10\x00\x20\x00\x30\x00\x40\x00\x50\x00\x60"
        result = resample_pcm16(audio, 24000, 16000)

        # Should be shorter (2/3 samples)
        assert len(result) < len(audio)
        # 6 samples * (2/3) = 4 samples = 8 bytes
        assert len(result) == 8

    def test_preserves_silence(self):
        """Test that silence remains silence after resampling."""
        silence = b"\x00\x00" * 100
        result = resample_pcm16(silence, 16000, 24000)

        # All bytes should still be zero
        assert all(b == 0 for b in result)


class TestChunkAudio:
    """Test chunk_audio function."""

    def test_single_chunk(self):
        """Test audio smaller than chunk size."""
        audio = b"\x00" * 100
        chunks = list(chunk_audio(audio, chunk_size=200))

        assert len(chunks) == 1
        assert chunks[0] == audio

    def test_exact_chunks(self):
        """Test audio that divides evenly into chunks."""
        audio = b"\x00" * 100
        chunks = list(chunk_audio(audio, chunk_size=25))

        assert len(chunks) == 4
        assert all(len(c) == 25 for c in chunks)

    def test_partial_last_chunk(self):
        """Test audio with partial last chunk."""
        audio = b"\x00" * 100
        chunks = list(chunk_audio(audio, chunk_size=30))

        assert len(chunks) == 4
        assert len(chunks[0]) == 30
        assert len(chunks[1]) == 30
        assert len(chunks[2]) == 30
        assert len(chunks[3]) == 10  # Partial

    def test_empty_audio(self):
        """Test empty audio."""
        chunks = list(chunk_audio(b""))
        assert len(chunks) == 0

    def test_default_chunk_size(self):
        """Test default chunk size is 4800 bytes."""
        audio = b"\x00" * 10000
        chunks = list(chunk_audio(audio))

        assert len(chunks[0]) == 4800
        assert len(chunks[1]) == 4800
        assert len(chunks[2]) == 400  # Remainder


class TestPcm16ToFloat32:
    """Test pcm16_to_float32 function."""

    def test_silence(self):
        """Test silence converts to zeros."""
        audio = b"\x00\x00" * 10
        samples = pcm16_to_float32(audio)

        assert len(samples) == 10
        assert all(s == 0.0 for s in samples)

    def test_max_positive(self):
        """Test maximum positive value."""
        # 32767 in little-endian
        audio = b"\xff\x7f"
        samples = pcm16_to_float32(audio)

        assert len(samples) == 1
        assert samples[0] == pytest.approx(32767 / 32768.0)

    def test_max_negative(self):
        """Test maximum negative value."""
        # -32768 in little-endian
        audio = b"\x00\x80"
        samples = pcm16_to_float32(audio)

        assert len(samples) == 1
        assert samples[0] == pytest.approx(-1.0)

    def test_empty_audio(self):
        """Test empty audio."""
        samples = pcm16_to_float32(b"")
        assert samples == []


class TestFloat32ToPcm16:
    """Test float32_to_pcm16 function."""

    def test_silence(self):
        """Test zeros convert to silence."""
        samples = [0.0] * 10
        audio = float32_to_pcm16(samples)

        assert len(audio) == 20  # 2 bytes per sample
        assert all(b == 0 for b in audio)

    def test_max_positive(self):
        """Test +1.0 converts to max positive."""
        samples = [1.0]
        audio = float32_to_pcm16(samples)

        # Should be 32767 in little-endian
        assert audio == b"\xff\x7f"

    def test_max_negative(self):
        """Test -1.0 converts to max negative."""
        samples = [-1.0]
        audio = float32_to_pcm16(samples)

        # Should be -32767 in little-endian (we use 32767 as max)
        assert audio == b"\x01\x80"

    def test_clipping(self):
        """Test values outside [-1, 1] are clipped."""
        samples = [2.0, -2.0]
        audio = float32_to_pcm16(samples)

        # Both should be clipped
        assert len(audio) == 4

    def test_roundtrip(self):
        """Test roundtrip conversion."""
        original = [0.0, 0.5, -0.5, 0.25]
        audio = float32_to_pcm16(original)
        recovered = pcm16_to_float32(audio)

        # Should be approximately equal (quantization error)
        for o, r in zip(original, recovered):
            assert o == pytest.approx(r, abs=1e-4)


class TestCalculateDurationMs:
    """Test calculate_duration_ms function."""

    def test_one_second_24k(self):
        """Test 1 second at 24kHz."""
        # 24000 samples * 2 bytes = 48000 bytes
        audio = b"\x00" * 48000
        duration = calculate_duration_ms(audio, 24000)

        assert duration == pytest.approx(1000.0)

    def test_one_second_16k(self):
        """Test 1 second at 16kHz."""
        # 16000 samples * 2 bytes = 32000 bytes
        audio = b"\x00" * 32000
        duration = calculate_duration_ms(audio, 16000)

        assert duration == pytest.approx(1000.0)

    def test_100ms_24k(self):
        """Test 100ms at 24kHz."""
        # 2400 samples * 2 bytes = 4800 bytes
        audio = b"\x00" * 4800
        duration = calculate_duration_ms(audio, 24000)

        assert duration == pytest.approx(100.0)

    def test_empty_audio(self):
        """Test empty audio is 0ms."""
        duration = calculate_duration_ms(b"")
        assert duration == 0.0


class TestSilencePcm16:
    """Test silence_pcm16 function."""

    def test_100ms_at_24k(self):
        """Test 100ms of silence at 24kHz."""
        audio = silence_pcm16(100, 24000)

        # 100ms * 24 samples/ms * 2 bytes = 4800 bytes
        assert len(audio) == 4800
        assert all(b == 0 for b in audio)

    def test_1000ms_at_16k(self):
        """Test 1000ms of silence at 16kHz."""
        audio = silence_pcm16(1000, 16000)

        # 1000ms * 16 samples/ms * 2 bytes = 32000 bytes
        assert len(audio) == 32000

    def test_default_sample_rate(self):
        """Test default sample rate is 24kHz."""
        audio = silence_pcm16(100)

        # Should use 24000 Hz default
        assert len(audio) == 4800

    def test_zero_duration(self):
        """Test 0ms returns empty."""
        audio = silence_pcm16(0)
        assert audio == b""
