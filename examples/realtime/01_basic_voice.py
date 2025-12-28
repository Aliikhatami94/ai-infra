#!/usr/bin/env python
"""Basic Realtime Voice Conversation Example.

This example demonstrates:
- Zero-config voice conversation setup
- Callback-based event handling
- Sending and receiving audio
- Basic conversation loop
- Voice Activity Detection (VAD)

ai-infra provides a unified interface for real-time voice
conversations with OpenAI Realtime API and Google Gemini Live.
"""

import asyncio

from ai_infra.llm.realtime import (
    RealtimeConfig,
    RealtimeVoice,
    VADMode,
)

# =============================================================================
# Example 1: Zero-Config Voice Setup
# =============================================================================


async def zero_config():
    """Create a voice session with automatic provider detection."""
    print("=" * 60)
    print("1. Zero-Config Voice Setup")
    print("=" * 60)

    # Check available providers
    configured = RealtimeVoice.configured_providers()
    available = RealtimeVoice.available_providers()

    print(f"\n  Available providers: {available}")
    print(f"  Configured providers: {configured}")

    if not configured:
        print("\n  ⚠ No providers configured!")
        print("    Set OPENAI_API_KEY or GOOGLE_API_KEY")
        return

    # Auto-detect provider from environment
    voice = RealtimeVoice()

    print(f"\n  Selected provider: {voice.provider_name}")
    print("  ✓ Ready for voice conversation")


# =============================================================================
# Example 2: Basic Callback Registration
# =============================================================================


async def basic_callbacks():
    """Register callbacks for audio and transcript events."""
    print("\n" + "=" * 60)
    print("2. Basic Callback Registration")
    print("=" * 60)

    if not RealtimeVoice.configured_providers():
        print("\n  ⚠ No providers configured")
        return

    voice = RealtimeVoice()

    # Register transcript callback
    @voice.on_transcript
    async def handle_transcript(text: str, is_final: bool):
        """Called when speech is transcribed."""
        status = "Final" if is_final else "Interim"
        print(f"  [{status}] {text}")

    # Register audio callback
    @voice.on_audio
    async def handle_audio(audio: bytes):
        """Called when AI generates audio response."""
        print(f"  [Audio] Received {len(audio)} bytes")

    # Register error callback
    @voice.on_error
    async def handle_error(error):
        """Called on errors."""
        print(f"  [Error] {error}")

    print("\n  Callbacks registered:")
    print("    - on_transcript: Handle speech-to-text")
    print("    - on_audio: Handle AI audio output")
    print("    - on_error: Handle errors")

    print("\n  ✓ Voice session ready with callbacks")


# =============================================================================
# Example 3: Simple Conversation Loop
# =============================================================================


async def simple_conversation():
    """Run a basic voice conversation."""
    print("\n" + "=" * 60)
    print("3. Simple Conversation Loop")
    print("=" * 60)

    if not RealtimeVoice.configured_providers():
        print("\n  ⚠ No providers configured")
        return

    voice = RealtimeVoice()

    # Track conversation
    transcripts = []

    @voice.on_transcript
    async def on_transcript(text: str, is_final: bool):
        if is_final:
            transcripts.append(text)
            print(f"  AI: {text}")

    @voice.on_audio
    async def on_audio(audio: bytes):
        # In real app: play audio through speakers
        pass

    print("\n  Conversation pattern:")
    print("""
    async with voice.connect() as session:
        # Send user audio
        await session.send_audio(microphone_data)

        # Or send text (TTS)
        await session.send_text("Hello, how are you?")

        # Wait for response
        await asyncio.sleep(2)
""")

    print("\n  Note: Actual audio I/O requires sounddevice or pyaudio")


# =============================================================================
# Example 4: Voice Activity Detection
# =============================================================================


async def voice_activity_detection():
    """Configure Voice Activity Detection (VAD)."""
    print("\n" + "=" * 60)
    print("4. Voice Activity Detection (VAD)")
    print("=" * 60)

    print("\n  VAD Modes:")
    print("    - SERVER: Provider handles speech detection (recommended)")
    print("    - MANUAL: You control when speech starts/ends")
    print("    - NONE: No automatic detection")

    # Server-side VAD (recommended)
    config_server = RealtimeConfig(
        vad_mode=VADMode.SERVER,
        silence_duration_ms=500,  # End turn after 500ms silence
        prefix_padding_ms=300,  # Include audio before speech
    )

    # Manual VAD
    _config_manual = RealtimeConfig(
        vad_mode=VADMode.MANUAL,
    )

    print("\n  Server VAD config:")
    print(f"    vad_mode: {config_server.vad_mode}")
    print(f"    silence_duration_ms: {config_server.silence_duration_ms}")
    print(f"    prefix_padding_ms: {config_server.prefix_padding_ms}")

    print("\n  With manual VAD, call session.commit_audio() to end turn")


# =============================================================================
# Example 5: Audio Format
# =============================================================================


async def audio_format():
    """Understand audio format requirements."""
    print("\n" + "=" * 60)
    print("5. Audio Format Requirements")
    print("=" * 60)

    print("\n  OpenAI Realtime API:")
    print("    - Format: PCM16 (16-bit signed, little-endian)")
    print("    - Sample rate: 24000 Hz")
    print("    - Channels: Mono (1 channel)")

    print("\n  Google Gemini Live:")
    print("    - Format: PCM16")
    print("    - Sample rate: 16000 Hz")
    print("    - Channels: Mono")

    print("\n  Helper utilities:")
    print("""
    from ai_infra.llm.realtime import (
        pcm16_to_float32,
        float32_to_pcm16,
        resample_pcm16,
        chunk_audio,
    )

    # Convert float32 audio to PCM16
    pcm_audio = float32_to_pcm16(float_audio)

    # Resample to correct rate
    resampled = resample_pcm16(pcm_audio, from_rate=44100, to_rate=24000)

    # Chunk for streaming
    for chunk in chunk_audio(audio_data, chunk_ms=100):
        await session.send_audio(chunk)
""")


# =============================================================================
# Example 6: Session Management
# =============================================================================


async def session_management():
    """Manage voice session lifecycle."""
    print("\n" + "=" * 60)
    print("6. Session Management")
    print("=" * 60)

    print("\n  Session lifecycle:")
    print("""
    voice = RealtimeVoice()

    # Connect and get session (async context manager)
    async with voice.connect() as session:
        # Check session state
        print(f"Session ID: {session.session_id}")
        print(f"Active: {session.is_active}")

        # Send audio
        await session.send_audio(audio_chunk)

        # Send text (converted to speech by AI)
        await session.send_text("Hello!")

        # Interrupt AI response
        await session.interrupt()

        # Commit audio buffer (manual VAD)
        await session.commit_audio()

    # Session automatically closed on exit
""")

    print("\n  The context manager handles:")
    print("    - WebSocket connection")
    print("    - Session initialization")
    print("    - Cleanup on exit")
    print("    - Error handling")


# =============================================================================
# Example 7: Sending Audio Stream
# =============================================================================


async def sending_audio():
    """Stream audio to the voice session."""
    print("\n" + "=" * 60)
    print("7. Sending Audio Stream")
    print("=" * 60)

    print("\n  Pattern 1: Send chunks as they arrive")
    print("""
    async with voice.connect() as session:
        async for chunk in microphone_stream():
            await session.send_audio(chunk)
""")

    print("\n  Pattern 2: Send from file")
    print("""
    import wave

    with wave.open("audio.wav", "rb") as wav:
        chunk_size = 4800  # 100ms at 24kHz mono 16-bit
        while True:
            chunk = wav.readframes(chunk_size)
            if not chunk:
                break
            await session.send_audio(chunk)
            await asyncio.sleep(0.1)  # Simulate real-time
""")

    print("\n  Pattern 3: Using sounddevice")
    print("""
    import sounddevice as sd
    import asyncio

    async def mic_stream():
        q = asyncio.Queue()

        def callback(indata, frames, time, status):
            q.put_nowait(bytes(indata))

        with sd.InputStream(
            samplerate=24000,
            channels=1,
            dtype='int16',
            callback=callback,
        ):
            while True:
                chunk = await q.get()
                yield chunk
""")


# =============================================================================
# Example 8: Receiving Audio
# =============================================================================


async def receiving_audio():
    """Handle audio output from AI."""
    print("\n" + "=" * 60)
    print("8. Receiving Audio Output")
    print("=" * 60)

    print("\n  Pattern 1: Play immediately")
    print("""
    import sounddevice as sd
    import numpy as np

    @voice.on_audio
    async def play_audio(audio: bytes):
        # Convert to numpy array
        samples = np.frombuffer(audio, dtype=np.int16)
        # Play through speakers
        sd.play(samples, samplerate=24000)
""")

    print("\n  Pattern 2: Queue for playback")
    print("""
    audio_queue = asyncio.Queue()

    @voice.on_audio
    async def queue_audio(audio: bytes):
        await audio_queue.put(audio)

    # Separate task for playback
    async def playback_loop():
        while True:
            chunk = await audio_queue.get()
            play_audio(chunk)
""")

    print("\n  Pattern 3: Save to file")
    print("""
    audio_buffer = bytearray()

    @voice.on_audio
    async def collect_audio(audio: bytes):
        audio_buffer.extend(audio)

    # After conversation, save to WAV
    import wave
    with wave.open("response.wav", "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(24000)
        wav.writeframes(bytes(audio_buffer))
""")


# =============================================================================
# Example 9: Text Input
# =============================================================================


async def text_input():
    """Send text instead of audio."""
    print("\n" + "=" * 60)
    print("9. Text Input (Text-to-Speech)")
    print("=" * 60)

    print("\n  You can send text and receive audio response:")
    print("""
    async with voice.connect() as session:
        # Send text - AI responds with speech
        await session.send_text("What's the weather like today?")

        # Wait for and play audio response
        await asyncio.sleep(5)
""")

    print("\n  Useful for:")
    print("    - Testing without microphone")
    print("    - Accessibility features")
    print("    - Hybrid text/voice interfaces")


# =============================================================================
# Example 10: Error Handling
# =============================================================================


async def error_handling():
    """Handle errors in voice sessions."""
    print("\n" + "=" * 60)
    print("10. Error Handling")
    print("=" * 60)

    print("\n  Register error callback:")
    print("""
    @voice.on_error
    async def handle_error(error: RealtimeError):
        print(f"Error: {error.message}")
        if error.code:
            print(f"Code: {error.code}")
""")

    print("\n  Common errors:")
    print("    - RealtimeConnectionError: WebSocket connection failed")
    print("    - RealtimeError: General API error")
    print("    - Rate limit exceeded")
    print("    - Invalid audio format")

    print("\n  Best practices:")
    print("""
    try:
        async with voice.connect() as session:
            await session.send_audio(audio)
    except RealtimeConnectionError as e:
        print(f"Connection failed: {e}")
        # Retry logic
    except RealtimeError as e:
        print(f"Session error: {e}")
    finally:
        # Cleanup resources
        pass
""")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Basic Realtime Voice Examples")
    print("=" * 60)
    print("\nai-infra provides unified real-time voice conversations.")
    print("Supports OpenAI Realtime API and Google Gemini Live.\n")

    await zero_config()
    await basic_callbacks()
    await simple_conversation()
    await voice_activity_detection()
    await audio_format()
    await session_management()
    await sending_audio()
    await receiving_audio()
    await text_input()
    await error_handling()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. RealtimeVoice() auto-selects configured provider")
    print("  2. Use @voice.on_transcript for speech-to-text")
    print("  3. Use @voice.on_audio for AI audio output")
    print("  4. async with voice.connect() manages session lifecycle")
    print("  5. Audio format: PCM16, 24kHz (OpenAI) or 16kHz (Gemini)")


if __name__ == "__main__":
    asyncio.run(main())
