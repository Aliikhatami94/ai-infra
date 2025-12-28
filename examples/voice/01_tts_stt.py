#!/usr/bin/env python
"""Text-to-Speech and Speech-to-Text Example.

This example demonstrates:
- Zero-config TTS/STT with automatic provider detection
- Provider discovery and configuration checking
- Converting text to speech (TTS)
- Transcribing audio to text (STT)
- Streaming audio generation
- Async operations

ai-infra provides a unified interface for voice synthesis and
transcription with support for OpenAI, Google Cloud, ElevenLabs,
and Deepgram.

Required API Keys (at least one per category):
TTS: OPENAI_API_KEY, ELEVEN_API_KEY, or GOOGLE_APPLICATION_CREDENTIALS
STT: OPENAI_API_KEY, DEEPGRAM_API_KEY, or GOOGLE_APPLICATION_CREDENTIALS
"""

import asyncio
import tempfile

from ai_infra import STT, TTS

# =============================================================================
# Example 1: Zero-Config TTS Setup
# =============================================================================


def zero_config_tts():
    """Create TTS with automatic provider detection."""
    print("=" * 60)
    print("1. Zero-Config TTS Setup")
    print("=" * 60)

    try:
        # Auto-detect provider from environment
        tts = TTS()
        print(f"\n  Provider: {tts.provider}")
        print(f"  Voice: {tts.voice}")
        print(f"  Model: {tts.model}")
        print("  ✓ TTS ready")
    except ValueError as e:
        print(f"\n  ⚠ No TTS provider configured: {e}")
        print("    Set OPENAI_API_KEY, ELEVEN_API_KEY, or GOOGLE_APPLICATION_CREDENTIALS")


# =============================================================================
# Example 2: Zero-Config STT Setup
# =============================================================================


def zero_config_stt():
    """Create STT with automatic provider detection."""
    print("\n" + "=" * 60)
    print("2. Zero-Config STT Setup")
    print("=" * 60)

    try:
        # Auto-detect provider from environment
        stt = STT()
        print(f"\n  Provider: {stt.provider}")
        print(f"  Model: {stt.model}")
        print(f"  Language: {stt.language or 'auto-detect'}")
        print("  ✓ STT ready")
    except ValueError as e:
        print(f"\n  ⚠ No STT provider configured: {e}")
        print("    Set OPENAI_API_KEY, DEEPGRAM_API_KEY, or GOOGLE_APPLICATION_CREDENTIALS")


# =============================================================================
# Example 3: Explicit Provider Configuration
# =============================================================================


def explicit_provider():
    """Configure TTS/STT with specific providers."""
    print("\n" + "=" * 60)
    print("3. Explicit Provider Configuration")
    print("=" * 60)

    # TTS with specific provider
    print("\n  TTS with OpenAI:")
    try:
        tts = TTS(provider="openai", voice="nova", model="tts-1-hd")
        print(f"    Provider: {tts.provider}")
        print(f"    Voice: {tts.voice}")
        print(f"    Model: {tts.model}")
    except Exception as e:
        print(f"    ⚠ {e}")

    # STT with specific provider
    print("\n  STT with Deepgram:")
    try:
        stt = STT(provider="deepgram", model="nova-2")
        print(f"    Provider: {stt.provider}")
        print(f"    Model: {stt.model}")
    except Exception as e:
        print(f"    ⚠ {e}")


# =============================================================================
# Example 4: Text-to-Speech Conversion
# =============================================================================


def text_to_speech():
    """Convert text to audio."""
    print("\n" + "=" * 60)
    print("4. Text-to-Speech Conversion")
    print("=" * 60)

    try:
        tts = TTS()
    except ValueError:
        print("\n  ⚠ No TTS provider configured")
        return

    text = "Hello! This is a demonstration of ai-infra text to speech capabilities."

    # Get audio bytes
    print("\n  Converting text to speech...")
    print(f"  Text: '{text[:50]}...'")

    try:
        audio_bytes = tts.speak(text)
        print(f"  ✓ Generated {len(audio_bytes):,} bytes of audio")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        return

    # Save to file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts.speak_to_file(text, f.name)
        print(f"  ✓ Saved to: {f.name}")


# =============================================================================
# Example 5: Speech-to-Text Transcription
# =============================================================================


def speech_to_text():
    """Transcribe audio to text."""
    print("\n" + "=" * 60)
    print("5. Speech-to-Text Transcription")
    print("=" * 60)

    try:
        stt = STT()
        print(f"\n  Provider: {stt.provider}")
    except ValueError:
        print("\n  ⚠ No STT provider configured")
        return

    # Note: This example shows the API, but requires an actual audio file
    print("\n  Transcription API demonstration:")
    print("  ")
    print("  # From file:")
    print("  result = stt.transcribe('audio.mp3')")
    print("  print(result.text)")
    print("  ")
    print("  # With timestamps:")
    print("  result = stt.transcribe('audio.mp3', timestamps=True)")
    print("  for segment in result.segments:")
    print("      print(f'{segment.start:.2f}s: {segment.text}')")
    print("  ")
    print("  # With language hint:")
    print("  result = stt.transcribe('audio.mp3', language='en')")


# =============================================================================
# Example 6: Different Voices and Formats
# =============================================================================


def voices_and_formats():
    """Use different voices and audio formats."""
    print("\n" + "=" * 60)
    print("6. Different Voices and Formats")
    print("=" * 60)

    try:
        tts = TTS(provider="openai")
    except ValueError:
        print("\n  ⚠ OpenAI TTS not configured")
        return

    from ai_infra.llm.multimodal.models import AudioFormat

    text = "Testing different voices."

    # Available OpenAI voices
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    print("\n  OpenAI TTS voices:")
    for voice in voices:
        print(f"    - {voice}")

    # Generate with different voice
    print("\n  Generating with 'nova' voice...")
    try:
        audio = tts.speak(text, voice="nova")
        print(f"  ✓ Generated {len(audio):,} bytes")
    except Exception as e:
        print(f"  ⚠ Error: {e}")

    # Different formats
    print("\n  Audio formats:")
    for fmt in [AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OGG]:
        try:
            audio = tts.speak(text, output_format=fmt)
            print(f"    {fmt.value}: {len(audio):,} bytes")
        except Exception as e:
            print(f"    {fmt.value}: ⚠ {e}")


# =============================================================================
# Example 7: Async Operations
# =============================================================================


async def async_operations():
    """Use async versions for concurrent processing."""
    print("\n" + "=" * 60)
    print("7. Async Operations")
    print("=" * 60)

    try:
        tts = TTS()
    except ValueError:
        print("\n  ⚠ No TTS provider configured")
        return

    texts = [
        "First sentence to convert.",
        "Second sentence to convert.",
        "Third sentence to convert.",
    ]

    print("\n  Converting multiple texts concurrently...")

    try:
        # Process all texts concurrently
        tasks = [tts.aspeak(text) for text in texts]
        results = await asyncio.gather(*tasks)

        for i, (text, audio) in enumerate(zip(texts, results), 1):
            print(f"    {i}. '{text[:30]}...' -> {len(audio):,} bytes")

        print("  ✓ All conversions complete")
    except Exception as e:
        print(f"  ⚠ Error: {e}")


# =============================================================================
# Example 8: Streaming TTS
# =============================================================================


def streaming_tts():
    """Stream audio for real-time playback."""
    print("\n" + "=" * 60)
    print("8. Streaming TTS")
    print("=" * 60)

    try:
        tts = TTS(provider="openai")
    except ValueError:
        print("\n  ⚠ OpenAI TTS not configured")
        return

    text = "This is a longer text that will be streamed in chunks for real-time playback."

    print("\n  Streaming audio generation...")
    print(f"  Text: '{text[:50]}...'")

    try:
        total_bytes = 0
        chunk_count = 0
        for chunk in tts.stream(text):
            chunk_count += 1
            total_bytes += len(chunk)
            print(f"    Chunk {chunk_count}: {len(chunk):,} bytes")

        print(f"  ✓ Total: {total_bytes:,} bytes in {chunk_count} chunks")
    except Exception as e:
        print(f"  ⚠ Error: {e}")


# =============================================================================
# Example 9: Round-Trip TTS + STT
# =============================================================================


def round_trip():
    """Convert text to speech and back to text."""
    print("\n" + "=" * 60)
    print("9. Round-Trip TTS + STT")
    print("=" * 60)

    try:
        tts = TTS()
        stt = STT()
    except ValueError as e:
        print(f"\n  ⚠ Provider not configured: {e}")
        return

    original_text = "Hello, this is a test of the round trip conversion."

    print(f"\n  Original: '{original_text}'")
    print(f"  TTS Provider: {tts.provider}")
    print(f"  STT Provider: {stt.provider}")

    # Convert text to speech
    print("\n  Step 1: Converting to speech...")
    try:
        audio = tts.speak(original_text)
        print(f"    ✓ Generated {len(audio):,} bytes")
    except Exception as e:
        print(f"    ⚠ TTS Error: {e}")
        return

    # Convert back to text
    print("  Step 2: Transcribing audio...")
    try:
        result = stt.transcribe(audio)
        print(f"    ✓ Transcribed: '{result.text}'")
    except Exception as e:
        print(f"    ⚠ STT Error: {e}")
        return

    # Compare
    print("\n  Comparison:")
    print(f"    Original:    '{original_text}'")
    print(f"    Transcribed: '{result.text}'")


# =============================================================================
# Example 10: Error Handling
# =============================================================================


def error_handling():
    """Handle common errors gracefully."""
    print("\n" + "=" * 60)
    print("10. Error Handling")
    print("=" * 60)

    print("\n  Common errors and handling:")

    # No provider configured
    print("\n  1. No provider configured:")
    print("     try:")
    print("         tts = TTS()")
    print("     except ValueError as e:")
    print("         print(f'No provider: {e}')")

    # Invalid provider
    print("\n  2. Invalid provider:")
    print("     try:")
    print("         tts = TTS(provider='invalid')")
    print("     except ValueError as e:")
    print("         print(f'Invalid: {e}')")

    # API errors
    print("\n  3. API errors:")
    print("     try:")
    print("         audio = tts.speak('text')")
    print("     except Exception as e:")
    print("         logger.error(f'TTS failed: {e}')")
    print("         # Handle gracefully or retry")

    # File not found (STT)
    print("\n  4. File not found:")
    print("     try:")
    print("         result = stt.transcribe_file('missing.mp3')")
    print("     except FileNotFoundError:")
    print("         print('Audio file not found')")

    print("\n  Best practices:")
    print("    - Always wrap API calls in try/except")
    print("    - Check provider availability before operations")
    print("    - Validate file paths before transcription")
    print("    - Log errors for debugging")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ai-infra TTS/STT Examples")
    print("=" * 60)
    print("\nThese examples demonstrate the Text-to-Speech and")
    print("Speech-to-Text capabilities of ai-infra.")
    print()

    # Sync examples
    zero_config_tts()
    zero_config_stt()
    explicit_provider()
    text_to_speech()
    speech_to_text()
    voices_and_formats()
    streaming_tts()
    round_trip()
    error_handling()

    # Async examples
    asyncio.run(async_operations())

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nFor realtime voice conversations, see:")
    print("  examples/realtime/01_basic_voice.py")


if __name__ == "__main__":
    main()
