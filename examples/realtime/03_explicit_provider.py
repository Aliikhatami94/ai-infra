#!/usr/bin/env python
"""Explicit Provider Selection for Realtime Voice Example.

This example demonstrates:
- Selecting specific providers (OpenAI, Gemini)
- Provider-specific configuration
- Voice selection per provider
- Model selection per provider
- Provider capabilities comparison

Choose the right provider based on your needs:
latency, cost, features, and voice options.
"""

import asyncio
import os

from ai_infra.llm.realtime import (
    RealtimeConfig,
    RealtimeVoice,
    VADMode,
)
from ai_infra.llm.realtime.gemini import GeminiRealtimeProvider
from ai_infra.llm.realtime.openai import OpenAIRealtimeProvider

# =============================================================================
# Example 1: OpenAI Realtime API
# =============================================================================


async def openai_provider():
    """Use OpenAI Realtime API explicitly."""
    print("=" * 60)
    print("1. OpenAI Realtime API")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n  ⚠ OPENAI_API_KEY not set")
        print("    Set the environment variable to use OpenAI")
    else:
        print("\n  ✓ OPENAI_API_KEY configured")

    # Explicit OpenAI selection
    voice = RealtimeVoice(provider="openai")

    print(f"\n  Provider: {voice.provider_name}")

    # OpenAI-specific configuration
    config = RealtimeConfig(
        model="gpt-4o-realtime-preview",
        voice="alloy",
        vad_mode=VADMode.SERVER,
        system_prompt="You are a helpful assistant.",
    )

    _voice_configured = RealtimeVoice(provider="openai", config=config)

    print("\n  OpenAI Configuration:")
    print(f"    Model: {config.model}")
    print(f"    Voice: {config.voice}")
    print(f"    VAD: {config.vad_mode}")


# =============================================================================
# Example 2: Google Gemini Live API
# =============================================================================


async def gemini_provider():
    """Use Google Gemini Live API explicitly."""
    print("\n" + "=" * 60)
    print("2. Google Gemini Live API")
    print("=" * 60)

    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        print("\n  ⚠ GOOGLE_API_KEY / GEMINI_API_KEY not set")
        print("    Set the environment variable to use Gemini")
    else:
        print("\n  ✓ Google API key configured")

    # Explicit Gemini selection
    voice = RealtimeVoice(provider="gemini")

    print(f"\n  Provider: {voice.provider_name}")

    # Gemini-specific configuration
    config = RealtimeConfig(
        model="gemini-2.0-flash-exp",
        voice="Kore",  # Gemini voice
        vad_mode=VADMode.SERVER,
        system_prompt="You are a helpful assistant.",
    )

    _voice_configured = RealtimeVoice(provider="gemini", config=config)

    print("\n  Gemini Configuration:")
    print(f"    Model: {config.model}")
    print(f"    Voice: {config.voice}")
    print(f"    VAD: {config.vad_mode}")


# =============================================================================
# Example 3: Available Voices
# =============================================================================


async def available_voices():
    """List available voices for each provider."""
    print("\n" + "=" * 60)
    print("3. Available Voices")
    print("=" * 60)

    # OpenAI voices
    openai_voices = OpenAIRealtimeProvider.list_voices()
    print(f"\n  OpenAI voices ({len(openai_voices)}):")
    for voice in openai_voices:
        print(f"    - {voice}")

    # Gemini voices
    gemini_voices = GeminiRealtimeProvider.list_voices()
    print(f"\n  Gemini voices ({len(gemini_voices)}):")
    for voice in gemini_voices:
        print(f"    - {voice}")

    print("\n  Voice selection:")
    print("""
    # OpenAI - use voice parameter
    config = RealtimeConfig(voice="nova")

    # Gemini - use voice parameter
    config = RealtimeConfig(voice="Kore")
""")


# =============================================================================
# Example 4: Available Models
# =============================================================================


async def available_models():
    """List available models for each provider."""
    print("\n" + "=" * 60)
    print("4. Available Models")
    print("=" * 60)

    # OpenAI models
    openai_models = OpenAIRealtimeProvider.list_models()
    print("\n  OpenAI Realtime models:")
    for model in openai_models:
        print(f"    - {model}")

    # Gemini models
    gemini_models = GeminiRealtimeProvider.list_models()
    print("\n  Gemini Live models:")
    for model in gemini_models:
        print(f"    - {model}")

    # Default models
    print("\n  Default models:")
    print(f"    OpenAI: {OpenAIRealtimeProvider.get_default_model()}")
    print(f"    Gemini: {GeminiRealtimeProvider.get_default_model()}")


# =============================================================================
# Example 5: Provider Configuration
# =============================================================================


async def provider_configuration():
    """Check if providers are configured."""
    print("\n" + "=" * 60)
    print("5. Provider Configuration Status")
    print("=" * 60)

    providers = [
        ("OpenAI", OpenAIRealtimeProvider, "OPENAI_API_KEY"),
        ("Gemini", GeminiRealtimeProvider, "GOOGLE_API_KEY or GEMINI_API_KEY"),
    ]

    print("\n  Provider status:")
    for name, provider_class, env_var in providers:
        configured = provider_class.is_configured()
        status = "✓ Configured" if configured else "✗ Not configured"
        print(f"    {name}: {status}")
        if not configured:
            print(f"      Set: {env_var}")

    # Using RealtimeVoice class methods
    print(f"\n  All available: {RealtimeVoice.available_providers()}")
    print(f"  Configured: {RealtimeVoice.configured_providers()}")


# =============================================================================
# Example 6: Provider-Specific Features
# =============================================================================


async def provider_features():
    """Compare provider-specific features."""
    print("\n" + "=" * 60)
    print("6. Provider-Specific Features")
    print("=" * 60)

    features = {
        "OpenAI Realtime": {
            "Audio Format": "PCM16 @ 24kHz",
            "VAD": "Server-side (excellent)",
            "Tool Calling": "Full support",
            "Latency": "~300ms",
            "Voices": "6 options",
            "Special": "Interruption handling",
        },
        "Gemini Live": {
            "Audio Format": "PCM16 @ 16kHz",
            "VAD": "Server-side",
            "Tool Calling": "Full support",
            "Latency": "~200-400ms",
            "Voices": "Multiple options",
            "Special": "Multimodal (video support)",
        },
    }

    for provider, feats in features.items():
        print(f"\n  {provider}:")
        for key, value in feats.items():
            print(f"    {key}: {value}")


# =============================================================================
# Example 7: Environment Variable Priority
# =============================================================================


async def env_priority():
    """Understand environment variable priority."""
    print("\n" + "=" * 60)
    print("7. Environment Variable Priority")
    print("=" * 60)

    print("\n  Auto-selection priority (first configured wins):")
    print("    1. REALTIME_VOICE_PROVIDER (explicit override)")
    print("    2. OPENAI_API_KEY → OpenAI")
    print("    3. GOOGLE_API_KEY or GEMINI_API_KEY → Gemini")

    print("\n  Force specific provider:")
    print("""
    # Via environment
    export REALTIME_VOICE_PROVIDER=gemini

    # Via code
    voice = RealtimeVoice(provider="gemini")
""")

    print("\n  Current environment:")
    env_vars = [
        "REALTIME_VOICE_PROVIDER",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask API keys
            if "KEY" in var or "TOKEN" in var:
                display = value[:8] + "..."
            else:
                display = value
            print(f"    {var}: {display}")
        else:
            print(f"    {var}: (not set)")


# =============================================================================
# Example 8: Custom Provider Instance
# =============================================================================


async def custom_provider():
    """Use provider instance directly."""
    print("\n" + "=" * 60)
    print("8. Custom Provider Instance")
    print("=" * 60)

    print("\n  Create provider instance directly:")
    print("""
    from ai_infra.llm.realtime.openai import OpenAIRealtimeProvider

    # Create with custom config
    config = RealtimeConfig(
        model="gpt-4o-realtime-preview",
        voice="shimmer",
    )

    provider = OpenAIRealtimeProvider(config=config)

    # Use provider directly
    async with provider.session() as session:
        await session.send_audio(audio_data)

    # Or pass to RealtimeVoice
    voice = RealtimeVoice(provider=provider)
""")


# =============================================================================
# Example 9: Switching Providers
# =============================================================================


async def switching_providers():
    """Switch between providers dynamically."""
    print("\n" + "=" * 60)
    print("9. Switching Providers")
    print("=" * 60)

    print("\n  Switch providers at runtime:")
    print("""
    # Start with OpenAI
    voice = RealtimeVoice(provider="openai")

    async with voice.connect() as session:
        # Use OpenAI
        pass

    # Switch to Gemini
    voice = RealtimeVoice(provider="gemini")

    async with voice.connect() as session:
        # Now using Gemini
        pass
""")

    print("\n  Fallback pattern:")
    print("""
    def get_voice():
        configured = RealtimeVoice.configured_providers()

        # Prefer OpenAI, fallback to Gemini
        if "openai" in configured:
            return RealtimeVoice(provider="openai")
        elif "gemini" in configured:
            return RealtimeVoice(provider="gemini")
        else:
            raise RuntimeError("No voice provider configured")
""")


# =============================================================================
# Example 10: Provider Selection Guide
# =============================================================================


async def selection_guide():
    """Guide for choosing the right provider."""
    print("\n" + "=" * 60)
    print("10. Provider Selection Guide")
    print("=" * 60)

    print("\n  Choose OpenAI when:")
    print("    ✓ You need the best voice quality")
    print("    ✓ Interruption handling is critical")
    print("    ✓ Using GPT-4 level reasoning")
    print("    ✓ Extensive tool calling")

    print("\n  Choose Gemini when:")
    print("    ✓ You need multimodal (video) support")
    print("    ✓ Cost optimization is priority")
    print("    ✓ Already using Google Cloud")
    print("    ✓ Need 16kHz audio (smaller bandwidth)")

    print("\n  Comparison summary:")
    print("""
    | Feature          | OpenAI      | Gemini      |
    |------------------|-------------|-------------|
    | Voice Quality    | Excellent   | Very Good   |
    | Latency          | ~300ms      | ~200-400ms  |
    | Tool Calling     | Full        | Full        |
    | Interruptions    | Excellent   | Good        |
    | Video Support    | No          | Yes         |
    | Audio Rate       | 24kHz       | 16kHz       |
""")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Explicit Provider Selection")
    print("=" * 60)
    print("\nChoose the right provider for your voice application.\n")

    await openai_provider()
    await gemini_provider()
    await available_voices()
    await available_models()
    await provider_configuration()
    await provider_features()
    await env_priority()
    await custom_provider()
    await switching_providers()
    await selection_guide()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Use provider='openai' or provider='gemini' explicitly")
    print("  2. Each provider has different voices and models")
    print("  3. OpenAI: 24kHz audio, best interruption handling")
    print("  4. Gemini: 16kHz audio, multimodal support")
    print("  5. Check is_configured() before using a provider")


if __name__ == "__main__":
    asyncio.run(main())
