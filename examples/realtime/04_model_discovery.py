#!/usr/bin/env python
"""Model and Provider Discovery for Realtime Voice Example.

This example demonstrates:
- Discovering available providers
- Listing models per provider
- Listing voices per provider
- Provider capabilities
- Runtime configuration discovery

Use discovery APIs to build adaptive UIs and
fallback logic for your voice applications.
"""

import asyncio

from ai_infra.llm.realtime import (
    RealtimeVoice,
)

# =============================================================================
# Example 1: List All Providers
# =============================================================================


async def list_providers():
    """List all available realtime voice providers."""
    print("=" * 60)
    print("1. List All Providers")
    print("=" * 60)

    # All supported providers (whether configured or not)
    available = RealtimeVoice.available_providers()
    print(f"\n  Available providers: {available}")

    # Only configured providers (with API keys)
    configured = RealtimeVoice.configured_providers()
    print(f"  Configured providers: {configured}")

    print("\n  Provider details:")
    for provider in available:
        is_ready = provider in configured
        status = "✓ Ready" if is_ready else "✗ Not configured"
        print(f"    {provider}: {status}")


# =============================================================================
# Example 2: List Models Per Provider
# =============================================================================


async def list_models():
    """List available models for each provider."""
    print("\n" + "=" * 60)
    print("2. List Models Per Provider")
    print("=" * 60)

    # Using the unified interface
    all_models = RealtimeVoice.list_models()
    print(f"\n  All models ({len(all_models)} total):")
    for model_info in all_models:
        print(f"    - {model_info['model']} ({model_info['provider']})")

    # Models for specific provider
    print("\n  Filter by provider:")
    openai_models = RealtimeVoice.list_models(provider="openai")
    print(f"    OpenAI: {[m['model'] for m in openai_models]}")

    gemini_models = RealtimeVoice.list_models(provider="gemini")
    print(f"    Gemini: {[m['model'] for m in gemini_models]}")


# =============================================================================
# Example 3: List Voices Per Provider
# =============================================================================


async def list_voices():
    """List available voices for each provider."""
    print("\n" + "=" * 60)
    print("3. List Voices Per Provider")
    print("=" * 60)

    # All voices across providers
    all_voices = RealtimeVoice.list_voices()
    print(f"\n  All voices ({len(all_voices)} total):")

    # Group by provider
    by_provider = {}
    for voice_info in all_voices:
        provider = voice_info["provider"]
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(voice_info["voice"])

    for provider, voices in by_provider.items():
        print(f"\n  {provider.title()} voices:")
        for voice in voices:
            print(f"    - {voice}")

    # Filter by provider
    print("\n  Filter example:")
    print("    openai_voices = RealtimeVoice.list_voices(provider='openai')")


# =============================================================================
# Example 4: Get Provider Capabilities
# =============================================================================


async def get_capabilities():
    """Get detailed capabilities for each provider."""
    print("\n" + "=" * 60)
    print("4. Provider Capabilities")
    print("=" * 60)

    capabilities = RealtimeVoice.get_capabilities()

    for provider, caps in capabilities.items():
        print(f"\n  {provider.title()}:")
        print(f"    Audio Input Format: {caps.get('input_format', 'PCM16')}")
        print(f"    Audio Sample Rate: {caps.get('sample_rate', 'varies')}")
        print(f"    VAD Support: {caps.get('vad_modes', [])}")
        print(f"    Tool Calling: {caps.get('tool_calling', False)}")
        print(f"    Multimodal: {caps.get('multimodal', False)}")
        print(f"    Max Audio Length: {caps.get('max_audio_seconds', 'unlimited')}")


# =============================================================================
# Example 5: Check Provider Configuration
# =============================================================================


async def check_configuration():
    """Check what's needed to configure each provider."""
    print("\n" + "=" * 60)
    print("5. Configuration Requirements")
    print("=" * 60)

    requirements = {
        "openai": {
            "env_vars": ["OPENAI_API_KEY"],
            "optional": ["OPENAI_ORG_ID"],
            "docs": "https://platform.openai.com/api-keys",
        },
        "gemini": {
            "env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],  # Either works
            "optional": ["GOOGLE_PROJECT_ID"],
            "docs": "https://ai.google.dev/",
        },
    }

    configured = RealtimeVoice.configured_providers()

    for provider, reqs in requirements.items():
        status = "✓ Configured" if provider in configured else "✗ Missing"
        print(f"\n  {provider.title()} ({status}):")
        print(f"    Required: {' or '.join(reqs['env_vars'])}")
        if reqs["optional"]:
            print(f"    Optional: {', '.join(reqs['optional'])}")
        print(f"    Docs: {reqs['docs']}")


# =============================================================================
# Example 6: Default Values
# =============================================================================


async def default_values():
    """Get default model and voice per provider."""
    print("\n" + "=" * 60)
    print("6. Default Values")
    print("=" * 60)

    defaults = RealtimeVoice.get_defaults()

    for provider, values in defaults.items():
        print(f"\n  {provider.title()} defaults:")
        print(f"    Model: {values.get('model', 'N/A')}")
        print(f"    Voice: {values.get('voice', 'N/A')}")
        print(f"    VAD Mode: {values.get('vad_mode', 'server')}")
        print(f"    Sample Rate: {values.get('sample_rate', 'N/A')} Hz")


# =============================================================================
# Example 7: Model Information
# =============================================================================


async def model_information():
    """Get detailed information about specific models."""
    print("\n" + "=" * 60)
    print("7. Model Information")
    print("=" * 60)

    model_info = {
        "gpt-4o-realtime-preview": {
            "provider": "openai",
            "type": "Preview",
            "features": ["voice", "text", "tool-calling", "interruptions"],
            "context_window": "128K tokens",
            "pricing": "Variable (audio + text tokens)",
        },
        "gemini-2.0-flash-exp": {
            "provider": "gemini",
            "type": "Experimental",
            "features": ["voice", "text", "video", "tool-calling"],
            "context_window": "1M tokens",
            "pricing": "Based on usage",
        },
    }

    for model, info in model_info.items():
        print(f"\n  {model}:")
        print(f"    Provider: {info['provider']}")
        print(f"    Type: {info['type']}")
        print(f"    Features: {', '.join(info['features'])}")
        print(f"    Context: {info['context_window']}")


# =============================================================================
# Example 8: Voice Characteristics
# =============================================================================


async def voice_characteristics():
    """Get characteristics of different voices."""
    print("\n" + "=" * 60)
    print("8. Voice Characteristics")
    print("=" * 60)

    # OpenAI voice descriptions
    openai_voices = {
        "alloy": "Neutral, balanced (default)",
        "echo": "Deep, warm",
        "fable": "Expressive, storytelling",
        "onyx": "Deep, authoritative",
        "nova": "Energetic, youthful",
        "shimmer": "Soft, gentle",
    }

    print("\n  OpenAI Voices:")
    for voice, description in openai_voices.items():
        print(f"    {voice}: {description}")

    # Gemini voices
    gemini_voices = {
        "Kore": "Balanced, clear",
        "Charon": "Deep, authoritative",
        "Fenrir": "Dynamic, expressive",
        "Aoede": "Warm, friendly",
        "Puck": "Playful, energetic",
    }

    print("\n  Gemini Voices:")
    for voice, description in gemini_voices.items():
        print(f"    {voice}: {description}")


# =============================================================================
# Example 9: Build Provider Selector UI
# =============================================================================


async def build_selector():
    """Example: Build a provider/voice selector UI."""
    print("\n" + "=" * 60)
    print("9. Build Provider Selector UI")
    print("=" * 60)

    print("\n  Pattern for building a settings UI:")
    print("""
    def get_voice_options():
        '''Return options for voice selection dropdown.'''
        options = []

        for provider in RealtimeVoice.configured_providers():
            voices = RealtimeVoice.list_voices(provider=provider)
            for voice_info in voices:
                options.append({
                    "value": f"{provider}:{voice_info['voice']}",
                    "label": f"{voice_info['voice']} ({provider})",
                    "provider": provider,
                })

        return options

    def get_model_options():
        '''Return options for model selection dropdown.'''
        options = []

        for provider in RealtimeVoice.configured_providers():
            models = RealtimeVoice.list_models(provider=provider)
            for model_info in models:
                options.append({
                    "value": model_info['model'],
                    "label": f"{model_info['model']}",
                    "provider": provider,
                })

        return options
""")


# =============================================================================
# Example 10: Fallback Configuration
# =============================================================================


async def fallback_configuration():
    """Configure fallback providers."""
    print("\n" + "=" * 60)
    print("10. Fallback Configuration")
    print("=" * 60)

    print("\n  Fallback pattern for reliability:")
    print("""
    def create_voice_with_fallback(
        preferred: str = "openai",
        fallback: str = "gemini"
    ) -> RealtimeVoice:
        '''Create voice with fallback provider.'''
        configured = RealtimeVoice.configured_providers()

        if preferred in configured:
            return RealtimeVoice(provider=preferred)
        elif fallback in configured:
            print(f"Using fallback: {fallback}")
            return RealtimeVoice(provider=fallback)
        else:
            raise RuntimeError(
                f"No providers configured. Need {preferred} or {fallback}"
            )

    # Usage
    voice = create_voice_with_fallback(
        preferred="openai",
        fallback="gemini"
    )
""")

    print("\n  Runtime fallback:")
    print("""
    async def run_with_fallback():
        providers = ["openai", "gemini"]

        for provider in providers:
            if provider not in RealtimeVoice.configured_providers():
                continue

            try:
                voice = RealtimeVoice(provider=provider)
                async with voice.connect() as session:
                    # Use session
                    return
            except Exception as e:
                print(f"{provider} failed: {e}, trying next...")
                continue

        raise RuntimeError("All providers failed")
""")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Model and Provider Discovery")
    print("=" * 60)
    print("\nDiscover providers, models, and voices at runtime.\n")

    await list_providers()
    await list_models()
    await list_voices()
    await get_capabilities()
    await check_configuration()
    await default_values()
    await model_information()
    await voice_characteristics()
    await build_selector()
    await fallback_configuration()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey discovery APIs:")
    print("  RealtimeVoice.available_providers() - All supported")
    print("  RealtimeVoice.configured_providers() - Ready to use")
    print("  RealtimeVoice.list_models(provider=...) - Available models")
    print("  RealtimeVoice.list_voices(provider=...) - Available voices")
    print("  RealtimeVoice.get_capabilities() - Provider features")
    print("  RealtimeVoice.get_defaults() - Default values")


if __name__ == "__main__":
    asyncio.run(main())
