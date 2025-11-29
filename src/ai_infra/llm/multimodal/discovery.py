"""Multimodal discovery - list providers, models, voices for TTS/STT.

This module provides discovery functions for multimodal capabilities:
- List TTS providers and voices
- List STT providers and models
- List audio-capable LLM models

Example:
    ```python
    from ai_infra.llm.multimodal import discovery

    # TTS
    print(discovery.list_tts_providers())
    print(discovery.list_tts_voices("openai"))

    # STT
    print(discovery.list_stt_providers())
    print(discovery.list_stt_models("openai"))

    # Audio LLMs
    print(discovery.list_audio_models("openai"))
    ```
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# =============================================================================
# TTS Discovery
# =============================================================================

TTS_PROVIDERS = {
    "openai": {
        "name": "OpenAI TTS",
        "models": ["tts-1", "tts-1-hd"],
        "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        "env_var": "OPENAI_API_KEY",
        "default_model": "tts-1",
        "default_voice": "alloy",
    },
    "elevenlabs": {
        "name": "ElevenLabs",
        "models": ["eleven_multilingual_v2", "eleven_turbo_v2"],
        "voices": [],  # Dynamic - fetched from API
        "env_var": "ELEVENLABS_API_KEY",
        "default_model": "eleven_multilingual_v2",
        "default_voice": "Rachel",
    },
    "google": {
        "name": "Google Cloud TTS",
        "models": ["standard", "neural2", "studio"],
        "voices": [],  # Dynamic - many voices
        "env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "default_model": "neural2",
        "default_voice": "en-US-Neural2-C",
    },
}


def list_tts_providers() -> List[str]:
    """List all supported TTS providers.

    Returns:
        List of provider names.
    """
    return list(TTS_PROVIDERS.keys())


def list_tts_voices(provider: str = "openai") -> List[str]:
    """List available voices for a TTS provider.

    Args:
        provider: Provider name (openai, elevenlabs, google).

    Returns:
        List of voice names.
    """
    info = TTS_PROVIDERS.get(provider)
    if not info:
        raise ValueError(f"Unknown TTS provider: {provider}. Available: {list_tts_providers()}")
    return info["voices"]


def list_tts_models(provider: str = "openai") -> List[str]:
    """List available models for a TTS provider.

    Args:
        provider: Provider name.

    Returns:
        List of model names.
    """
    info = TTS_PROVIDERS.get(provider)
    if not info:
        raise ValueError(f"Unknown TTS provider: {provider}. Available: {list_tts_providers()}")
    return info["models"]


def get_tts_provider_info(provider: str) -> Dict[str, Any]:
    """Get full info for a TTS provider.

    Args:
        provider: Provider name.

    Returns:
        Dict with provider details.
    """
    info = TTS_PROVIDERS.get(provider)
    if not info:
        raise ValueError(f"Unknown TTS provider: {provider}")
    return info


def is_tts_configured(provider: str) -> bool:
    """Check if a TTS provider is configured (API key set).

    Args:
        provider: Provider name.

    Returns:
        True if the provider's env var is set.
    """
    info = TTS_PROVIDERS.get(provider)
    if not info:
        return False
    return bool(os.environ.get(info["env_var"]))


def get_default_tts_provider() -> Optional[str]:
    """Get the first configured TTS provider.

    Returns:
        Provider name or None if none configured.
    """
    for provider in TTS_PROVIDERS:
        if is_tts_configured(provider):
            return provider
    return None


# =============================================================================
# STT Discovery
# =============================================================================

STT_PROVIDERS = {
    "openai": {
        "name": "OpenAI Whisper",
        "models": ["whisper-1"],
        "env_var": "OPENAI_API_KEY",
        "default_model": "whisper-1",
        "features": ["timestamps", "word_timestamps", "language_detection"],
    },
    "deepgram": {
        "name": "Deepgram",
        "models": ["nova-2", "nova", "enhanced", "base"],
        "env_var": "DEEPGRAM_API_KEY",
        "default_model": "nova-2",
        "features": ["streaming", "timestamps", "diarization", "smart_formatting"],
    },
    "google": {
        "name": "Google Cloud Speech-to-Text",
        "models": ["default", "latest_long", "latest_short"],
        "env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "default_model": "default",
        "features": ["streaming", "timestamps", "speaker_diarization"],
    },
}


def list_stt_providers() -> List[str]:
    """List all supported STT providers.

    Returns:
        List of provider names.
    """
    return list(STT_PROVIDERS.keys())


def list_stt_models(provider: str = "openai") -> List[str]:
    """List available models for an STT provider.

    Args:
        provider: Provider name.

    Returns:
        List of model names.
    """
    info = STT_PROVIDERS.get(provider)
    if not info:
        raise ValueError(f"Unknown STT provider: {provider}. Available: {list_stt_providers()}")
    return info["models"]


def get_stt_provider_info(provider: str) -> Dict[str, Any]:
    """Get full info for an STT provider.

    Args:
        provider: Provider name.

    Returns:
        Dict with provider details.
    """
    info = STT_PROVIDERS.get(provider)
    if not info:
        raise ValueError(f"Unknown STT provider: {provider}")
    return info


def is_stt_configured(provider: str) -> bool:
    """Check if an STT provider is configured (API key set).

    Args:
        provider: Provider name.

    Returns:
        True if the provider's env var is set.
    """
    info = STT_PROVIDERS.get(provider)
    if not info:
        return False
    return bool(os.environ.get(info["env_var"]))


def get_default_stt_provider() -> Optional[str]:
    """Get the first configured STT provider.

    Returns:
        Provider name or None if none configured.
    """
    for provider in STT_PROVIDERS:
        if is_stt_configured(provider):
            return provider
    return None


# =============================================================================
# Audio LLM Discovery
# =============================================================================

AUDIO_LLMS = {
    "openai": {
        "input": ["gpt-4o-audio-preview"],
        "output": ["gpt-4o-audio-preview"],
        "realtime": ["gpt-4o-realtime-preview"],
        "env_var": "OPENAI_API_KEY",
    },
    "google": {
        "input": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"],
        "output": ["gemini-2.0-flash-exp"],
        "realtime": [],
        "env_var": "GOOGLE_API_KEY",
    },
}


def list_audio_input_models(provider: str = "openai") -> List[str]:
    """List models that support audio input.

    Args:
        provider: Provider name.

    Returns:
        List of model names.
    """
    info = AUDIO_LLMS.get(provider)
    if not info:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(AUDIO_LLMS.keys())}")
    return info["input"]


def list_audio_output_models(provider: str = "openai") -> List[str]:
    """List models that support audio output.

    Args:
        provider: Provider name.

    Returns:
        List of model names.
    """
    info = AUDIO_LLMS.get(provider)
    if not info:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(AUDIO_LLMS.keys())}")
    return info["output"]


def list_realtime_models(provider: str = "openai") -> List[str]:
    """List models that support realtime audio streaming.

    Args:
        provider: Provider name.

    Returns:
        List of model names.
    """
    info = AUDIO_LLMS.get(provider)
    if not info:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(AUDIO_LLMS.keys())}")
    return info["realtime"]
