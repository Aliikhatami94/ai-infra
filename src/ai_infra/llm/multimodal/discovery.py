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

from typing import Any

from ai_infra.providers import ProviderCapability, ProviderRegistry

# =============================================================================
# Helper to build legacy format from registry
# =============================================================================


def _build_tts_providers() -> dict[str, dict[str, Any]]:
    """Build TTS_PROVIDERS dict from registry."""
    result = {}
    for name in ProviderRegistry.list_for_capability(ProviderCapability.TTS):
        provider = ProviderRegistry.get(name)
        if not provider:
            continue
        cap = provider.get_capability(ProviderCapability.TTS)
        if not cap:
            continue
        result[name] = {
            "name": provider.display_name,
            "models": cap.models or [],
            "voices": cap.voices or [],
            "env_var": provider.env_var,
            "default_model": cap.default_model,
            "default_voice": cap.default_voice,
        }
    return result


def _build_stt_providers() -> dict[str, dict[str, Any]]:
    """Build STT_PROVIDERS dict from registry."""
    result = {}
    for name in ProviderRegistry.list_for_capability(ProviderCapability.STT):
        provider = ProviderRegistry.get(name)
        if not provider:
            continue
        cap = provider.get_capability(ProviderCapability.STT)
        if not cap:
            continue
        result[name] = {
            "name": provider.display_name,
            "models": cap.models or [],
            "env_var": provider.env_var,
            "default_model": cap.default_model,
            "features": cap.features or [],
        }
    return result


def _build_audio_llms() -> dict[str, dict[str, Any]]:
    """Build AUDIO_LLMS dict from registry (for realtime capability)."""
    result = {}
    for name in ProviderRegistry.list_for_capability(ProviderCapability.REALTIME):
        provider = ProviderRegistry.get(name)
        if not provider:
            continue
        cap = provider.get_capability(ProviderCapability.REALTIME)
        if not cap:
            continue
        # Build audio LLM structure
        result[name] = {
            "input": cap.extra.get("audio_input_models", []) if cap.extra else [],
            "output": cap.extra.get("audio_output_models", []) if cap.extra else [],
            "realtime": cap.models or [],
            "env_var": provider.env_var,
        }
    return result


# =============================================================================
# Legacy constants (built from registry for backwards compatibility)
# =============================================================================

TTS_PROVIDERS = _build_tts_providers()
STT_PROVIDERS = _build_stt_providers()
AUDIO_LLMS = _build_audio_llms()


# =============================================================================
# TTS Discovery
# =============================================================================


def list_tts_providers() -> list[str]:
    """List all supported TTS providers.

    Returns:
        List of provider names.
    """
    return ProviderRegistry.list_for_capability(ProviderCapability.TTS)


def list_tts_voices(provider: str = "openai") -> list[str]:
    """List available voices for a TTS provider.

    Args:
        provider: Provider name (openai, elevenlabs, google).

    Returns:
        List of voice names.
    """
    config = ProviderRegistry.get(provider)
    if not config:
        raise ValueError(f"Unknown TTS provider: {provider}. Available: {list_tts_providers()}")
    cap = config.get_capability(ProviderCapability.TTS)
    if not cap:
        raise ValueError(f"Provider {provider} does not support TTS")
    return cap.voices or []


def list_tts_models(provider: str = "openai") -> list[str]:
    """List available models for a TTS provider.

    Args:
        provider: Provider name.

    Returns:
        List of model names.
    """
    config = ProviderRegistry.get(provider)
    if not config:
        raise ValueError(f"Unknown TTS provider: {provider}. Available: {list_tts_providers()}")
    cap = config.get_capability(ProviderCapability.TTS)
    if not cap:
        raise ValueError(f"Provider {provider} does not support TTS")
    return cap.models or []


def get_tts_provider_info(provider: str) -> dict[str, Any]:
    """Get full info for a TTS provider.

    Args:
        provider: Provider name.

    Returns:
        Dict with provider details.
    """
    # Use legacy dict for backwards compatibility
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
    return ProviderRegistry.is_configured(provider)


def get_default_tts_provider() -> str | None:
    """Get the first configured TTS provider.

    Returns:
        Provider name or None if none configured.
    """
    return ProviderRegistry.get_default_for_capability(ProviderCapability.TTS)


# =============================================================================
# STT Discovery
# =============================================================================


def list_stt_providers() -> list[str]:
    """List all supported STT providers.

    Returns:
        List of provider names.
    """
    return ProviderRegistry.list_for_capability(ProviderCapability.STT)


def list_stt_models(provider: str = "openai") -> list[str]:
    """List available models for an STT provider.

    Args:
        provider: Provider name.

    Returns:
        List of model names.
    """
    config = ProviderRegistry.get(provider)
    if not config:
        raise ValueError(f"Unknown STT provider: {provider}. Available: {list_stt_providers()}")
    cap = config.get_capability(ProviderCapability.STT)
    if not cap:
        raise ValueError(f"Provider {provider} does not support STT")
    return cap.models or []


def get_stt_provider_info(provider: str) -> dict[str, Any]:
    """Get full info for an STT provider.

    Args:
        provider: Provider name.

    Returns:
        Dict with provider details.
    """
    # Use legacy dict for backwards compatibility
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
    return ProviderRegistry.is_configured(provider)


def get_default_stt_provider() -> str | None:
    """Get the first configured STT provider.

    Returns:
        Provider name or None if none configured.
    """
    return ProviderRegistry.get_default_for_capability(ProviderCapability.STT)


# =============================================================================
# Audio LLM Discovery
# =============================================================================


def list_audio_input_models(provider: str = "openai") -> list[str]:
    """List models that support audio input.

    Args:
        provider: Provider name.

    Returns:
        List of model names.
    """
    config = ProviderRegistry.get(provider)
    if not config:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {ProviderRegistry.list_for_capability(ProviderCapability.REALTIME)}"
        )
    cap = config.get_capability(ProviderCapability.REALTIME)
    if cap and cap.extra:
        result = cap.extra.get("audio_input_models", [])
        return list(result) if result else []
    return []


def list_audio_output_models(provider: str = "openai") -> list[str]:
    """List models that support audio output.

    Args:
        provider: Provider name.

    Returns:
        List of model names.
    """
    config = ProviderRegistry.get(provider)
    if not config:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {ProviderRegistry.list_for_capability(ProviderCapability.REALTIME)}"
        )
    cap = config.get_capability(ProviderCapability.REALTIME)
    if cap and cap.extra:
        result = cap.extra.get("audio_output_models", [])
        return list(result) if result else []
    return []


def list_realtime_models(provider: str = "openai") -> list[str]:
    """List models that support realtime audio streaming.

    Args:
        provider: Provider name.

    Returns:
        List of model names.
    """
    config = ProviderRegistry.get(provider)
    if not config:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {ProviderRegistry.list_for_capability(ProviderCapability.REALTIME)}"
        )
    cap = config.get_capability(ProviderCapability.REALTIME)
    return cap.models or [] if cap else []
