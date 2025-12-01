"""
Dynamic model and provider discovery for ai-infra.

This module provides functions to discover available providers and models
at runtime by querying the provider APIs directly.

NOTE: This module now uses the centralized provider registry at
`ai_infra.providers`. The constants and functions here are maintained
for backwards compatibility but delegate to the registry.

Usage:
    from ai_infra.llm.providers.discovery import (
        list_providers,
        list_models,
        list_all_models,
        is_provider_configured,
    )

    # List all supported providers
    providers = list_providers()

    # List models for a specific provider
    models = list_models("openai")

    # Check if a provider has API key configured
    is_configured = is_provider_configured("openai")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from ai_infra.providers import ProviderCapability, ProviderRegistry

log = logging.getLogger(__name__)

# =============================================================================
# DEPRECATED: These constants are maintained for backwards compatibility.
# Use ai_infra.providers.ProviderRegistry instead.
# =============================================================================


# Provider configuration - now sourced from registry
# Kept for backwards compatibility with code that imports these directly
def _get_provider_env_vars() -> Dict[str, str]:
    """Get provider env vars from registry (backwards compat)."""
    result = {}
    for name in ProviderRegistry.list_for_capability(ProviderCapability.CHAT):
        config = ProviderRegistry.get(name)
        if config:
            result[name] = config.env_var
    return result


def _get_provider_alt_env_vars() -> Dict[str, List[str]]:
    """Get alternative env vars from registry (backwards compat)."""
    result = {}
    for name in ProviderRegistry.list_for_capability(ProviderCapability.CHAT):
        config = ProviderRegistry.get(name)
        if config and config.alt_env_vars:
            result[name] = config.alt_env_vars
    return result


# Backwards compatibility - these are now computed from registry
PROVIDER_ENV_VARS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
}

PROVIDER_ALT_ENV_VARS: Dict[str, List[str]] = {
    "google_genai": ["GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"],
}

SUPPORTED_PROVIDERS: List[str] = ["openai", "anthropic", "google_genai", "xai"]

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "ai-infra"
CACHE_FILE = CACHE_DIR / "models.json"
CACHE_TTL_SECONDS = 3600  # 1 hour


def list_providers() -> List[str]:
    """
    List all supported provider names for CHAT capability.

    Returns:
        List of provider names: ["openai", "anthropic", "google_genai", "xai"]
    """
    return ProviderRegistry.list_for_capability(ProviderCapability.CHAT)


def list_configured_providers() -> List[str]:
    """
    List providers that have API keys configured.

    Returns:
        List of provider names that have their API key env var set.
    """
    return ProviderRegistry.list_configured_for_capability(ProviderCapability.CHAT)


def is_provider_configured(provider: str) -> bool:
    """
    Check if a provider has its API key configured.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")

    Returns:
        True if the provider's API key environment variable is set.

    Raises:
        ValueError: If provider is not supported.
    """
    config = ProviderRegistry.get(provider)
    if not config:
        supported = ProviderRegistry.list_for_capability(ProviderCapability.CHAT)
        raise ValueError(f"Unknown provider: {provider}. Supported: {', '.join(supported)}")
    return ProviderRegistry.is_configured(provider)


def get_api_key(provider: str) -> Optional[str]:
    """
    Get the API key for a provider.

    Args:
        provider: Provider name

    Returns:
        API key string or None if not configured.
    """
    return ProviderRegistry.get_api_key(provider)


def get_default_provider() -> Optional[str]:
    """
    Auto-detect the default provider based on configured API keys.

    Checks providers in priority order:
    1. OpenAI (most common)
    2. Anthropic
    3. Google GenAI
    4. xAI

    Returns:
        Provider name if one is configured, None otherwise.

    Example:
        >>> # With OPENAI_API_KEY set
        >>> get_default_provider()
        'openai'
    """
    # Priority order for auto-detection
    priority_order = ["openai", "anthropic", "google_genai", "xai"]
    return ProviderRegistry.get_default_for_capability(
        ProviderCapability.CHAT, priority=priority_order
    )


# -----------------------------------------------------------------------------
# Per-provider model fetchers
# -----------------------------------------------------------------------------


def _list_openai_models() -> List[str]:
    """Fetch models from OpenAI API."""
    try:
        import openai

        client = openai.OpenAI()
        models = client.models.list()
        # Filter to chat/completion models, exclude embeddings/audio/etc
        chat_models = [
            m.id
            for m in models.data
            if any(prefix in m.id for prefix in ("gpt-4", "gpt-3.5", "o1", "o3", "chatgpt"))
            and "realtime" not in m.id
            and "audio" not in m.id
        ]
        return sorted(set(chat_models))
    except Exception as e:
        log.warning(f"Failed to fetch OpenAI models: {e}")
        return []


def _list_anthropic_models() -> List[str]:
    """Fetch models from Anthropic API."""
    try:
        import anthropic

        client = anthropic.Anthropic()
        models = client.models.list()
        return sorted([m.id for m in models.data])
    except Exception as e:
        log.warning(f"Failed to fetch Anthropic models: {e}")
        return []


def _list_google_models() -> List[str]:
    """Fetch models from Google GenAI API."""
    try:
        from google import genai

        # Google SDK uses GOOGLE_API_KEY by default, but we support multiple
        api_key = get_api_key("google_genai")
        client = genai.Client(api_key=api_key)
        models = client.models.list()
        # Filter to generative models
        gen_models = [
            m.name.replace("models/", "")
            for m in models
            if hasattr(m, "name") and "gemini" in m.name.lower()
        ]
        return sorted(set(gen_models))
    except Exception as e:
        log.warning(f"Failed to fetch Google GenAI models: {e}")
        return []


def _list_xai_models() -> List[str]:
    """Fetch models from xAI API (OpenAI-compatible)."""
    try:
        import openai

        client = openai.OpenAI(
            api_key=get_api_key("xai"),
            base_url="https://api.x.ai/v1",
        )
        models = client.models.list()
        return sorted([m.id for m in models.data])
    except Exception as e:
        log.warning(f"Failed to fetch xAI models: {e}")
        return []


# Fetcher dispatch
_FETCHERS = {
    "openai": _list_openai_models,
    "anthropic": _list_anthropic_models,
    "google_genai": _list_google_models,
    "xai": _list_xai_models,
}


# -----------------------------------------------------------------------------
# Caching
# -----------------------------------------------------------------------------


def _load_cache() -> Dict[str, any]:
    """Load cache from disk."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: Dict[str, any]) -> None:
    """Save cache to disk."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        log.warning(f"Failed to save cache: {e}")


def _is_cache_valid(cache: Dict[str, any], provider: str) -> bool:
    """Check if cache entry is still valid."""
    if provider not in cache:
        return False
    entry = cache[provider]
    if "timestamp" not in entry:
        return False
    age = time.time() - entry["timestamp"]
    return age < CACHE_TTL_SECONDS


def clear_cache() -> None:
    """Clear the model cache."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        log.info("Model cache cleared")


# -----------------------------------------------------------------------------
# Main discovery functions
# -----------------------------------------------------------------------------


def list_models(
    provider: str,
    *,
    refresh: bool = False,
    use_cache: bool = True,
) -> List[str]:
    """
    List available models for a specific provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        refresh: Force refresh from API, bypassing cache
        use_cache: Whether to use cached results (default: True)

    Returns:
        List of model IDs available from the provider.

    Raises:
        ValueError: If provider is not supported.
        RuntimeError: If provider is not configured (no API key).
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. " f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    if not is_provider_configured(provider):
        raise RuntimeError(
            f"Provider '{provider}' is not configured. "
            f"Set {PROVIDER_ENV_VARS[provider]} environment variable."
        )

    # Check cache
    if use_cache and not refresh:
        cache = _load_cache()
        if _is_cache_valid(cache, provider):
            log.debug(f"Using cached models for {provider}")
            return cache[provider]["models"]

    # Fetch from API
    log.info(f"Fetching models from {provider}...")
    fetcher = _FETCHERS.get(provider)
    if not fetcher:
        return []

    models = fetcher()

    # Update cache
    if use_cache and models:
        cache = _load_cache()
        cache[provider] = {
            "models": models,
            "timestamp": time.time(),
        }
        _save_cache(cache)

    return models


def list_all_models(
    *,
    refresh: bool = False,
    use_cache: bool = True,
    skip_unconfigured: bool = True,
) -> Dict[str, List[str]]:
    """
    List models for all configured providers.

    Args:
        refresh: Force refresh from API, bypassing cache
        use_cache: Whether to use cached results
        skip_unconfigured: Skip providers without API keys (default: True)

    Returns:
        Dict mapping provider name to list of model IDs.
        Example: {"openai": ["gpt-4o", "gpt-4o-mini", ...], ...}
    """
    result: Dict[str, List[str]] = {}

    for provider in SUPPORTED_PROVIDERS:
        if not is_provider_configured(provider):
            if skip_unconfigured:
                log.debug(f"Skipping {provider} (not configured)")
                continue
            else:
                result[provider] = []
                continue

        try:
            models = list_models(provider, refresh=refresh, use_cache=use_cache)
            result[provider] = models
        except Exception as e:
            log.warning(f"Failed to list models for {provider}: {e}")
            result[provider] = []

    return result


# -----------------------------------------------------------------------------
# Convenience exports
# -----------------------------------------------------------------------------

__all__ = [
    "list_providers",
    "list_configured_providers",
    "list_models",
    "list_all_models",
    "is_provider_configured",
    "get_api_key",
    "clear_cache",
    "SUPPORTED_PROVIDERS",
    "PROVIDER_ENV_VARS",
]
