"""Provider and model discovery for image generation.

This module provides functions to discover available providers and models,
including live fetching from provider APIs.

Usage:
    from ai_infra.imagegen.discovery import (
        list_providers,
        list_configured_providers,
        list_models,
        list_available_models,
        is_provider_configured,
    )

    # List all supported providers
    providers = list_providers()

    # List configured providers (have API keys)
    configured = list_configured_providers()

    # List static models for a provider
    models = list_models("google")

    # Fetch live models from API
    live_models = list_available_models("openai")
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_infra.imagegen.models import AVAILABLE_MODELS, ImageGenProvider

log = logging.getLogger(__name__)

# Supported providers
SUPPORTED_PROVIDERS = [p.value for p in ImageGenProvider]

# Environment variables for each provider
PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "stability": "STABILITY_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
}

# Cache settings
CACHE_TTL = 3600  # 1 hour
CACHE_DIR = Path.home() / ".cache" / "ai_infra" / "imagegen"


def list_providers() -> List[str]:
    """List all supported image generation providers.

    Returns:
        List of provider names.
    """
    return SUPPORTED_PROVIDERS.copy()


def is_provider_configured(provider: str) -> bool:
    """Check if a provider has an API key configured.

    Args:
        provider: Provider name.

    Returns:
        True if the provider has an API key set.
    """
    if provider not in SUPPORTED_PROVIDERS:
        return False

    env_var = PROVIDER_ENV_VARS.get(provider)
    if not env_var:
        return False

    # Google can use either GOOGLE_API_KEY or GEMINI_API_KEY
    if provider == "google":
        return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

    return bool(os.getenv(env_var))


def get_api_key(provider: str) -> Optional[str]:
    """Get the API key for a provider.

    Args:
        provider: Provider name.

    Returns:
        API key if configured, None otherwise.
    """
    if provider == "google":
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    env_var = PROVIDER_ENV_VARS.get(provider)
    return os.getenv(env_var) if env_var else None


def list_configured_providers() -> List[str]:
    """List providers that have API keys configured.

    Returns:
        List of configured provider names.
    """
    return [p for p in SUPPORTED_PROVIDERS if is_provider_configured(p)]


def list_models(provider: str) -> List[str]:
    """List static/known models for a provider.

    Args:
        provider: Provider name.

    Returns:
        List of model names.

    Raises:
        ValueError: If provider is not supported.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    p = ImageGenProvider(provider.lower())
    return AVAILABLE_MODELS.get(p, [])


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------


def _get_cache_path() -> Path:
    """Get the cache file path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / "models_cache.json"


def _load_cache() -> Dict[str, Any]:
    """Load the cache from disk."""
    cache_path = _get_cache_path()
    if cache_path.exists():
        try:
            import json

            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Save the cache to disk."""
    import json

    cache_path = _get_cache_path()
    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        log.warning(f"Failed to save cache: {e}")


def _is_cache_valid(cache: Dict[str, Any], provider: str) -> bool:
    """Check if cache entry is valid (not expired)."""
    if provider not in cache:
        return False
    entry = cache[provider]
    if "timestamp" not in entry or "models" not in entry:
        return False
    return time.time() - entry["timestamp"] < CACHE_TTL


def clear_cache() -> None:
    """Clear the model cache."""
    cache_path = _get_cache_path()
    if cache_path.exists():
        cache_path.unlink()
        log.info("Cache cleared")


# -----------------------------------------------------------------------------
# Live model fetchers
# -----------------------------------------------------------------------------


def _fetch_openai_models() -> List[str]:
    """Fetch available image models from OpenAI API."""
    import openai

    client = openai.OpenAI()
    models = client.models.list()

    # Filter to image generation models
    image_models = [
        m.id for m in models.data if m.id.startswith("dall-e") or "image" in m.id.lower()
    ]

    return sorted(image_models)


def _fetch_google_models() -> List[str]:
    """Fetch available image models from Google API."""
    from google import genai

    api_key = get_api_key("google")
    client = genai.Client(api_key=api_key)

    # List models and filter for image generation
    image_models = []
    for model in client.models.list():
        name = model.name
        # Models that can generate images
        if "imagen" in name.lower() or "image" in name.lower():
            # Remove 'models/' prefix if present
            model_id = name.replace("models/", "")
            image_models.append(model_id)

    return sorted(image_models)


def _fetch_stability_models() -> List[str]:
    """Fetch available models from Stability AI API."""
    import httpx

    api_key = get_api_key("stability")

    try:
        response = httpx.get(
            "https://api.stability.ai/v1/engines/list",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        engines = response.json()
        return sorted([e["id"] for e in engines])
    except Exception as e:
        log.warning(f"Failed to fetch Stability models: {e}")
        return list_models("stability")  # Fall back to static list


def _fetch_replicate_models() -> List[str]:
    """Fetch popular image models from Replicate.

    Note: Replicate has thousands of models, so we return curated image models.
    """
    # Replicate doesn't have a simple "list image models" API
    # Return our curated list of popular image generation models
    return list_models("replicate")


_FETCHERS = {
    "openai": _fetch_openai_models,
    "google": _fetch_google_models,
    "stability": _fetch_stability_models,
    "replicate": _fetch_replicate_models,
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def list_available_models(
    provider: str,
    *,
    refresh: bool = False,
    use_cache: bool = True,
) -> List[str]:
    """List available models for a provider by querying the API.

    This fetches live data from the provider's API to get the current
    list of available models.

    Args:
        provider: Provider name (e.g., "openai", "google").
        refresh: Force refresh from API, bypassing cache.
        use_cache: Whether to use cached results (default: True).

    Returns:
        List of model IDs available from the provider.

    Raises:
        ValueError: If provider is not supported.
        RuntimeError: If provider is not configured (no API key).

    Example:
        >>> list_available_models("openai")
        ['dall-e-2', 'dall-e-3']

        >>> list_available_models("google", refresh=True)
        ['gemini-2.0-flash-exp-image-generation', 'gemini-2.5-flash-image', ...]
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
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
    log.info(f"Fetching image models from {provider}...")
    fetcher = _FETCHERS.get(provider)
    if not fetcher:
        return list_models(provider)  # Fall back to static list

    try:
        models = fetcher()
    except Exception as e:
        log.warning(f"Failed to fetch models from {provider}: {e}")
        return list_models(provider)  # Fall back to static list

    # Update cache
    if use_cache and models:
        cache = _load_cache()
        cache[provider] = {
            "models": models,
            "timestamp": time.time(),
        }
        _save_cache(cache)

    return models


def list_all_available_models(
    *,
    refresh: bool = False,
    use_cache: bool = True,
    skip_unconfigured: bool = True,
) -> Dict[str, List[str]]:
    """List models for all configured providers.

    Args:
        refresh: Force refresh from API, bypassing cache.
        use_cache: Whether to use cached results.
        skip_unconfigured: Skip providers without API keys (default: True).

    Returns:
        Dict mapping provider name to list of model IDs.
        Example: {"openai": ["dall-e-2", "dall-e-3"], ...}
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
            models = list_available_models(provider, refresh=refresh, use_cache=use_cache)
            result[provider] = models
        except Exception as e:
            log.warning(f"Failed to list models for {provider}: {e}")
            result[provider] = []

    return result


__all__ = [
    "list_providers",
    "list_configured_providers",
    "list_models",
    "list_available_models",
    "list_all_available_models",
    "is_provider_configured",
    "get_api_key",
    "clear_cache",
    "SUPPORTED_PROVIDERS",
    "PROVIDER_ENV_VARS",
]
