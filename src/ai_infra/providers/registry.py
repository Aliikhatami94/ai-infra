"""Central provider registry for ai-infra.

This module provides the ProviderRegistry class which serves as the single
source of truth for all provider configurations. All modules in ai-infra
should use this registry instead of maintaining their own provider configs.

Example:
    >>> from ai_infra.providers import ProviderRegistry, ProviderCapability
    >>>
    >>> # List all providers
    >>> ProviderRegistry.list_all()
    ['openai', 'anthropic', 'google_genai', ...]
    >>>
    >>> # List providers for a specific capability
    >>> ProviderRegistry.list_for_capability(ProviderCapability.TTS)
    ['openai', 'elevenlabs', 'google_genai']
    >>>
    >>> # Check if a provider is configured
    >>> ProviderRegistry.is_configured("openai")
    True
    >>>
    >>> # Get API key
    >>> ProviderRegistry.get_api_key("openai")
    'sk-...'
"""

from __future__ import annotations

import logging
import os

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig

log = logging.getLogger(__name__)


class ProviderRegistry:
    """Central registry for all AI provider configurations.

    This class maintains a registry of all available providers and their
    capabilities. Provider modules register themselves when imported.

    The registry is lazily initialized - provider configs are only loaded
    when first accessed.

    Class Attributes:
        _providers: Dict mapping provider name to ProviderConfig.
        _initialized: Whether provider modules have been imported.
    """

    _providers: dict[str, ProviderConfig] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, config: ProviderConfig) -> None:
        """Register a provider configuration.

        This is called by individual provider modules to add their config
        to the registry.

        Args:
            config: The provider configuration to register.

        Example:
            >>> from ai_infra.providers.registry import ProviderRegistry
            >>> from ai_infra.providers.base import ProviderConfig
            >>> config = ProviderConfig(name="my_provider", ...)
            >>> ProviderRegistry.register(config)
        """
        cls._providers[config.name] = config
        log.debug(f"Registered provider: {config.name}")

    @classmethod
    def get(cls, name: str) -> ProviderConfig | None:
        """Get provider configuration by name.

        Args:
            name: The provider name (e.g., "openai", "anthropic").

        Returns:
            ProviderConfig if found, None otherwise.

        Example:
            >>> config = ProviderRegistry.get("openai")
            >>> config.display_name
            'OpenAI'
        """
        cls._ensure_initialized()
        return cls._providers.get(name)

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names sorted alphabetically.

        Example:
            >>> ProviderRegistry.list_all()
            ['anthropic', 'cohere', 'deepgram', 'elevenlabs', 'google_genai', ...]
        """
        cls._ensure_initialized()
        return sorted(cls._providers.keys())

    @classmethod
    def list_configured(cls) -> list[str]:
        """List providers that have API keys configured.

        Returns:
            List of provider names that have their API key set.

        Example:
            >>> ProviderRegistry.list_configured()
            ['openai', 'anthropic']  # Only if these have API keys set
        """
        cls._ensure_initialized()
        return [name for name in cls._providers if cls.is_configured(name)]

    @classmethod
    def list_for_capability(cls, cap: ProviderCapability) -> list[str]:
        """List providers that support a specific capability.

        Args:
            cap: The capability to filter by.

        Returns:
            List of provider names that support the capability.

        Example:
            >>> ProviderRegistry.list_for_capability(ProviderCapability.TTS)
            ['elevenlabs', 'google_genai', 'openai']
        """
        cls._ensure_initialized()
        return sorted(name for name, config in cls._providers.items() if config.has_capability(cap))

    @classmethod
    def list_configured_for_capability(cls, cap: ProviderCapability) -> list[str]:
        """List configured providers that support a specific capability.

        Args:
            cap: The capability to filter by.

        Returns:
            List of provider names that support the capability AND have API keys.

        Example:
            >>> ProviderRegistry.list_configured_for_capability(ProviderCapability.CHAT)
            ['openai']  # Only if OPENAI_API_KEY is set
        """
        return [name for name in cls.list_for_capability(cap) if cls.is_configured(name)]

    @classmethod
    def is_configured(cls, name: str) -> bool:
        """Check if a provider has its API key configured.

        Checks the primary env var first, then any alternative env vars.

        Args:
            name: The provider name.

        Returns:
            True if the provider has an API key set.

        Example:
            >>> os.environ["OPENAI_API_KEY"] = "sk-test"
            >>> ProviderRegistry.is_configured("openai")
            True
        """
        config = cls.get(name)
        if not config:
            return False
        if os.environ.get(config.env_var):
            return True
        return any(os.environ.get(var) for var in config.alt_env_vars)

    @classmethod
    def get_api_key(cls, name: str) -> str | None:
        """Get the API key for a provider.

        Checks the primary env var first, then any alternative env vars.

        Args:
            name: The provider name.

        Returns:
            The API key string, or None if not configured.

        Example:
            >>> os.environ["OPENAI_API_KEY"] = "sk-test"
            >>> ProviderRegistry.get_api_key("openai")
            'sk-test'
        """
        config = cls.get(name)
        if not config:
            return None
        key = os.environ.get(config.env_var)
        if key:
            return key
        for var in config.alt_env_vars:
            key = os.environ.get(var)
            if key:
                return key
        return None

    @classmethod
    def get_env_var(cls, name: str) -> str | None:
        """Get the primary environment variable name for a provider.

        Args:
            name: The provider name.

        Returns:
            The env var name (e.g., "OPENAI_API_KEY"), or None if unknown provider.
        """
        config = cls.get(name)
        return config.env_var if config else None

    @classmethod
    def get_default_for_capability(
        cls,
        cap: ProviderCapability,
        priority: list[str] | None = None,
    ) -> str | None:
        """Get the first configured provider for a capability.

        This is useful for auto-selecting a provider when the user hasn't
        specified one explicitly.

        Args:
            cap: The capability required.
            priority: Optional list of providers to check first, in order.
                     Providers not in this list are checked after, alphabetically.

        Returns:
            Provider name if one is configured, None otherwise.

        Example:
            >>> # With OPENAI_API_KEY set
            >>> ProviderRegistry.get_default_for_capability(ProviderCapability.CHAT)
            'openai'
            >>> # With custom priority
            >>> ProviderRegistry.get_default_for_capability(
            ...     ProviderCapability.CHAT,
            ...     priority=["anthropic", "openai"]
            ... )
            'anthropic'  # If ANTHROPIC_API_KEY is set
        """
        providers = cls.list_for_capability(cap)
        if priority:
            # Reorder: priority list first, then the rest alphabetically
            priority_set = set(priority)
            providers = [p for p in priority if p in providers] + [
                p for p in providers if p not in priority_set
            ]
        for name in providers:
            if cls.is_configured(name):
                return name
        return None

    @classmethod
    def get_capability_config(cls, name: str, cap: ProviderCapability) -> CapabilityConfig | None:
        """Get capability configuration for a specific provider.

        Convenience method to get capability config in one call.

        Args:
            name: The provider name.
            cap: The capability to get config for.

        Returns:
            CapabilityConfig if found, None otherwise.

        Example:
            >>> config = ProviderRegistry.get_capability_config("openai", ProviderCapability.CHAT)
            >>> config.default_model
            'gpt-4o-mini'
        """
        provider = cls.get(name)
        return provider.get_capability(cap) if provider else None

    @classmethod
    def get_models(cls, name: str, cap: ProviderCapability) -> list[str]:
        """Get list of models for a provider's capability.

        Args:
            name: The provider name.
            cap: The capability to get models for.

        Returns:
            List of model names, empty if provider or capability not found.
        """
        config = cls.get_capability_config(name, cap)
        return config.models if config else []

    @classmethod
    def get_default_model(cls, name: str, cap: ProviderCapability) -> str | None:
        """Get the default model for a provider's capability.

        Args:
            name: The provider name.
            cap: The capability to get the default model for.

        Returns:
            Default model name, None if provider or capability not found.
        """
        config = cls.get_capability_config(name, cap)
        return config.default_model if config else None

    @classmethod
    def get_voices(cls, name: str, cap: ProviderCapability) -> list[str]:
        """Get list of voices for a provider's capability.

        Args:
            name: The provider name.
            cap: The capability to get voices for (usually TTS or REALTIME).

        Returns:
            List of voice names, empty if not applicable.
        """
        config = cls.get_capability_config(name, cap)
        return config.voices if config else []

    @classmethod
    def get_default_voice(cls, name: str, cap: ProviderCapability) -> str | None:
        """Get the default voice for a provider's capability.

        Args:
            name: The provider name.
            cap: The capability to get the default voice for.

        Returns:
            Default voice name, None if not applicable.
        """
        config = cls.get_capability_config(name, cap)
        return config.default_voice if config else None

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Lazy-load all provider configs.

        This imports all provider modules, which triggers their registration.
        Only runs once per process.
        """
        if cls._initialized:
            return

        # Import all provider modules to trigger registration
        # Using relative imports to avoid circular dependencies
        from ai_infra.providers import (  # noqa: F401
            anthropic,
            cohere,
            deepgram,
            elevenlabs,
            google,
            openai,
            replicate,
            stability,
            voyage,
            xai,
        )

        cls._initialized = True
        log.debug(f"Provider registry initialized with {len(cls._providers)} providers")

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing only).

        Clears all registered providers and resets initialization state.
        """
        cls._providers.clear()
        cls._initialized = False


# =============================================================================
# Convenience functions (module-level)
# =============================================================================


def get_provider(name: str) -> ProviderConfig | None:
    """Get provider configuration by name.

    Args:
        name: The provider name.

    Returns:
        ProviderConfig if found, None otherwise.
    """
    return ProviderRegistry.get(name)


def get_provider_config(name: str, capability: ProviderCapability) -> CapabilityConfig | None:
    """Get capability configuration for a provider.

    Args:
        name: The provider name.
        capability: The capability to get config for.

    Returns:
        CapabilityConfig if found, None otherwise.
    """
    return ProviderRegistry.get_capability_config(name, capability)


def list_providers() -> list[str]:
    """List all registered provider names.

    Returns:
        Sorted list of provider names.
    """
    return ProviderRegistry.list_all()


def list_configured_providers() -> list[str]:
    """List providers that have API keys configured.

    Returns:
        List of provider names with API keys set.
    """
    return ProviderRegistry.list_configured()


def list_providers_for_capability(cap: ProviderCapability) -> list[str]:
    """List providers that support a capability.

    Args:
        cap: The capability to filter by.

    Returns:
        Sorted list of provider names.
    """
    return ProviderRegistry.list_for_capability(cap)


def is_provider_configured(name: str) -> bool:
    """Check if a provider has its API key configured.

    Args:
        name: The provider name.

    Returns:
        True if API key is set.
    """
    return ProviderRegistry.is_configured(name)


def get_default_provider(cap: ProviderCapability) -> str | None:
    """Get the first configured provider for a capability.

    Args:
        cap: The capability required.

    Returns:
        Provider name if one is configured, None otherwise.
    """
    return ProviderRegistry.get_default_for_capability(cap)


def get_api_key(name: str) -> str | None:
    """Get the API key for a provider.

    Args:
        name: The provider name.

    Returns:
        API key string, or None if not configured.
    """
    return ProviderRegistry.get_api_key(name)
