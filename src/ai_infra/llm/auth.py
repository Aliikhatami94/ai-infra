"""Authentication helpers for AI providers.

This module provides utilities for managing API keys, especially for
BYOK (Bring Your Own Key) scenarios where users provide temporary keys.

Example:
    from ai_infra.llm.auth import temporary_api_key

    # User provides their own OpenAI key
    with temporary_api_key("openai", user_provided_key):
        result = await agent.astream("Hello")
    # Original key restored automatically
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager, contextmanager

# Map provider names to their environment variable names
PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


@contextmanager
def temporary_api_key(provider: str, api_key: str):
    """Temporarily set provider API key, restore original on exit.

    This is useful for BYOK (Bring Your Own Key) scenarios where users
    provide their own API keys that should only be active for specific
    operations.

    Args:
        provider: Provider name (openai, anthropic, google, xai, etc.)
        api_key: API key to use temporarily

    Raises:
        ValueError: If provider is not recognized

    Example:
        ```python
        # Use user's OpenAI key for this operation
        with temporary_api_key("openai", user_provided_key):
            agent = Agent(provider="openai")
            result = agent.run("Hello")
        # Original OPENAI_API_KEY restored here

        # Works with streaming
        with temporary_api_key("anthropic", user_key):
            async for event in agent.astream("Hello"):
                print(event.content)
        ```
    """
    # Normalize provider name
    provider_lower = provider.lower()

    env_var = PROVIDER_ENV_VARS.get(provider_lower)
    if not env_var:
        raise ValueError(
            f"Unknown provider: {provider}. Known providers: {list(PROVIDER_ENV_VARS.keys())}"
        )

    # Save original value
    original = os.environ.get(env_var)

    # Set temporary key
    os.environ[env_var] = api_key

    try:
        yield
    finally:
        # Restore original value
        if original is not None:
            os.environ[env_var] = original
        elif env_var in os.environ:
            # Key wasn't set before, remove it
            del os.environ[env_var]


@asynccontextmanager
async def atemporary_api_key(provider: str, api_key: str):
    """Async version of temporary_api_key().

    Provides the same BYOK functionality but works with async context managers.

    Args:
        provider: Provider name (openai, anthropic, google, xai, etc.)
        api_key: API key to use temporarily

    Raises:
        ValueError: If provider is not recognized

    Example:
        ```python
        # FastAPI endpoint with BYOK
        @app.post("/chat")
        async def chat(message: str, provider: str, api_key: str):
            async with atemporary_api_key(provider, api_key):
                async for event in agent.astream(message):
                    yield event.to_dict()
        ```
    """
    with temporary_api_key(provider, api_key):
        yield


def add_provider_mapping(provider: str, env_var: str) -> None:
    """Add a custom provider to environment variable mapping.

    This allows supporting additional providers not included by default.

    Args:
        provider: Provider name (lowercase recommended)
        env_var: Environment variable name for the provider's API key

    Example:
        ```python
        # Add support for a custom provider
        add_provider_mapping("custom_llm", "CUSTOM_LLM_API_KEY")

        # Now can use with temporary_api_key
        with temporary_api_key("custom_llm", user_key):
            ...
        ```
    """
    PROVIDER_ENV_VARS[provider.lower()] = env_var


def get_provider_env_var(provider: str) -> str | None:
    """Get the environment variable name for a provider.

    Args:
        provider: Provider name

    Returns:
        Environment variable name, or None if provider not recognized

    Example:
        ```python
        env_var = get_provider_env_var("openai")
        # "OPENAI_API_KEY"
        ```
    """
    return PROVIDER_ENV_VARS.get(provider.lower())


__all__ = [
    "PROVIDER_ENV_VARS",
    "add_provider_mapping",
    "atemporary_api_key",
    "get_provider_env_var",
    "temporary_api_key",
]
