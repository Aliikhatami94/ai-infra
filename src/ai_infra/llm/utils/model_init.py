from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model


def build_model_key(provider: str, model_name: str) -> str:
    return f"{provider}:{model_name}"


def initialize_model(key: str, provider: str, **kwargs):
    """Initialize a chat model with the provider's API key.

    Args:
        key: The model key in format "provider:model_name"
        provider: The provider name (openai, anthropic, etc.)
        **kwargs: Additional kwargs to pass to init_chat_model
    """
    # Extract model name from key (format: "provider:model_name")
    model_name = key.split(":", 1)[1] if ":" in key else key

    # Remove any conflicting kwargs before passing to init_chat_model
    kwargs.pop("model", None)
    kwargs.pop("model_provider", None)

    return init_chat_model(
        model_name,
        model_provider=provider,
        api_key=os.environ.get(f"{provider.upper()}_API_KEY"),
        **kwargs,
    )


def sanitize_model_kwargs(
    model_kwargs: dict[str, Any], banned: list[str] | None = None
) -> dict[str, Any]:
    """Remove agent/tool-only kwargs from a model kwargs dict (mutates input)."""
    if not model_kwargs:
        return model_kwargs
    banned = banned or ["tools", "tool_choice", "parallel_tool_calls", "force_once"]
    for b in banned:
        model_kwargs.pop(b, None)
    return model_kwargs
