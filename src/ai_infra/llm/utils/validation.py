from __future__ import annotations

from ai_infra.llm.providers import Providers


def validate_provider(provider: str) -> None:
    """Validate that the provider is supported."""
    provider_names: list[str] = [
        v for k, v in Providers.__dict__.items() if not k.startswith("__") and not callable(v)
    ]
    if provider not in provider_names:
        raise ValueError(f"Unknown provider: {provider}. Supported: {provider_names}")


# NOTE: validate_model removed - we now allow any model string
# Users can use LLM.list_models() to discover available models dynamically
