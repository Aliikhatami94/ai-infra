from __future__ import annotations
from typing import List
from ai_infra.llm.providers import Providers
from ai_infra.llm.providers.models import Models

def validate_provider(provider: str) -> None:
    """Validate that the provider is supported."""
    provider_names: List[str] = [
        v for k, v in Providers.__dict__.items()
        if not k.startswith("__") and not callable(v)
    ]
    if provider not in provider_names:
        raise ValueError(f"Unknown provider: {provider}")

def validate_model(provider: str, model_name: str) -> None:
    """Validate that the model is supported for the given provider."""
    valid_models = getattr(Models, provider)
    if model_name not in [m.value for m in valid_models]:
        raise ValueError(f"Invalid model_name '{model_name}' for provider '{provider}'.")

def validate_provider_and_model(provider: str, model_name: str) -> None:
    validate_provider(provider)
    validate_model(provider, model_name)