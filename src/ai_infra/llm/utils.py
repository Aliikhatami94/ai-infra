import os
from langchain.chat_models import init_chat_model

from ai_infra.llm.providers import Providers
from ai_infra.llm.models import Models

# Validation Functions
def validate_provider(provider: str) -> None:
    """Validate that the provider is supported."""
    provider_names = [v for k, v in Providers.__dict__.items() if not k.startswith('__') and not callable(v)]
    if provider not in provider_names:
        raise ValueError(f"Unknown provider: {provider}")

def validate_model(provider: str, model_name: str) -> None:
    """Validate that the model is supported for the given provider."""
    valid_models = getattr(Models, provider)
    if model_name not in [m.value for m in valid_models]:
        raise ValueError(f"Invalid model_name '{model_name}' for provider '{provider}'.")

def validate_provider_and_model(provider: str, model_name: str) -> None:
    """Validate both provider and model in a single call."""
    validate_provider(provider)
    validate_model(provider, model_name)

# Model Utility Functions
def build_model_key(provider: str, model_name: str) -> str:
    """Build a unique key for caching models."""
    return f"{provider}:{model_name}"

def initialize_model(key: str, provider: str, **kwargs):
    """Initialize a chat model with the given parameters."""
    return init_chat_model(
        key,
        api_key=os.environ.get(f"{provider.upper()}_API_KEY"),
        **kwargs
    )
