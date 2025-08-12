import os
from ai_infra.llm.providers import Providers
from ai_infra.llm.models import Models
from langchain.chat_models import init_chat_model

def validate_provider(provider: str) -> None:
    provider_names = [v for k, v in Providers.__dict__.items() if not k.startswith('__') and not callable(v)]
    if provider not in provider_names:
        raise ValueError(f"Unknown provider: {provider}")

def validate_model(provider: str, model_name: str) -> None:
    valid_models = getattr(Models, provider)
    if model_name not in [m.value for m in valid_models]:
        raise ValueError(f"Invalid model_name '{model_name}' for provider '{provider}'.")

def build_model_key(provider: str, model_name: str) -> str:
    return f"{provider}:{model_name}"

def initialize_model(key: str, provider: str, **kwargs):
    return init_chat_model(
        key,
        api_key=os.environ.get(f"{provider.upper()}_API_KEY"),
        **kwargs
    )

def validate_provider_and_model(provider: str, model_name: str) -> None:
    validate_provider(provider)
    validate_model(provider, model_name)
