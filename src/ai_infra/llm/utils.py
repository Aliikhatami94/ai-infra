import os
from langchain.chat_models import init_chat_model
from typing import Dict, Any, List, Tuple, Optional, Callable
import asyncio, random

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

def sanitize_model_kwargs(model_kwargs: Dict[str, Any], banned: Optional[List[str]] = None) -> Dict[str, Any]:
    """Remove agent/tool-only kwargs from a model kwargs dict (in place).

    Returns the mutated dict for chaining.
    """
    if not model_kwargs:
        return model_kwargs
    banned = banned or ["tools", "tool_choice", "parallel_tool_calls", "force_once"]
    for b in banned:
        model_kwargs.pop(b, None)
    return model_kwargs

async def with_retry(afn: Callable[[], Any], *, max_tries: int = 3, base: float = 0.5, jitter: float = 0.2):
    """Generic exponential backoff retry around an awaited call factory.

    Parameters:
        afn: zero-arg async callable to execute.
        max_tries: maximum attempts (>=1)
        base: base backoff seconds (grows exponentially)
        jitter: max random jitter seconds added each attempt
    """
    last = None
    for i in range(max_tries):
        try:
            return await afn()
        except Exception as e:  # pragma: no cover - defensive
            last = e
            if i == max_tries - 1:
                break
            await asyncio.sleep(base * (2 ** i) + random.random() * jitter)
    raise last if last else RuntimeError("Retry failed with unknown error")


def run_with_fallbacks(
    messages: List[Dict[str, Any]],
    candidates: List[Tuple[str, str]],
    run_single: Callable[[str, str], Any],
) -> Any:
    """Try (provider, model) pairs until one succeeds or exhaust.

    run_single: callable(provider, model_name) -> result
    """
    last_err = None
    for provider, model_name in candidates:
        try:
            return run_single(provider, model_name)
        except Exception as e:  # pragma: no cover - defensive
            last_err = e
            continue
    raise last_err or RuntimeError("All fallbacks failed")
