from typing import Any

from ai_infra.llm.defaults import DEFAULT_MODELS

from .model_init import build_model_key, initialize_model
from .validation import validate_provider


def _norm_key(s: str) -> str:
    return (s or "").strip().lower().replace("-", "_")


def _resolve_effective_model(provider: str, model_name: str | None) -> str:
    """
    If model_name is None, use the default model for the provider.
    Otherwise, return model_name as-is (any model string is allowed).
    """
    if model_name is None:
        norm_provider = _norm_key(provider)
        return DEFAULT_MODELS.get(norm_provider, "gpt-4o-mini")
    return model_name.strip()


class ModelRegistry:
    """Lightweight model cache / registry per provider+model key."""

    def __init__(self):
        self._models: dict[str, Any] = {}

    def resolve_model_name(self, provider: str, model_name: str | None) -> str:
        return _resolve_effective_model(provider, model_name)

    def get_or_create(self, provider: str, model_name: str | None, **kwargs) -> Any:
        eff_model = self.resolve_model_name(provider, model_name)
        validate_provider(provider)  # Only validate provider, allow any model string
        key = build_model_key(provider, eff_model)
        if key not in self._models:
            self._models[key] = initialize_model(key, provider, **(kwargs or {}))
        return self._models[key]

    def get(self, provider: str, model_name: str | None) -> Any:
        eff_model = self.resolve_model_name(provider, model_name)
        key = build_model_key(provider, eff_model)
        return self._models.get(key)
