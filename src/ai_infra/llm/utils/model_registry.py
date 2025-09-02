from typing import Any, Dict

from ai_infra.llm.utils import (
    validate_provider_and_model,
    build_model_key,
    initialize_model,
)
from ai_infra.llm.providers.models import Models

def _norm_key(s: str) -> str:
    return (s or "").strip().lower().replace("-", "_")

def _norm_member_name(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", (s or "").strip().lower()).strip("_")

def _resolve_effective_model(provider: str, model_name: str | None) -> str:
    """
    If model_name is None or 'default', use Models.<provider>.default.value.
    If model_name matches an Enum member name (normalized), use that member's value.
    Otherwise, treat model_name as a raw id and return it as-is.
    """
    ns = getattr(Models, _norm_key(provider), None)
    if ns is None:
        return (model_name or "").strip()
    if model_name is None:
        return ns.default.value
    member = getattr(ns, _norm_member_name(model_name), None)
    return member.value if member is not None else model_name

class ModelRegistry:
    """Lightweight model cache / registry per provider+model key."""
    def __init__(self):
        self._models: Dict[str, Any] = {}

    def resolve_model_name(self, provider: str, model_name: str | None) -> str:
        return _resolve_effective_model(provider, model_name)

    def get_or_create(self, provider: str, model_name: str | None, **kwargs) -> Any:
        eff_model = self.resolve_model_name(provider, model_name)
        validate_provider_and_model(provider, eff_model)
        key = build_model_key(provider, eff_model)
        if key not in self._models:
            self._models[key] = initialize_model(key, provider, **(kwargs or {}))
        return self._models[key]

    def get(self, provider: str, model_name: str | None) -> Any:
        eff_model = self.resolve_model_name(provider, model_name)
        key = build_model_key(provider, eff_model)
        return self._models.get(key)