from __future__ import annotations
from functools import lru_cache
from typing import Dict, Optional
from pydantic import BaseModel, Field, model_validator

from ai_infra.llm.models import OpenAIModels, AnthropicModels, GoogleGenAIModels, XAIModels

class ProviderConfig(BaseModel):
    temperature: float = 1.0
    models: BaseModel

def _default_providers() -> Dict[str, ProviderConfig]:
    return {
        "openai": ProviderConfig(models=OpenAIModels()),
        "anthropic": ProviderConfig(models=AnthropicModels()),
        "google_genai": ProviderConfig(models=GoogleGenAIModels()),
        "xai": ProviderConfig(models=XAIModels()),
    }

class LLMSettings(BaseModel):  # <- BaseModel, not BaseSettings
    # If you truly want no default, set this to None
    default_provider: Optional[str] = None
    providers: Dict[str, ProviderConfig] = Field(default_factory=_default_providers)

    @model_validator(mode="after")
    def _validate_default_in_providers(self) -> "LLMSettings":
        if self.default_provider is not None and self.default_provider not in self.providers:
            raise ValueError(
                f"default_provider '{self.default_provider}' not in providers {list(self.providers.keys())}"
            )
        return self

@lru_cache
def get_llm_settings() -> LLMSettings:
    # Pure in-code defaults, no env/ini/yaml
    return LLMSettings()