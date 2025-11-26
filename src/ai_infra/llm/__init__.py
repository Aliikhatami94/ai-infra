# Backward-compatible deprecated aliases
from ai_infra.llm.core import LLM, Agent, BaseLLMCore, CoreAgent, CoreLLM
from ai_infra.llm.defaults import MODEL, PROVIDER
from ai_infra.llm.providers import Providers
from ai_infra.llm.providers.models import Models
from ai_infra.llm.tools import tools_from_functions
from ai_infra.llm.utils.settings import ModelSettings

__all__ = [
    # New names (preferred)
    "LLM",
    "Agent",
    "BaseLLMCore",
    "ModelSettings",
    "Models",
    "Providers",
    "PROVIDER",
    "MODEL",
    "tools_from_functions",
    # Deprecated aliases (backward compatibility)
    "CoreLLM",
    "CoreAgent",
]
