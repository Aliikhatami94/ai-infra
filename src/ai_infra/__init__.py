import os
from dotenv import load_dotenv, find_dotenv

if not os.environ.get("AI_INFRA_ENV_LOADED"):
    load_dotenv(find_dotenv(usecwd=True))
    os.environ["AI_INFRA_ENV_LOADED"] = "1"

# Re-export primary public API components
from ai_infra.llm.core import CoreLLM
from ai_infra.graph.core import CoreGraph
from ai_infra.llm.providers import Providers
from ai_infra.llm.providers.models import Models

__all__ = [
    "CoreGraph",
    "Models",
    "Providers",
    "CoreMCP",
]

