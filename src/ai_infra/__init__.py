import os
from dotenv import load_dotenv, find_dotenv

if not os.environ.get("AI_INFRA_ENV_LOADED"):
    load_dotenv(find_dotenv(usecwd=True))
    os.environ["AI_INFRA_ENV_LOADED"] = "1"

# Re-export primary public API components
from .llm.core import CoreLLM  # noqa: E402
from .llm.settings import ModelSettings  # noqa: E402
from .llm.providers import Providers  # noqa: E402
from .llm.models import Models  # noqa: E402

__all__ = [
    "CoreLLM",
    "ModelSettings",
    "Models",
    "Providers",
]

