from dotenv import load_dotenv, find_dotenv

from .llm.base import BaseLLM
from .llm.context import LLMContext
from .llm.settings import LLMSettings, get_llm_settings

load_dotenv(find_dotenv(usecwd=True), override=False)

__all__ = [
    "BaseLLM",
    "LLMContext",
    "LLMSettings",
    "get_llm_settings",
]

__version__ = "0.1.1"