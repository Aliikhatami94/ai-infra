from __future__ import annotations
import os
from typing import Any, Dict, Optional, List
from langchain.chat_models import init_chat_model

def build_model_key(provider: str, model_name: str) -> str:
    return f"{provider}:{model_name}"

def initialize_model(key: str, provider: str, **kwargs):
    """Initialize a chat model with the provider's API key."""
    return init_chat_model(
        key,
        api_key=os.environ.get(f"{provider.upper()}_API_KEY"),
        **kwargs
    )

def sanitize_model_kwargs(model_kwargs: Dict[str, Any], banned: Optional[List[str]] = None) -> Dict[str, Any]:
    """Remove agent/tool-only kwargs from a model kwargs dict (mutates input)."""
    if not model_kwargs:
        return model_kwargs
    banned = banned or ["tools", "tool_choice", "parallel_tool_calls", "force_once"]
    for b in banned:
        model_kwargs.pop(b, None)
    return model_kwargs