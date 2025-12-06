"""FastAPI integration for ai-infra."""

from ai_infra.fastapi.chat import add_agent_endpoint
from ai_infra.fastapi.models import ChatRequest, ChatResponse

__all__ = [
    "add_agent_endpoint",
    "ChatRequest",
    "ChatResponse",
]
