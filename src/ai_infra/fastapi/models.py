"""FastAPI models for agent endpoints."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Standard chat request model.

    Extend this for your app-specific fields:
        class MyChatRequest(ChatRequest):
            tenant_id: str
            session_id: str
    """

    message: str = Field(..., description="User message")

    # Optional overrides
    provider: Optional[str] = Field(None, description="LLM provider (openai, anthropic, etc.)")
    model: Optional[str] = Field(None, description="Model name")

    # History
    history: List[Dict[str, str]] = Field(default_factory=list, description="Conversation history")

    # Streaming
    stream: bool = Field(False, description="Enable SSE streaming")

    # Advanced
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ChatResponse(BaseModel):
    """
    Standard chat response model.

    Extend for app-specific fields:
        class MyChatResponse(ChatResponse):
            session_id: str
            credits_used: int
    """

    response: str = Field(..., description="Agent response")

    # Metadata
    provider: Optional[str] = Field(None, description="Provider used")
    model: Optional[str] = Field(None, description="Model used")

    # Stats
    tools_called: int = Field(0, description="Number of tools called")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
