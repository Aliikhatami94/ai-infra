from pydantic import BaseModel
from typing import List

class OpenAIModels(BaseModel):
    models: List[str] = [
        "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-nano", "gpt-5-mini", "gpt-5-chat"
    ]

class AnthropicModels(BaseModel):
    models: List[str] = [
        "claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"
    ]

class GoogleGenAIModels(BaseModel):
    models: List[str] = [
        "gemini-2.5-flash"
    ]

class XAIModels(BaseModel):
    models: List[str] = [
        "grok-3", "grok-3-mini"
    ]

