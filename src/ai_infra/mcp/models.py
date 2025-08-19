# ai_infra/mcp/models.py
from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal, Awaitable, Callable
from pydantic import BaseModel, model_validator, Field

ToolFn = Callable[..., str | Awaitable[str]]

class ToolDef(BaseModel):
    fn: Optional[ToolFn] = Field(default=None, exclude=True)
    name: Optional[str] = None
    description: Optional[str] = None

class McpServerConfig(BaseModel):
    transport: Literal["stdio", "streamable_http", "sse"]

    # http-like
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    # stdio
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

    # opts
    stateless_http: Optional[bool] = None
    json_response: Optional[bool] = None
    oauth: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate(self):
        if self.transport in ("streamable_http", "sse") and not self.url:
            raise ValueError(f"{self.transport} requires 'url'")
        if self.transport == "stdio" and not self.command:
            raise ValueError("Remote stdio requires 'command'")
        return self