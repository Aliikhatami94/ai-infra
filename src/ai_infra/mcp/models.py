from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal, Awaitable, Callable
from pydantic import BaseModel, model_validator, Field

# ---------- prompts/tools ----------
ToolFn = Callable[..., str | Awaitable[str]]
class ToolDef(BaseModel):
    fn: Optional[ToolFn] = Field(default=None, exclude=True)
    name: Optional[str] = None
    description: Optional[str] = None

class Prompt(BaseModel):
    contents: Optional[List[str]] = None

# ---------- REMOTE (no module_path) ----------
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
        if self.transport in ("streamable_http", "sse"):
            if not self.url:
                raise ValueError("Remote HTTP/SSE requires 'url'.")
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("Remote stdio requires 'command'.")
        return self

class RemoteServer(BaseModel):
    name: str
    config: McpServerConfig
    tools: Optional[List[ToolDef]] = None

class RemoteMcp(BaseModel):
    """Remote-only MCP config; accepts prompts."""
    prompts: List[Prompt] = []
    servers: List[RemoteServer] = []