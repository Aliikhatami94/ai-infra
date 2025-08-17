from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, model_validator

# ---------- prompts/tools ----------
class Prompt(BaseModel):
    contents: Optional[List[str]] = None

# ---------- REMOTE (no module_path) ----------
class RemoteServerInfo(BaseModel):
    name: str
    description: Optional[str] = None

class RemoteServerConfig(BaseModel):
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
    info: RemoteServerInfo
    config: RemoteServerConfig

class OpenMcp(BaseModel):
    """Remote-only MCP config; accepts prompts."""
    prompts: List[Prompt] = []
    servers: List[RemoteServer] = []