from __future__ import annotations
from pydantic import ConfigDict, model_validator
from typing import Dict, Any, List, Optional, Union, Awaitable, Callable
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

# Prompts
class Prompt(BaseModel):
    contents: Optional[List[str]] = None


# Tool definitions
ToolFn = Callable[..., Union[str, Awaitable[str]]]

class ToolDef(BaseModel):
    fn: Optional[ToolFn] = Field(default=None, exclude=True)
    name: Optional[str] = None
    description: Optional[str] = None


# Server metadata and configuration
class ServerMetadata(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None

class ServerConfig(BaseModel):
    command: Optional[str] = None
    args: Optional[Any] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    transport: str

class Server(BaseModel):
    metadata: ServerMetadata
    module: FastMCP | None = Field(default=None, exclude=True)
    tools: Optional[List[ToolDef]] = None
    config: ServerConfig

    # allow non-pydantic types like FastMCP
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_module(self):
        mod = self.module
        if mod is None:
            return self
        missing = []
        # check attributes without touching lazy properties
        cls = type(mod)
        if not (hasattr(cls, "session_manager") or "session_manager" in dir(cls)):
            missing.append("session_manager")
        if not hasattr(mod, "streamable_http_app"):
            missing.append("streamable_http_app()")
        if missing:
            raise ValueError(
                f"FastMCP at path='{self.path}' missing: {', '.join(missing)}"
            )
        return self


#  MCP configuration
class McpConfig(BaseModel):
    name: str
    host: str
    prompts: List[Prompt]
    servers: List[Server]