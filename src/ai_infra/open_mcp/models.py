from __future__ import annotations
from pydantic import ConfigDict
from typing import Dict, Any, List, Optional, Union, Awaitable, Callable, Literal
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
    name: str
    description: Optional[str] = None

class ServerConfig(BaseModel):
    # Transport selection + parameters for each transport
    transport: Literal["stdio", "streamable_http", "sse"] = "streamable_http"

    # Common
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    # stdio
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

    # streamable-http flavors
    stateless_http: Optional[bool] = None
    json_response: Optional[bool] = None  # stateless w/o SSE stream

    # auth (optional)
    oauth: Optional[Dict[str, Any]] = None  # client metadata, scopes, etc.

class Server(BaseModel):
    metadata: ServerMetadata
    module: FastMCP | None = Field(default=None, exclude=True)
    tools: Optional[List[ToolDef]] = None
    config: ServerConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

#  MCP configuration
class McpConfig(BaseModel):
    name: str
    host: str
    prompts: List[Prompt]
    servers: List[Server]