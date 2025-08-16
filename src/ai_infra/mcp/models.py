from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import FastMCP


class ServerConfig(BaseModel):
    command: Optional[str] = None
    args: Optional[Any] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    module: Optional[FastMCP] = None
    transport: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

class Server(BaseModel):
    id: str
    name: str
    description: str
    config: ServerConfig

class McpConfig(BaseModel):
    name: str
    host: str
    prompts: Dict[str, Any]
    servers: List[Server]