from __future__ import annotations

from typing import Dict, Any, Optional, List

from mcp.server import FastMCP
from pydantic import BaseModel


class ServerConfig(BaseModel):
    command: Optional[str] = None
    args: Optional[Any] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    module: FastMCP = None
    transport: str

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