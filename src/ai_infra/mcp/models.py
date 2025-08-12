from typing import Dict, Any, Optional
from pydantic import BaseModel


class ServerConfig(BaseModel):
    command: Optional[str]
    args: Optional[Any]
    url: Optional[str]
    headers: Optional[Dict[str, str]]
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
    servers: Dict[str, Server]