from typing import Dict, Any
from pydantic import BaseModel


class ServerConfig(BaseModel):
    url: str
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