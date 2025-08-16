from __future__ import annotations
from pydantic import BaseModel, ConfigDict, model_validator
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP


class ServerConfig(BaseModel):
    command: Optional[str] = None
    args: Optional[Any] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    transport: str

class Server(BaseModel):
    id: str
    name: str
    description: str
    module: FastMCP | None = None
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

class McpConfig(BaseModel):
    name: str
    host: str
    prompts: Dict[str, Any]
    servers: List[Server]