from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, model_validator


class HostedServerInfo(BaseModel):
    name: str
    module_path: str  # e.g. "pkg.mod:mcp" or "pkg.mod:app"

class HostedServerConfig(BaseModel):
    # Typically you’ll mount a streamable-http FastMCP ASGI app
    transport: Literal["streamable_http", "stdio"] = "streamable_http"
    # where to mount in FastAPI (preferred). If omitted, you may derive from a URL.
    mount_path: Optional[str] = None
    # allow using a URL as a mount hint; your mounting util can extract the path.
    url: Optional[str] = None

    @model_validator(mode="after")
    def _validate(self):
        if self.transport == "streamable_http":
            if not (self.mount_path or self.url):
                raise ValueError("Hosted streamable_http requires 'mount_path' or 'url' (as mount hint).")
        if self.transport == "stdio":
            # only if you actually plan to spawn something; otherwise ignore
            pass
        return self

class HostedServer(BaseModel):
    info: HostedServerInfo
    config: HostedServerConfig

class HostedMcp(BaseModel):
    """Hosted-only MCP config; no prompts needed."""
    servers: List[HostedServer] = []