from __future__ import annotations
from typing import Literal, Optional, Union, List, Dict, Any, Awaitable, Callable
from pydantic import BaseModel, Field, ConfigDict, model_validator

# ---- prompts/tools (unchanged) ----
class Prompt(BaseModel):
    contents: Optional[List[str]] = None

ToolFn = Callable[..., Union[str, Awaitable[str]]]
class ToolDef(BaseModel):
    fn: Optional[ToolFn] = Field(default=None, exclude=True)
    name: Optional[str] = None
    description: Optional[str] = None

# ---- discriminated variants ----
class HostedServerInfo(BaseModel):
    type: Literal["hosted"] = "hosted"
    name: str
    description: Optional[str] = None
    module_path: str
    model_config = ConfigDict(extra="forbid")

class RemoteServerInfo(BaseModel):
    type: Literal["remote"] = "remote"
    name: str
    description: Optional[str] = None
    model_config = ConfigDict(extra="forbid")  # forbids module_path

# ---- transport config ----
class ServerConfig(BaseModel):
    transport: Literal["stdio", "streamable_http", "sse"] = "streamable_http"
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    stateless_http: Optional[bool] = None
    json_response: Optional[bool] = None
    oauth: Optional[Dict[str, Any]] = None

# ---- server ----
class Server(BaseModel):
    info: Union[HostedServerInfo, RemoteServerInfo] = Field(discriminator="type")
    config: ServerConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_cross_fields(self):
        if self.info.type == "hosted":
            # allow `config.url` as a mount hint or an explicit `mount_path`
            if self.config.transport in ("streamable_http", "sse"):
                if not (getattr(self.config, "mount_path", None) or self.config.url):
                    raise ValueError(
                        "Hosted HTTP/SSE requires either 'mount_path' or 'url' (used as a mount hint)."
                    )
            # if you require hosted+stdio to provide a command, keep this:
            if self.config.transport == "stdio" and not self.config.command:
                raise ValueError("Hosted+stdio requires 'command' if you intend to spawn a process.")
        else:  # remote
            if self.config.transport in ("streamable_http", "sse") and not self.config.url:
                raise ValueError("Remote HTTP/SSE servers require 'url'.")
            if self.config.transport == "stdio" and not self.config.command:
                raise ValueError("Remote stdio servers require 'command'.")
        return self

# ---- top-level config ----
class OpenMcp(BaseModel):
    prompts: List[Prompt] = []
    servers: List[Server] = []