# ai_infra/mcp/models.py
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, model_validator


class McpServerConfig(BaseModel):
    transport: Literal["stdio", "streamable_http", "sse"]

    # http-like
    url: Optional[str] = None
    headers: Optional[dict[str, str]] = None

    # stdio
    command: Optional[str] = None
    args: Optional[list[str]] = None
    env: Optional[dict[str, str]] = None

    # opts
    stateless_http: Optional[bool] = None
    json_response: Optional[bool] = None
    oauth: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate(self):
        if self.transport in ("streamable_http", "sse") and not self.url:
            raise ValueError(f"{self.transport} requires 'url'")
        if self.transport == "stdio" and not self.command:
            raise ValueError("Remote stdio requires 'command'")
        return self
