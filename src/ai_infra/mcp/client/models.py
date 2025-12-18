# ai_infra/mcp/models.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, model_validator


class McpServerConfig(BaseModel):
    transport: Literal["stdio", "streamable_http", "sse"]

    # http-like
    url: str | None = None
    headers: dict[str, str] | None = None

    # stdio
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None

    # opts
    stateless_http: bool | None = None
    json_response: bool | None = None
    oauth: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate(self):
        if self.transport in ("streamable_http", "sse") and not self.url:
            raise ValueError(f"{self.transport} requires 'url'")
        if self.transport == "stdio" and not self.command:
            raise ValueError("Remote stdio requires 'command'")
        return self
