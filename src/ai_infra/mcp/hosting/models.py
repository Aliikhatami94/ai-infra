from __future__ import annotations
from typing import List
from pydantic import BaseModel
from ai_infra.mcp.models import McpServerConfig


class HostedServer(BaseModel):
    name: str
    module_path: str
    config: McpServerConfig

class HostedMcp(BaseModel):
    """Hosted-only MCP config; no prompts needed."""
    servers: List[HostedServer] = []