# Model setup
from ai_infra.mcp.models import (
    RemoteServer,
    McpServerConfig,
)
from ai_infra.mcp.hosting.models import (
    HostedMcp,
    HostedServer,
)

# Main MCP classes and functions
from ai_infra.mcp.hosting import build_mcp_from_tools, add_mcp_to_fastapi
from ai_infra.mcp.openapi import build_mcp_from_openapi, load_openapi