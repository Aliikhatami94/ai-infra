# Model setup
from ai_infra.mcp.client.core import CoreMCPClient, MCPClient
from ai_infra.mcp.client.models import McpServerConfig

# Backward-compatible deprecated aliases
# Main MCP classes (new names)
from ai_infra.mcp.server.core import CoreMCPServer, MCPServer
from ai_infra.mcp.server.openapi import load_openapi, load_spec
from ai_infra.mcp.server.tools import mcp_from_functions

__all__ = [
    # New names (preferred)
    "MCPServer",
    "MCPClient",
    "McpServerConfig",
    "load_openapi",
    "load_spec",
    "mcp_from_functions",
    # Deprecated aliases (backward compatibility)
    "CoreMCPServer",
    "CoreMCPClient",
]
