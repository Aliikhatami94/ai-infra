"""MCP Client exceptions.

Re-exports from ai_infra.errors for backward compatibility.
All MCP exceptions should be imported from ai_infra.errors or this module.
"""

from ai_infra.errors import (
    MCPConnectionError,
    MCPError,
    MCPServerError,
    MCPTimeoutError,
    MCPToolError,
)

__all__ = [
    "MCPConnectionError",
    "MCPError",
    "MCPServerError",
    "MCPTimeoutError",
    "MCPToolError",
]
