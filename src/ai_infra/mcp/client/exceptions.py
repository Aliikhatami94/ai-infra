"""MCP Client exceptions."""

from __future__ import annotations

from typing import Any, Dict, Optional


class MCPError(Exception):
    """Base exception for MCP errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class MCPServerError(MCPError):
    """Error related to MCP server operations (connection, discovery, etc)."""

    def __init__(
        self,
        message: str,
        server_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.server_name = server_name


class MCPToolError(MCPError):
    """Error related to MCP tool calls."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        server_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.tool_name = tool_name
        self.server_name = server_name


class MCPTimeoutError(MCPError):
    """Timeout during MCP operations."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.operation = operation
        self.timeout = timeout


class MCPConnectionError(MCPServerError):
    """Error establishing or maintaining connection to MCP server."""

    pass
