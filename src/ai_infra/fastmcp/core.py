import os

try:
    # FastMCP 2.x
    from fastmcp import FastMCP
    from fastmcp.client import Client as _FastMCPClient
    # Optional OpenAPI routing/types if you want to pass custom maps
    try:
        # Prefer experimental parser if requested
        if os.getenv("FASTMCP_EXPERIMENTAL_ENABLE_NEW_OPENAPI_PARSER"):
            from fastmcp.experimental.server.openapi import RouteMap, MCPType  # type: ignore
        else:
            from fastmcp.server.openapi import RouteMap, MCPType  # type: ignore
    except Exception:
        RouteMap = MCPType = None  # type: ignore
except ImportError as e:
    raise RuntimeError("fastmcp package is required. `pip install fastmcp`") from e