from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Iterable
import asyncio

os.environ["FASTMCP_EXPERIMENTAL_ENABLE_NEW_OPENAPI_PARSER"] = "true"

import httpx
from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.client import Client

__all__ = ["FastAPI"]

class CoreMCPServer:
    """
    Thin convenience wrapper around FastMCP.

    - Keep all FastMCP power available via `.mcp`
    - Add simple constructors for OpenAPI/FastAPI
    - Add helper to mount/run/list tools without extra boilerplate
    """

    def __init__(
            self,
            name: str | None = None,
            mcp: FastMCP | None = None,
            *,
            tags: Optional[Iterable[str]] = None,   # keep for your own API if you like
    ):
        self.mcp: FastMCP = mcp or FastMCP(name or "MCP Server")
        self._server_tags = set(tags or [])  # optional: store if you want

    # ---- Factories ----
    @classmethod
    def from_openapi(
            cls,
            openapi_spec: Dict[str, Any] | str,
            *,
            client: httpx.AsyncClient | None = None,
            name: str | None = None,
            route_maps: Any | None = None,
            route_map_fn: Any | None = None,
            mcp_component_fn: Any | None = None,
            mcp_names: Dict[str, str] | None = None,
            tags: Optional[Iterable[str]] = None,
            **fastmcp_kwargs: Any,
    ) -> "CoreMCPServer":
        ...
        mcp = FastMCP.from_openapi(
            openapi_spec=openapi_spec,
            client=client,
            name=name,
            route_maps=route_maps,
            route_map_fn=route_map_fn,
            mcp_component_fn=mcp_component_fn,
            mcp_names=mcp_names,
            tags=set(tags or []),
            **fastmcp_kwargs,
        )
        return cls(name=name, mcp=mcp, tags=tags)

    @classmethod
    def from_fastapi(
            cls,
            app: "FastAPI",
            *,
            name: str | None = None,
            route_maps: Any | None = None,
            route_map_fn: Any | None = None,
            mcp_component_fn: Any | None = None,
            mcp_names: Dict[str, str] | None = None,
            tags: Optional[Iterable[str]] = None,
            httpx_client_kwargs: Dict[str, Any] | None = None,
            **fastmcp_kwargs: Any,
    ) -> "CoreMCPServer":
        mcp = FastMCP.from_fastapi(
            app=app,
            name=name,
            route_maps=route_maps,
            route_map_fn=route_map_fn,
            mcp_component_fn=mcp_component_fn,
            mcp_names=mcp_names,
            tags=set(tags or []),
            httpx_client_kwargs=httpx_client_kwargs or {},
            **fastmcp_kwargs,
        )
        return cls(name=name, mcp=mcp, tags=tags)

    # ---- Passthroughs / Ergonomics ----
    @property
    def name(self) -> str:
        return self.mcp.name

    def tool(self, *dargs, **dkwargs):
        """
        Passthrough to @FastMCP.tool decorator.
        Usage:

            @server.tool
            def my_tool(...): ...

        or

            @server.tool(name="list_users")
            def list_users(...): ...
        """
        return self.mcp.tool(*dargs, **dkwargs)

    def add_tool(self, fn: Callable | None = None, /, **kwargs):
        """
        Convenience wrapper to add a tool from a callable without decorator.

        Example:
            server.add_tool(lambda x: x + 1, name="inc", description="Increment")
        """
        if fn is None:
            # Allow usage as decorator: server.add_tool(name="...")(func)
            def _decorator(func: Callable):
                self.mcp.tool(**kwargs)(func)
                return func
            return _decorator

        self.mcp.tool(**kwargs)(fn)
        return fn

    def http_app(self, path: str = "/mcp", **kwargs):
        """
        Return an ASGI app for mounting (FastAPI/Starlette):
            app.mount("/llm", server.http_app("/mcp"))
        """
        return self.mcp.http_app(path=path, **kwargs)

    def run(self, *args, **kwargs):
        """Run as a standalone MCP server (stdio/stdin/stdout)."""
        self.mcp.run(*args, **kwargs)

    # ---- Simple “test it” helpers (in-memory client) ----
    async def list_tools(self) -> list[Any]:
        """List tools via an in-memory client (no network needed)."""
        async with Client(self.mcp) as client:
            return await client.list_tools()

    async def call_tool(self, name: str, args: Dict[str, Any] | None = None) -> Any:
        """Call a tool via an in-memory client."""
        async with Client(self.mcp) as client:
            return await client.call_tool(name, args or {})

    # Handy accessor for folks who want the raw FastMCP:
    def unwrap(self) -> FastMCP:
        return self.mcp
