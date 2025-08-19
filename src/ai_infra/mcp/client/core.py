from __future__ import annotations
from typing import Dict, Any, AsyncContextManager
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    StreamableHttpConnection, StdioConnection, SSEConnection
)
from langchain_mcp_adapters.tools import load_mcp_tools

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

from ai_infra.mcp.models import RemoteServer, ToolDef


class CoreMCPClient:
    def __init__(self, config: list[dict] | list[RemoteServer]):
        if isinstance(config, list):
            self.config = [
                s if isinstance(s, RemoteServer) else RemoteServer.model_validate(s)
                for s in config
            ]
        else:
            raise TypeError("Config must be a list of servers")

    # ---------- per-server session helpers (raw MCP) ----------

    @staticmethod
    def _set_server_info_on_session(session: ClientSession, init_result):
        """Store initialize()'s server info on the session for easy access."""
        # try common field names across library versions
        info = (
                getattr(init_result, "server_info", None)
                or getattr(init_result, "serverInfo", None)
                or getattr(init_result, "serverinfo", None)
        )
        if info is None:
            return
        if is_dataclass(info):
            info = asdict(info)
        elif hasattr(info, "model_dump"):
            info = info.model_dump()
        session.mcp_server_info = info

    def _open_session(self, srv: RemoteServer) -> AsyncContextManager[ClientSession]:
        t = srv.config.transport

        if t == "stdio":
            params = StdioServerParameters(
                command=srv.config.command,
                args=srv.config.args or [],
                env=srv.config.env or {},
            )
            parent_ctx = stdio_client(params)

            @asynccontextmanager
            async def ctx():
                async with parent_ctx as (read, write):
                    async with ClientSession(read, write) as session:
                        init_result = await session.initialize()
                        self._set_server_info_on_session(session, init_result)
                        yield session
            return ctx()

        if t == "streamable_http":
            if not srv.config.url:
                raise ValueError(f"{srv.name}: 'url' is required for streamable_http")
            parent_ctx = streamablehttp_client(srv.config.url, headers=srv.config.headers)

            @asynccontextmanager
            async def ctx():
                async with parent_ctx as (read, write, _closer):
                    async with ClientSession(read, write) as session:
                        init_result = await session.initialize()
                        self._set_server_info_on_session(session, init_result)
                        yield session
            return ctx()

        if t == "sse":
            if not srv.config.url:
                raise ValueError(f"{srv.name}: 'url' is required for sse")
            parent_ctx = sse_client(srv.config.url, headers=srv.config.headers or None)

            @asynccontextmanager
            async def ctx():
                async with parent_ctx as (read, write):
                    async with ClientSession(read, write) as session:
                        init_result = await session.initialize()
                        self._set_server_info_on_session(session, init_result)
                        yield session
            return ctx()

        raise ValueError(f"Unknown transport: {t}")

    def _find_server(self, name: str) -> RemoteServer:
        for s in self.config:
            if s.name == name:
                return s
        lowered = name.lower()
        candidates = [s for s in self.config if s.name.lower().startswith(lowered)]
        if len(candidates) == 1:
            return candidates[0]
        names = ", ".join(s.name for s in self.config)
        raise ValueError(f"Server '{name}' not found. Available: {names}")

    # ---------- public API ----------

    def get_client(self, server_name: str) -> AsyncContextManager[ClientSession]:
        srv = self._find_server(server_name)
        return self._open_session(srv)  # returns @asynccontextmanager

    async def list_clients(self) -> MultiServerMCPClient:
        mapping: Dict[str, Any] = {}
        for srv in self.config:
            t = srv.config.transport
            if t == "streamable_http":
                mapping[srv.name] = StreamableHttpConnection(
                    transport="streamable_http",
                    url=srv.config.url,               # type: ignore[arg-type]
                    headers=srv.config.headers or None,
                )
            elif t == "stdio":
                mapping[srv.name] = StdioConnection(
                    transport="stdio",
                    command=srv.config.command,       # type: ignore[arg-type]
                    args=srv.config.args or [],
                    env=srv.config.env or {},
                )
            elif t == "sse":
                mapping[srv.name] = SSEConnection(
                    transport="sse",
                    url=srv.config.url,               # type: ignore[arg-type]
                    headers=srv.config.headers or None,
                )
            else:
                raise ValueError(f"Unknown transport: {t}")

        return MultiServerMCPClient(mapping)

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        srv = self._find_server(server_name)
        async with self._open_session(srv) as session:
            res = await session.call_tool(tool_name, arguments=arguments)
            if getattr(res, "structuredContent", None):
                return {"structured": res.structuredContent}
            texts = [c.text for c in (res.content or []) if hasattr(c, "text")]
            return {"content": "\n".join(texts)}

    async def list_tools(self):
        ms_client = await self.list_clients()
        return await ms_client.get_tools()

    async def get_metadata(self):
        ms_client = await self.list_clients()
        for server in self.config:
            async with ms_client.session(server.name) as session:
                tools = await load_mcp_tools(session)
            server.tools = [ToolDef(name=t.name, description=t.description) for t in tools]
        # return normalized list form
        return [s.model_dump(exclude_unset=True, exclude_none=True) for s in self.config]