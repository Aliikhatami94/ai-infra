# ai_infra/mcp/client/core.py
from __future__ import annotations
from typing import Dict, Any, List, AsyncIterator, AsyncContextManager, Tuple
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

from ai_infra.mcp.models import McpServerConfig, ToolDef


class CoreMCPClient:
    """
    Config = list[McpServerConfig-like dicts]. No names required.
    We discover server names from MCP initialize() and map them.
    """

    def __init__(self, config: List[dict] | List[McpServerConfig]):
        if not isinstance(config, list):
            raise TypeError("Config must be a list of server configs")
        self._configs: List[McpServerConfig] = [
            c if isinstance(c, McpServerConfig) else McpServerConfig.model_validate(c)
            for c in config
        ]
        self._by_name: Dict[str, McpServerConfig] = {}
        self._discovered: bool = False

    # ---------- utils ----------

    @staticmethod
    def _extract_server_info(init_result) -> Dict[str, Any] | None:
        info = (
                getattr(init_result, "server_info", None)
                or getattr(init_result, "serverInfo", None)
                or getattr(init_result, "serverinfo", None)
        )
        if info is None:
            return None
        if is_dataclass(info):
            return asdict(info)
        if hasattr(info, "model_dump"):
            return info.model_dump()
        if isinstance(info, dict):
            return info
        return None

    @staticmethod
    def _uniq_name(base: str, used: set[str]) -> str:
        if base not in used:
            return base
        i = 2
        while f"{base}#{i}" in used:
            i += 1
        return f"{base}#{i}"

    # ---------- low-level open session from config ----------

    def _open_session_from_config(self, cfg: McpServerConfig) -> AsyncContextManager[ClientSession]:
        t = cfg.transport

        if t == "stdio":
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args or [],
                env=cfg.env or {},
            )
            parent_ctx = stdio_client(params)

            @asynccontextmanager
            async def ctx():
                async with parent_ctx as (read, write):
                    async with ClientSession(read, write) as session:
                        init_result = await session.initialize()
                        info = self._extract_server_info(init_result) or {}
                        session.mcp_server_info = info
                        yield session
            return ctx()

        if t == "streamable_http":
            if not cfg.url:
                raise ValueError("'url' is required for streamable_http")
            parent_ctx = streamablehttp_client(cfg.url, headers=cfg.headers)

            @asynccontextmanager
            async def ctx():
                async with parent_ctx as (read, write, _closer):
                    async with ClientSession(read, write) as session:
                        init_result = await session.initialize()
                        info = self._extract_server_info(init_result) or {}
                        session.mcp_server_info = info
                        yield session
            return ctx()

        if t == "sse":
            if not cfg.url:
                raise ValueError("'url' is required for sse")
            parent_ctx = sse_client(cfg.url, headers=cfg.headers or None)

            @asynccontextmanager
            async def ctx():
                async with parent_ctx as (read, write):
                    async with ClientSession(read, write) as session:
                        init_result = await session.initialize()
                        info = self._extract_server_info(init_result) or {}
                        session.mcp_server_info = info
                        yield session
            return ctx()

        raise ValueError(f"Unknown transport: {t}")

    # ---------- discovery ----------

    async def discover(self) -> Dict[str, McpServerConfig]:
        """
        Connect briefly to each server to learn its declared name, then
        build a stable name→config map. Run once (idempotent).
        """
        if self._discovered:
            return self._by_name

        name_map: Dict[str, McpServerConfig] = {}
        used: set[str] = set()

        for cfg in self._configs:
            async with self._open_session_from_config(cfg) as session:
                info = getattr(session, "mcp_server_info", {}) or {}
                base = str(info.get("name") or "server").strip() or "server"
                name = self._uniq_name(base, used)
                used.add(name)
                name_map[name] = cfg

        self._by_name = name_map
        self._discovered = True
        return name_map

    def server_names(self) -> List[str]:
        """Return discovered names (call discover() first if you need them now)."""
        return list(self._by_name.keys())

    # ---------- public API ----------

    def get_client(self, server_name: str) -> AsyncContextManager[ClientSession]:
        """
        Open a ready-to-use MCP ClientSession by discovered name.
        NOTE: You must call `await client.discover()` at least once before using names.
        """
        if server_name not in self._by_name:
            raise ValueError(
                f"Unknown server '{server_name}'. "
                f"Known: {', '.join(self._by_name) or '(none discovered yet)'}"
            )
        cfg = self._by_name[server_name]
        return self._open_session_from_config(cfg)

    async def list_clients(self) -> MultiServerMCPClient:
        """
        Build a MultiServerMCPClient mapping discovered_name → Connection.
        """
        await self.discover()
        mapping: Dict[str, Any] = {}
        for name, cfg in self._by_name.items():
            if cfg.transport == "streamable_http":
                mapping[name] = StreamableHttpConnection(
                    transport="streamable_http",
                    url=cfg.url,  # type: ignore[arg-type]
                    headers=cfg.headers or None,
                )
            elif cfg.transport == "stdio":
                mapping[name] = StdioConnection(
                    transport="stdio",
                    command=cfg.command,  # type: ignore[arg-type]
                    args=cfg.args or [],
                    env=cfg.env or {},
                )
            elif cfg.transport == "sse":
                mapping[name] = SSEConnection(
                    transport="sse",
                    url=cfg.url,  # type: ignore[arg-type]
                    headers=cfg.headers or None,
                )
            else:
                raise ValueError(f"Unknown transport: {cfg.transport}")
        return MultiServerMCPClient(mapping)

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self._discovered:
            await self.discover()
        async with self.get_client(server_name) as session:
            res = await session.call_tool(tool_name, arguments=arguments)
            if getattr(res, "structuredContent", None):
                return {"structured": res.structuredContent}
            texts = [c.text for c in (res.content or []) if hasattr(c, "text")]
            return {"content": "\n".join(texts)}

    async def list_tools(self):
        ms_client = await self.list_clients()
        return await ms_client.get_tools()

    async def get_metadata(self):
        """
        Return lightweight metadata: discovered names + available tools.
        """
        ms_client = await self.list_clients()
        out: List[Dict[str, Any]] = []
        for name in self.server_names():
            async with ms_client.session(name) as session:
                tools = await load_mcp_tools(session)
            out.append({
                "name": name,
                "tools": [ToolDef(name=t.name, description=t.description).model_dump(exclude_none=True) for t in tools],
            })
        return out