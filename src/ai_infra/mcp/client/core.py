from __future__ import annotations

import difflib
import traceback
from typing import Dict, Any, List, AsyncContextManager
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

from ai_infra.mcp.client.models import McpServerConfig
from ai_infra.mcp.server.tools import ToolDef


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
        self._errors: List[Dict[str, Any]] = []   # <-- NEW

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

    def last_errors(self) -> List[Dict[str, Any]]:
        """Return error records from the last discover() run."""
        return list(self._errors)

    def _cfg_identity(self, cfg: McpServerConfig) -> str:
        """Human-friendly identity for error messages."""
        if cfg.transport == "stdio":
            return f"stdio: {cfg.command or '<missing command>'} {' '.join(cfg.args or [])}"
        return f"{cfg.transport}: {cfg.url or '<missing url>'}"

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

    async def discover(self, strict: bool = False) -> Dict[str, McpServerConfig]:
        """
        Probe each server to learn its MCP-declared name.
        - strict=False (default): collect errors and continue (partial success).
        - strict=True: raise ExceptionGroup with all failures.
        """
        # reset state each time (optional: make it idempotent and skip already-known)
        self._by_name = {}
        self._errors = []
        self._discovered = False

        name_map: Dict[str, McpServerConfig] = {}
        used: set[str] = set()
        failures: List[BaseException] = []

        for cfg in self._configs:
            ident = self._cfg_identity(cfg)
            try:
                async with self._open_session_from_config(cfg) as session:
                    info = getattr(session, "mcp_server_info", {}) or {}
                    base = str(info.get("name") or "server").strip() or "server"
                    name = self._uniq_name(base, used)
                    used.add(name)
                    name_map[name] = cfg
            except Exception as e:
                # record detailed error but keep going
                tb = "".join(traceback.format_exception(e))
                self._errors.append({
                    "config": {
                        "transport": cfg.transport,
                        "url": cfg.url,
                        "command": cfg.command,
                        "args": cfg.args,
                    },
                    "identity": ident,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": tb,
                })
                failures.append(e)

        self._by_name = name_map
        self._discovered = True

        if strict and failures:
            # Raise a summarized ExceptionGroup
            raise ExceptionGroup(
                f"MCP discovery failed for {len(failures)} server(s)",
                failures
            )

        return name_map

    def server_names(self) -> List[str]:
        return list(self._by_name.keys())

    # ---------- public API ----------

    def get_client(self, server_name: str) -> AsyncContextManager[ClientSession]:
        if server_name not in self._by_name:
            # suggest close matches to reduce frustration
            suggestions = difflib.get_close_matches(server_name, self.server_names(), n=3, cutoff=0.5)
            suggest_msg = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            known = ", ".join(self.server_names()) or "(none discovered yet)"
            raise ValueError(f"Unknown server '{server_name}'. Known: {known}.{suggest_msg}")
        cfg = self._by_name[server_name]
        return self._open_session_from_config(cfg)

    async def list_clients(self) -> MultiServerMCPClient:
        # ensure we have discovered mapping
        if not self._discovered:
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