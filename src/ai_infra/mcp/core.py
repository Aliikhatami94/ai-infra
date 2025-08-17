from pydantic import BaseModel
from typing import List, AsyncIterator, Dict, Any, AsyncContextManager
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from contextlib import asynccontextmanager

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client
from mcp import ClientSession

from ai_infra.mcp.models import OpenMcp, ToolDef, Prompt, RemoteServer
from .utils import (
    make_system_messages as _make_system_messages
)

__all__ = ["RemoteServer"]


class CoreMCP:
    """
    Main entry point for interacting with the MCP system.

    Args:
        config (dict | OpenMcp): Configuration dictionary or OpenMcp model.
    """
    def __init__(self, config):
        if isinstance(config, OpenMcp):
            self.config = config
        else:
            if isinstance(config, BaseModel):
                config = config.model_dump()
            self.config = OpenMcp.model_validate(config)

    def _open_session(self, cfg: "RemoteServer") -> AsyncContextManager[ClientSession]:
        t = cfg.config.transport

        if t == "stdio":
            params = dict(
                command=cfg.config.command,
                args=cfg.config.args or [],
                env=cfg.config.env or {},
            )
            client_ctx = stdio_client(**params)

            @asynccontextmanager
            async def ctx() -> AsyncIterator[ClientSession]:
                async with client_ctx as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session

            return ctx()

        elif t == "streamable_http":
            if not cfg.config.url:
                # FIX: name (was name)
                raise ValueError(f"{cfg.name}: url required for streamable_http")

            client_ctx = streamablehttp_client(cfg.config.url, headers=cfg.config.headers)

            @asynccontextmanager
            async def ctx() -> AsyncIterator[ClientSession]:
                async with client_ctx as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session

            return ctx()

        elif t == "sse":
            raise NotImplementedError("SSE client hookup not implemented")

        else:
            raise ValueError(f"Unknown transport: {t}")

    def _find_server(self, server_name: str) -> RemoteServer:
        # exact match
        for s in self.config.servers:
            if s.name == server_name:
                return s
        # case-insensitive / startswith fallback (optional)
        lowered = server_name.lower()
        candidates = [s for s in self.config.servers if s.name.lower().startswith(lowered)]
        if len(candidates) == 1:
            return candidates[0]
        names = ", ".join(s.name for s in self.config.servers)
        raise ValueError(f"Server '{server_name}' not found. Available: {names}")

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        srv = self._find_server(server_name)
        async with self._open_session(srv) as session:
            res = await session.call_tool(tool_name, arguments=arguments)
            if getattr(res, "structuredContent", None):
                return {"structured": res.structuredContent}
            texts = [c.text for c in res.content if hasattr(c, "text")]
            return {"content": "\n".join(texts)}

    async def get_metadata(self):
        client = await self.get_client()

        for server in self.config.servers:
            server_key = server.name
            async with client.session(server_key) as session:
                tools = await load_mcp_tools(session)

            server.tools = [
                ToolDef(name=t.name, description=t.description)
                for t in tools
            ]

        return self.config.model_dump(exclude_unset=True, exclude_none=True)

    async def get_server_prompt(self, additional_context: List[Prompt] = None) -> List[SystemMessage]:
        return _make_system_messages(self.config.prompts or [], additional_context)

    async def get_server_setup(self) -> dict:
        servers = self.config.servers
        server_setup = {}
        for server in servers:
            server_setup[server.name] = server.config.model_dump(exclude_unset=True, exclude_none=True)
        return server_setup

    async def get_client(self):
        server_setup = await self.get_server_setup()
        return MultiServerMCPClient(server_setup)

    async def list_tools(self):
        client = await self.get_client()
        tools = await client.get_tools()
        return tools