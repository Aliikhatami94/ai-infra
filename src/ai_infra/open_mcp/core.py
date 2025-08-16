from typing import List, Optional, AsyncIterator, Dict, Any
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client
from mcp import ClientSession
from mcp.shared.metadata_utils import get_display_name

from .models import McpConfig, ToolDef, Prompt
from .utils import (
    resolve_arg_path as _resolve_arg_path,
    make_system_messages as _make_system_messages
)


class CoreMCP:
    """
    Main entry point for interacting with the MCP system.

    Args:
        config (dict | McpConfig): Configuration dictionary or McpConfig model.
    """
    def __init__(self, config: dict | McpConfig):
        self.config = config if isinstance(config, McpConfig) else McpConfig(**config)

    # ---------- Transport factory ----------
    async def _open_session(self, cfg: "Server"):
        t = cfg.config.transport
        if t == "stdio":
            params = dict(command=cfg.config.command, args=cfg.config.args or [], env=cfg.config.env or {})
            client_ctx = stdio_client(**params)
            async def _ctx() -> AsyncIterator[ClientSession]:
                async with client_ctx as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session
            return _ctx

        elif t == "streamable_http":
            if not cfg.config.url:
                raise ValueError(f"{cfg.metadata.name}: url required for streamable_http")
            client_ctx = streamablehttp_client(cfg.config.url, headers=cfg.config.headers)
            async def _ctx() -> AsyncIterator[ClientSession]:
                async with client_ctx as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session
            return _ctx

        elif t == "sse":
            # (If you still use SSE, plug the SSE client here similarly)
            raise NotImplementedError("SSE client hookup not implemented")
        else:
            raise ValueError(f"Unknown transport: {t}")

    # ---------- Discovery helpers ----------
    async def list_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {}
        for srv in self.config.servers:
            open_session = await self._open_session(srv)
            async with open_session() as session:
                tools_resp = await session.list_tools()
                result[srv.metadata.name] = [
                    {"name": t.name, "title": get_display_name(t), "description": t.description}
                    for t in tools_resp.tools
                ]
        return result

    async def list_resources(self) -> Dict[str, List[Dict[str, Any]]]:
        result = {}
        for srv in self.config.servers:
            open_session = await self._open_session(srv)
            async with open_session() as session:
                r = await session.list_resources()
                result[srv.metadata.name] = [{"uri": res.uri, "title": get_display_name(res)} for res in r.resources]
        return result

    async def list_prompts(self) -> Dict[str, List[Dict[str, Any]]]:
        result = {}
        for srv in self.config.servers:
            open_session = await self._open_session(srv)
            async with open_session() as session:
                p = await session.list_prompts()
                result[srv.metadata.name] = [{"name": pr.name, "title": pr.title or pr.name} for pr in p.prompts]
        return result

    # ---------- Action helpers ----------
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        srv = next(s for s in self.config.servers if s.metadata.name == server_name)
        open_session = await self._open_session(srv)
        async with open_session() as session:
            res = await session.call_tool(tool_name, arguments=arguments)
            # Prefer structured content if present
            if getattr(res, "structuredContent", None):
                return {"structured": res.structuredContent}
            # Fallback to text content
            texts = [c.text for c in res.content if hasattr(c, "text")]
            return {"content": "\n".join(texts)}

    async def read_resource(self, server_name: str, uri: str) -> str:
        srv = next(s for s in self.config.servers if s.metadata.name == server_name)
        open_session = await self._open_session(srv)
        async with open_session() as session:
            res = await session.read_resource(uri)
            for c in res.contents:
                if hasattr(c, "text"):
                    return c.text
            return "[non-text resource]"

    async def get_prompt(self, server_name: str, name: str, arguments: Optional[Dict[str, str]] = None):
        srv = next(s for s in self.config.servers if s.metadata.name == server_name)
        open_session = await self._open_session(srv)
        async with open_session() as session:
            pr = await session.get_prompt(name, arguments=arguments or {})
            # Return plain text for convenience
            blocks = []
            for m in pr.messages:
                # messages contain role + content block(s)
                if hasattr(m.content, "text"):
                    blocks.append(m.content.text)
            return "\n".join(blocks)

    @staticmethod
    def _process_config_dict(cfg, host: Optional[str], resolve_arg_path) -> dict:
        d = cfg.model_dump(exclude_unset=True, exclude_none=True)

        # Only resolve args that look like filesystem paths
        if "args" in d and d["args"]:
            resolved = []
            for a in d["args"]:
                if isinstance(a, str) and (
                        a.startswith("/")       # absolute *nix
                        or a.startswith("./")   # relative
                        or a.startswith("../")  # relative up
                        or "\\" in a            # Windows-style path
                ):
                    try:
                        resolved.append(resolve_arg_path(a))
                    except FileNotFoundError:
                        # If it looks like a path but we can't resolve, leave it as-is
                        resolved.append(a)
                else:
                    # Flags like "-y" or program names, leave them
                    resolved.append(a)
            d["args"] = resolved

        # Only prefix host for HTTP-style URLs, and only if the url is relative
        if d.get("url") and host and not d["url"].startswith(("http://", "https://")):
            d["url"] = host.rstrip("/") + "/" + d["url"].lstrip("/")

        return d

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

        return self.config.model_dump()

    async def get_server_prompt(self, additional_context: List[Prompt] = None) -> List[SystemMessage]:
        return _make_system_messages(self.config.prompts or [], additional_context)

    async def get_server_setup(self) -> dict:
        servers = self.config.servers
        server_setup = {}
        for server in servers:
            server_setup[server.info.name] = server.config.model_dump(exclude_unset=True, exclude_none=True)
        return server_setup

    async def get_client(self):
        server_setup = await self.get_server_setup()
        return MultiServerMCPClient(server_setup)

    async def get_tools(self):
        client = await self.get_client()
        tools = await client.get_tools()
        return tools