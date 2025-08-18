from __future__ import annotations

from typing import Any, Dict, Optional, Iterable, Literal, Protocol, runtime_checkable
from dataclasses import dataclass
import asyncio

# In-memory (FastMCP ↔ Client)
from fastmcp import FastMCP
from fastmcp.client import Client as FastMCPClient

# Reference MCP client (stdio / streamable_http / sse)
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

# Optional: if you have a tiny wrapper around FastMCP already
try:
    from ai_infra.fastmcp.server.core import CoreMCPServer
except Exception:
    CoreMCPServer = None


TransportKind = Literal["in_memory", "stdio", "streamable_http", "sse"]


@dataclass
class ConnectionConfig:
    """
    A normalized configuration for a single MCP server connection.
    """
    name: str
    transport: TransportKind

    # in_memory
    mcp: Optional[FastMCP] = None  # or CoreMCPServer.mcp

    # stdio
    command: Optional[str] = None
    args: Optional[list[str]] = None
    env: Optional[dict[str, str]] = None

    # streamable_http / sse
    url: Optional[str] = None
    headers: Optional[dict[str, str]] = None

@runtime_checkable
class SupportsCallToolResult(Protocol):
    data: Any | None
    structured_content: Any | None
    structuredContent: Any | None  # some versions use camelCase
    content: list[Any] | None


class CoreMCPClient:
    """
    One client API for multiple MCP transports.

    - in_memory: uses fastmcp.client.Client(FastMCP)
    - stdio:     uses reference MCP client over subprocess stdio
    - streamable_http: uses reference MCP client over Streamable HTTP

    Example:
        cfgs = [
            ConnectionConfig(name="LocalAPI", transport="in_memory", mcp=fastmcp_server),
            ConnectionConfig(name="Wikipedia", transport="stdio",
                             command="npx", args=["-y", "wikipedia-mcp"]),
            ConnectionConfig(name="Weather", transport="streamable_http",
                             url="http://localhost:8000/mcp", headers={"Authorization": "Bearer xyz"}),
        ]
        async with CoreMCPClient(cfgs) as c:
            await c.connect_all()
            print(await c.list_tools("Wikipedia"))
            res = await c.call_tool("Wikipedia", "search", {"query": "FastMCP"})
            print(CoreMCPClient.extract_payload(res))
    """

    def __init__(self, connections: Iterable[ConnectionConfig]):
        self._configs: dict[str, ConnectionConfig] = {c.name: c for c in connections}

        # Live client/session handles per connection
        self._inmem_clients: dict[str, FastMCPClient] = {}
        self._entered: bool = False

        self._remote_sessions: dict[str, ClientSession] = {}
        self._remote_contexts: dict[str, Any] = {}

    # ---------- Async context management ----------
    async def __aenter__(self) -> "CoreMCPClient":
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Close remote sessions first
        for name, session in list(self._remote_sessions.items()):
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
            finally:
                self._remote_sessions.pop(name, None)

        # Then close their transport contexts
        for name, ctx in list(self._remote_contexts.items()):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
            finally:
                self._remote_contexts.pop(name, None)

        # Close in-memory clients
        for name, client in list(self._inmem_clients.items()):
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
            finally:
                self._inmem_clients.pop(name, None)

    # ---------- Connection helpers ----------
    def has(self, name: str) -> bool:
        return name in self._configs

    async def connect(self, name: str) -> None:
        """
        Connect a single configured server by name.
        Safe to call multiple times; it’s a no-op if already connected.
        """
        cfg = self._require_cfg(name)

        # In-memory FastMCP
        if cfg.transport == "in_memory":
            if name in self._inmem_clients:
                return
            if cfg.mcp is None:
                raise ValueError(f"[{name}] in_memory requires FastMCP instance (cfg.mcp).")
            client = FastMCPClient(cfg.mcp)
            await client.__aenter__()
            self._inmem_clients[name] = client
            return

        # Already connected to a remote session?
        if name in self._remote_sessions:
            return

        # STDIO transport
        if cfg.transport == "stdio":
            if not cfg.command:
                raise ValueError(f"[{name}] stdio requires 'command'.")

            # Prefer new-style API that accepts StdioServerParameters (allows env reliably)
            ctx = None
            try:
                # If the import exists, use it
                server_params = StdioServerParameters(
                    command=cfg.command,
                    args=cfg.args or [],
                    env=cfg.env or {},
                )
                ctx = stdio_client(server_params)
            except NameError:
                # Fallback for older versions: try stdio_client(command, args, env?) signatures
                import inspect
                sig = inspect.signature(stdio_client)
                if "env" in sig.parameters:
                    ctx = stdio_client(cfg.command, cfg.args or [], cfg.env or {})
                else:
                    ctx = stdio_client(cfg.command, cfg.args or [])

            read, write = await ctx.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            self._remote_sessions[name] = session
            # IMPORTANT: store the transport context so we can close it later
            self._remote_contexts[name] = ctx
            return

        # Streamable HTTP transport
        if cfg.transport == "streamable_http":
            if not cfg.url:
                raise ValueError(f"[{name}] streamable_http requires 'url'.")
            ctx = streamablehttp_client(cfg.url, headers=cfg.headers or {})
            read, write, _ = await ctx.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            self._remote_sessions[name] = session
            # Store context for clean shutdown
            self._remote_contexts[name] = ctx
            return

        # SSE (if you add it later)
        if cfg.transport == "sse":
            if not cfg.url:
                raise ValueError(f"[{name}] sse requires 'url'.")
            # optional headers are supported
            ctx = sse_client(cfg.url, headers=cfg.headers or {})
            read, write = await ctx.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            self._remote_sessions[name] = session
            self._remote_contexts[name] = ctx  # store so __aexit__ can close it
            return

        raise ValueError(f"[{name}] Unknown transport: {cfg.transport}")

    async def connect_all(self) -> None:
        await asyncio.gather(*(self.connect(n) for n in self._configs.keys()))

    async def disconnect(self, name: str) -> None:
        if name in self._inmem_clients:
            try:
                await self._inmem_clients[name].__aexit__(None, None, None)
            finally:
                self._inmem_clients.pop(name, None)

        if name in self._remote_sessions:
            try:
                await self._remote_sessions[name].__aexit__(None, None, None)
            finally:
                self._remote_sessions.pop(name, None)
                ctx = self._remote_contexts.pop(name, None)
                if ctx is not None:
                    try:
                        await ctx.__aexit__(None, None, None)
                    except Exception:
                        pass

    # ---------- High-level operations ----------
    async def list_server_tools(self, name: str) -> list[Any]:
        """
        Return the tool descriptors for a connected server.
        """
        if name in self._inmem_clients:
            return await self._inmem_clients[name].list_tools()

        if name in self._remote_sessions:
            # Reference MCP client
            session = self._remote_sessions[name]
            # Prefer the explicit API if available
            if hasattr(session, "list_tools"):
                return await session.list_tools()
            # Fallback (older clients sometimes: tools in capabilities)
            raise RuntimeError("This MCP session does not support list_tools().")

        raise RuntimeError(f"[{name}] Not connected. Call connect(name) first.")

    @staticmethod
    def _coerce_tools_list(obj) -> list:
        """Normalize both reference-client and FastMCP-client responses to a list[Tool]."""
        # Reference MCP client returns ListToolsResult(meta, nextCursor, tools=[...])
        tools = getattr(obj, "tools", None)
        if tools is not None:
            return list(tools)
        # FastMCP's in-memory client returns list[Tool]
        if isinstance(obj, list):
            return obj
        # Fallback: nothing / unknown shape
        return []

    async def list_tools(
            self,
            *,
            connect_missing: bool = True,
            dedupe: bool = True,
    ) -> list:
        """
        Return a single flattened list of tools from all connected servers.
        No server names, just tools.

        - connect_missing=True will auto-connect all configured servers first
        - dedupe=True keeps the first tool per unique .name
        """
        if connect_missing:
            await self.connect_all()

        # Gather per-server tool lists concurrently
        names = list(self._inmem_clients.keys() | self._remote_sessions.keys())
        per_server = await asyncio.gather(*(self.list_server_tools(n) for n in names))

        # Normalize & flatten
        flat: list = []
        for result in per_server:
            flat.extend(self._coerce_tools_list(result))

        if dedupe:
            seen = set()
            deduped = []
            for t in flat:
                # prefer the 'name' attribute; fall back to dict key if a dict-like tool
                key = getattr(t, "name", None) or (isinstance(t, dict) and t.get("name"))
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                deduped.append(t)
            return deduped

        return flat

    async def call_tool(self, name: str, tool: str, args: Dict[str, Any] | None = None) -> Any:
        """
        Call a tool on a connected server and return the raw result object:
        - For in_memory: fastmcp.CallToolResult
        - For remote:     MCP result object (with .content, .structuredContent, etc.)
        """
        args = args or {}

        if name in self._inmem_clients:
            return await self._inmem_clients[name].call_tool(tool, args)

        if name in self._remote_sessions:
            session = self._remote_sessions[name]
            return await session.call_tool(tool, arguments=args)

        raise RuntimeError(f"[{name}] Not connected. Call connect(name) first.")

    # ---------- Utilities ----------
    @staticmethod
    def extract_payload(result: SupportsCallToolResult | Any) -> Any:
        """Prefer .data, then structured_content/structuredContent, then first text chunk."""
        # 1) .data (best for JSON/primitive results)
        data = getattr(result, "data", None)
        if data is not None:
            return data

        # 2) structured content (snake or camel)
        sc = getattr(result, "structured_content", None)
        if sc is None:
            sc = getattr(result, "structuredContent", None)
        if sc is not None:
            return sc

        # 3) first text content chunk
        content = getattr(result, "content", None) or []
        for c in content:
            txt = getattr(c, "text", None)
            if txt:
                return txt

        return None

    def _require_cfg(self, name: str) -> ConnectionConfig:
        try:
            return self._configs[name]
        except KeyError:
            raise KeyError(f"No connection named '{name}'. Known: {list(self._configs)}")