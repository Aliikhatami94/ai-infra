from __future__ import annotations
import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ai_infra.mcp.server.tools import ToolDef, _mcp_from_tools
from ai_infra.mcp.client.core import CoreMCPClient

Executor = Callable[[str, Dict[str, Any]], Awaitable[Any]]  # (tool_name, args) -> result


def _infer_client_cfg_from_openmcp(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best-effort: build a single-server config usable by CoreMCPClient from OpenMCP doc."""
    server = (doc or {}).get("server") or {}
    transport = server.get("transport")
    endpoint = server.get("endpoint")
    x_vendor = (doc or {}).get("x-vendor") or {}

    if transport == "streamable_http" and endpoint:
        cfg = {"transport": "streamable_http", "url": endpoint}
        headers = x_vendor.get("http", {}).get("headers") or x_vendor.get("headers")
        if headers:
            cfg["headers"] = headers
        return cfg

    if transport == "sse" and endpoint:
        cfg = {"transport": "sse", "url": endpoint}
        headers = x_vendor.get("http", {}).get("headers") or x_vendor.get("headers")
        if headers:
            cfg["headers"] = headers
        return cfg

    if transport == "stdio":
        stdio = x_vendor.get("stdio") or {}
        command = stdio.get("command") or server.get("command") or endpoint  # endpoint may be just "npx"
        if not command:
            return None
        return {
            "transport": "stdio",
            "command": command,
            "args": stdio.get("args", []),
            "env": stdio.get("env", {}),
        }

    # websocket (if you support it in CoreMCPClient) could be added similarly
    return None


def _make_default_executor(doc: Dict[str, Any]) -> Executor:
    """
    Auto-proxy tools to the remote MCP described by this OpenMCP document.
    Requires the doc to contain enough info to construct a CoreMCPClient config.
    """
    cfg = _infer_client_cfg_from_openmcp(doc)
    if cfg is None:
        raise ValueError(
            "OpenMCP doc does not contain enough transport/endpoint/vendor hints "
            "to auto-connect. Provide an `executor` explicitly."
        )

    # Lazy client, shared per-process. CoreMCPClient opens/closes sessions per call_tool anyway.
    client = CoreMCPClient([cfg])
    _discover_once_lock = asyncio.Lock()
    _discovered = {"ok": False, "server": None}

    async def _ensure_discovered() -> str:
        if _discovered["ok"] and _discovered["server"]:
            return _discovered["server"]
        async with _discover_once_lock:
            if _discovered["ok"] and _discovered["server"]:
                return _discovered["server"]
            await client.discover()
            names = client.server_names()
            if not names:
                raise RuntimeError("Failed to connect to remote MCP (no servers discovered).")
            # choose the first discovered server (OpenMCP describes exactly one server)
            _discovered["ok"] = True
            _discovered["server"] = names[0]
            return _discovered["server"]

    async def _exec(tool_name: str, arguments: Dict[str, Any]) -> Any:
        server_name = await _ensure_discovered()
        # call_tool returns either {"structured": ...} or {"content": "..."} in this client
        res = await client.call_tool(server_name, tool_name, arguments)
        # Normalize result to a dict; if plain content exists, wrap it
        if isinstance(res, dict):
            return res.get("structured") or res
        return {"result": str(res)}

    return _exec


def _build_tool_defs_from_openmcp(
        doc: Dict[str, Any],
        executor: Executor,
) -> List[ToolDef]:
    """Translate OpenMCP tools into ToolDefs that invoke `executor`."""
    tools = (doc or {}).get("tools") or []
    out: List[ToolDef] = []
    for t in tools:
        name = t.get("name")
        if not name:
            # skip unnamed tools
            continue
        desc = t.get("description") or ""
        args_schema = t.get("args_schema") or {}
        output_schema = t.get("output_schema") or None

        async def _make_fn(tool_name: str):
            async def _fn(**kwargs):
                # Accept kwargs matching args_schema. Pass-through to executor.
                result = await executor(tool_name, kwargs)
                # If no declared output schema, ensure a simple shape
                if output_schema is None and not isinstance(result, dict):
                    return {"result": str(result)}
                return result
            return _fn

        # bind a fresh function per tool
        # (need a closure to freeze the current tool_name)
        async_fn = asyncio.get_event_loop().run_until_complete(_make_fn(name))  # we’re in sync context here
        out.append(ToolDef(fn=async_fn, name=name, description=desc,
                           args_schema=args_schema, output_schema=output_schema))
    return out


def _mcp_from_openmcp(
        doc: Dict[str, Any],
        *,
        executor: Optional[Executor] = None,
        name: Optional[str] = None,
):
    """
    Build a FastMCP (server) from an OpenMCP document.

    - If `executor` is provided: your function runs the tool calls.
    - Else: we auto-proxy to the MCP described in `doc.server` (streamable_http / sse / stdio),
      using CoreMCPClient under the hood.
    """
    if executor is None:
        executor = _make_default_executor(doc)

    mcp_name = name or ((doc.get("info") or {}).get("title") or (doc.get("server") or {}).get("name") or "mcp")
    tool_defs = _build_tool_defs_from_openmcp(doc, executor)
    return _mcp_from_tools(name=mcp_name, tools=tool_defs)