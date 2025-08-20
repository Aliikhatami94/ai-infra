# ai_infra/mcp/server/openmcp/builder.py
from __future__ import annotations
import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ai_infra.mcp.server.tools import ToolDef, _mcp_from_tools
from ai_infra.mcp.client.core import CoreMCPClient

Executor = Callable[[str, Dict[str, Any]], Awaitable[Any]]  # (tool_name, args) -> result


def _infer_client_cfg_from_openmcp(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best-effort: construct a CoreMCPClient config from OpenMCP doc."""
    server = (doc or {}).get("server") or {}
    transport = server.get("transport")
    endpoint = server.get("endpoint")
    x_vendor = (doc or {}).get("x-vendor") or {}

    if transport == "streamable_http" and endpoint:
        cfg = {"transport": "streamable_http", "url": endpoint}
        headers = (x_vendor.get("http") or {}).get("headers") or x_vendor.get("headers")
        if headers:
            cfg["headers"] = headers
        return cfg

    if transport == "sse" and endpoint:
        cfg = {"transport": "sse", "url": endpoint}
        headers = (x_vendor.get("http") or {}).get("headers") or x_vendor.get("headers")
        if headers:
            cfg["headers"] = headers
        return cfg

    if transport == "stdio":
        stdio = x_vendor.get("stdio") or {}
        command = stdio.get("command") or server.get("command") or endpoint
        if not command:
            return None
        return {
            "transport": "stdio",
            "command": command,
            "args": stdio.get("args", []),
            "env": stdio.get("env", {}),
        }

    # Add websocket/etc. here if your CoreMCPClient supports them
    return None


def _make_default_executor(doc: Dict[str, Any], client_config: Optional[Dict[str, Any]]) -> Executor:
    """
    Auto-proxy tools to the MCP described by the doc (or an explicit client_config).
    """
    cfg = client_config or _infer_client_cfg_from_openmcp(doc)
    if cfg is None:
        raise ValueError(
            "OpenMCP doc does not contain enough transport/endpoint/vendor hints to auto-connect. "
            "Pass `client_config` to add_openmcp(...), or supply an explicit `executor`."
        )

    client = CoreMCPClient([cfg])
    _discover_once_lock = asyncio.Lock()
    _discovered = {"ok": False, "server": None}

    async def _ensure_discovered() -> str:
        if _discovered["ok"] and _discovered["server"]:
            return _discovered["server"]  # type: ignore[return-value]
        async with _discover_once_lock:
            if _discovered["ok"] and _discovered["server"]:
                return _discovered["server"]  # type: ignore[return-value]
            await client.discover()
            names = client.server_names()
            if not names:
                raise RuntimeError("Failed to connect to remote MCP (no servers discovered).")
            _discovered["ok"] = True
            _discovered["server"] = names[0]
            return names[0]

    async def _exec(tool_name: str, arguments: Dict[str, Any]) -> Any:
        server_name = await _ensure_discovered()
        res = await client.call_tool(server_name, tool_name, arguments)
        # Normalize
        if isinstance(res, dict):
            return res.get("structured") or res
        return {"result": str(res)}

    return _exec


def _build_tool_defs_from_openmcp(
        doc: Dict[str, Any],
        executor: Executor,
) -> List[ToolDef]:
    """
    Translate OpenMCP tools into ToolDefs backed by the provided executor.
    No event-loop tricks; ToolDef gets a real async function per tool.
    """
    tools = (doc or {}).get("tools") or []
    out: List[ToolDef] = []

    for t in tools:
        name = t.get("name")
        if not name:
            continue
        desc = t.get("description") or ""
        args_schema = t.get("args_schema") or {}
        output_schema = t.get("output_schema") or None

        async def tool_impl(_tool_name=name, _output_schema=output_schema, **kwargs):
            result = await executor(_tool_name, kwargs)
            if _output_schema is None and not isinstance(result, dict):
                return {"result": str(result)}
            return result

        out.append(
            ToolDef(
                fn=tool_impl,
                name=name,
                description=desc,
                args_schema=args_schema,
                output_schema=output_schema,
            )
        )

    return out


def _mcp_from_openmcp(
        doc: Dict[str, Any],
        *,
        executor: Optional[Executor] = None,
        client_config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
):
    """
    Build a FastMCP from an OpenMCP document.
    - If `executor` is given, it runs tool calls.
    - Else we auto-proxy using `client_config` or endpoints in the doc.
    """
    if executor is None:
        executor = _make_default_executor(doc, client_config)

    mcp_name = name or ((doc.get("info") or {}).get("title") or (doc.get("server") or {}).get("name") or "mcp")
    tool_defs = _build_tool_defs_from_openmcp(doc, executor)
    return _mcp_from_tools(name=mcp_name, tools=tool_defs)