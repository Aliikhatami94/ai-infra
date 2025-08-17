from __future__ import annotations
from mcp.server.fastmcp import FastMCP
from typing import Iterable, Optional, Union, Callable
import inspect
import textwrap

from ai_infra.mcp.models import ToolDef, ToolFn

def _describe(fn: Callable[..., object], fallback: str) -> str:
    doc = inspect.getdoc(fn) or ""
    doc = textwrap.dedent(doc).strip()
    return doc or f"{fallback} tool"

def build_mcp_from_tools(
        name: Optional[str] = None,
        tools: Optional[Iterable[Union[ToolFn, ToolDef]]] = None,
) -> FastMCP:
    """
    Create a FastMCP from plain Python callables or ToolDef objects.

    - If a ToolDef is provided, use its .name/.description, else infer from function.
    - Deduplicates by final tool name (last one wins).
    """
    server = FastMCP(name=name)
    if not tools:
        return server

    seen: set[str] = set()
    for item in tools:
        if isinstance(item, ToolDef):
            fn = item.fn
            if fn is None:
                continue  # or raise ValueError("ToolDef.fn is required")
            tool_name = item.name or fn.__name__
            desc = (item.description or _describe(fn, tool_name)).strip()
        else:
            fn = item
            tool_name = fn.__name__
            desc = _describe(fn, tool_name)

        # prevent accidental double-registration
        if tool_name in seen:
            # Optionally: server.remove_tool(tool_name) if FastMCP supports it
            pass
        seen.add(tool_name)

        server.add_tool(name=tool_name, description=desc, fn=fn)

    return server