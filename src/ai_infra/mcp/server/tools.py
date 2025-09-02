from __future__ import annotations
from mcp.server.fastmcp import FastMCP
from typing import Iterable, Optional, Union, Callable, Awaitable
import inspect
import textwrap
from pydantic import BaseModel, Field


ToolFn = Callable[..., str | Awaitable[str]]

class ToolDef(BaseModel):
    fn: Optional[ToolFn] = Field(default=None, exclude=True)
    name: Optional[str] = None
    description: Optional[str] = None

def _describe(fn: Callable[..., object], fallback: str) -> str:
    doc = inspect.getdoc(fn) or ""
    doc = textwrap.dedent(doc).strip()
    return doc or f"{fallback} tool"

def mcp_from_functions(
        *,
        name: Optional[str],
        functions: Iterable[Union[ToolFn, ToolDef]] | None,
) -> FastMCP:
    """
    Create a FastMCP from plain functions or ToolDef objects.
    - If a ToolDef is provided, use its .name/.description, else infer from function.
    - Deduplicates by final tool name (last one wins).
    """
    server = FastMCP(name=name)
    if not functions:
        return server

    seen: set[str] = set()
    for item in functions:
        if isinstance(item, ToolDef):
            fn = getattr(item, "fn", None)
            if fn is None:
                continue  # or raise ValueError("ToolDef.fn is required")
            tool_name = getattr(item, "name", None) or fn.__name__
            desc = (getattr(item, "description", None) or _describe(fn, tool_name)).strip()
        else:
            fn = item
            tool_name = fn.__name__
            desc = _describe(fn, tool_name)

        # best-effort dedupe; last one wins
        if tool_name in seen:
            # If FastMCP ever supports removal/replacement, we could call it here.
            pass
        seen.add(tool_name)

        server.add_tool(name=tool_name, description=desc, fn=fn)

    return server
