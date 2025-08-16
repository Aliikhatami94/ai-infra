from mcp.server.fastmcp import FastMCP
from typing import Iterable, Union
from .models import ToolDef, ToolFn


def setup_mcp_server(tools: Iterable[Union[ToolFn, ToolDef]]) -> FastMCP:
    server = FastMCP()
    for item in tools:
        if isinstance(item, ToolDef):
            fn = item.fn
            name = (item.name or (fn.__name__ if fn else "tool")).strip()
            desc = (item.description or (fn.__doc__ if fn and fn.__doc__ else f"{name} tool")).strip()
        else:
            fn = item
            name = fn.__name__
            desc = (fn.__doc__ or f"{name} tool").strip()

        if fn is None:
            # Skip registering if no callable was supplied (metadata-only tool)
            continue

        server.add_tool(name=name, description=desc, fn=fn)
    return server