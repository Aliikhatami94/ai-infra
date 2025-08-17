from mcp.server.fastmcp import FastMCP
from typing import Iterable, Union, Optional

from ai_infra.mcp.models import ToolDef, ToolFn

def setup_mcp_server(name: Optional[str], tools: Optional[Iterable[Union[ToolFn, ToolDef]]]) -> FastMCP:
    server = FastMCP(name=name)
    for item in tools:
        if isinstance(item, ToolDef):
            fn = item.fn
            name = item.name or fn.__name__
            desc = item.description or (fn.__doc__ or f"{name} tool").strip()
        else:
            fn = item
            name = fn.__name__
            desc = (fn.__doc__ or f"{name} tool").strip()

        server.add_tool(
            name=name,
            description=desc,
            fn=fn,
        )
    return server