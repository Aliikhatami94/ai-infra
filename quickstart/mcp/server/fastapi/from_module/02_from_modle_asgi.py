from mcp.server.fastmcp import FastMCP

_f = FastMCP("from-module-asgi")

def hi(name: str) -> str:
    """Say hi (prebuilt ASGI)."""
    return f"hi {name}"

_f.add_tool(hi)

# Build the streamable app here and attach its manager
app = _f.streamable_http_app()
app.state.session_manager = _f.session_manager