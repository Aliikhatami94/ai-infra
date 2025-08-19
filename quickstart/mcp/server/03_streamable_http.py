from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-streamable")

def ping(text: str) -> str:
    """Echo back text with a 'pong:' prefix."""
    return f"pong: {text}"

mcp.add_tool(ping)
streamable_app = mcp.streamable_http_app()
streamable_app.state.session_manager = mcp.session_manager
