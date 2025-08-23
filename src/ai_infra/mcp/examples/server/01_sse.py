from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sse-demo")

def ping(text: str = "pong") -> str:
    """Echo text back."""
    return f"echo:{text}"

mcp.add_tool(ping)
mcp_app = mcp.sse_app()