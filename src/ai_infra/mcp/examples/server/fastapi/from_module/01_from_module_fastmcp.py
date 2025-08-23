from mcp.server.fastmcp import FastMCP

mcp = FastMCP("from-module")

def echo(msg: str) -> str:
    """Echo message (from module FastMCP)."""
    return f"echo:{msg}"

mcp.add_tool(echo)