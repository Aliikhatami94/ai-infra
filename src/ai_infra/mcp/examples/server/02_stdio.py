from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-stdio")

def ping(text: str) -> str:
    """Echo back text with a 'pong:' prefix."""
    return f"pong: {text}"

mcp.add_tool(ping)

if __name__ == '__main__':
    mcp.run(transport="stdio")