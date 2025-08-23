from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from ai_infra.mcp.server.core import CoreMCPServer

app = FastAPI()
mcp_server = CoreMCPServer(strict=True)

# FastMCP #1 — streamable
m1 = FastMCP("streamable-from-code")

def whoami() -> str:
    """Return a simple id."""
    return "mcp:streamable-from-code"

m1.add_tool(whoami)

# FastMCP #2 — sse
m2 = FastMCP("sse-from-code")

def ping(text: str = "pong") -> str:
    """Echo text (SSE)."""
    return f"echo:{text}"

m2.add_tool(ping)

# Tell CoreMCPServer which transport to build
mcp_server.add_fastmcp(m1, "/streamable", transport="streamable_http")  # lifecycle auto-managed
mcp_server.add_fastmcp(m2, "/sse", transport="sse")                     # no manager required

mcp_server.attach_to_fastapi(app)

# Run: uvicorn quickstarts.02_add_fastmcp_main:app --reload