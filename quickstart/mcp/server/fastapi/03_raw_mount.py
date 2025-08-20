from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from ai_infra.mcp.server.core import CoreMCPServer, MCPMount

app = FastAPI()
mcp_server = CoreMCPServer(strict=True)

# Build a streamable FastMCP
fm = FastMCP("mount-raw")

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

fm.add_tool(add)
streamable_app = fm.streamable_http_app()

# Example: you can decide lifecycle handling explicitly via MCPMount
mount = MCPMount(
    path="/raw",
    app=streamable_app,
    session_manager=fm.session_manager,  # include to enable lifecycle
    require_manager=True,                # enforce lifecycle (raises if missing)
)

mcp_server.add(mount)          # add the raw mount record
mcp_server.attach_to_fastapi(app)

# Run: uvicorn quickstarts.04_mcpmount_main:app --reload