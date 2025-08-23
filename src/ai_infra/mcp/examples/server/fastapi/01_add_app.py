from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from ai_infra.mcp.server.core import CoreMCPServer

app = FastAPI()
mcp_server = CoreMCPServer(strict=True)

# --- Streamable HTTP (exposes /streamable/mcp) ---
streamable = FastMCP("streamable-demo")

def say_hello(name: str) -> str:
    """Say hello (streamable)."""
    return f"Hello, {name}!"

streamable.add_tool(say_hello)

streamable_app = streamable.streamable_http_app()
# FastMCP exposes a session manager; put it on app.state for lifecycle mgmt.
streamable_app.state.session_manager = streamable.session_manager

# --- SSE (exposes /sse/messages/) ---
sse = FastMCP("sse-demo")

def ping(text: str = "pong") -> str:
    """Echo back text (SSE)."""
    return f"echo:{text}"

sse.add_tool(ping)
sse_app = sse.sse_app()  # no session manager needed for server-side

# Mount both
mcp_server.add_app("/streamable", streamable_app)           # requires manager (auto-detected)
mcp_server.add_app("/sse", sse_app, require_manager=False)  # skip lifecycle

# One-liner: mount + wire lifespans
mcp_server.attach_to_fastapi(app)

# Run: uvicorn quickstarts.01_add_app_main:app --reload