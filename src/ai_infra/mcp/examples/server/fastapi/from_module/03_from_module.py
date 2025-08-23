from fastapi import FastAPI
from ai_infra.mcp.server.core import CoreMCPServer

app = FastAPI()
mcp_server = CoreMCPServer(strict=True)

# Case A: module exports a FastMCP named `mcp`
mcp_server.add_from_module(
    "quickstarts.mcp.server.from_module.from_module_fastmcp:mcp",
    "/mod-fastmcp",
    transport="streamable_http",  # needed so we know which transport app to build
)

# Case B: module exports a prebuilt ASGI app named `app`
mcp_server.add_from_module(
    "quickstarts.mcp.server.from_module.from_module_asgi:app",
    "/mod-asgi",                  # transport ignored (already an ASGI app)
)

mcp_server.attach_to_fastapi(app)

# Run: uvicorn quickstarts.03_add_from_module_main:app --reload