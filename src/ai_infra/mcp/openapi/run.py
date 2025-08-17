# run_server.py
import asyncio
from fastapi import FastAPI
from ai_infra.mcp.openapi.core import load_spec, build_mcp_from_openapi

spec = load_spec("apiframeworks.json")  # or JSON
mcp = build_mcp_from_openapi(spec)  # or override base with
print(mcp)# build_mcp_from_openapi(spec, base_url="https://api.example.com")

app = FastAPI()
app.mount("/openapi-mcp", mcp.streamable_http_app())

# If you need an MCP session manager lifespan elsewhere:
# async with mcp.session_manager.run(): ...