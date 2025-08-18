import os
import httpx
import asyncio
import json
from pathlib import Path

os.environ["FASTMCP_EXPERIMENTAL_ENABLE_NEW_OPENAPI_PARSER"] = "true"

from ai_infra.fastmcp.server.core import CoreMCPServer

def say_hello(name: str) -> str:
    """Example tool that greets a user."""
    return f"Hello, {name}!"

path_to_spec = Path(__file__).parent / "apiframeworks.json"
spec = json.load(open(path_to_spec))

api_client = httpx.AsyncClient(
    base_url="https://0.0.0.0:8000",
)
openapi_server = CoreMCPServer.from_openapi(
    openapi_spec=spec,
    name="OpenAPI MCP Server",
    client=api_client,
)

openapi_server.tool(say_hello)
tools = asyncio.run(openapi_server.list_tools())
print(tools)
