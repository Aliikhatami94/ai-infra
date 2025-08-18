import asyncio
from ai_infra.fastmcp.client.core import ConnectionConfig, CoreMCPClient
from mcp.client.sse import sse_client
from mcp import ClientSession

cfg = [
    ConnectionConfig(
        name="sse",
        transport="sse",
        url="http://0.0.0.0:8000/sse-mcp/sse",
        # headers={"Authorization": "Bearer ..."}
    ),
]

client = CoreMCPClient(cfg)

async def get_tools():
    async with client as c:
        await c.connect("sse")
        tools = await c.list_server_tools("sse")
        print(tools)

async def call_tool():
    async with client as c:
        await c.connect("sse")  # or: await c.connect("weather")
        res = await c.call_tool("sse", "echo", {"msg": "Hello SSE MCP!"})
        print(c.extract_payload(res))  # nice helper

asyncio.run(call_tool())