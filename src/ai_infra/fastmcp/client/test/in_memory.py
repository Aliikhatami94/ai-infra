import asyncio
from fastmcp import FastMCP

from ai_infra.fastmcp.client.core import ConnectionConfig, CoreMCPClient

mcp = FastMCP("Local")
@mcp.tool
def greet(name: str) -> str:
    return f"hello {name}"

client_cfg = [
    ConnectionConfig(name="Local", transport="in_memory", mcp=mcp)
]

async def demo():
    async with CoreMCPClient(client_cfg) as c:
        await c.connect("Local")
        tools = await c.list_tools("Local")
        print("Tools:", [t.name for t in tools])
        res = await c.call_tool("Local", "greet", {"name": "Ford"})
        print("Payload:", CoreMCPClient.extract_payload(res))

asyncio.run(demo())