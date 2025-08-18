import asyncio

from ai_infra.fastmcp.client.core import ConnectionConfig, CoreMCPClient

cfg = [
    ConnectionConfig(
        name="Wikipedia",
        transport="stdio",
        command="npx",
        args=["-y", "wikipedia-mcp"],
    ),
]

async def demo():
    async with CoreMCPClient(cfg) as c:
        await c.connect("Wikipedia")
        tools = await c.list_tools("Wikipedia")
        print(tools)

asyncio.run(demo())