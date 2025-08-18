import asyncio

from ai_infra.fastmcp.client.core import ConnectionConfig, CoreMCPClient

cfg = [
    ConnectionConfig(
        name="weather",
        transport="streamable_http",
        url="http://0.0.0.0:8000/weather-mcp/mcp"
    ),
    ConnectionConfig(
        name="Wikipedia",
        transport="stdio",
        command="npx",
        args=["-y", "wikipedia-mcp"],
    ),
]

async def demo():
    async with CoreMCPClient(cfg) as c:
        await c.connect("weather")
        tools = await c.list_tools()
        print(tools)

asyncio.run(demo())