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

client = CoreMCPClient(cfg)

async def get_tools():
    async with client as c:
        await c.connect("weather")
        tools = await c.list_tools()
        print(tools)

async def call_tool():
    async with client as c:
        await c.connect_all()  # or: await c.connect("weather")
        res = await c.call_tool("weather", "get_weather", {"city": "New York"})
        print(c.extract_payload(res))  # nice helper

asyncio.run(call_tool())