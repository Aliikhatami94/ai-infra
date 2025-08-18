import asyncio

from ai_infra.fastmcp.client.core import ConnectionConfig, CoreMCPClient

cfg = [
    ConnectionConfig(
        name="weather",
        transport="streamable_http",
        url="http://0.0.0.0:8000/weather-mcp/mcp"
    ),
]

client = CoreMCPClient(cfg)

async def get_tools():
    async with client as c:
        await c.connect("weather")
        tools = await c.list_server_tools("weather")
        print(tools)

async def call_tool():
    async with client as c:
        await c.connect("weather")  # or: await c.connect("weather")
        res = await c.call_tool("weather", "get_weather", {"city": "New York"})
        print(c.extract_payload(res))  # nice helper

asyncio.run(call_tool())