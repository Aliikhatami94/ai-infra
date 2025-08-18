import asyncio
from ai_infra.fastmcp.client.core import ConnectionConfig, CoreMCPClient

cfg = [
    ConnectionConfig(
        name="weather",
        transport="sse",
        url="http://127.0.0.1:8000/weather-mcp/sse",
        # headers={"Authorization": "Bearer ..."}  # optional
    ),
]

async def main():
    async with CoreMCPClient(cfg) as c:
        await c.connect("weather")
        tools = await c.list_server_tools("weather")
        print("Tools:", tools)
        res = await c.call_tool("weather", "get_weather", {"city": "New York"})
        print("Payload:", CoreMCPClient.extract_payload(res))

asyncio.run(main())