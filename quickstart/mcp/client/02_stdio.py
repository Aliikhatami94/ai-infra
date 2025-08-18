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

client = CoreMCPClient(cfg)

async def get_tools():
    async with CoreMCPClient(cfg) as c:
        await c.connect("Wikipedia")
        tools = await c.list_server_tools("Wikipedia")
        print(tools)

async def call_tool():
    async with client as c:
        tool = "Wikipedia"
        await c.connect(tool)  # or: await c.connect("weather")
        res = await c.call_tool(tool, "search", {"query": "Capital of France"})
        print(c.extract_payload(res))  # nice helper

asyncio.run(get_tools())