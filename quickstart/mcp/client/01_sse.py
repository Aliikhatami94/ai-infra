import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

async def main():
    url = "http://0.0.0.0:8000/sse-demo/sse"

    # SSE client returns (read, write) — exactly two values.
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(tools)

if __name__ == "__main__":
    asyncio.run(main())