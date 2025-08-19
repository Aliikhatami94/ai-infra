# client_streamable.py
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

async def main():
    url = "http://0.0.0.0:8000/streamable-app/mcp"
    async with streamablehttp_client(url) as (read, write, closer):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(tools)

if __name__ == "__main__":
    asyncio.run(main())