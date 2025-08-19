import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools

async def main():
    url = "http://0.0.0.0:8000/openapi-api/mcp"
    async with streamablehttp_client(url) as (read, write, closer):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print(tools)

if __name__ == "__main__":
    asyncio.run(main())