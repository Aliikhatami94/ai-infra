from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

async def main():
    # Use the exact MCP base URL your server exposes (often .../mcp)
    url = "http://0.0.0.0:8000/apiframeworks-mcp/mcp"

    # streamablehttp_client returns (read, write, closer)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print(tools)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())