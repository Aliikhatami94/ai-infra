import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools

async def main():
    params = StdioServerParameters(
        command="npx",
        args=["-y", "wikipedia-mcp"],
    )

    # stdio_client -> (read, write)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print(tools)

if __name__ == "__main__":
    asyncio.run(main())