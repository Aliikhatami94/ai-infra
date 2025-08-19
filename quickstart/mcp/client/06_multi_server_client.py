import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection

async def main():
    client = MultiServerMCPClient({
        "streamable": StreamableHttpConnection(
            transport="streamable_http",
            url="http://0.0.0.0:8000/streamable-app/mcp",
        ),
        "openapi": StreamableHttpConnection(
            transport="streamable_http",
            url="http://0.0.0.0:8000/openapi-app/mcp",
        )
    })

    tools = await client.get_tools()
    print(tools)

if __name__ == "__main__":
    asyncio.run(main())