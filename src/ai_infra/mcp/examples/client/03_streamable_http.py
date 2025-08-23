import asyncio

from langchain_mcp_adapters.tools import load_mcp_tools
from ai_infra.mcp.client.core import CoreMCPClient

cfg = [
    {
        "transport": "streamable_http",
        "url": "http://0.0.0.0:8000/streamable-app/mcp",
    }
]

client = CoreMCPClient(cfg)

async def main():
    async with client.get_client("streamable-app") as session:
        # session is already initialized by get_client()
        tools = await load_mcp_tools(session)
        print(tools)

if __name__ == "__main__":
    asyncio.run(main())