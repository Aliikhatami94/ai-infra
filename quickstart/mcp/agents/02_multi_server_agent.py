import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection

from ai_infra import Providers, Models
from ai_infra.llm import CoreAgent


# also available: StdioConnection, SSEConnection, WebsocketConnection

async def main():
    client = MultiServerMCPClient({
        "streamable": StreamableHttpConnection(
            transport="streamable_http",
            url="http://0.0.0.0:8000/streamable-app/mcp",  # must end with /mcp
        ),
        "openapi": StreamableHttpConnection(
            transport="streamable_http",
            url="http://0.0.0.0:8000/openapi-app/mcp",  # must end with /mcp
        )
    })

    tools = await client.get_tools()          # merged tools from all servers
    agent = CoreAgent()
    resp = await agent.arun_agent(
        messages=[{"role": "user", "content": "Call my ping endpoint. what did you get?"}],
        provider=Providers.google_genai,
        model_name=Models.google_genai.gemini_2_5_flash.value,
        tools=tools,
        model_kwargs={"temperature": 0.7},
    )
    print(resp)

if __name__ == "__main__":
    asyncio.run(main())