import asyncio

from ai_infra.llm import Agent, Providers
from ai_infra.mcp.client import MCPClient
from ai_infra.mcp.client.models import McpServerConfig

cfg = [McpServerConfig(transport="streamable_http", url="http://0.0.0.0:8000/streamable-app/mcp")]


async def main():
    client = MCPClient(cfg)
    tools = await client.list_tools()
    agent = Agent()
    resp = await agent.arun_agent(
        messages=[{"role": "user", "content": "How is the weather in Chicago today?"}],
        provider=Providers.google_genai,
        model_name="gemini-2.5-flash",
        tools=tools,
        model_kwargs={"temperature": 0.7},
    )
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())
