import asyncio

from ai_infra.llm import CoreAgent, Providers, Models
from ai_infra.mcp.client.models import McpServerConfig
from ai_infra.mcp.client.core import CoreMCPClient

cfg = [
    McpServerConfig(
        transport="streamable_http",
        url="http://0.0.0.0:8000/streamable-app/mcp"
    )
]

async def main():
    client = CoreMCPClient(cfg)
    tools = await client.list_tools()
    agent = CoreAgent()
    resp = await agent.arun_agent(
        messages=[{"role": "user", "content": "How is the weather in Chicago today?"}],
        provider=Providers.google_genai,
        model_name=Models.google_genai.gemini_2_5_flash.value,
        tools=tools,
        model_kwargs={"temperature": 0.7},
    )
    print(resp)

if __name__ == "__main__":
    asyncio.run(main())