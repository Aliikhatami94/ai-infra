import asyncio

from ai_infra import Providers, Models
from ai_infra.llm import CoreAgent
from ai_infra.mcp.client.core import CoreMCPClient

cfg = [
    {
        "transport": "streamable_http",
        "url": "http://0.0.0.0:8000/streamable-app/mcp",
    },
    {
        "transport": "sse",
        "url": "http://0.0.0.0:8000/sse-demo/sse",
    },
    {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "wikipedia-mcp"]
    }
]

async def main():
    client = CoreMCPClient(cfg)
    tools = await client.list_tools()
    agent = CoreAgent()
    resp = await agent.arun_agent(
        messages=[{"role": "user", "content": "What is the capital of france? use wikipedia to find out."}],
        provider=Providers.mistralai,
        model_name=Models.mistralai.codestral_latest.value,
        tools=tools,
        model_kwargs={"temperature": 0.7},
    )
    print(resp)

if __name__ == "__main__":
    asyncio.run(main())