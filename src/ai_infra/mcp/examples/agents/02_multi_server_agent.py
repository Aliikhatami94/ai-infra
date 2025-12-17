import asyncio
from typing import Any

from ai_infra import Providers
from ai_infra.llm import Agent
from ai_infra.mcp.client import MCPClient

cfg: list[dict[str, Any]] = [
    {
        "transport": "streamable_http",
        "url": "http://0.0.0.0:8000/streamable-app/mcp",
    },
    {
        "transport": "sse",
        "url": "http://0.0.0.0:8000/sse-demo/sse",
    },
    {"transport": "stdio", "command": "npx", "args": ["-y", "wikipedia-mcp"]},
]


async def main():
    client = MCPClient(cfg)
    tools = await client.list_tools()
    agent = Agent()
    resp = await agent.arun_agent(
        messages=[
            {
                "role": "user",
                "content": "What is the capital of france? use wikipedia to find out.",
            }
        ],
        provider=Providers.openai,
        model_name="gpt-4o",
        tools=tools,
        model_kwargs={"temperature": 0.7},
    )
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())
