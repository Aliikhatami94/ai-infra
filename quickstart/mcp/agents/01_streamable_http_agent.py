import asyncio

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

from ai_infra.llm import CoreAgent, Providers, Models


async def main():
    url = "http://0.0.0.0:8000/openapi-app/mcp"
    async with streamablehttp_client(url) as (read, write, _closer):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = CoreAgent()
            resp = await agent.arun_agent(
                messages=[{"role": "user", "content": "Ping my apiframeworks and tell me what you get"}],
                provider=Providers.openai,
                model_name=Models.openai.gpt_5_mini.value,
                tools=tools,
                model_kwargs={"temperature": 0.7},
            )
            print(resp)

if __name__ == "__main__":
    asyncio.run(main())
