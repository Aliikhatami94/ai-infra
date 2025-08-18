import asyncio

from ai_infra import RemoteMcp
from ai_infra.mcp.core import CoreMCP
from ai_infra.llm import Models, Providers, CoreAgent

from ai_infra.mcp.models import RemoteServer, RemoteServerConfig

remote_mcp_config = RemoteMcp(
    servers=[
        RemoteServer(
            name="Spotify API",
            config=RemoteServerConfig(
                url="http://0.0.0.0:8000/spotify-mcp/mcp",
                transport="streamable_http",
            )
        )
    ]
)

agent = CoreAgent()

async def _get_openapi_agent():
    mcp = CoreMCP(config=remote_mcp_config)
    tools = await mcp.list_tools()
    res = await agent.arun_agent(
        messages=[{"role": "user", "content": '''can you tell me my profile info'''}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4_1_mini.value,
        tools=tools,
    )
    print(res)

def main():
    asyncio.run(_get_openapi_agent())

if __name__ == "__main__":
    main()