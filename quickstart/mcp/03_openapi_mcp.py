import asyncio

from ai_infra import RemoteMcp
from ai_infra.mcp.core import CoreMCP
from ai_infra.llm import CoreAgent

from ai_infra.mcp.models import RemoteServer, McpServerConfig

remote_mcp_config = RemoteMcp(
    servers=[
        RemoteServer(
            config=McpServerConfig(
                url="http://0.0.0.0:8000/spotify-mcp/mcp",
                transport="streamable_http",
            )
        ),
        RemoteServer(
            config=McpServerConfig(
                url="http://0.0.0.0:8000/apiframeworks-mcp/mcp",
                transport="streamable_http",
            )
        ),
        RemoteServer(
            config=McpServerConfig(
                url="http://0.0.0.0:8000/weather-mcp/mcp",
                transport="streamable_http",
            )
        )
    ]
)

agent = CoreAgent()

async def _get_openapi_agent():
    mcp = CoreMCP(config=remote_mcp_config)
    tools = await mcp.list_tools()
    # res = await agent.arun_agent(
    #     messages=[{"role": "user", "content": "Tell me all the tools available"}],
    #     provider=Providers.openai,
    #     model_name=Models.openai.gpt_5_mini.value,
    #     tools=tools,
    # )
    print(tools)

def main():
    asyncio.run(_get_openapi_agent())

if __name__ == "__main__":
    main()