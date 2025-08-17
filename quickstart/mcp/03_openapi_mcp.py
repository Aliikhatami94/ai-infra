import asyncio

from ai_infra import RemoteMcp
from ai_infra.mcp.core import CoreMCP
from ai_infra.llm import Models, Providers, CoreAgent

from ai_infra.mcp.models import Prompt, RemoteServer, RemoteServerConfig
from ai_infra.mcp.openapi.builder import load_spec

remote_mcp_config = RemoteMcp(
    prompts=[Prompt(contents=[
        "Your name is AskOpenMCP.",
        "You are a helpful assistant.",
    ])],
    servers=[
        RemoteServer(
            name="openapi_mcp",
            config=RemoteServerConfig(
                url="http://0.0.0.0:8000/openapi-mcp/mcp",
                transport="streamable_http",
            )
        )
    ]
)


agent = CoreAgent()

async def _get_openapi_agent():
    mcp = CoreMCP(config=remote_mcp_config)
    tools = await mcp.list_tools()
    spec = load_spec("./resources/apiframeworks.json")
    res = await agent.arun_agent(
        messages=[{"role": "user", "content": f"tell me the weather in New York City. after Can you tell me the governance score of my OpenAPI spec? {spec}."}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4_1_mini.value,
        tools=tools,
    )
    print(res)

def main():
    asyncio.run(_get_openapi_agent())

if __name__ == "__main__":
    main()