import asyncio

from ai_infra import RemoteMcp
from ai_infra.mcp.core import CoreMCP
from ai_infra.llm import Models, Providers, CoreAgent

from ai_infra.mcp.models import RemoteServer, RemoteServerConfig
from ai_infra.mcp.openapi.builder import load_spec

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
        messages=[{"role": "user", "content": '''here is the headers for spority: { "_headers": { "Authorization": "Bearer BQDPIUXYD3CKb_1Ez0ItHpOiWkYEzO3j71zUoVCZosNKhoiGs1S7nVr-_X7pnuXmxP5ujefrtOA5OJZgRrhh5ftuSqmqYDhrjGvwToir4a5mxl9m2f5zZpExwdQ_dPalGjuaAxQfFH5qgO_a1h-sOVMkepvu17IHotq2vZQ28tZET42zkNXa7gmcZzNxkG9K3ilGTt5KLAm1MSlhX-1xSRSNLVqxwnIXGiEryjuuAF692eq54_QtHjchQhlLYg5zEBoQE_Zyjw1wQMF_EgKnnyFFSzV2fjbLjrCCFWgwYTZR8pNSf2LJGBqvI1Uft6hI" } } can you get me kayne albums from spotify'''}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4_1_mini.value,
        tools=tools,
    )
    print(res)

def main():
    asyncio.run(_get_openapi_agent())

if __name__ == "__main__":
    main()