from src.ai_infra.open_mcp.core import CoreMCP
from src.ai_infra.open_mcp.models import McpConfig, Server, ServerConfig, RemoteServerInfo
from ai_infra.llm import Models, Providers, CoreAgent

config = McpConfig(
    prompts=[],
    servers=[
        Server(
            info=RemoteServerInfo(
                name="wikipedia_mcp",
                description="Wikipedia MCP server"
            ),
            config=ServerConfig(
                command="npx",
                args=["-y", "wikipedia-mcp"],
                transport="stdio",
            )
        ),
        Server(
            info=RemoteServerInfo(
                name="test_server",
                description="Test server",
            ),
            config=ServerConfig(
                url="http://0.0.0.0:8000/test-mcp/mcp",
                transport="streamable_http",
            )
        )
    ]
)
mcp = CoreMCP(config=config)

# Example usage in an async context
import asyncio

agent = CoreAgent()

async def ask_agent():
    tools = await mcp.get_tools()
    res = await agent.arun_agent(
        messages=[{"role": "user", "content": "What is the capital of France? search wikipedia. then tell me the weather in New York City."}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4_1_mini.value,
        tools=tools
    )
    print(res)

# Run the example
if __name__ == "__main__":
    asyncio.run(ask_agent())
    # print(asyncio.run(mcp.get_server_setup()))

