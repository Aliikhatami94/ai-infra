from src.ai_infra.open_mcp.core import CoreMCP
from src.ai_infra.open_mcp.models import McpConfig, Server, ServerConfig, ServerMetadata
from ai_infra.llm import Models, Providers, CoreAgent

# Example config following the models in src/ai_infra/mcp/models.py
config = McpConfig(
    name="test_mcp",
    host="http://0.0.0.0:8000",
    prompts=[],
    servers=[
        Server(
            metadata=ServerMetadata(
                id="wikipedia_mcp",
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
            metadata=ServerMetadata(
                id="test-mcp",
                name="test_server",
                description="Test server"
            ),
            config=ServerConfig(
                url="/test-mcp/mcp",
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

