from src.ai_infra.open_mcp.core import CoreMCP
from src.ai_infra.open_mcp.models import McpConfig, Server, ServerConfig, ServerMetadata, Prompt
from ai_infra.llm import Models, Providers, CoreAgent

# Example config following the models in src/ai_infra/mcp/models.py
mcp_config = McpConfig(
    name="test_mcp",
    host="http://0.0.0.0:8000",
    prompts=[
        Prompt(contents=[
            "Your name is Alex."
        ])
    ],
    servers=[
        Server(
            metadata=ServerMetadata(
                id="test-mcp",
                name="test_server",
                description="Test server",
            ),
            config=ServerConfig(
                url="/test-mcp/mcp",
                transport="streamable_http",
            ),
        )
    ],
)
mcp = CoreMCP(config=mcp_config)

# Example usage in an async context
import asyncio

agent = CoreAgent()

async def ask_agent():
    tools = await mcp.get_tools()
    res = await agent.arun_agent(
        messages=[{"role": "user", "content": "How is the weather in Paris?"}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4_1_mini.value,
        tools=tools
    )
    print(res)

# Run the example
if __name__ == "__main__":
    asyncio.run(ask_agent())

