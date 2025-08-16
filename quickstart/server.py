from src.ai_infra.mcp.core import CoreMCP
from src.ai_infra.mcp.models import McpConfig, Server, ServerConfig, ServerMetadata, Prompt

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

async def main():
    metadata = await mcp.get_tools()
    print(metadata)

# Run the example
if __name__ == "__main__":
    asyncio.run(main())

