from src.ai_infra.mcp import CoreMCP
from src.ai_infra.mcp.models import McpConfig, Server, ServerConfig

# Example config following the models in src/ai_infra/mcp/models.py
example_config = McpConfig(
    name="ExampleMCP",
    host="http://localhost:8000",
    prompts={
        "greeting": ["Hello!", "How can I help you?"]
    },
    servers={
        "server1": Server(
            id="server1",
            name="Test Server",
            description="A test server for MCP.",
            config=ServerConfig(
                url="/test-server/mcp",
                transport="streamable_http"
            )
        ),
        "server2": Server(
            id="server2",
            name="Another Server",
            description="Another test server for MCP.",
            config=ServerConfig(
                command="python",
                args=["./graph.py"],
                transport="stdio"
            )
        )}
)

# Instantiate CoreMCP with the config
core_mcp = CoreMCP(config=example_config)

# Example usage in an async context
import asyncio

async def main():
    metadata = await core_mcp.get_server_setup()
    print("MCP Server configuration metadata:")
    print(metadata)

# Run the example
if __name__ == "__main__":
    asyncio.run(main())

