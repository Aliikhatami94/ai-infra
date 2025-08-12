from src.ai_infra.mcp import CoreMCP

# Example config following the models in src/ai_infra/mcp/models.py
example_config = {
    "name": "ExampleMCP",
    "host": "http://localhost:8000",
    "prompts": {
        "greeting": ["Hello!", "How can I help you?"]
    },
    "servers": {
        "server1": {
            "id": "server1",
            "name": "Test Server",
            "description": "A test server for MCP.",
            "config": {
                "url": "/test-server/mcp",
                "transport": "streamable_http"
            }
        }
    }
}

# Instantiate CoreMCP with the config
core_mcp = CoreMCP(config=example_config)

# Example usage in an async context
import asyncio

async def main():
    metadata = await core_mcp.get()
    print("Agent Metadata:", metadata)

# Run the example
if __name__ == "__main__":
    asyncio.run(main())

