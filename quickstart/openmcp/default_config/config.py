from src.ai_infra.openmcp.models import OpenMcp, Server, ServerConfig, RemoteServerInfo, Prompt

config = OpenMcp(
    prompts=[Prompt(contents=[
        "Your name is AskOpenMCP.",
        "You are a helpful assistant.",
    ])],
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