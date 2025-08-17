import json
import yaml
from pathlib import Path
from src.ai_infra.mcp.models import OpenMcp, Server, ServerConfig, RemoteServerInfo, Prompt

json_path = Path(__file__).parent / "config.json"
json_text = json_path.read_text(encoding="utf-8")
json_config = json.loads(json_text)

yaml_path = Path(__file__).parent / "config.yaml"
yaml_text = yaml_path.read_text(encoding="utf-8")
yaml_config = yaml.safe_load(yaml_text)

openmcp_config = OpenMcp(
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