import json
import yaml
from pathlib import Path
from src.ai_infra.mcp.models import OpenMcp, RemoteServer, RemoteServerConfig, Prompt

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
        RemoteServer(
            name="wikipedia_mcp",
            config=RemoteServerConfig(
                command="npx",
                args=["-y", "wikipedia-mcp"],
                transport="stdio",
            )
        ),
        RemoteServer(
            name="test_server",
            config=RemoteServerConfig(
                url="http://0.0.0.0:8000/test-mcp/mcp",
                transport="streamable_http",
            )
        )
    ]
)