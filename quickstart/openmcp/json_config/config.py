import json
from pathlib import Path

json_path = Path(__file__).parent / "template.json"
json_text = json_path.read_text(encoding="utf-8")
mcp_json_config = json.loads(json_text)