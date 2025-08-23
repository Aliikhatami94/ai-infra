import json
import yaml
import httpx
from pathlib import Path
from typing import Dict, Any

from ai_infra.mcp import _mcp_from_openapi

OpenAPISpec = Dict[str, Any]
path_to_spec = (Path(__file__).resolve().parents[1] / "resources" / "apiframeworks.json").resolve()

def load_openapi(path_or_str: str | Path) -> OpenAPISpec:
    p = Path(path_or_str)
    text = p.read_text(encoding="utf-8") if p.exists() else str(path_or_str)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)

spec = load_openapi(path_to_spec)

client = httpx.AsyncClient(
    base_url="http://0.0.0.0:8000",
    timeout=30.0,
)

mcp = _mcp_from_openapi(spec, client=client)   # <- injected client
streamable_app = mcp.streamable_http_app()
streamable_app.state.session_manager = mcp.session_manager
