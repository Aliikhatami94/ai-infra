import json
from pathlib import Path
from typing import Any

import httpx
import yaml

from ai_infra.mcp.server.openapi import _mcp_from_openapi

OpenAPISpec = dict[str, Any]
path_to_spec = (Path(__file__).resolve().parents[1] / "resources" / "apiframeworks.json").resolve()


def load_openapi(path_or_str: str | Path) -> OpenAPISpec:
    p = Path(path_or_str)
    text = p.read_text(encoding="utf-8") if p.exists() else str(path_or_str)
    try:
        result: OpenAPISpec = json.loads(text)
        return result
    except json.JSONDecodeError:
        result = yaml.safe_load(text)
        return dict(result) if result else {}


spec = load_openapi(path_to_spec)

client = httpx.AsyncClient(
    base_url="http://0.0.0.0:8000",
    timeout=30.0,
)

mcp, _cleanup, _report = _mcp_from_openapi(spec, client=client)
streamable_app = mcp.streamable_http_app()
streamable_app.state.session_manager = mcp.session_manager
