from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import yaml

from .models import OpenAPISpec

__all__ = ["load_openapi", "load_spec"]


def load_openapi(source: Union[str, Path, dict]) -> OpenAPISpec:
    """Load OpenAPI spec from various sources.

    Supports:
    - Dict: Returns as-is (already parsed)
    - URL (http/https): Fetches from remote
    - Local file path: Reads JSON or YAML
    - Raw JSON/YAML string: Parses directly

    Example:
        # URL
        spec = load_openapi("https://api.example.com/openapi.json")

        # Local file
        spec = load_openapi("./openapi.json")
        spec = load_openapi(Path("./openapi.yaml"))

        # Dict (passthrough)
        spec = load_openapi({"openapi": "3.1.0", "paths": {...}})

        # Raw JSON/YAML string
        spec = load_openapi('{"openapi": "3.1.0", ...}')
    """
    # Already a dict - return as-is
    if isinstance(source, dict):
        return source

    source_str = str(source)

    # URL - fetch remotely
    if source_str.startswith(("http://", "https://")):
        return _fetch_openapi_url(source_str)

    # Local file path
    p = Path(source_str)
    if p.exists() and p.is_file():
        return _load_openapi_file(p)

    # Raw JSON/YAML string
    return _parse_openapi_string(source_str)


def _fetch_openapi_url(url: str) -> OpenAPISpec:
    """Fetch OpenAPI spec from a URL."""
    import httpx

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        # Try JSON first
        if "json" in content_type or url.endswith(".json"):
            return resp.json()

        # Try YAML
        if "yaml" in content_type or url.endswith((".yaml", ".yml")):
            return yaml.safe_load(resp.text)

        # Auto-detect from content
        return _parse_openapi_string(resp.text)


def _load_openapi_file(path: Path) -> OpenAPISpec:
    """Load OpenAPI spec from a local file."""
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        return json.loads(text)
    elif path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    else:
        # Auto-detect
        return _parse_openapi_string(text)


def _parse_openapi_string(text: str) -> OpenAPISpec:
    """Parse OpenAPI spec from raw JSON or YAML string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def load_spec(source: Union[str, Path, dict]) -> OpenAPISpec:
    """Alias for load_openapi."""
    return load_openapi(source)
