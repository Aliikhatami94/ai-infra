from __future__ import annotations

import json
from pathlib import Path

import yaml

from .models import OpenAPISpec

__all__ = ["load_openapi", "load_spec"]


def load_openapi(source: str | Path | dict) -> OpenAPISpec:
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


def _fetch_openapi_url(
    url: str,
    *,
    timeout: float = 30.0,
    retries: int = 0,
) -> OpenAPISpec:
    """Fetch OpenAPI spec from a URL.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        retries: Number of retries on failure

    Raises:
        OpenAPINetworkError: On network or HTTP errors
        OpenAPIParseError: On parse errors
    """
    import httpx

    # Import error classes (avoid circular import)
    from .builder import OpenAPINetworkError, OpenAPIParseError

    last_error: Exception | None = None
    attempts = retries + 1

    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url)

                if resp.status_code >= 400:
                    raise OpenAPINetworkError(
                        f"HTTP {resp.status_code}: {resp.reason_phrase}",
                        url=url,
                        status_code=resp.status_code,
                    )

                content_type = resp.headers.get("content-type", "")

                # Try JSON first
                if "json" in content_type or url.endswith(".json"):
                    try:
                        from typing import cast

                        return cast("OpenAPISpec", resp.json())
                    except json.JSONDecodeError as e:
                        raise OpenAPIParseError(f"Invalid JSON: {e}") from e

                # Try YAML
                if "yaml" in content_type or url.endswith((".yaml", ".yml")):
                    try:
                        from typing import cast

                        return cast("OpenAPISpec", yaml.safe_load(resp.text))
                    except yaml.YAMLError as e:
                        raise OpenAPIParseError(f"Invalid YAML: {e}") from e

                # Auto-detect from content
                return _parse_openapi_string(resp.text)

        except httpx.TimeoutException as e:
            last_error = OpenAPINetworkError(f"Request timeout: {e}", url=url)
            if attempt < attempts - 1:
                continue
        except httpx.ConnectError as e:
            last_error = OpenAPINetworkError(f"Connection error: {e}", url=url)
            if attempt < attempts - 1:
                continue
        except httpx.HTTPError as e:
            last_error = OpenAPINetworkError(f"HTTP error: {e}", url=url)
            if attempt < attempts - 1:
                continue
        except (OpenAPINetworkError, OpenAPIParseError):
            raise

    if last_error:
        raise last_error

    raise OpenAPINetworkError("Unknown error fetching URL", url=url)


def _load_openapi_file(path: Path) -> OpenAPISpec:
    """Load OpenAPI spec from a local file."""
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        from typing import cast

        return cast("OpenAPISpec", json.loads(text))
    elif path.suffix in (".yaml", ".yml"):
        from typing import cast

        return cast("OpenAPISpec", yaml.safe_load(text))
    else:
        # Auto-detect
        return _parse_openapi_string(text)


def _parse_openapi_string(text: str) -> OpenAPISpec:
    """Parse OpenAPI spec from raw JSON or YAML string."""
    try:
        from typing import cast

        return cast("OpenAPISpec", json.loads(text))
    except json.JSONDecodeError:
        from typing import cast

        return cast("OpenAPISpec", yaml.safe_load(text))


def load_spec(source: str | Path | dict) -> OpenAPISpec:
    """Alias for load_openapi."""
    return load_openapi(source)
