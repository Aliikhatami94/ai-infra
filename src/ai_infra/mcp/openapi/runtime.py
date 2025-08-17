from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from .models import OpenAPISpec, Operation


def sanitize_tool_name(s: str) -> str:
    """Return a safe tool name (alphanumeric + underscores, not starting/ending with underscore)."""
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "op"


def op_tool_name(path: str, method: str, opid: Optional[str]) -> str:
    """Derive a tool name from operationId if present, else from method + path."""
    if opid:
        return sanitize_tool_name(opid)
    return sanitize_tool_name(f"{method.lower()}_{path.strip('/').replace('/', '_')}")


def pick_base_url(spec: OpenAPISpec, override: Optional[str] = None) -> str:
    """Return an effective base URL from an override or the first server entry."""
    if override:
        return override.rstrip("/")
    servers = spec.get("servers") or []
    if servers:
        return str(servers[0].get("url", "")).rstrip("/") or ""
    return ""


def collect_params(op: Operation) -> Dict[str, List[Dict[str, Any]]]:
    """Collect parameters grouped by 'in' location (path, query, header)."""
    out = {"path": [], "query": [], "header": []}
    for p in (op.get("parameters") or []):
        loc = p.get("in")
        if loc in out:
            out[loc].append(p)
    return out


def has_request_body(op: Operation) -> bool:
    """True if the operation defines at least one requestBody content entry."""
    return bool(op.get("requestBody", {}).get("content"))


def extract_body_content_type(op: Operation) -> str:
    """Pick a reasonable content-type for the request body.

    Preference order: json, form-url-encoded, text/plain; else the first declared; defaults to application/json.
    """
    content = op.get("requestBody", {}).get("content", {})
    for ct in ("application/json", "application/x-www-form-urlencoded", "text/plain"):
        if ct in content:
            return ct
    return next(iter(content.keys())) if content else "application/json"

__all__ = [
    "sanitize_tool_name",
    "op_tool_name",
    "pick_base_url",
    "collect_params",
    "has_request_body",
    "extract_body_content_type",
]

