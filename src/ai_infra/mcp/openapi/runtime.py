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


def merge_parameters(path_item: Dict[str, Any] | None, op: Operation) -> List[Dict[str, Any]]:
    """Merge path-level and operation-level parameters with op-level overriding.

    Does not resolve $ref. Skips invalid parameter objects.
    """
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for src in (path_item.get("parameters") if path_item else []) or []:
        if isinstance(src, dict) and {"in", "name"} <= src.keys():
            merged.append(src)
            seen.add((src["in"], src["name"]))
    for src in (op.get("parameters") or []):
        if isinstance(src, dict) and {"in", "name"} <= src.keys():
            key = (src["in"], src["name"])
            if key in seen:
                for i, existing in enumerate(merged):
                    if (existing.get("in"), existing.get("name")) == key:
                        merged[i] = src
                        break
            else:
                merged.append(src)
                seen.add(key)
    return merged


def split_params(params: List[Dict[str, Any]]):
    path_params: List[Dict[str, Any]] = []
    query_params: List[Dict[str, Any]] = []
    header_params: List[Dict[str, Any]] = []
    cookie_params: List[Dict[str, Any]] = []
    for p in params:
        loc = p.get("in")
        if loc == "path":
            path_params.append(p)
        elif loc == "query":
            query_params.append(p)
        elif loc == "header":
            header_params.append(p)
        elif loc == "cookie":
            cookie_params.append(p)
    return path_params, query_params, header_params, cookie_params


def pick_effective_base_url(spec: OpenAPISpec, path_item: Dict[str, Any] | None, op: Operation | None, override: Optional[str]) -> str:
    """Return base URL honoring precedence: override > op.servers > path.servers > root.servers.
    """
    if override:
        return override.rstrip("/")
    for node in (op or {}, path_item or {}, spec):  # type: ignore[arg-type]
        servers = node.get("servers") or []  # type: ignore[assignment]
        if servers:
            return str(servers[0].get("url", "")).rstrip("/") or ""
    return ""

__all__ = [
    "sanitize_tool_name",
    "op_tool_name",
    "pick_base_url",
    "collect_params",
    "has_request_body",
    "extract_body_content_type",
    "merge_parameters",
    "split_params",
    "pick_effective_base_url",
]
