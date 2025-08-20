from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from .models import OpenAPISpec, Operation

def sanitize_tool_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "op"

def op_tool_name(path: str, method: str, opid: Optional[str]) -> str:
    if opid:
        return sanitize_tool_name(opid)
    return sanitize_tool_name(f"{method.lower()}_{path.strip('/').replace('/', '_')}")

def pick_effective_base_url_with_source(
        spec: OpenAPISpec,
        path_item: Dict[str, Any] | None,
        op: Operation | None,
        override: Optional[str],
) -> tuple[str, str]:
    """
    Returns (url, source) where source ∈ {"override","operation","path","root","none"}.
    """
    if override:
        return override.rstrip("/"), "override"
    for source, node in (("operation", op or {}), ("path", path_item or {}), ("root", spec or {})):
        servers = node.get("servers") or []
        if servers:
            url = str(servers[0].get("url", "")).rstrip("/")
            if url:
                return url, source
    return "", "none"

def collect_params(op: Operation) -> Dict[str, List[Dict[str, Any]]]:
    out = {"path": [], "query": [], "header": []}
    for p in (op.get("parameters") or []):
        loc = p.get("in")
        if loc in out:
            out[loc].append(p)
    return out

def has_request_body(op: Operation) -> bool:
    return bool(op.get("requestBody", {}).get("content"))

def extract_body_content_type(op: Operation) -> str:
    content = op.get("requestBody", {}).get("content", {})
    for ct in ("application/json", "application/x-www-form-urlencoded", "text/plain"):
        if ct in content:
            return ct
    return next(iter(content.keys())) if content else "application/json"

def merge_parameters(path_item: Dict[str, Any] | None, op: Operation) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for src in (path_item.get("parameters") if path_item else []) or []:
        if isinstance(src, dict) and {"in", "name"} <= src.keys():
            merged.append(src); seen.add((src["in"], src["name"]))
    for src in (op.get("parameters") or []):
        if isinstance(src, dict) and {"in", "name"} <= src.keys():
            key = (src["in"], src["name"])
            if key in seen:
                for i, existing in enumerate(merged):
                    if (existing.get("in"), existing.get("name")) == key:
                        merged[i] = src; break
            else:
                merged.append(src); seen.add(key)
    return merged

def split_params(params: List[Dict[str, Any]]):
    path_params: List[Dict[str, Any]] = []
    query_params: List[Dict[str, Any]] = []
    header_params: List[Dict[str, Any]] = []
    cookie_params: List[Dict[str, Any]] = []
    for p in params:
        loc = p.get("in")
        if loc == "path":   path_params.append(p)
        elif loc == "query":  query_params.append(p)
        elif loc == "header": header_params.append(p)
        elif loc == "cookie": cookie_params.append(p)
    return path_params, query_params, header_params, cookie_params

def pick_effective_base_url(spec: OpenAPISpec, path_item: Dict[str, Any] | None, op: Operation | None, override: Optional[str]) -> str:
    if override:
        return override.rstrip("/")
    for node in (op or {}, path_item or {}, spec):  # op → path → root
        servers = node.get("servers") or []
        if servers:
            return str(servers[0].get("url", "")).rstrip("/") or ""
    return ""