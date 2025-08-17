# openapi_to_mcp.py
from __future__ import annotations
import json, yaml, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx

from mcp.server.fastmcp import FastMCP

def load_spec(path_or_str: str | Path) -> Dict[str, Any]:
    p = Path(path_or_str)
    text = p.read_text(encoding="utf-8") if p.exists() else str(path_or_str)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)

def sanitize_tool_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "op"

def op_tool_name(path: str, method: str, opid: Optional[str]) -> str:
    if opid:  # prefer operationId if present
        return sanitize_tool_name(opid)
    return sanitize_tool_name(f"{method.lower()}_{path.strip('/').replace('/', '_')}")

def pick_base_url(spec: Dict[str, Any], override: Optional[str] = None) -> str:
    if override:
        return override.rstrip("/")
    servers = spec.get("servers") or []
    if servers:
        return str(servers[0].get("url", "")).rstrip("/") or ""
    return ""

def collect_params(op: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out = {"path": [], "query": [], "header": []}
    for p in (op.get("parameters") or []):
        loc = p.get("in")
        if loc in out:
            out[loc].append(p)
    return out

def has_request_body(op: Dict[str, Any]) -> bool:
    return bool(op.get("requestBody", {}).get("content"))

def extract_body_content_type(op: Dict[str, Any]) -> str:
    # choose a reasonable default
    content = op.get("requestBody", {}).get("content", {})
    for ct in ("application/json", "application/x-www-form-urlencoded", "text/plain"):
        if ct in content:
            return ct
    # fallback to first
    return next(iter(content.keys())) if content else "application/json"

def build_mcp_from_openapi(spec: Dict[str, Any], base_url: Optional[str] = None) -> FastMCP:
    if not isinstance(spec, dict):
        spec = load_spec(spec)  # your loader that handles JSON/YAML or file path
    mcp = FastMCP(spec.get("info", {}).get("title") or "OpenAPI MCP")

    root_base = pick_base_url(spec, base_url)

    for path, path_item in (spec.get("paths") or {}).items():
        for method, op in path_item.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete", "head", "options"}:
                continue
            name = op_tool_name(path, method, op.get("operationId"))
            desc = op.get("summary") or op.get("description") or f"{method.upper()} {path}"
            params = collect_params(op)
            wants_body = has_request_body(op)
            body_ct = extract_body_content_type(op) if wants_body else None

            # capture loop vars with defaults
            def register_tool(name=name, desc=desc, method=method, path=path,
                              params=params, wants_body=wants_body, body_ct=body_ct):
                @mcp.tool(name=name, description=desc)
                async def tool(**kwargs) -> str:
                    # prepare URL
                    url_base = (kwargs.pop("_base_url", None) or root_base).rstrip("/")
                    if not url_base:
                        return "Error: no base URL provided (spec.servers[] missing and _base_url not set)."

                    # path params
                    url_path = path
                    for p in params["path"]:
                        pname = p.get("name")
                        if p.get("required") and pname not in kwargs:
                            return f"Missing required path param: {pname}"
                        if pname in kwargs:
                            url_path = url_path.replace("{" + pname + "}", str(kwargs.pop(pname)))

                    # query params
                    q = {}
                    for p in params["query"]:
                        pname = p.get("name")
                        if pname in kwargs:
                            q[pname] = kwargs.pop(pname)
                        elif p.get("required"):
                            return f"Missing required query param: {pname}"

                    # header params
                    headers = {}
                    for p in params["header"]:
                        pname = p.get("name")
                        if pname in kwargs:
                            headers[pname] = str(kwargs.pop(pname))
                        elif p.get("required"):
                            return f"Missing required header: {pname}"

                    data = None
                    json_body = None

                    if wants_body:
                        body_arg = kwargs.pop("body", None)
                        if body_arg is None and op.get("requestBody", {}).get("required"):
                            return "Missing required request body: pass as 'body=<json/dict/str>'."
                        if body_arg is not None:
                            if body_ct == "application/json":
                                json_body = body_arg  # httpx will serialize
                                headers.setdefault("Content-Type", "application/json")
                            elif body_ct == "application/x-www-form-urlencoded":
                                data = body_arg
                                headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
                            else:
                                # send as raw text/bytes for other content types
                                data = body_arg
                                headers.setdefault("Content-Type", body_ct)

                    # auth helpers (optional): env or passthrough
                    # e.g., support _api_key or _auth_header from kwargs
                    if "_api_key" in kwargs:
                        headers.setdefault("Authorization", f"Bearer {kwargs.pop('_api_key')}")
                    if "_headers" in kwargs and isinstance(kwargs["_headers"], dict):
                        headers.update(kwargs.pop("_headers"))

                    # anything left in kwargs gets added to query by default
                    for k, v in list(kwargs.items()):
                        q[k] = v

                    full_url = f"{url_base}{url_path}"
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.request(method.upper(), full_url, params=q, headers=headers,
                                                    json=json_body, data=data)
                    # return JSON if possible, else text
                    ct = resp.headers.get("content-type", "")
                    if "application/json" in ct:
                        try:
                            return json.dumps(resp.json(), indent=2)
                        except Exception:
                            return resp.text
                    return resp.text
            register_tool()
    return mcp