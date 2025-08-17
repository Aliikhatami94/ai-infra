from __future__ import annotations
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Union, Optional
import base64
import httpx
from mcp.server.fastmcp import FastMCP

from .models import OpenAPISpec, OperationContext
from .constants import ALLOWED_METHODS
from .runtime import (
    op_tool_name,
    has_request_body,
    extract_body_content_type,
    merge_parameters,
    split_params,
    pick_effective_base_url,
)

__all__ = ["build_mcp_from_openapi", "load_openapi", "load_spec"]

# ---------------- Security Resolver -----------------
class SecurityResolver:
    """Pragmatic security handler: apiKey (header/query), http bearer/basic.

    It inspects spec.components.securitySchemes and effective security (op.security or root.security).
    Users may still override by passing _api_key, _basic_auth, or explicit header/query kwargs.
    """
    def __init__(self, header_api_keys=None, query_api_keys=None, bearer=False, basic=False):
        self.header_api_keys = header_api_keys or []
        self.query_api_keys = query_api_keys or []
        self.bearer = bearer
        self.basic = basic

    @classmethod
    def from_spec(cls, spec: OpenAPISpec, op: dict) -> "SecurityResolver":
        effective = op.get("security", spec.get("security"))
        schemes = (spec.get("components", {}) or {}).get("securitySchemes", {}) or {}
        header_keys: list[str] = []
        query_keys: list[str] = []
        bearer = False
        basic = False
        if effective:
            for requirement in effective:
                if not isinstance(requirement, dict):
                    continue
                for name in requirement.keys():
                    sch = schemes.get(name) or {}
                    t = sch.get("type")
                    if t == "http" and sch.get("scheme") == "bearer":
                        bearer = True
                    elif t == "http" and sch.get("scheme") == "basic":
                        basic = True
                    elif t == "apiKey":
                        where = sch.get("in")
                        keyname = sch.get("name")
                        if keyname:
                            if where == "header":
                                header_keys.append(keyname)
                            elif where == "query":
                                query_keys.append(keyname)
        return cls(header_api_keys=header_keys, query_api_keys=query_keys, bearer=bearer, basic=basic)

    def apply(self, headers: dict, query: dict, kwargs: dict):
        # merge user-provided headers first
        if "_headers" in kwargs and isinstance(kwargs["_headers"], dict):
            headers.update(kwargs.pop("_headers"))
        # bearer token via _api_key
        if self.bearer and "_api_key" in kwargs:
            headers.setdefault("Authorization", f"Bearer {kwargs.pop('_api_key')}")
        # basic auth via _basic_auth -> tuple/user:pass
        if self.basic and "_basic_auth" in kwargs:
            cred = kwargs.pop("_basic_auth")
            if isinstance(cred, (list, tuple)) and len(cred) == 2:
                token = base64.b64encode(f"{cred[0]}:{cred[1]}".encode()).decode()
            else:
                token = str(cred)
            headers.setdefault("Authorization", f"Basic {token}")
        # apiKey mappings
        for k in list(kwargs.keys()):
            if k in self.header_api_keys:
                headers.setdefault(k, str(kwargs.pop(k)))
            if k in self.query_api_keys:
                query.setdefault(k, kwargs.pop(k))

# --------------- Spec Loading ------------------

def load_openapi(path_or_str: str | Path) -> OpenAPISpec:
    p = Path(path_or_str)
    text = p.read_text(encoding="utf-8") if p.exists() else str(path_or_str)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)

def load_spec(path_or_str: str | Path) -> OpenAPISpec:  # backward compat alias
    return load_openapi(path_or_str)

# --------------- Operation Context --------------

def _make_operation_context(path: str, method: str, path_item: dict, op: dict) -> OperationContext:
    merged = merge_parameters(path_item, op)
    path_params, query_params, header_params, cookie_params = split_params(merged)
    wants_body = has_request_body(op)
    body_ct = extract_body_content_type(op) if wants_body else None
    return OperationContext(
        name=op_tool_name(path, method, op.get("operationId")),
        description=op.get("summary") or op.get("description") or f"{method.upper()} {path}",
        method=method.upper(),
        path=path,
        path_params=path_params,
        query_params=query_params,
        header_params=header_params,
        cookie_params=cookie_params,
        wants_body=wants_body,
        body_content_type=body_ct,
        body_required=bool(op.get("requestBody", {}).get("required")) if wants_body else False,
    )

# --------------- Tool Registration --------------

def _register_operation_tool(mcp: FastMCP, *, base_url: str, spec: OpenAPISpec, op: dict, op_ctx: OperationContext) -> None:
    security = SecurityResolver.from_spec(spec, op)

    @mcp.tool(name=op_ctx.name, description=op_ctx.full_description())
    async def tool(**kwargs) -> str:  # type: ignore[override]
        # effective base url precedence: user override > computed
        url_base = (kwargs.pop("_base_url", None) or base_url).rstrip("/")
        if not url_base:
            return "Error: no base URL provided (servers missing and _base_url not set)."

        errors: list[str] = []

        # path params
        url_path = op_ctx.path
        for p in op_ctx.path_params:
            pname = p.get("name")
            if p.get("required") and pname not in kwargs:
                errors.append(f"Missing required path param: {pname}")
                continue
            if pname in kwargs:
                url_path = url_path.replace("{" + pname + "}", str(kwargs.pop(pname)))

        # query / header / cookie params
        query: Dict[str, Any] = {}
        headers: Dict[str, str] = {}
        cookies: Dict[str, str] = {}

        for p in op_ctx.query_params:
            pname = p.get("name")
            if pname in kwargs:
                query[pname] = kwargs.pop(pname)
            elif p.get("required"):
                errors.append(f"Missing required query param: {pname}")
        for p in op_ctx.header_params:
            pname = p.get("name")
            if pname in kwargs:
                headers[pname] = str(kwargs.pop(pname))
            elif p.get("required"):
                errors.append(f"Missing required header: {pname}")
        for p in op_ctx.cookie_params:
            pname = p.get("name")
            if pname in kwargs:
                cookies[pname] = str(kwargs.pop(pname))
            elif p.get("required"):
                errors.append(f"Missing required cookie: {pname}")

        # body handling (multipart, json, form, octet-stream, text, fallback)
        data = None
        json_body = None
        files = None
        if op_ctx.wants_body:
            body_arg = kwargs.pop("body", None)
            if body_arg is None and op_ctx.body_required:
                errors.append("Missing required request body: pass as 'body=<json/dict/str>'.")
            elif body_arg is not None:
                ct = op_ctx.body_content_type
                if ct == "application/json":
                    json_body = body_arg
                    headers.setdefault("Content-Type", "application/json")
                elif ct == "application/x-www-form-urlencoded":
                    data = body_arg
                    headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
                elif ct == "multipart/form-data":
                    files = kwargs.pop("_files", None)
                    if files is None:
                        if isinstance(body_arg, dict):
                            # convert dict into multipart parts (simple heuristic)
                            files = {k: (k, v) for k, v in body_arg.items()}
                        else:
                            files = {"file": ("file", body_arg)}
                    # httpx sets boundary automatically; don't set explicit Content-Type
                elif ct in ("text/plain", "application/octet-stream"):
                    data = body_arg
                    headers.setdefault("Content-Type", ct)
                else:
                    data = body_arg
                    if ct:
                        headers.setdefault("Content-Type", ct)

        if errors:
            return "Validation errors:\n" + "\n".join(f" - {e}" for e in errors)

        # Apply security AFTER collecting explicit params so user overrides win
        security.apply(headers, query, kwargs)

        # leftover kwargs -> query params (loose mapping)
        for k, v in list(kwargs.items()):
            query[k] = v

        full_url = f"{url_base}{url_path}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                op_ctx.method,
                full_url,
                params=query,
                headers=headers,
                cookies=cookies,
                json=json_body,
                data=data,
                files=files,
            )
        content_type = resp.headers.get("content-type", "")
        result: Dict[str, Any] = {
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "url": str(resp.request.url),
            "method": resp.request.method,
        }
        # attach body
        if "application/json" in content_type:
            try:
                result["json"] = resp.json()
            except Exception:
                result["text"] = resp.text
        else:
            try:
                # some APIs send JSON with wrong/missing content-type
                result["json"] = resp.json()
            except Exception:
                result["text"] = resp.text
        # default serialization as JSON string (to remain tool-friendly)
        return json.dumps(result, indent=2, default=str)

# --------------- Builder -----------------------

def build_mcp_from_openapi(spec: Union[dict, str, Path], base_url: str | None = None) -> FastMCP:
    if not isinstance(spec, dict):
        spec = load_openapi(spec)
    mcp = FastMCP(spec.get("info", {}).get("title") or "OpenAPI MCP")
    paths = spec.get("paths") or {}

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, op in path_item.items():
            if method.lower() not in ALLOWED_METHODS:
                continue
            if not isinstance(op, dict):
                continue
            op_ctx = _make_operation_context(path, method, path_item, op)
            effective_base = pick_effective_base_url(spec, path_item, op, override=base_url)
            _register_operation_tool(
                mcp,
                base_url=effective_base,
                spec=spec,
                op=op,
                op_ctx=op_ctx,
            )
    return mcp