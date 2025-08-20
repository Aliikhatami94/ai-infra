from __future__ import annotations
import base64
import logging
import os
import httpx
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Callable, Awaitable, Tuple
from pydantic import BaseModel, Field, create_model, ConfigDict, conlist
from mcp.server.fastmcp import FastMCP

from .models import OpenAPISpec, OperationContext, BuildReport, OpReport
from .constants import ALLOWED_METHODS
from .runtime import (
    op_tool_name, has_request_body, extract_body_content_type, merge_parameters,
    split_params, pick_effective_base_url_with_source,
)
from .io import load_openapi

__all__ = ["_mcp_from_openapi"]
log = logging.getLogger(__name__)

# ---------------------- Diagnostics logging ----------------------

def _maybe_log_report(report: BuildReport, report_log: bool) -> None:
    if not report_log:
        return
    log.info(
        "[OpenAPI→MCP] title=%s tools=%d/%d skipped=%d warnings=%d",
        report.title, report.registered_tools, report.total_ops,
        report.skipped_ops, len(report.warnings)
    )
    for op in report.ops:
        log.debug(
            "[OpenAPI→MCP] %s %s -> tool=%s base=%s (%s) params={path:%d query:%d header:%d cookie:%d} "
            "body(%s, req=%s) fields=%d media=%s warn=%s",
            op.method, op.path, op.tool_name, op.base_url, op.base_url_source,
            op.params.get("path", 0), op.params.get("query", 0),
            op.params.get("header", 0), op.params.get("cookie", 0),
            op.body_content_type, op.body_required, op.input_model_fields,
            op.media_types_seen, op.warnings
        )
    for w in report.warnings:
        log.warning("[OpenAPI→MCP][global-warning] %s", w)

# ---------------------- Security ----------------------

class SecurityResolver:
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
                    elif t == "oauth2":
                        bearer = True
                    elif t == "apiKey":
                        where = sch.get("in"); keyname = sch.get("name")
                        if keyname:
                            if where == "header": header_keys.append(keyname)
                            elif where == "query": query_keys.append(keyname)
        return cls(header_api_keys=header_keys, query_api_keys=query_keys, bearer=bearer, basic=basic)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "header_api_keys": list(self.header_api_keys),
            "query_api_keys": list(self.query_api_keys),
            "bearer": bool(self.bearer),
            "basic": bool(self.basic),
        }

    def apply(self, headers: dict, query: dict, kwargs: dict):
        if "_headers" in kwargs and isinstance(kwargs["_headers"], dict):
            headers.update(kwargs.pop("_headers"))
        if self.bearer and "_api_key" in kwargs:
            headers.setdefault("Authorization", f"Bearer {kwargs.pop('_api_key')}")
        if self.basic and "_basic_auth" in kwargs:
            cred = kwargs.pop("_basic_auth")
            if isinstance(cred, (list, tuple)) and len(cred) == 2:
                token = base64.b64encode(f"{cred[0]}:{cred[1]}".encode()).decode()
            else:
                token = str(cred)
            headers.setdefault("Authorization", f"Basic {token}")
        for k in list(kwargs.keys()):
            if k in self.header_api_keys:
                headers.setdefault(k, str(kwargs.pop(k)))
            if k in self.query_api_keys:
                query.setdefault(k, kwargs.pop(k))

# ---------------------- Context helpers ----------------------

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

# ---------------------- Schema helpers ----------------------

def _resolve_ref(schema: Dict[str, Any], spec: OpenAPISpec) -> Dict[str, Any]:
    ref = schema.get("$ref")
    if not ref or not isinstance(ref, str):
        return schema
    if not ref.startswith("#/"):
        return schema
    parts = ref.lstrip("#/").split("/")
    node: Any = spec
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return schema
        node = node[p]
    return node if isinstance(node, dict) else schema

def _py_type_from_schema(schema: Dict[str, Any], spec: OpenAPISpec | None = None) -> Any:
    if spec is not None and isinstance(schema, dict) and "$ref" in schema:
        schema = _resolve_ref(schema, spec)

    t = (schema or {}).get("type")
    fmt = (schema or {}).get("format")

    if t == "string":
        return bytes if fmt in {"binary", "byte"} else str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        items = (schema or {}).get("items") or {}
        return List[_py_type_from_schema(items, spec)]  # type: ignore[index]
    if t == "object" or ("properties" in (schema or {})):
        from pydantic import BaseModel, create_model, ConfigDict
        props = (schema or {}).get("properties") or {}
        reqs = set((schema or {}).get("required") or [])
        fields: dict[str, tuple[object, object]] = {}
        for k, v in props.items():
            typ = _py_type_from_schema(v or {}, spec)
            default = ... if k in reqs else None
            fields[k] = (typ, default)
        Model = create_model(
            "AnonModel",
            __base__=BaseModel,
            __config__=ConfigDict(populate_by_name=True, protected_namespaces=()),
            **fields
        )
        return Model

    return Any

# ---------------------- Input / Output models ----------------------

def _build_input_model(op_ctx: OperationContext, path_item: dict, op: dict, spec: OpenAPISpec) -> type[BaseModel]:
    fields: dict[str, tuple[object, object]] = {}

    def _extract_param_type(param: Dict[str, Any]) -> Any:
        schema = param.get("schema") or {}
        return _py_type_from_schema(schema, spec)

    for p in op_ctx.path_params + op_ctx.query_params + op_ctx.header_params + op_ctx.cookie_params:
        name = p.get("name")
        if not name:
            continue
        typ = _extract_param_type(p)
        required = p.get("required", False) or (p.get("in") == "path")
        default = ... if required else None
        fields[name] = (typ, default)

    if op_ctx.wants_body:
        req = (op.get("requestBody") or {})
        content = (req.get("content") or {})
        body_schema = ((content.get(op_ctx.body_content_type) or {}).get("schema")) \
                      or ((content.get("application/json") or {}).get("schema")) \
                      or {}
        body_typ = _py_type_from_schema(body_schema, spec) if body_schema else Any
        fields["body"] = (body_typ, ... if op_ctx.body_required else None)

        if op_ctx.body_content_type == "multipart/form-data":
            fields["files"] = (Optional[Dict[str, Any]], Field(default=None, alias="_files"))

    BasicAuthList = conlist(str, min_length=2, max_length=2)
    fields["headers"]    = (Optional[Dict[str, str]], Field(default=None, alias="_headers"))
    fields["api_key"]    = (Optional[str],           Field(default=None, alias="_api_key"))
    fields["basic_auth"] = (Optional[Union[str, BasicAuthList]], Field(default=None, alias="_basic_auth"))
    fields["base_url"]   = (Optional[str],           Field(default=None, alias="_base_url"))

    Model = create_model(
        "Input_" + op_ctx.name,
        __base__=BaseModel,
        __config__=ConfigDict(populate_by_name=True, protected_namespaces=()),
        **fields,
        )
    return Model

def _pick_response_schema(op: dict, spec: OpenAPISpec) -> tuple[Optional[dict], Optional[str]]:
    responses = (op.get("responses") or {})
    for status, resp in sorted(responses.items(), key=lambda kv: kv[0]):
        try:
            code = int(status)
        except Exception:
            continue
        if 200 <= code < 300 and isinstance(resp, dict):
            content = (resp.get("content") or {})
            if "application/json" in content:
                schema = (content["application/json"].get("schema")) or {}
                return (_resolve_ref(schema, spec), "application/json")
    for _status, resp in responses.items():
        if not isinstance(resp, dict): continue
        content = (resp.get("content") or {})
        for ct, cnode in content.items():
            schema = (cnode or {}).get("schema")
            if schema:
                return (_resolve_ref(schema, spec), ct)
    return (None, None)

def _build_output_model(op_ctx: OperationContext, op: dict, spec: OpenAPISpec) -> type[BaseModel]:
    """
    Envelope: status, headers, url, method, and payload as either:
      - alias 'json' (typed if we discovered a schema), OR
      - 'text'
    Uses aliases to avoid shadowing BaseModel.json().
    """
    resp_schema, resp_ct = _pick_response_schema(op, spec)

    fields: dict[str, tuple[object, object]] = {
        "status": (int, ...),
        "headers": (Dict[str, str], ...),
        "url": (str, ...),
        "method": (str, ...),
    }

    # use internal names with alias="json"/"text"
    if resp_schema and (resp_ct == "application/json"):
        payload_type = _py_type_from_schema(resp_schema, spec)
        fields["payload_json"] = (Optional[payload_type], Field(default=None, alias="json"))  # type: ignore[name-defined]
        fields["payload_text"] = (Optional[str], Field(default=None, alias="text"))
    else:
        fields["payload_json"] = (Optional[Any], Field(default=None, alias="json"))
        fields["payload_text"] = (Optional[str], Field(default=None, alias="text"))

    Model = create_model(
        "Output_" + op_ctx.name,
        __base__=BaseModel,
        __config__=ConfigDict(populate_by_name=True, protected_namespaces=()),
        **fields,
        )
    return Model

# ---------------------- Tool registration ----------------------

def _register_operation_tool(
        mcp: FastMCP,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        spec: OpenAPISpec,
        op: dict,
        op_ctx: OperationContext,
        report: BuildReport,
        base_url_source: str,
) -> None:
    warnings: List[str] = []
    InputModel  = _build_input_model(op_ctx, path_item={}, op=op, spec=spec)
    OutputModel = _build_output_model(op_ctx, op, spec)
    security = SecurityResolver.from_spec(spec, op)

    media_types = list(((op.get("requestBody") or {}).get("content") or {}).keys())
    if op_ctx.wants_body and media_types and op_ctx.body_content_type not in media_types:
        warnings.append(
            f"Chosen content-type {op_ctx.body_content_type!r} not present in requestBody keys={media_types!r}"
        )
    if len(media_types) > 1:
        preferred = ("application/json", "application/x-www-form-urlencoded", "multipart/form-data", "text/plain")
        if not any(mt in preferred for mt in media_types):
            warnings.append(f"Multiple body media types; defaulting to {op_ctx.body_content_type!r}")

    for p in (op.get("parameters") or []):
        if p.get("deprecated"):
            warnings.append(f"Parameter '{p.get('name')}' is deprecated")
        style = p.get("style")
        explode = p.get("explode")
        if style not in (None, "form", "simple", "matrix", "label", "spaceDelimited", "pipeDelimited", "deepObject"):
            warnings.append(f"Unrecognized style={style!r} for param '{p.get('name')}'")
        if explode not in (None, True, False):
            warnings.append(f"Unrecognized explode={explode!r} for param '{p.get('name')}'")

    def _has_var(url: str) -> bool:
        return "{" in url and "}" in url
    if base_url and _has_var(base_url):
        warnings.append(f"Base URL contains server variables; not expanded: {base_url!r}")

    if op_ctx.wants_body and op_ctx.body_content_type not in (
            None, "application/json", "application/x-www-form-urlencoded", "multipart/form-data",
            "text/plain", "application/octet-stream"
    ):
        warnings.append(f"Unsupported content-type mapped as raw data: {op_ctx.body_content_type!r}")

    if not base_url:
        warnings.append("No effective base URL: spec.servers empty and no override; tool will require _base_url.")

    op_rep = OpReport(
        operation_id=op.get("operationId"),
        tool_name=op_ctx.name,
        method=op_ctx.method,
        path=op_ctx.path,
        base_url=base_url or "",
        base_url_source=base_url_source,
        has_body=op_ctx.wants_body,
        body_content_type=op_ctx.body_content_type,
        body_required=op_ctx.body_required,
        params={
            "path":   len(op_ctx.path_params),
            "query":  len(op_ctx.query_params),
            "header": len(op_ctx.header_params),
            "cookie": len(op_ctx.cookie_params),
        },
        security=security.as_dict(),
        media_types_seen=media_types,
        warnings=[],
    )
    # record input model field count
    try:
        op_rep.input_model_fields = len(getattr(InputModel, "model_fields", {}))
    except Exception:
        pass

    async def tool(args: Optional[InputModel] = None) -> OutputModel:
        # Allow completely empty calls (e.g., ping) by treating None as {}
        payload = (args.model_dump(by_alias=True, exclude_none=True) if args is not None else {})

        url_base   = (payload.pop("_base_url", None) or base_url).rstrip("/")
        api_key    = payload.pop("_api_key", None)
        basic_auth = payload.pop("_basic_auth", None)
        headers_in = payload.pop("_headers", None) or {}

        if not url_base:
            # Keep structure consistent even on error: still return OutputModel
            out = {
                "status": 0,
                "headers": {},
                "url": "",
                "method": op_ctx.method,
                "json": None,
                "text": "Error: no base URL provided (servers missing and _base_url not set).",
            }
            return OutputModel.model_validate(out)

        errors: list[str] = []

        url_path = op_ctx.path
        for p in op_ctx.path_params:
            pname = p.get("name")
            if p.get("required") and pname not in payload:
                errors.append(f"Missing required path param: {pname}")
                continue
            if pname in payload:
                url_path = url_path.replace("{" + pname + "}", str(payload.pop(pname)))

        query: Dict[str, Any] = {}
        headers: Dict[str, str] = {}
        cookies: Dict[str, str] = {}

        for p in op_ctx.query_params:
            pname = p.get("name")
            if pname in payload:
                query[pname] = payload.pop(pname)
            elif p.get("required"):
                errors.append(f"Missing required query param: {pname}")

        for p in op_ctx.header_params:
            pname = p.get("name")
            if pname in payload:
                headers[pname] = str(payload.pop(pname))
            elif p.get("required"):
                errors.append(f"Missing required header: {pname}")

        for p in op_ctx.cookie_params:
            pname = p.get("name")
            if pname in payload:
                cookies[pname] = str(payload.pop(pname))
            elif p.get("required"):
                errors.append(f"Missing required cookie: {pname}")

        data = json_body = files = None
        if op_ctx.wants_body:
            body_arg = payload.pop("body", None)
            if body_arg is None and op_ctx.body_required:
                errors.append("Missing required request body: pass 'body'.")
            elif body_arg is not None:
                ct = op_ctx.body_content_type
                if ct == "application/json":
                    json_body = body_arg; headers.setdefault("Content-Type", "application/json")
                elif ct == "application/x-www-form-urlencoded":
                    data = body_arg; headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
                elif ct == "multipart/form-data":
                    files = payload.pop("_files", None)
                    if files is None:
                        if isinstance(body_arg, dict):
                            files = {k: (k, v) for k, v in body_arg.items()}
                        else:
                            files = {"file": ("file", body_arg)}
                elif ct in ("text/plain", "application/octet-stream"):
                    data = body_arg; headers.setdefault("Content-Type", ct)
                else:
                    data = body_arg
                    if ct: headers.setdefault("Content-Type", ct)

        if errors:
            out = {
                "status": 0,
                "headers": {},
                "url": f"{url_base}{url_path}",
                "method": op_ctx.method,
                "json": None,
                "text": "Validation errors:\n" + "\n".join(f" - {e}" for e in errors),
            }
            return OutputModel(**out)

        security.apply(headers, query, {"_api_key": api_key, "_basic_auth": basic_auth, "_headers": headers_in})

        for k, v in list(payload.items()):
            if not str(k).startswith("_"):
                query[k] = v
            payload.pop(k, None)

        full_url = f"{url_base}{url_path}"
        resp = await client.request(
            op_ctx.method, full_url,
            params=query or None, headers=headers or None, cookies=cookies or None,
            json=json_body, data=data, files=files,
        )

        content_type = resp.headers.get("content-type", "")
        out = {
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "url": str(resp.request.url),
            "method": resp.request.method,
            "json": None,
            "text": None,
        }
        if "application/json" in content_type:
            try:
                out["json"] = resp.json()
            except Exception:
                out["text"] = resp.text
        else:
            try:
                out["json"] = resp.json()
            except Exception:
                out["text"] = resp.text

        return OutputModel.model_validate(out)

    # expose schemas to MCP (input/outputSchema) via annotations
    tool.__annotations__ = {"args": Optional[InputModel], "return": OutputModel}
    mcp.add_tool(name=op_ctx.name, description=op_ctx.full_description(), fn=tool)

    op_rep.warnings.extend(warnings)
    report.ops.append(op_rep)

# ---------------------- Builder entrypoint ----------------------

def _mcp_from_openapi(
        spec: Union[dict, str, Path],
        *,
        client: httpx.AsyncClient | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
        base_url: str | None = None,
        strict_names: bool = False,
        report_log: Optional[bool] = None,
) -> Tuple[FastMCP, Optional[Callable[[], Awaitable[None]]], BuildReport]:
    """
    Build a FastMCP from OpenAPI and return (mcp, async_cleanup, report).
    """
    if not isinstance(spec, dict):
        spec = load_openapi(spec)

    title = (spec.get("info", {}) or {}).get("title") or "OpenAPI MCP"
    report = BuildReport(title=title)
    if report_log is None:
        report_log = (os.getenv("MCP_OPENAPI_DEBUG", "0") == "1")

    own_client = False
    if client is None:
        client = client_factory() if client_factory else httpx.AsyncClient(timeout=30.0)
        own_client = True

    mcp = FastMCP(title)
    seen_tool_names: set[str] = set()
    paths = spec.get("paths") or {}
    total_ops = 0

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            report.warnings.append(f"Path item is not an object: {path}")
            continue
        for method, op in path_item.items():
            if method.lower() not in ALLOWED_METHODS or not isinstance(op, dict):
                continue
            total_ops += 1

            op_ctx = _make_operation_context(path, method, path_item, op)

            base_tool = op_ctx.name
            if base_tool in seen_tool_names:
                msg = f"Duplicate tool name '{base_tool}' from operationId/path; renaming."
                if strict_names:
                    raise ValueError(msg)
                report.warnings.append(msg)
                i = 2
                new_name = f"{base_tool}_{i}"
                while new_name in seen_tool_names:
                    i += 1
                    new_name = f"{base_tool}_{i}"
                op_ctx.name = new_name
            seen_tool_names.add(op_ctx.name)

            effective_base, source = pick_effective_base_url_with_source(
                spec, path_item, op, override=base_url
            )

            try:
                _register_operation_tool(
                    mcp,
                    client=client,
                    base_url=effective_base or "",
                    spec=spec,
                    op=op,
                    op_ctx=op_ctx,
                    report=report,
                    base_url_source=source,
                )
                report.registered_tools += 1
            except Exception as e:
                report.skipped_ops += 1
                warn = f"Failed to register tool for {method.upper()} {path}: {type(e).__name__}: {e}"
                report.warnings.append(warn)
                log.debug(warn, exc_info=True)

    report.total_ops = total_ops

    async_cleanup: Optional[Callable[[], Awaitable[None]]] = None
    if own_client:
        async def _cleanup() -> None:
            await client.aclose()
        async_cleanup = _cleanup

    _maybe_log_report(report, report_log)

    return mcp, async_cleanup, report