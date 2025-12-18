from __future__ import annotations

import base64
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, conlist, create_model

from .constants import ALLOWED_METHODS
from .io import load_openapi
from .models import (
    AuthConfig,
    BuildReport,
    OpenAPIOptions,
    OpenAPISpec,
    OperationContext,
    OpReport,
)
from .runtime import (
    extract_body_content_type,
    has_request_body,
    merge_parameters,
    op_tool_name,
    pick_effective_base_url_with_source,
    serialize_query_param,
    split_params,
)

__all__ = ["AuthConfig", "OpenAPIError", "OpenAPIOptions", "_mcp_from_openapi"]
log = logging.getLogger(__name__)


# ---------------------- Error Classes ----------------------


class OpenAPIError(Exception):
    """Base error for OpenAPI→MCP conversion errors."""

    pass


class OpenAPIParseError(OpenAPIError):
    """Error parsing OpenAPI specification."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


class OpenAPINetworkError(OpenAPIError):
    """Error fetching OpenAPI specification from URL."""

    def __init__(self, message: str, url: str, status_code: int | None = None):
        super().__init__(message)
        self.url = url
        self.status_code = status_code


class OpenAPIValidationError(OpenAPIError):
    """Error validating OpenAPI specification."""

    def __init__(self, message: str, path: str | None = None):
        super().__init__(message)
        self.path = path


# ---------------------- Diagnostics logging ----------------------


def _maybe_log_report(report: BuildReport, report_log: bool) -> None:
    if not report_log:
        return
    log.info(
        "[OpenAPI→MCP] title=%s tools=%d/%d skipped=%d warnings=%d",
        report.title,
        report.registered_tools,
        report.total_ops,
        report.skipped_ops,
        len(report.warnings),
    )
    for op in report.ops:
        log.debug(
            "[OpenAPI→MCP] %s %s -> tool=%s base=%s (%s) params={path:%d query:%d header:%d cookie:%d} "
            "body(%s, req=%s) fields=%d media=%s warn=%s",
            op.method,
            op.path,
            op.tool_name,
            op.base_url,
            op.base_url_source,
            op.params.get("path", 0),
            op.params.get("query", 0),
            op.params.get("header", 0),
            op.params.get("cookie", 0),
            op.body_content_type,
            op.body_required,
            op.input_model_fields,
            op.media_types_seen,
            op.warnings,
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
    def from_spec(cls, spec: OpenAPISpec, op: dict) -> SecurityResolver:
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
                        where = sch.get("in")
                        keyname = sch.get("name")
                        if keyname:
                            if where == "header":
                                header_keys.append(keyname)
                            elif where == "query":
                                query_keys.append(keyname)
        return cls(
            header_api_keys=header_keys,
            query_api_keys=query_keys,
            bearer=bearer,
            basic=basic,
        )

    def as_dict(self) -> dict[str, Any]:
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


async def _apply_auth_config(
    auth_config: AuthConfig | None,
    headers: dict[str, str],
    query: dict[str, Any],
) -> None:
    """Apply AuthConfig to request headers/query params."""
    if not auth_config:
        return

    # Apply static headers
    for k, v in auth_config.headers.items():
        headers.setdefault(k, v)

    # Apply query params
    for k, v in auth_config.query.items():
        query.setdefault(k, v)

    # Apply basic auth
    if auth_config.basic:
        u, p = auth_config.basic
        token = base64.b64encode(f"{u}:{p}".encode()).decode()
        headers.setdefault("Authorization", f"Basic {token}")

    # Apply bearer token
    if auth_config.bearer:
        headers.setdefault("Authorization", f"Bearer {auth_config.bearer}")

    # Apply dynamic bearer token
    if auth_config.bearer_fn:
        import asyncio

        fn = auth_config.bearer_fn
        if asyncio.iscoroutinefunction(fn):
            token = await fn()
        else:
            token = fn()
        headers.setdefault("Authorization", f"Bearer {token}")


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

# Track refs being resolved to detect circular references
_resolving_refs: set = set()


def _resolve_ref(
    schema: dict[str, Any],
    spec: OpenAPISpec,
    visited: set | None = None,
) -> dict[str, Any]:
    """Resolve $ref with circular reference detection."""
    ref = schema.get("$ref")
    if not ref or not isinstance(ref, str):
        return schema
    if not ref.startswith("#/"):
        return schema

    # Detect circular reference
    if visited is None:
        visited = set()
    if ref in visited:
        # Circular reference - return a placeholder
        return {"type": "object", "description": f"[Circular ref: {ref}]"}
    visited.add(ref)

    parts = ref.lstrip("#/").split("/")
    node: Any = spec
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return schema
        node = node[p]

    if isinstance(node, dict):
        # Recursively resolve nested refs
        if "$ref" in node:
            return _resolve_ref(node, spec, visited)
        return node
    return schema


def _merge_allof_schemas(schemas: list[dict[str, Any]], spec: OpenAPISpec) -> dict[str, Any]:
    """Merge allOf schemas into a single schema."""
    result: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    for schema in schemas:
        resolved = _resolve_ref(schema, spec) if "$ref" in schema else schema

        # Merge properties
        props = resolved.get("properties") or {}
        result["properties"].update(props)

        # Merge required
        required = resolved.get("required") or []
        result["required"].extend(required)

        # Take first non-None type
        if "type" in resolved and result.get("type") == "object":
            result["type"] = resolved["type"]

        # Merge description
        if "description" in resolved and "description" not in result:
            result["description"] = resolved["description"]

    # Dedupe required
    result["required"] = list(set(result["required"]))

    return result


def _py_type_from_schema(
    schema: dict[str, Any],
    spec: OpenAPISpec | None = None,
    visited: set | None = None,
) -> Any:
    """Convert OpenAPI schema to Python type with full schema composition support."""
    if visited is None:
        visited = set()

    if spec is not None and isinstance(schema, dict) and "$ref" in schema:
        ref = schema.get("$ref", "")
        if ref in visited:
            # Circular reference - return Any to avoid infinite recursion
            return Any
        visited = visited | {ref}
        schema = _resolve_ref(schema, spec)

    # Handle schema composition (allOf, oneOf, anyOf)
    if "allOf" in schema:
        merged = _merge_allof_schemas(schema["allOf"], spec or {})
        return _py_type_from_schema(merged, spec, visited)

    if "oneOf" in schema or "anyOf" in schema:
        # For oneOf/anyOf, we use Union of all possible types
        variants = schema.get("oneOf") or schema.get("anyOf") or []
        if len(variants) == 1:
            return _py_type_from_schema(variants[0], spec, visited)
        if len(variants) == 0:
            return Any
        # Build Union type using | operator at runtime
        import functools

        types = [_py_type_from_schema(v, spec, visited) for v in variants]
        return functools.reduce(lambda a, b: a | b, types)

    t = (schema or {}).get("type")
    fmt = (schema or {}).get("format")

    # Handle enum
    enum_values = (schema or {}).get("enum")
    if enum_values:
        from typing import Literal

        # Create Literal type for enum
        try:
            return Literal[tuple(enum_values)]
        except Exception:
            pass  # Fall through to regular type

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
        item_type = _py_type_from_schema(items, spec, visited)
        return list[item_type]  # type: ignore[valid-type]
    if t == "object" or ("properties" in (schema or {})):
        from pydantic import BaseModel, ConfigDict, create_model
        from pydantic import Field as PydanticField

        props = (schema or {}).get("properties") or {}
        reqs = set((schema or {}).get("required") or [])
        fields: dict[str, Any] = {}

        for k, v in props.items():
            prop_schema = v or {}
            typ = _py_type_from_schema(prop_schema, spec, visited)

            # Preserve description and other metadata
            description = prop_schema.get("description")
            default_val = prop_schema.get("default")

            if k in reqs:
                if description:
                    fields[k] = (typ, PydanticField(..., description=description))
                else:
                    fields[k] = (typ, ...)
            else:
                if default_val is not None:
                    if description:
                        fields[k] = (
                            typ,
                            PydanticField(default=default_val, description=description),
                        )
                    else:
                        fields[k] = (typ, default_val)
                else:
                    if description:
                        fields[k] = (
                            typ | None,
                            PydanticField(default=None, description=description),
                        )
                    else:
                        fields[k] = (typ | None, None)

        # Generate unique model name to avoid conflicts
        model_name = schema.get("title") or "AnonModel"
        model_name = model_name.replace(" ", "_").replace("-", "_")

        Model = create_model(
            model_name,
            __base__=BaseModel,
            __config__=ConfigDict(populate_by_name=True, protected_namespaces=()),
            **fields,
        )
        return Model

    return Any


# ---------------------- Input / Output models ----------------------


def _build_input_model(
    op_ctx: OperationContext, path_item: dict, op: dict, spec: OpenAPISpec
) -> type[BaseModel]:
    fields: dict[str, Any] = {}

    def _extract_param_type(param: dict[str, Any]) -> Any:
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
        req = op.get("requestBody") or {}
        content = req.get("content") or {}
        body_schema = (
            ((content.get(op_ctx.body_content_type) or {}).get("schema"))
            or ((content.get("application/json") or {}).get("schema"))
            or {}
        )
        body_typ = _py_type_from_schema(body_schema, spec) if body_schema else Any
        fields["body"] = (body_typ, ... if op_ctx.body_required else None)

        if op_ctx.body_content_type == "multipart/form-data":
            fields["files"] = (
                dict[str, Any] | None,
                Field(default=None, alias="_files"),
            )

    BasicAuthList = conlist(str, min_length=2, max_length=2)
    fields["headers"] = (
        dict[str, str] | None,
        Field(default=None, alias="_headers"),
    )
    fields["api_key"] = (str | None, Field(default=None, alias="_api_key"))
    fields["basic_auth"] = (
        str | BasicAuthList | None,
        Field(default=None, alias="_basic_auth"),
    )
    fields["base_url"] = (str | None, Field(default=None, alias="_base_url"))

    Model = create_model(
        "Input_" + op_ctx.name,
        __base__=BaseModel,
        __config__=ConfigDict(populate_by_name=True, protected_namespaces=()),
        **fields,
    )
    return Model


def _pick_response_schema(op: dict, spec: OpenAPISpec) -> tuple[dict | None, str | None]:
    responses = op.get("responses") or {}
    for status, resp in sorted(responses.items(), key=lambda kv: kv[0]):
        try:
            code = int(status)
        except Exception:
            continue
        if 200 <= code < 300 and isinstance(resp, dict):
            content = resp.get("content") or {}
            if "application/json" in content:
                schema = (content["application/json"].get("schema")) or {}
                return (_resolve_ref(schema, spec), "application/json")
    for _status, resp in responses.items():
        if not isinstance(resp, dict):
            continue
        content = resp.get("content") or {}
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

    fields: dict[str, Any] = {
        "status": (int, ...),
        "headers": (dict[str, str], ...),
        "url": (str, ...),
        "method": (str, ...),
    }

    # use internal names with alias="json"/"text"
    if resp_schema and (resp_ct == "application/json"):
        payload_type = _py_type_from_schema(resp_schema, spec)
        fields["payload_json"] = (
            payload_type | None,
            Field(default=None, alias="json"),
        )
        fields["payload_text"] = (str | None, Field(default=None, alias="text"))
    else:
        fields["payload_json"] = (Any | None, Field(default=None, alias="json"))
        fields["payload_text"] = (str | None, Field(default=None, alias="text"))

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
    auth_config: AuthConfig | None = None,
    # Performance options
    rate_limiter: Any | None = None,  # RateLimiter
    cache: Any | None = None,  # ResponseCache
    deduplicator: Any | None = None,  # RequestDeduplicator
    rate_limit_retry: bool = True,
    rate_limit_max_retries: int = 3,
    auto_paginate: bool = False,
    max_pages: int = 10,
) -> None:
    warnings: list[str] = []
    InputModel = _build_input_model(op_ctx, path_item={}, op=op, spec=spec)
    OutputModel = _build_output_model(op_ctx, op, spec)
    security = SecurityResolver.from_spec(spec, op)

    media_types = list(((op.get("requestBody") or {}).get("content") or {}).keys())
    if op_ctx.wants_body and media_types and op_ctx.body_content_type not in media_types:
        warnings.append(
            f"Chosen content-type {op_ctx.body_content_type!r} not present in requestBody keys={media_types!r}"
        )
    if len(media_types) > 1:
        preferred = (
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
        )
        if not any(mt in preferred for mt in media_types):
            warnings.append(
                f"Multiple body media types; defaulting to {op_ctx.body_content_type!r}"
            )

    for p in op.get("parameters") or []:
        if p.get("deprecated"):
            warnings.append(f"Parameter '{p.get('name')}' is deprecated")
        style = p.get("style")
        explode = p.get("explode")
        if style not in (
            None,
            "form",
            "simple",
            "matrix",
            "label",
            "spaceDelimited",
            "pipeDelimited",
            "deepObject",
        ):
            warnings.append(f"Unrecognized style={style!r} for param '{p.get('name')}'")
        if explode not in (None, True, False):
            warnings.append(f"Unrecognized explode={explode!r} for param '{p.get('name')}'")

    def _has_var(url: str) -> bool:
        return "{" in url and "}" in url

    if base_url and _has_var(base_url):
        warnings.append(f"Base URL contains server variables; not expanded: {base_url!r}")

    if op_ctx.wants_body and op_ctx.body_content_type not in (
        None,
        "application/json",
        "application/x-www-form-urlencoded",
        "multipart/form-data",
        "text/plain",
        "application/octet-stream",
    ):
        warnings.append(
            f"Unsupported content-type mapped as raw data: {op_ctx.body_content_type!r}"
        )

    if not base_url:
        warnings.append(
            "No effective base URL: spec.servers empty and no override; tool will require _base_url."
        )

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
            "path": len(op_ctx.path_params),
            "query": len(op_ctx.query_params),
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

    async def tool(args: InputModel | None = None) -> OutputModel:  # type: ignore[valid-type]
        # Allow completely empty calls (e.g., ping) by treating None as {}
        payload: dict[str, Any] = (
            args.model_dump(by_alias=True, exclude_none=True) if args is not None else {}
        )

        url_base = (payload.pop("_base_url", None) or base_url).rstrip("/")
        api_key = payload.pop("_api_key", None)
        basic_auth = payload.pop("_basic_auth", None)
        headers_in = payload.pop("_headers", None) or {}

        if not url_base:
            # Keep structure consistent even on error: still return OutputModel
            out_err: dict[str, Any] = {
                "status": 0,
                "headers": {},
                "url": "",
                "method": op_ctx.method,
                "json": None,
                "text": "Error: no base URL provided (servers missing and _base_url not set).",
            }
            return OutputModel.model_validate(out_err)

        errors: list[str] = []

        url_path = op_ctx.path
        for p in op_ctx.path_params:
            pname = p.get("name")
            if p.get("required") and pname not in payload:
                errors.append(f"Missing required path param: {pname}")
                continue
            if pname in payload:
                url_path = url_path.replace("{" + pname + "}", str(payload.pop(pname)))

        query: dict[str, Any] = {}
        headers: dict[str, str] = {}
        cookies: dict[str, str] = {}

        # Handle query params with style/explode for arrays
        for p in op_ctx.query_params:
            pname = p.get("name")
            if pname in payload:
                value = payload.pop(pname)
                style = p.get("style")
                explode = p.get("explode")
                # Use serialize_query_param for proper array handling
                # pname is validated above (checked against payload keys)
                serialized = serialize_query_param(str(pname), value, style=style, explode=explode)
                query.update(serialized)
            elif p.get("required"):
                errors.append(f"Missing required query param: {pname}")

        for p in op_ctx.header_params:
            pname = p.get("name")
            if pname in payload:
                if pname is not None:
                    headers[pname] = str(payload.pop(pname))
            elif p.get("required"):
                errors.append(f"Missing required header: {pname}")

        for p in op_ctx.cookie_params:
            pname = p.get("name")
            if pname in payload:
                if pname is not None:
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
                    json_body = body_arg
                    headers.setdefault("Content-Type", "application/json")
                elif ct == "application/x-www-form-urlencoded":
                    data = body_arg
                    headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
                elif ct == "multipart/form-data":
                    files = payload.pop("_files", None)
                    if files is None:
                        if isinstance(body_arg, dict):
                            files = {k: (k, v) for k, v in body_arg.items()}
                        else:
                            files = {"file": ("file", body_arg)}
                elif ct in ("text/plain", "application/octet-stream"):
                    data = body_arg
                    headers.setdefault("Content-Type", ct)
                else:
                    data = body_arg
                    if ct:
                        headers.setdefault("Content-Type", ct)

        if errors:
            out_validation: dict[str, Any] = {
                "status": 0,
                "headers": {},
                "url": f"{url_base}{url_path}",
                "method": op_ctx.method,
                "json": None,
                "text": "Validation errors:\n" + "\n".join(f" - {e}" for e in errors),
            }
            return OutputModel.model_validate(out_validation)

        # Apply auth from OpenAPI security schemes
        security.apply(
            headers,
            query,
            {"_api_key": api_key, "_basic_auth": basic_auth, "_headers": headers_in},
        )

        # Apply auth from auth_config (overrides/supplements security schemes)
        await _apply_auth_config(auth_config, headers, query)

        for k, v in list(payload.items()):
            if not str(k).startswith("_"):
                query[k] = v
            payload.pop(k, None)

        full_url = f"{url_base}{url_path}"

        # Check cache first (for safe methods)
        cache_key = None
        if cache and cache.should_cache(op_ctx.method):
            cache_key = cache.make_key(op_ctx.method, full_url, query)
            cached = cache.get(cache_key)
            if cached is not None:
                return OutputModel.model_validate(cached)

        # Apply rate limiting
        if rate_limiter:
            await rate_limiter.acquire()

        # Define the actual request function
        async def _do_request():
            import asyncio

            retries_left = rate_limit_max_retries if rate_limit_retry else 0
            last_resp = None

            while True:
                try:
                    resp = await client.request(
                        op_ctx.method,
                        full_url,
                        params=query or None,
                        headers=headers or None,
                        cookies=cookies or None,
                        json=json_body,
                        data=data,
                        files=files,
                    )
                    last_resp = resp

                    # Handle rate limiting (429)
                    if resp.status_code == 429 and retries_left > 0:
                        retries_left -= 1
                        # Try to get retry-after header
                        retry_after = resp.headers.get("retry-after", "1")
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            wait_time = 1.0
                        await asyncio.sleep(min(wait_time, 30))  # Cap at 30s
                        continue

                    return resp

                except httpx.TimeoutException as e:
                    return {"error": f"Request timeout: {e}"}
                except httpx.ConnectError as e:
                    return {"error": f"Connection error: {e}"}
                except httpx.HTTPError as e:
                    return {"error": f"HTTP error: {e}"}

            return last_resp

        # Use deduplicator if enabled
        if deduplicator:
            dedup_key = cache_key or f"{op_ctx.method}:{full_url}:{query}"
            resp = await deduplicator.execute(dedup_key, _do_request)
        else:
            resp = await _do_request()

        # Handle error dict from _do_request
        if isinstance(resp, dict) and "error" in resp:
            return OutputModel.model_validate(
                {
                    "status": 0,
                    "headers": {},
                    "url": full_url,
                    "method": op_ctx.method,
                    "json": None,
                    "text": resp["error"],
                }
            )

        content_type = resp.headers.get("content-type", "")
        out: dict[str, Any] = {
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "url": str(resp.request.url),
            "method": resp.request.method,
            "json": None,
            "text": None,
        }

        # Parse response based on content type
        if "application/json" in content_type:
            try:
                json_data = resp.json()
                out["json"] = json_data

                # Auto-pagination for JSON responses
                if auto_paginate and op_ctx.method.upper() == "GET":
                    all_items = []
                    current_data = json_data
                    pages_fetched = 1

                    # Detect if response is paginated (has items/data array + next link)
                    items_key = None
                    for key in ("items", "data", "results", "records"):
                        if isinstance(current_data, dict) and key in current_data:
                            if isinstance(current_data[key], list):
                                items_key = key
                                break

                    if items_key:
                        all_items.extend(current_data[items_key])

                        while pages_fetched < max_pages:
                            # Look for next page link
                            next_url = None
                            if isinstance(current_data, dict):
                                # Common pagination patterns
                                next_url = (
                                    current_data.get("next")
                                    or current_data.get("next_page")
                                    or current_data.get("nextPage")
                                    or (current_data.get("links", {}) or {}).get("next")
                                    or (current_data.get("_links", {}) or {})
                                    .get("next", {})
                                    .get("href")
                                )

                            if not next_url:
                                break

                            # Fetch next page
                            try:
                                if rate_limiter:
                                    await rate_limiter.acquire()
                                next_resp = await client.get(next_url, headers=headers)
                                if next_resp.status_code != 200:
                                    break
                                current_data = next_resp.json()
                                if items_key in current_data:
                                    all_items.extend(current_data[items_key])
                                pages_fetched += 1
                            except Exception:
                                break

                        # Replace items with all collected items
                        if pages_fetched > 1:
                            from typing import cast

                            json_out = cast("dict[str, Any]", out["json"])
                            json_out[items_key] = all_items
                            json_out["_paginated"] = True
                            json_out["_pages_fetched"] = pages_fetched

            except Exception:
                out["text"] = resp.text
        elif (
            "text/" in content_type
            or "application/xml" in content_type
            or "text/xml" in content_type
        ):
            out["text"] = resp.text
        elif "application/octet-stream" in content_type:
            # Binary data - encode as base64
            import base64 as b64

            out["text"] = b64.b64encode(resp.content).decode("ascii")
        else:
            # Try JSON first, fall back to text
            try:
                out["json"] = resp.json()
            except Exception:
                out["text"] = resp.text

        # Store in cache if applicable
        if cache_key and cache and resp.status_code < 400:
            cache.set(cache_key, out)

        return OutputModel.model_validate(out)

    # expose schemas to MCP (input/outputSchema) via annotations
    tool.__annotations__ = {"args": InputModel | None, "return": OutputModel}
    mcp.add_tool(name=op_ctx.name, description=op_ctx.full_description(), fn=tool)

    op_rep.warnings.extend(warnings)
    report.ops.append(op_rep)


# ---------------------- Builder entrypoint ----------------------


def _mcp_from_openapi(
    spec: dict | str | Path,
    *,
    client: httpx.AsyncClient | None = None,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
    base_url: str | None = None,
    strict_names: bool = False,
    report_log: bool | None = None,
    # NEW: Options for filtering and customization
    options: OpenAPIOptions | None = None,
    # Convenience shortcuts (applied to options)
    tool_prefix: str | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    include_methods: list[str] | None = None,
    exclude_methods: list[str] | None = None,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    include_operations: list[str] | None = None,
    exclude_operations: list[str] | None = None,
    tool_name_fn: Callable[[str, str, dict], str] | None = None,
    tool_description_fn: Callable[[dict], str] | None = None,
    auth: Any = None,
    endpoint_auth: dict[str, Any] | None = None,
) -> tuple[FastMCP, Callable[[], Awaitable[None]] | None, BuildReport]:
    """
    Build a FastMCP from OpenAPI and return (mcp, async_cleanup, report).

    Args:
        spec: OpenAPI spec as dict, file path, or URL
        client: Existing httpx.AsyncClient to use
        client_factory: Factory function to create client
        base_url: Override base URL for all requests
        strict_names: Raise on duplicate tool names
        report_log: Log build report (default: MCP_OPENAPI_DEBUG=1)
        options: Full OpenAPIOptions object for advanced configuration

        # Convenience shortcuts (merged into options):
        tool_prefix: Prefix all tool names (e.g., "github" -> "github_get_user")
        include_paths: Only include paths matching these glob patterns
        exclude_paths: Exclude paths matching these glob patterns
        include_methods: Only include these HTTP methods
        exclude_methods: Exclude these HTTP methods
        include_tags: Only include operations with these tags
        exclude_tags: Exclude operations with these tags
        include_operations: Only include these operationIds
        exclude_operations: Exclude these operationIds
        tool_name_fn: Custom function(method, path, operation) -> name
        tool_description_fn: Custom function(operation) -> description
        auth: Authentication config (dict, tuple, str, callable, or AuthConfig)
        endpoint_auth: Per-endpoint auth overrides (pattern -> auth)

    Returns:
        Tuple of (FastMCP, async_cleanup_fn, BuildReport)

    Example:
        # Zero-config (all endpoints)
        mcp, cleanup, report = _mcp_from_openapi("https://api.example.com/openapi.json")

        # With filtering
        mcp, cleanup, report = _mcp_from_openapi(
            "https://api.github.com/openapi.json",
            tool_prefix="github",
            include_paths=["/repos/*", "/users/*"],
            exclude_methods=["DELETE"],
            auth={"Authorization": "Bearer ghp_xxx"},
        )
    """
    # Merge shortcut options into OpenAPIOptions
    if options is None:
        options = OpenAPIOptions()

    # Apply shortcuts (only if not already set in options)
    if tool_prefix and not options.tool_prefix:
        options.tool_prefix = tool_prefix
    if include_paths and not options.include_paths:
        options.include_paths = include_paths
    if exclude_paths and not options.exclude_paths:
        options.exclude_paths = exclude_paths
    if include_methods and not options.include_methods:
        options.include_methods = include_methods
    if exclude_methods and not options.exclude_methods:
        options.exclude_methods = exclude_methods
    if include_tags and not options.include_tags:
        options.include_tags = include_tags
    if exclude_tags and not options.exclude_tags:
        options.exclude_tags = exclude_tags
    if include_operations and not options.include_operations:
        options.include_operations = include_operations
    if exclude_operations and not options.exclude_operations:
        options.exclude_operations = exclude_operations
    if tool_name_fn and not options.tool_name_fn:
        options.tool_name_fn = tool_name_fn
    if tool_description_fn and not options.tool_description_fn:
        options.tool_description_fn = tool_description_fn
    if auth and not options.auth:
        options.auth = AuthConfig.from_value(auth)
    if endpoint_auth and not options.endpoint_auth:
        options.endpoint_auth = endpoint_auth

    if not isinstance(spec, dict):
        spec = load_openapi(spec)

    title = (spec.get("info", {}) or {}).get("title") or "OpenAPI MCP"
    report = BuildReport(title=title)
    if report_log is None:
        report_log = os.getenv("MCP_OPENAPI_DEBUG", "0") == "1"

    own_client = False
    if client is None:
        client = client_factory() if client_factory else httpx.AsyncClient(timeout=30.0)
        own_client = True

    mcp = FastMCP(title)
    seen_tool_names: set[str] = set()
    paths = spec.get("paths") or {}
    total_ops = 0
    filtered_ops = 0

    # Create performance objects based on options
    rate_limiter = None
    cache = None
    deduplicator = None

    if options.rate_limit:
        from .runtime import RateLimiter

        rate_limiter = RateLimiter(rate=options.rate_limit)

    if options.cache_ttl:
        from .runtime import ResponseCache

        cache = ResponseCache(
            ttl=options.cache_ttl,
            methods=options.cache_methods or ["GET"],
        )

    if options.dedupe_requests:
        from .runtime import RequestDeduplicator

        deduplicator = RequestDeduplicator()

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            report.warnings.append(f"Path item is not an object: {path}")
            continue
        for method, op in path_item.items():
            if method.lower() not in ALLOWED_METHODS or not isinstance(op, dict):
                continue
            total_ops += 1

            # Apply filtering from options
            if not options.should_include_operation(path, method, op):
                filtered_ops += 1
                continue

            op_ctx = _make_operation_context(path, method, path_item, op)

            # Apply custom tool naming from options
            base_tool = options.get_tool_name(op_ctx.name, method, path, op)
            op_ctx.name = base_tool

            # Apply custom description from options
            op_ctx.description = options.get_tool_description(op_ctx.description, op)

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

            # Get auth config for this path
            path_auth = options.get_auth_for_path(path)

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
                    auth_config=path_auth,
                    # Performance options
                    rate_limiter=rate_limiter,
                    cache=cache,
                    deduplicator=deduplicator,
                    rate_limit_retry=options.rate_limit_retry,
                    rate_limit_max_retries=options.rate_limit_max_retries,
                    auto_paginate=options.auto_paginate,
                    max_pages=options.max_pages,
                )
                report.registered_tools += 1
            except Exception as e:
                report.skipped_ops += 1
                warn = (
                    f"Failed to register tool for {method.upper()} {path}: {type(e).__name__}: {e}"
                )
                report.warnings.append(warn)
                log.debug(warn, exc_info=True)

    report.total_ops = total_ops
    report.filtered_ops = filtered_ops

    async_cleanup: Callable[[], Awaitable[None]] | None = None
    if own_client:

        async def _cleanup() -> None:
            await client.aclose()

        async_cleanup = _cleanup

    _maybe_log_report(report, report_log)

    return mcp, async_cleanup, report
