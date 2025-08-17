from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Union, Optional
import httpx

from mcp.server.fastmcp import FastMCP

from .utils import (
    load_spec,
    op_tool_name,
    pick_base_url,
    collect_params,
    has_request_body,
    extract_body_content_type,
)
from .models import OperationContext
from .constants import ALLOWED_METHODS

__all__ = ["build_mcp_from_openapi"]


def _make_operation_context(path: str, method: str, op: Dict[str, Any]) -> OperationContext:
    params = collect_params(op)
    wants_body = has_request_body(op)
    body_ct = extract_body_content_type(op) if wants_body else None
    return OperationContext(
        name=op_tool_name(path, method, op.get("operationId")),
        description=op.get("summary") or op.get("description") or f"{method.upper()} {path}",
        method=method.upper(),
        path=path,
        path_params=params["path"],
        query_params=params["query"],
        header_params=params["header"],
        wants_body=wants_body,
        body_content_type=body_ct,
        body_required=bool(op.get("requestBody", {}).get("required")) if wants_body else False,
    )


def _register_operation_tool(mcp: FastMCP, *, root_base: str, op_ctx: OperationContext) -> None:
    """Register one OpenAPI operation as an MCP tool using precomputed metadata."""

    @mcp.tool(name=op_ctx.name, description=op_ctx.full_description())
    async def tool(**kwargs) -> str:  # type: ignore[override]
        url_base = (kwargs.pop("_base_url", None) or root_base).rstrip("/")
        if not url_base:
            return "Error: no base URL provided (spec.servers[] missing and _base_url not set)."

        errors: list[str] = []

        # Path param substitution
        url_path = op_ctx.path
        for p in op_ctx.path_params:
            pname = p.get("name")
            if p.get("required") and pname not in kwargs:
                errors.append(f"Missing required path param: {pname}")
                continue
            if pname in kwargs:
                url_path = url_path.replace("{" + pname + "}", str(kwargs.pop(pname)))

        # Query params extraction
        query: Dict[str, Any] = {}
        for p in op_ctx.query_params:
            pname = p.get("name")
            if pname in kwargs:
                query[pname] = kwargs.pop(pname)
            elif p.get("required"):
                errors.append(f"Missing required query param: {pname}")

        # Header params extraction
        headers: Dict[str, str] = {}
        for p in op_ctx.header_params:
            pname = p.get("name")
            if pname in kwargs:
                headers[pname] = str(kwargs.pop(pname))
            elif p.get("required"):
                errors.append(f"Missing required header: {pname}")

        data = None
        json_body = None
        if op_ctx.wants_body:
            body_arg = kwargs.pop("body", None)
            if body_arg is None and op_ctx.body_required:
                errors.append("Missing required request body: pass as 'body=<json/dict/str>'.")
            elif body_arg is not None:
                if op_ctx.body_content_type == "application/json":
                    json_body = body_arg
                    headers.setdefault("Content-Type", "application/json")
                elif op_ctx.body_content_type == "application/x-www-form-urlencoded":
                    data = body_arg
                    headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
                else:
                    data = body_arg
                    if op_ctx.body_content_type:
                        headers.setdefault("Content-Type", op_ctx.body_content_type)

        if errors:
            return "Validation errors:\n" + "\n".join(f" - {e}" for e in errors)

        if "_api_key" in kwargs:
            headers.setdefault("Authorization", f"Bearer {kwargs.pop('_api_key')}")
        if "_headers" in kwargs and isinstance(kwargs["_headers"], dict):
            headers.update(kwargs.pop("_headers"))

        for k, v in list(kwargs.items()):
            query[k] = v

        full_url = f"{url_base}{url_path}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                op_ctx.method,
                full_url,
                params=query,
                headers=headers,
                json=json_body,
                data=data,
            )
        ct_hdr = resp.headers.get("content-type", "")
        if "application/json" in ct_hdr:
            try:
                return json.dumps(resp.json(), indent=2)
            except Exception:
                return resp.text
        return resp.text


def build_mcp_from_openapi(spec: Union[dict, str, Path], base_url: str | None = None) -> FastMCP:
    """Build a FastMCP instance from an OpenAPI spec (dict, file path, or raw string).

    Public contract:
      spec: dict OR path to JSON/YAML OR raw JSON/YAML string
      base_url: optional override for spec.servers[0].url
    Returns:
      FastMCP with one tool per supported operation.
    """
    if not isinstance(spec, dict):
        spec = load_spec(spec)
    mcp = FastMCP(spec.get("info", {}).get("title") or "OpenAPI MCP")
    root_base = pick_base_url(spec, base_url)

    paths = spec.get("paths") or {}
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):  # defensive
            continue
        for method, op in path_item.items():
            if method.lower() not in ALLOWED_METHODS:
                continue
            if not isinstance(op, dict):
                continue
            op_ctx = _make_operation_context(path, method, op)
            _register_operation_tool(
                mcp,
                root_base=root_base,
                op_ctx=op_ctx,
            )
    return mcp