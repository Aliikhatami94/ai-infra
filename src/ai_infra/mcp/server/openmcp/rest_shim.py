# ai_infra/mcp/server/openmcp/rest_shim.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Callable, Awaitable, List

from jsonschema import Draft7Validator, ValidationError
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse

from ai_infra.mcp.client.core import CoreMCPClient
from ai_infra.mcp.client.models import McpServerConfig


def _select_openmcp_doc(openmcp: Dict[str, Any] | List[Any], select: Optional[str]) -> Dict[str, Any]:
    """
    - If openmcp is a dict and looks like a bundle {name: doc, ...}, select by key.
    - If it's already a single doc (has 'server' or 'tools'), return as-is.
    """
    if isinstance(openmcp, dict):
        if "server" in openmcp or "tools" in openmcp:
            return openmcp  # single
        if select:
            if select not in openmcp:
                raise ValueError(f"OpenMCP bundle has no entry '{select}'. "
                                 f"Available: {', '.join(openmcp.keys())}")
            return openmcp[select]
        # fallback if exactly one
        if len(openmcp) == 1:
            return next(iter(openmcp.values()))
        raise ValueError("OpenMCP bundle contains multiple servers; pass select='<name>'.")
    raise TypeError("OpenMCP must be a dict (single doc or bundle).")


def _coerce_json_schema(schema: Any) -> Optional[Dict[str, Any]]:
    if schema is None:
        return None
    if hasattr(schema, "model_json_schema"):
        try:
            return schema.model_json_schema()
        except Exception:
            pass
    if hasattr(schema, "schema"):
        try:
            return schema.schema()
        except Exception:
            pass
    if isinstance(schema, str):
        try:
            return json.loads(schema)
        except Exception:
            return {"description": schema}
    if isinstance(schema, dict):
        return schema
    return {"description": str(schema)}


def _mk_validator(schema: Optional[Dict[str, Any]]) -> Optional[Draft7Validator]:
    if not schema or not isinstance(schema, dict):
        return None
    try:
        return Draft7Validator(schema)
    except Exception:
        return None


def _resolve_server_name(doc: Dict[str, Any]) -> str:
    return (
            (doc.get("info") or {}).get("title")
            or (doc.get("server") or {}).get("name")
            or "server"
    )


async def _proxy_call_tool(backend_cfg: McpServerConfig, server_name: str,
                           tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    client = CoreMCPClient([backend_cfg.model_dump()])
    await client.discover()
    names = client.server_names()
    # use declared name if present, else only server available
    target = server_name if server_name in names else (names[0] if len(names) == 1 else None)
    if not target:
        raise RuntimeError(f"Backend server '{server_name}' not found. Available: {', '.join(names) or '(none)'}")
    return await client.call_tool(target, tool, args)

def fastapi_from_openmcp(
        openmcp_doc: Dict[str, Any],
        *,
        backend_config: Optional[Dict[str, Any]] = None,
        handlers: Optional[Dict[str, Callable[..., Awaitable[Any]]]] = None,
) -> FastAPI:
    """
    Create a FastAPI app that exposes OpenMCP tools as REST endpoints:
      POST /tools/{tool_name}  -> body validated with args_schema; returns output_schema
    Also exposes:
      GET /prompts
      GET /resources
      GET /resource-templates
      GET /roots
    """
    # Title/desc/version for OpenAPI
    info = openmcp_doc.get("info") or {}
    title = info.get("title") or (openmcp_doc.get("server") or {}).get("name") or "MCP"
    description = info.get("description") or "OpenMCP REST shim"
    version = info.get("version") or "0.1.0"

    app = FastAPI(title=title, description=description, version=version)

    backend_cfg_model: Optional[McpServerConfig] = None
    if backend_config:
        backend_cfg_model = McpServerConfig.model_validate(backend_config)

    server_name = _resolve_server_name(openmcp_doc)

    # ---- Tools
    for td in openmcp_doc.get("tools", []) or []:
        tool_name = td.get("name")
        if not tool_name:
            continue

        in_schema = _coerce_json_schema(td.get("args_schema"))
        out_schema = _coerce_json_schema(td.get("output_schema"))
        in_validator = _mk_validator(in_schema)
        out_validator = _mk_validator(out_schema)

        async def route(payload: Dict[str, Any] = Body(...), __tool=tool_name,
                        __in=in_validator, __out=out_validator):
            args = payload.get("args", payload)
            if __in:
                try:
                    # Validate either direct args or {"args": ...} depending on schema shape
                    maybe = {"args": args} if "properties" in (__in.schema or {}) and "args" in (__in.schema["properties"]) else args
                    __in.validate(maybe)
                except ValidationError as e:
                    raise HTTPException(status_code=422, detail=f"input validation failed: {e.message}")

            # Proxy, handler, or stub
            if backend_cfg_model is not None:
                result = await _proxy_call_tool(backend_cfg_model, server_name, __tool, args)
            elif handlers and __tool in handlers:
                h = handlers[__tool]
                result = await h(**args) if hasattr(h, "__call__") and h.__code__.co_flags & 0x80 else h(**args)  # await if coroutine
            else:
                raise HTTPException(status_code=501, detail=f"Tool '{__tool}' has no backend/handler.")

            # Normalize to dict for validation
            out_obj = result if isinstance(result, dict) else {"result": result}
            if __out:
                try:
                    __out.validate(out_obj)
                except ValidationError as e:
                    raise HTTPException(status_code=500, detail=f"output validation failed: {e.message}")
            return JSONResponse(out_obj)

        app.post(f"/tools/{tool_name}", tags=["tools"], summary=td.get("description") or tool_name)(route)

    # ---- Optional endpoints for prompts/resources/templates/roots
    prompts = openmcp_doc.get("prompts") or []
    resources = openmcp_doc.get("resources") or []
    templates = openmcp_doc.get("resource_templates") or []
    roots = openmcp_doc.get("roots") or []

    if prompts:
        @app.get("/prompts", tags=["prompts"])
        async def list_prompts():
            return prompts

    if resources:
        @app.get("/resources", tags=["resources"])
        async def list_resources():
            return resources

    if templates:
        @app.get("/resource-templates", tags=["resources"])
        async def list_templates():
            return templates

    if roots:
        @app.get("/roots", tags=["resources"])
        async def list_roots():
            return roots

    return app