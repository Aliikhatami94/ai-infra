from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Callable, Awaitable, List
from pathlib import Path

import httpx
from jsonschema import Draft7Validator, ValidationError  # pip install jsonschema
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server  # only for completeness if ever used directly

from ai_infra.mcp.client.core import CoreMCPClient
from ai_infra.mcp.client.models import McpServerConfig


def _coerce_json_schema(schema: Any) -> Optional[Dict[str, Any]]:
    if schema is None:
        return None
    if hasattr(schema, "model_json_schema"):
        try:
            return schema.model_json_schema()  # pydantic v2
        except Exception:
            pass
    if hasattr(schema, "schema"):
        try:
            return schema.schema()  # pydantic v1
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
        # Don't block server if schema is a bit off; just skip validation
        return None


async def _proxy_call_tool(
        client_cfg: McpServerConfig,
        target_server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    One-off proxy call: spin a tiny client for a single backend and forward call.
    We rely on initialize() to resolve the server_name (which you already do in discover()).
    """
    client = CoreMCPClient([client_cfg.model_dump()])
    await client.discover()
    # resolve discovered name (prefer exact, else only server)
    name = target_server_name
    if name not in client.server_names():
        # pick the only server if there is exactly one
        discovered = client.server_names()
        if len(discovered) == 1:
            name = discovered[0]
        else:
            raise RuntimeError(
                f"Backend server '{target_server_name}' not found. "
                f"Available: {', '.join(discovered) or '(none)'}"
            )
    return await client.call_tool(name, tool_name, arguments)


def _resolve_server_name(doc: Dict[str, Any]) -> str:
    # Prefer info.title, else server.name, else "server"
    return (
            (doc.get("info") or {}).get("title")
            or (doc.get("server") or {}).get("name")
            or "server"
    )


def _extract_doc_lists(doc: Dict[str, Any]) -> tuple[list, list, list, list]:
    tools = doc.get("tools") or []
    prompts = doc.get("prompts") or []
    resources = doc.get("resources") or []
    templates = doc.get("resource_templates") or []
    return tools, prompts, resources, templates


def _validate_via(schema_validator: Optional[Draft7Validator], data: Any, which: str) -> None:
    if not schema_validator:
        return
    try:
        schema_validator.validate(data)
    except ValidationError as e:
        raise ValueError(f"{which} validation failed: {e.message}") from e


def _normalize_tool_io_schemas(tool_def: Dict[str, Any]):
    in_schema = _coerce_json_schema(tool_def.get("args_schema"))
    out_schema = _coerce_json_schema(tool_def.get("output_schema"))
    return in_schema, out_schema


def _make_tool_handler(
        tool_def: Dict[str, Any],
        *,
        backend_cfg: Optional[McpServerConfig],
        handlers: Optional[Dict[str, Callable[..., Awaitable[Any]]]],
        server_name: str,
):
    """
    Returns an async function implementing the tool:
      - proxy to backend if available,
      - else call user handler if provided,
      - else raise NotImplemented
    with best-effort JSON-schema validation in/out.
    """
    name = tool_def.get("name")
    if not name:
        raise ValueError("OpenMCP tool missing 'name'.")

    in_schema, out_schema = _normalize_tool_io_schemas(tool_def)
    in_validator = _mk_validator(in_schema)
    out_validator = _mk_validator(out_schema)

    async def _handler(**kwargs):
        # MCPS tool schema in this library is typically { "args": <schema> } or direct fields;
        # We accept either – prefer kwargs verbatim; if "args" present, unwrap it.
        args = kwargs.get("args", kwargs)
        _validate_via(in_validator, {"args": args} if in_schema and "properties" in in_schema and "args" in (in_schema.get("properties") or {}) else args, "input")

        # Mode 1: proxy
        if backend_cfg is not None:
            result = await _proxy_call_tool(backend_cfg, server_name, name, args)
            # Normalize result to dict-like to validate
            to_validate = result if isinstance(result, dict) else {"result": result}
            _validate_via(out_validator, to_validate, "output")
            return to_validate

        # Mode 2: local handler
        if handlers and name in handlers:
            out = await handlers[name](**args) if asyncio.iscoroutinefunction(handlers[name]) else handlers[name](**args)
            to_validate = out if isinstance(out, dict) else {"result": out}
            _validate_via(out_validator, to_validate, "output")
            return to_validate

        # Mode 3: stub
        raise NotImplementedError(f"Tool '{name}' has no backend/handler.")

    return _handler, in_schema, out_schema

def _register_tool_with_fallback(mcp, td, handler, in_schema, out_schema):
    """
    Register a tool with FastMCP. If this FastMCP version supports schema kwargs,
    use them; otherwise, fall back to name/description only.
    """
    name = td.get("name")
    desc = (td.get("description") or None)

    # Try modern signature first
    try:
        return mcp.tool(
            name=name,
            description=desc,
            input_schema=in_schema,
            output_schema=out_schema,
        )(handler)
    except TypeError:
        # Older FastMCP: no schema kwargs supported
        return mcp.tool(
            name=name,
            description=desc,
        )(handler)

def _register_prompts_resources(
        mcp: FastMCP,
        *,
        doc: Dict[str, Any],
        backend_cfg: Optional[McpServerConfig],
        server_name: str,
):
    """
    Best-effort to expose prompts/resources/roots. In proxy mode, forward to backend.
    In stub mode they list but read returns NotImplemented.
    """
    # Prompts (list + get)
    prompts = doc.get("prompts") or []
    if prompts:
        @mcp.prompt("list")
        async def _list_prompts():
            # MCPS doc-level listing
            return [{"name": p.get("name"), "description": p.get("description")} for p in prompts]

        @mcp.prompt("get")
        async def _get_prompt(name: str):
            for p in prompts:
                if p.get("name") == name:
                    return p
            raise ValueError(f"Prompt '{name}' not found.")

    # Resources (list + read)
    resources = doc.get("resources") or []
    if resources:
        @mcp.resource("list")
        async def _list_resources():
            return resources

        @mcp.resource("read")
        async def _read_resource(uri: str):
            if backend_cfg is None:
                raise NotImplementedError("No backend to read resources from.")
            # Proxy read via client:
            client = CoreMCPClient([backend_cfg.model_dump()])
            await client.discover()
            names = client.server_names()
            target = server_name if server_name in names else (names[0] if len(names) == 1 else None)
            if not target:
                raise RuntimeError("Backend not available to read resource.")
            async with client.get_client(target) as session:
                res = await session.read_resource(uri)
                # Return raw (FastMCP will serialize content)
                return res

    # Resource templates (list)
    templates = doc.get("resource_templates") or []
    if templates:
        @mcp.resource("templates")
        async def _list_templates():
            return templates
    # Roots (list)
    roots = doc.get("roots") or []
    if roots:
        @mcp.resource("roots")
        async def _list_roots():
            return roots


def _mcp_from_openmcp(
        doc: Dict[str, Any],
        *,
        name: Optional[str] = None,
        backend_config: Optional[Dict[str, Any]] = None,
        handlers: Optional[Dict[str, Callable[..., Awaitable[Any]]]] = None,
) -> FastMCP:
    """
    Build a FastMCP instance from an OpenMCP (MCPS) document.

    Modes:
      - Proxy: provide backend_config (McpServerConfig-compatible dict)
      - Handler: provide handlers={tool_name: async fn}
      - Stub: neither provided -> tools raise NotImplemented when called
    """
    mcp_name = (
            name
            or (doc.get("server") or {}).get("name")
            or (doc.get("info") or {}).get("title")
            or "mcp"
    )

    mcp = FastMCP(name=mcp_name)

    # backend cfg -> pydantic model
    backend_cfg_model: Optional[McpServerConfig] = None
    if backend_config:
        backend_cfg_model = McpServerConfig.model_validate(backend_config)

    server_name = _resolve_server_name(doc)
    tools, _, _, _ = _extract_doc_lists(doc)

    # Register tools
    for td in tools:
        tool_name = td.get("name")
        if not tool_name:
            continue

        handler, in_schema, out_schema = _make_tool_handler(
            td,
            backend_cfg=backend_cfg_model,
            handlers=handlers,
            server_name=server_name,
        )

        _register_tool_with_fallback(mcp, td, handler, in_schema, out_schema)

    # Prompts / resources / roots
    _register_prompts_resources(
        mcp,
        doc=doc,
        backend_cfg=backend_cfg_model,
        server_name=server_name,
    )

    return mcp

def _select_openmcp_doc(
        openmcp: Dict[str, Any] | str | Path,
        *,
        select: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Accepts:
      - a single MCPS doc (dict with 'server'/'tools' keys) -> returns it
      - a bundle mapping {name -> MCPS doc}                 -> selects one entry

    If `openmcp` is a str/Path, it will be loaded (file or http(s) URL).
    """
    # 1) Load if needed
    if not isinstance(openmcp, dict):
        val = str(openmcp)
        if val.startswith("http://") or val.startswith("https://"):
            with httpx.Client(timeout=15.0) as c:
                r = c.get(val)
                r.raise_for_status()
                data = r.json()
        else:
            p = Path(val)
            if not p.exists():
                raise FileNotFoundError(f"OpenMCP file not found: {p}")
            data = json.loads(p.read_text())
    else:
        data = openmcp

    # 2) Single MCPS doc?
    if isinstance(data.get("server"), dict) or isinstance(data.get("tools"), list):
        return data  # looks like a single spec

    # 3) Otherwise expect a bundle {name -> doc}
    if not isinstance(data, dict):
        raise ValueError("OpenMCP input must be a MCPS document dict or a bundle {name: doc}.")

    keys = list(data.keys())
    if select:
        if select not in data:
            raise ValueError(
                f"OpenMCP bundle does not contain '{select}'. "
                f"Available: {', '.join(keys) or '(none)'}"
            )
        return data[select]

    if len(keys) == 1:
        # Only one entry, just return it
        return data[keys[0]]

    # Ambiguous bundle: require selection
    raise ValueError(
        "OpenMCP input appears to be a bundle of multiple servers. "
        f"Specify which one to mount via select=... . Available: {', '.join(keys)}"
    )