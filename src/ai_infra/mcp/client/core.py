from __future__ import annotations

import difflib
import traceback
import json
from typing import Dict, Any, List, AsyncContextManager, Optional
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    StreamableHttpConnection, StdioConnection, SSEConnection
)
from langchain_mcp_adapters.tools import load_mcp_tools

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client
from pydantic import BaseModel

from ai_infra.mcp.client.models import McpServerConfig


class CoreMCPClient:
    """
    Config = list[McpServerConfig-like dicts]. No names required.
    We discover server names from MCP initialize() and map them.
    """

    def __init__(self, config: List[dict] | List[McpServerConfig]):
        if not isinstance(config, list):
            raise TypeError("Config must be a list of server configs")
        self._configs: List[McpServerConfig] = [
            c if isinstance(c, McpServerConfig) else McpServerConfig.model_validate(c)
            for c in config
        ]
        self._by_name: Dict[str, McpServerConfig] = {}
        self._discovered: bool = False
        self._errors: List[Dict[str, Any]] = []   # <-- NEW

    # ---------- helpers for doc generation (NEW) ----------

    @staticmethod
    def _attr_or(dobj: Any, attr: str, default=None):
        """Get attr from object; if dict use key, else attribute, else default."""
        if hasattr(dobj, attr):
            return getattr(dobj, attr)
        if isinstance(dobj, dict):
            return dobj.get(attr, default)
        return default

    @staticmethod
    def _safe_schema(maybe_model: Any) -> Dict[str, Any] | None:
        if maybe_model is None:
            return None
        try:
            if isinstance(maybe_model, type) and issubclass(maybe_model, BaseModel):
                return maybe_model.model_json_schema()
            if hasattr(maybe_model, "model_json_schema"):
                return maybe_model.model_json_schema()
            if isinstance(maybe_model, dict):
                return maybe_model
            return None
        except Exception:
            return None

    @staticmethod
    def _safe_text(desc: Any) -> Optional[str]:
        return desc if isinstance(desc, str) and desc.strip() else None

    # ---------- utils ----------

    @staticmethod
    def _extract_server_info(init_result) -> Dict[str, Any] | None:
        info = (
                getattr(init_result, "server_info", None)
                or getattr(init_result, "serverInfo", None)
                or getattr(init_result, "serverinfo", None)
        )
        if info is None:
            return None
        if is_dataclass(info):
            return asdict(info)
        if hasattr(info, "model_dump"):
            return info.model_dump()
        if isinstance(info, dict):
            return info
        return None

    @staticmethod
    def _uniq_name(base: str, used: set[str]) -> str:
        if base not in used:
            return base
        i = 2
        while f"{base}#{i}" in used:
            i += 1
        return f"{base}#{i}"

    def last_errors(self) -> List[Dict[str, Any]]:
        """Return error records from the last discover() run."""
        return list(self._errors)

    def _cfg_identity(self, cfg: McpServerConfig) -> str:
        """Human-friendly identity for error messages."""
        if cfg.transport == "stdio":
            return f"stdio: {cfg.command or '<missing command>'} {' '.join(cfg.args or [])}"
        return f"{cfg.transport}: {cfg.url or '<missing url>'}"

    # ---------- low-level open session from config ----------

    def _open_session_from_config(self, cfg: McpServerConfig) -> AsyncContextManager[ClientSession]:
        t = cfg.transport

        if t == "stdio":
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args or [],
                env=cfg.env or {},
            )
            parent_ctx = stdio_client(params)

            @asynccontextmanager
            async def ctx():
                async with parent_ctx as (read, write):
                    async with ClientSession(read, write) as session:
                        init_result = await session.initialize()
                        info = self._extract_server_info(init_result) or {}
                        session.mcp_server_info = info
                        yield session
            return ctx()

        if t == "streamable_http":
            if not cfg.url:
                raise ValueError("'url' is required for streamable_http")
            parent_ctx = streamablehttp_client(cfg.url, headers=cfg.headers)

            @asynccontextmanager
            async def ctx():
                async with parent_ctx as (read, write, _closer):
                    async with ClientSession(read, write) as session:
                        init_result = await session.initialize()
                        info = self._extract_server_info(init_result) or {}
                        session.mcp_server_info = info
                        yield session
            return ctx()

        if t == "sse":
            if not cfg.url:
                raise ValueError("'url' is required for sse")
            parent_ctx = sse_client(cfg.url, headers=cfg.headers or None)

            @asynccontextmanager
            async def ctx():
                async with parent_ctx as (read, write):
                    async with ClientSession(read, write) as session:
                        init_result = await session.initialize()
                        info = self._extract_server_info(init_result) or {}
                        session.mcp_server_info = info
                        yield session
            return ctx()

        raise ValueError(f"Unknown transport: {t}")

    # ---------- discovery ----------

    async def discover(self, strict: bool = False) -> Dict[str, McpServerConfig]:
        """
        Probe each server to learn its MCP-declared name.
        - strict=False (default): collect errors and continue (partial success).
        - strict=True: raise ExceptionGroup with all failures.
        """
        self._by_name = {}
        self._errors = []
        self._discovered = False

        name_map: Dict[str, McpServerConfig] = {}
        used: set[str] = set()
        failures: List[BaseException] = []

        for cfg in self._configs:
            ident = self._cfg_identity(cfg)
            try:
                async with self._open_session_from_config(cfg) as session:
                    info = getattr(session, "mcp_server_info", {}) or {}
                    base = str(info.get("name") or "server").strip() or "server"
                    name = self._uniq_name(base, used)
                    used.add(name)
                    name_map[name] = cfg
            except Exception as e:
                tb = "".join(traceback.format_exception(e))
                self._errors.append({
                    "config": {
                        "transport": cfg.transport,
                        "url": cfg.url,
                        "command": cfg.command,
                        "args": cfg.args,
                    },
                    "identity": ident,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": tb,
                })
                failures.append(e)

        self._by_name = name_map
        self._discovered = True

        if strict and failures:
            raise ExceptionGroup(
                f"MCP discovery failed for {len(failures)} server(s)",
                failures
            )

        return name_map

    def server_names(self) -> List[str]:
        return list(self._by_name.keys())

    # ---------- public API ----------

    def get_client(self, server_name: str) -> AsyncContextManager[ClientSession]:
        if server_name not in self._by_name:
            suggestions = difflib.get_close_matches(server_name, self.server_names(), n=3, cutoff=0.5)
            suggest_msg = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            known = ", ".join(self.server_names()) or "(none discovered yet)"
            raise ValueError(f"Unknown server '{server_name}'. Known: {known}.{suggest_msg}")
        cfg = self._by_name[server_name]
        return self._open_session_from_config(cfg)

    async def list_clients(self) -> MultiServerMCPClient:
        if not self._discovered:
            await self.discover()
        mapping: Dict[str, Any] = {}
        for name, cfg in self._by_name.items():
            if cfg.transport == "streamable_http":
                mapping[name] = StreamableHttpConnection(
                    transport="streamable_http",
                    url=cfg.url,  # type: ignore[arg-type]
                    headers=cfg.headers or None,
                )
            elif cfg.transport == "stdio":
                mapping[name] = StdioConnection(
                    transport="stdio",
                    command=cfg.command,  # type: ignore[arg-type]
                    args=cfg.args or [],
                    env=cfg.env or {},
                )
            elif cfg.transport == "sse":
                mapping[name] = SSEConnection(
                    transport="sse",
                    url=cfg.url,  # type: ignore[arg-type]
                    headers=cfg.headers or None,
                )
            else:
                raise ValueError(f"Unknown transport: {cfg.transport}")
        return MultiServerMCPClient(mapping)

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self._discovered:
            await self.discover()
        async with self.get_client(server_name) as session:
            res = await session.call_tool(tool_name, arguments=arguments)
            if getattr(res, "structuredContent", None):
                return {"structured": res.structuredContent}
            texts = [c.text for c in (res.content or []) if hasattr(c, "text")]
            return {"content": "\n".join(texts)}

    async def list_tools(self):
        ms_client = await self.list_clients()
        return await ms_client.get_tools()

    async def list_resources(self, server_name: str):
        ms_client = await self.list_clients()
        return await ms_client.get_resources(server_name)

    async def list_prompts(self, server_name: str, prompt_name: str):
        ms_client = await self.list_clients()
        return await ms_client.get_prompt(server_name, prompt_name)

    async def get_openmcp(
            self,
            server_name: Optional[str] = None,
            *,
            schema_url: str = "https://meta.local/schemas/mcps-0.1.json",
    ) -> Dict[str, Any]:
        """
        Build an OpenAPI-like MCP Spec (MCPS) document for exactly one server.
        All top-level info is read from the server's initialize() metadata.
        If multiple servers are configured and `server_name` is not provided,
        raises a helpful error listing available names.
        """
        if not self._discovered:
            await self.discover()

        names = self.server_names()
        if not names:
            raise RuntimeError("No servers discovered; cannot generate docs.")

        if server_name is None:
            if len(names) > 1:
                raise ValueError(
                    "Multiple servers discovered; specify `server_name`. "
                    f"Available: {', '.join(names)}"
                )
            target = names[0]
        else:
            target = server_name
            if target not in self._by_name:
                # mirror your get_client() UX
                import difflib
                suggestions = difflib.get_close_matches(target, names, n=3, cutoff=0.5)
                suggest = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise ValueError(f"Unknown server '{target}'. Known: {', '.join(names)}.{suggest}")

        cfg = self._by_name[target]
        ms_client = await self.list_clients()

        tools, prompts, resources, templates, roots = [], [], [], [], []
        server_info: Dict[str, Any] = {}

        async with ms_client.session(target) as session:
            # captured at initialize() inside _open_session_from_config
            server_info = getattr(session, "mcp_server_info", {}) or {}

            # tools
            try:
                listed = await session.list_tools()
                # allow both raw list and typed response with `.tools`
                listed = getattr(listed, "tools", listed) or []
            except Exception:
                listed = []
            for t in listed:
                tools.append({
                    "name": getattr(t, "name", None),
                    "description": self._safe_text(getattr(t, "description", None)),
                    "args_schema": self._safe_schema(
                        getattr(t, "inputSchema", None) or getattr(t, "args_schema", None)
                    ),
                    "output_schema": self._safe_schema(
                        getattr(t, "outputSchema", None) or getattr(t, "output_schema", None)
                    ),
                    "examples": [],
                })

            # prompts
            try:
                pl = await session.list_prompts()
                pl = getattr(pl, "prompts", pl) or []
                for p in pl:
                    prompts.append({
                        "name": getattr(p, "name", None),
                        "description": self._safe_text(getattr(p, "description", None)),
                        "arguments_schema": self._safe_schema(getattr(p, "arguments_schema", None)),
                    })
            except Exception:
                pass

            # resources
            try:
                rl = await session.list_resources()
                rl = getattr(rl, "resources", rl) or []
                for r in rl:
                    resources.append({
                        "uri": getattr(r, "uri", None),
                        "name": getattr(r, "name", None),
                        "description": self._safe_text(getattr(r, "description", None)),
                        "mime_type": getattr(r, "mimeType", None),
                        "readable": True,
                    })
            except Exception:
                pass

            # resource templates
            try:
                tl = await session.list_resource_templates()
                tl = getattr(tl, "resource_templates", tl) or tl or []
                for tpl in tl:
                    vars_in = getattr(tpl, "variables", None) or []
                    variables = [{
                        "name": getattr(v, "name", None),
                        "description": self._safe_text(getattr(v, "description", None)),
                        "required": bool(getattr(v, "required", False)),
                    } for v in vars_in]
                    templates.append({
                        "uri_template": getattr(tpl, "uriTemplate", None),
                        "name": getattr(tpl, "name", None),
                        "description": self._safe_text(getattr(tpl, "description", None)),
                        "mime_type": getattr(tpl, "mimeType", None),
                        "variables": variables,
                    })
            except Exception:
                pass

            # roots
            try:
                base_roots = await session.list_roots()
                base_roots = getattr(base_roots, "roots", base_roots) or []
                for root in base_roots:
                    roots.append({
                        "uri": getattr(root, "uri", None),
                        "name": getattr(root, "name", None),
                        "description": self._safe_text(getattr(root, "description", None)),
                    })
            except Exception:
                pass

        # endpoint field
        endpoint = cfg.url or cfg.command or "stdio"

        # top-level info entirely from initialize()
        title = server_info.get("title") or server_info.get("name") or target
        description = self._safe_text(server_info.get("description"))
        version = (
                server_info.get("version")
                or server_info.get("semver")
                or "0.1.0"
        )

        # capabilities: prefer server-declared; fall back to inference
        info_caps = server_info.get("capabilities") or {}
        inferred_caps = {
            "tools": bool(tools),
            "resources": bool(resources or templates),
            "prompts": bool(prompts),
            "sampling": bool(info_caps.get("sampling", False)),
        }
        # merge booleans (server value wins when present)
        capabilities = {**inferred_caps, **{k: bool(v) for k, v in info_caps.items()}}

        return {
            "$schema": schema_url,
            "mcps_version": "0.1",
            "info": {
                "title": title,
                "description": description,
                "version": version,
            },
            "server": {
                "name": server_info.get("name") or title,
                "transport": cfg.transport,
                "endpoint": endpoint,
                "capabilities": capabilities,
            },
            "tools": tools,
            "prompts": prompts,
            "resources": resources,
            "resource_templates": templates,
            "roots": roots,
            "auth": {"notes": None},
            "x-vendor": {},
        }

    async def list_openmcp(
            self,
            *,
            schema_url: str = "https://meta.local/schemas/mcps-0.1.json",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return an MCPS doc per discovered server, keyed by server name.
        """
        if not self._discovered:
            await self.discover()
        result: Dict[str, Dict[str, Any]] = {}
        for name in self.server_names():
            result[name] = await self.get_openmcp(server_name=name, schema_url=schema_url)
        return result