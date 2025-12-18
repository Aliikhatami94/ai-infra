from __future__ import annotations

import contextlib
import importlib
import logging
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

import httpx

from ai_infra.mcp.server.openapi import _mcp_from_openapi
from ai_infra.mcp.server.tools import ToolDef, ToolFn, mcp_from_functions

from .models import MCPMount

Starlette: type | None
try:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except Exception:
    Starlette = None

log = logging.getLogger(__name__)


class MCPServer:
    """MCP Server for hosting one or more MCP endpoints."""

    def __init__(self, *, strict: bool = True, health_path: str = "/health") -> None:
        self._strict = strict
        self._health_path = health_path
        self._mounts: list[MCPMount] = []

    # ---------- add / compose ----------

    def add(self, *mounts: MCPMount) -> MCPServer:
        self._mounts.extend(mounts)
        return self

    def add_app(
        self,
        path: str,
        app: Any,
        *,
        name: str | None = None,
        session_manager: Any | None = None,
        require_manager: bool | None = None,
        async_cleanup: Callable[[], Awaitable[None]] | None = None,  # NEW
    ) -> MCPServer:
        m = MCPMount(
            path=normalize_mount(path),
            app=app,
            name=name,
            session_manager=session_manager,
            require_manager=require_manager,
            async_cleanup=async_cleanup,  # NEW
        )
        if m.require_manager is None:
            sm = m.session_manager or getattr(
                getattr(m.app, "state", None), "session_manager", None
            )
            m.require_manager = bool(sm)
        self._mounts.append(m)
        return self

    def add_fastmcp(
        self,
        mcp: Any,
        path: str,
        *,
        transport: str = "streamable_http",
        name: str | None = None,
        require_manager: bool | None = None,
        async_cleanup: Callable[[], Awaitable[None]] | None = None,  # NEW
    ) -> MCPServer:
        if transport == "streamable_http":
            sub_app = mcp.streamable_http_app()
            sm = getattr(mcp, "session_manager", None)
            if sm and not getattr(getattr(sub_app, "state", object()), "session_manager", None):
                sub_app.state.session_manager = sm
            if require_manager is None:
                require_manager = True
            return self.add_app(
                path,
                sub_app,
                name=name,
                session_manager=sm,
                require_manager=require_manager,
                async_cleanup=async_cleanup,
            )

        elif transport == "sse":
            sub_app = mcp.sse_app()
            if require_manager is None:
                require_manager = False
            return self.add_app(
                path,
                sub_app,
                name=name,
                session_manager=None,
                require_manager=require_manager,
                async_cleanup=async_cleanup,
            )

        elif transport == "websocket":
            sub_app = mcp.websocket_app()
            if require_manager is None:
                require_manager = False
            return self.add_app(
                path,
                sub_app,
                name=name,
                session_manager=None,
                require_manager=require_manager,
                async_cleanup=async_cleanup,
            )

        else:
            raise ValueError(f"Unknown transport: {transport}")

    def add_from_module(
        self,
        module_path: str,
        path: str,
        *,
        attr: str | None = None,
        transport: str | None = None,
        name: str | None = None,
        require_manager: bool | None = None,  # None = auto
    ) -> MCPServer:
        obj = import_object(module_path, attr=attr)
        # If it's a FastMCP (has .streamable_http_app), respect transport given
        if transport and hasattr(obj, "streamable_http_app"):
            return self.add_fastmcp(
                obj,
                path,
                transport=transport,
                name=name,
                require_manager=require_manager,
            )
        # Else assume it's an ASGI app
        return self.add_app(path, obj, name=name, require_manager=require_manager)

    def add_openapi(
        self,
        path: str,
        spec: dict | str | Path,
        *,
        transport: str = "streamable_http",
        client: httpx.AsyncClient | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
        base_url: str | None = None,
        name: str | None = None,
        report_log: bool | None = None,
        strict_names: bool = False,
        # Filtering options
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
        # Authentication
        auth: Any = None,
        endpoint_auth: dict[str, Any] | None = None,
    ) -> MCPServer:
        """Add OpenAPI spec as MCP tools.

        Args:
            path: Mount path for the MCP server
            spec: OpenAPI spec (URL, file path, or dict)
            transport: Transport type (streamable_http, sse, websocket)
            client: Existing httpx client
            client_factory: Factory for httpx client
            base_url: Override base URL for all requests
            name: Name for the MCP server
            report_log: Log build report
            strict_names: Raise on duplicate tool names

            # Filtering
            tool_prefix: Prefix all tool names
            include_paths: Only include paths matching these globs
            exclude_paths: Exclude paths matching these globs
            include_methods: Only include these HTTP methods
            exclude_methods: Exclude these HTTP methods
            include_tags: Only include operations with these tags
            exclude_tags: Exclude operations with these tags
            include_operations: Only include these operationIds
            exclude_operations: Exclude these operationIds
            tool_name_fn: Custom function(method, path, op) -> name
            tool_description_fn: Custom function(op) -> description

            # Authentication
            auth: Auth config (dict for headers, tuple for basic, str for bearer)
            endpoint_auth: Per-endpoint auth overrides (pattern -> auth)

        Example:
            # Zero-config
            server.add_openapi("/github", "https://api.github.com/openapi.json")

            # With filtering and auth
            server.add_openapi(
                "/github",
                "https://api.github.com/openapi.json",
                tool_prefix="github",
                include_paths=["/repos/*", "/users/*"],
                exclude_methods=["DELETE"],
                auth={"Authorization": "Bearer ghp_xxx"},
            )
        """
        res = _mcp_from_openapi(
            spec,
            client=client,
            client_factory=client_factory,
            base_url=base_url,
            strict_names=strict_names,
            report_log=report_log,
            tool_prefix=tool_prefix,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            include_methods=include_methods,
            exclude_methods=exclude_methods,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            include_operations=include_operations,
            exclude_operations=exclude_operations,
            tool_name_fn=tool_name_fn,
            tool_description_fn=tool_description_fn,
            auth=auth,
            endpoint_auth=endpoint_auth,
        )
        # back-compat unpack (2 or 3 items)
        if isinstance(res, tuple) and len(res) == 3:
            mcp, async_cleanup, report = res
            # optional: stash report for later introspection
            try:
                mcp.openapi_build_report = report  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            mcp, async_cleanup = res
            report = None

        return self.add_fastmcp(
            mcp,
            path,
            transport=transport,
            name=name,
            require_manager=None,
            async_cleanup=async_cleanup,
        )

    def add_tools(
        self,
        path: str,
        *,
        tools: Iterable[ToolFn | ToolDef] | None,
        name: str | None = None,
        transport: str = "streamable_http",
        require_manager: bool | None = None,  # None = auto
    ) -> MCPServer:
        """
        Build a FastMCP server from in-code tools and mount it.

        Example:
            server.add_tools(
                "/my-tools",
                tools=[say_hello, ToolDef(fn=foo, name="foo", description="...")],
                name="my-tools",
                transport="streamable_http",
            )
        """
        mcp = mcp_from_functions(name=name, functions=tools)
        return self.add_fastmcp(
            mcp,
            path,
            transport=transport,
            name=name,
            require_manager=require_manager,
        )

    def add_fastapi(
        self,
        path: str,
        *,
        app: Any | None = None,
        base_url: str | None = None,
        name: str | None = None,
        transport: str = "streamable_http",
        spec: dict | str | Path | None = None,
        openapi_url: str = "/openapi.json",
        client: httpx.AsyncClient | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | httpx.Timeout | None = 30.0,
        verify: bool | str | None = True,
        auth: httpx.Auth | tuple[str, str] | None = None,
    ) -> MCPServer:
        """
        Convert a FastAPI app (local) or a remote FastAPI service into an MCP server.
        """

        # ---------- resolve OpenAPI spec ----------
        resolved_spec: dict | str | Path | None = None

        if isinstance(spec, dict) or isinstance(spec, (str, Path)):
            resolved_spec = spec
        elif app is not None:
            if not hasattr(app, "openapi"):
                raise TypeError(
                    "Provided `app` does not look like a FastAPI application (missing .openapi())"
                )
            resolved_spec = app.openapi()
        elif base_url:
            url = base_url.rstrip("/") + openapi_url
            # verify defaults to True if not specified
            verify_val = verify if verify is not None else True
            with httpx.Client(
                headers=headers, timeout=timeout, verify=verify_val, auth=auth
            ) as sync_client:
                resp = sync_client.get(url)
                resp.raise_for_status()
                resolved_spec = resp.json()
        else:
            raise ValueError("You must provide either `app`, `base_url`, or an explicit `spec`.")

        # ---------- resolve Async client for tools ----------
        own_client = False
        # verify defaults to True if not specified
        verify_val = verify if verify is not None else True
        if client is not None:
            tools_client = client
        elif client_factory is not None:
            tools_client = client_factory()
            own_client = True
        elif app is not None:
            transport_obj = httpx.ASGITransport(app=app)
            tools_client = httpx.AsyncClient(
                transport=transport_obj,
                base_url=base_url or "http://app.local",
                headers=headers,
                timeout=timeout,
                verify=verify_val,
                auth=auth,
            )
            own_client = True
        elif base_url:
            tools_client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=timeout,
                verify=verify_val,
                auth=auth,
            )
            own_client = True
        else:
            raise ValueError(
                "Unable to build AsyncClient: no `app`, no `base_url`, and no provided client."
            )

        # ---------- infer base_url ----------
        inferred_base = base_url
        if inferred_base is None:
            try:
                inferred_base = str(tools_client.base_url) or None
            except Exception:
                inferred_base = None
        if inferred_base is None and app is not None:
            inferred_base = "http://app.local"

        # ---------- build MCP ----------
        mcp = _mcp_from_openapi(
            resolved_spec,
            client=tools_client,
            client_factory=None,
            base_url=inferred_base,
        )

        async_cleanup = tools_client.aclose if own_client else None
        resolved_name = name or (getattr(app, "title", None) if app is not None else None)

        return self.add_fastmcp(
            mcp,
            path,
            transport=transport,
            name=resolved_name,
            async_cleanup=async_cleanup,
        )

    # ---------- mounting + lifespan ----------

    def mount_all(self, root_app: Any) -> None:
        for m in self._mounts:
            root_app.mount(m.path, m.app)
            label = m.name or getattr(getattr(m.app, "state", object()), "mcp_name", None) or "mcp"
            log.info("Mounted MCP app '%s' at %s", label, m.path)

    def _iter_unique_session_managers(self) -> Iterable[tuple[str, Any]]:
        seen: set[int] = set()
        for m in self._mounts:
            sm = m.session_manager or getattr(
                getattr(m.app, "state", None), "session_manager", None
            )

            # Skip when not required or when auto-mode found none
            if not m.require_manager:
                log.debug(
                    "[MCP] Mount '%s' does not require a session manager; skipping.",
                    m.path,
                )
                continue
            if m.require_manager and sm is None:
                msg = f"[MCP] Sub-app at '{m.path}' has no session_manager."
                if self._strict:
                    raise RuntimeError(msg)
                log.warning(msg + " Skipping.")
                continue

            key = id(sm)
            if key in seen:
                continue
            seen.add(key)
            yield (m.name or m.path), sm

    @contextlib.asynccontextmanager
    async def lifespan(self, _app: Any):
        async with contextlib.AsyncExitStack() as stack:
            # Start session managers
            for label, sm in self._iter_unique_session_managers():
                log.info("Starting MCP session manager: %s", label)
                await stack.enter_async_context(sm.run())

            # Ensure per-mount extra cleanup runs on shutdown
            for m in self._mounts:
                if m.async_cleanup:
                    stack.push_async_callback(m.async_cleanup)

            yield

    def attach_to_fastapi(self, app: Any) -> None:
        self.mount_all(app)

        # Preserve any existing lifespan and merge with ours
        previous_lifespan = getattr(app.router, "lifespan_context", None)

        if previous_lifespan is None:
            # No existing lifespan, just use ours
            app.router.lifespan_context = self.lifespan
        else:
            # Merge: run both lifespans (existing first, then MCP)
            @contextlib.asynccontextmanager
            async def _merged_lifespan(a: Any):
                async with contextlib.AsyncExitStack() as stack:
                    # Enter previous lifespan first
                    await stack.enter_async_context(previous_lifespan(a))
                    # Then enter MCP lifespan
                    await stack.enter_async_context(self.lifespan(a))
                    yield

            app.router.lifespan_context = _merged_lifespan

    # ---------- discovery ----------

    def get_openmcp(self) -> dict:
        """Generate OpenMCP specification for this server.

        Returns an OpenAPI-like spec describing all MCP tools available
        on this server, which clients can use for discovery.

        Returns:
            Dict with OpenMCP specification

        Example:
            mcp = MCPServer(name="my-tools")
            mcp.add_openapi("/api", "https://api.example.com/openapi.json")

            spec = mcp.get_openmcp()
            # {
            #     "openmcp": "1.0.0",
            #     "info": {"title": "my-tools", ...},
            #     "servers": [...],
            #     "tools": [...]
            # }
        """
        tools = []
        servers = []

        for mount in self._mounts:
            # Try to get tools from the mounted app
            app = mount.app
            mcp_obj = getattr(getattr(app, "state", None), "mcp", None) or app

            # Get tools from FastMCP
            if hasattr(mcp_obj, "_tool_manager"):
                tool_manager = mcp_obj._tool_manager
                if hasattr(tool_manager, "_tools"):
                    for name, tool in tool_manager._tools.items():
                        tool_spec = {
                            "name": name,
                            "description": getattr(tool, "description", None) or "",
                            "path": mount.path,
                        }

                        # Try to get input schema
                        if hasattr(tool, "parameters"):
                            tool_spec["inputSchema"] = tool.parameters
                        elif hasattr(tool, "fn"):
                            fn = tool.fn
                            hints = getattr(fn, "__annotations__", {})
                            if hints:
                                # Extract from type hints
                                params = {}
                                for pname, ptype in hints.items():
                                    if pname == "return":
                                        continue
                                    params[pname] = {"type": _py_type_to_json(ptype)}
                                if params:
                                    tool_spec["inputSchema"] = {
                                        "type": "object",
                                        "properties": params,
                                    }

                        tools.append(tool_spec)

            # Get build report from OpenAPI
            report = getattr(mcp_obj, "openapi_build_report", None)
            if report and hasattr(report, "ops"):
                for op in report.ops:
                    if op.tool_name not in [t["name"] for t in tools]:
                        tools.append(
                            {
                                "name": op.tool_name,
                                "description": f"{op.method.upper()} {op.path}",
                                "path": mount.path,
                                "method": op.method,
                                "apiPath": op.path,
                                "baseUrl": op.base_url,
                            }
                        )

            servers.append(
                {
                    "path": mount.path,
                    "name": mount.name or mount.path,
                    "transport": "streamable_http",  # default
                }
            )

        return {
            "openmcp": "1.0.0",
            "info": {
                "title": "MCP Server",
                "version": "1.0.0",
                "description": f"MCP server with {len(tools)} tools across {len(servers)} endpoints",
            },
            "servers": servers,
            "tools": tools,
        }

    # ---------- standalone root ----------

    def build_asgi_root(self) -> Any:
        if Starlette is None:
            raise RuntimeError("Starlette is not installed. `pip install starlette`")

        async def health(_req):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[], lifespan=self.lifespan)
        if self._health_path:
            app.router.routes.append(Route(self._health_path, endpoint=health, methods=["GET"]))
        self.mount_all(app)
        return app

    def run_uvicorn(self, host: str = "0.0.0.0", port: int = 8000, log_level: str = "info"):
        import uvicorn

        uvicorn.run(self.build_asgi_root(), host=host, port=port, log_level=log_level)


# ---------- utils ----------


def normalize_mount(path: str) -> str:
    p = ("/" + path.strip("/")).rstrip("/")
    if p.endswith("/mcp"):
        p = p[:-4] or "/"
    return p or "/"


def import_object(module_path: str, *, attr: str | None = None) -> Any:
    if ":" in module_path and not attr:
        module_path, attr = module_path.split(":", 1)
        attr = attr or None

    module = importlib.import_module(module_path)
    if attr:
        obj = getattr(module, attr, None)
        if obj is None:
            raise ImportError(f"Attribute '{attr}' not found in module '{module_path}'")
        return obj

    for candidate in ("mcp", "app"):
        if hasattr(module, candidate):
            return getattr(module, candidate)

    raise ImportError(
        f"No obvious object found in '{module_path}'. "
        "Provide attr explicitly (e.g., 'pkg.mod:mcp') or export 'mcp'/'app'."
    )


def _py_type_to_json(py_type: Any) -> str:
    """Convert Python type to JSON Schema type."""
    if py_type is None:
        return "null"

    type_name = getattr(py_type, "__name__", str(py_type))

    type_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "NoneType": "null",
    }

    return type_map.get(type_name, "string")


# NOTE: CoreMCPServer alias removed - use MCPServer directly
