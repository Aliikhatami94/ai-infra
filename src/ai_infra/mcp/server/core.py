from __future__ import annotations

import httpx
import contextlib
import importlib
import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Union, Callable, Awaitable

from .models import MCPMount
from ai_infra.mcp.server.openapi import _mcp_from_openapi
from ai_infra.mcp.server.tools import mcp_from_functions, ToolDef, ToolFn

try:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except Exception:
    Starlette = None

log = logging.getLogger(__name__)


class CoreMCPServer:
    def __init__(self, *, strict: bool = True, health_path: str = "/health") -> None:
        self._strict = strict
        self._health_path = health_path
        self._mounts: list[MCPMount] = []

    # ---------- add / compose ----------

    def add(self, *mounts: MCPMount) -> "CoreMCPServer":
        self._mounts.extend(mounts)
        return self

    def add_app(
            self,
            path: str,
            app: Any,
            *,
            name: Optional[str] = None,
            session_manager: Any | None = None,
            require_manager: Optional[bool] = None,
            async_cleanup: Optional[Callable[[], Awaitable[None]]] = None,  # NEW
    ) -> "CoreMCPServer":
        m = MCPMount(
            path=normalize_mount(path),
            app=app,
            name=name,
            session_manager=session_manager,
            require_manager=require_manager,
            async_cleanup=async_cleanup,  # NEW
        )
        if m.require_manager is None:
            sm = m.session_manager or getattr(getattr(m.app, "state", None), "session_manager", None)
            m.require_manager = bool(sm)
        self._mounts.append(m)
        return self

    def add_fastmcp(
            self,
            mcp: Any,
            path: str,
            *,
            transport: str = "streamable_http",
            name: Optional[str] = None,
            require_manager: Optional[bool] = None,
            async_cleanup: Optional[Callable[[], Awaitable[None]]] = None,  # NEW
    ) -> "CoreMCPServer":
        if transport == "streamable_http":
            sub_app = mcp.streamable_http_app()
            sm = getattr(mcp, "session_manager", None)
            if sm and not getattr(getattr(sub_app, "state", object()), "session_manager", None):
                setattr(sub_app.state, "session_manager", sm)
            if require_manager is None:
                require_manager = True
            return self.add_app(path, sub_app, name=name, session_manager=sm,
                                require_manager=require_manager, async_cleanup=async_cleanup)

        elif transport == "sse":
            sub_app = mcp.sse_app()
            if require_manager is None:
                require_manager = False
            return self.add_app(path, sub_app, name=name, session_manager=None,
                                require_manager=require_manager, async_cleanup=async_cleanup)

        elif transport == "websocket":
            sub_app = mcp.websocket_app()
            if require_manager is None:
                require_manager = False
            return self.add_app(path, sub_app, name=name, session_manager=None,
                                require_manager=require_manager, async_cleanup=async_cleanup)

        else:
            raise ValueError(f"Unknown transport: {transport}")

    def add_from_module(
            self,
            module_path: str,
            path: str,
            *,
            attr: Optional[str] = None,
            transport: Optional[str] = None,
            name: Optional[str] = None,
            require_manager: Optional[bool] = None,  # None = auto
    ) -> "CoreMCPServer":
        obj = import_object(module_path, attr=attr)
        # If it's a FastMCP (has .streamable_http_app), respect transport given
        if transport and hasattr(obj, "streamable_http_app"):
            return self.add_fastmcp(obj, path, transport=transport, name=name, require_manager=require_manager)
        # Else assume it's an ASGI app
        return self.add_app(path, obj, name=name, require_manager=require_manager)

    def add_openapi(
            self,
            path: str,
            spec: Union[dict, str, Path],
            *,
            transport: str = "streamable_http",
            client: httpx.AsyncClient | None = None,
            client_factory: Callable[[], httpx.AsyncClient] | None = None,
            base_url: str | None = None,
            name: str | None = None,
            report_log: Optional[bool] = None,     # NEW: let callers force logging
            strict_names: bool = False,            # NEW: propagate strict names
    ) -> "CoreMCPServer":
        res = _mcp_from_openapi(
            spec,
            client=client,
            client_factory=client_factory,
            base_url=base_url,
            strict_names=strict_names,
            report_log=report_log,
        )
        # back-compat unpack (2 or 3 items)
        if isinstance(res, tuple) and len(res) == 3:
            mcp, async_cleanup, report = res
            # optional: stash report for later introspection
            try:
                setattr(mcp, "openapi_build_report", report)
            except Exception:
                pass
        else:
            mcp, async_cleanup = res  # type: ignore[misc]
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
            tools: Iterable[Union[ToolFn, ToolDef]] | None,
            name: Optional[str] = None,
            transport: str = "streamable_http",
            require_manager: Optional[bool] = None,  # None = auto
    ) -> "CoreMCPServer":
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
    ) -> "CoreMCPServer":
        """
        Convert a FastAPI app (local) or a remote FastAPI service into an MCP server.
        """

        # ---------- resolve OpenAPI spec ----------
        resolved_spec: dict | str | Path | None = None

        if isinstance(spec, dict) or isinstance(spec, (str, Path)):
            resolved_spec = spec
        elif app is not None:
            if not hasattr(app, "openapi"):
                raise TypeError("Provided `app` does not look like a FastAPI application (missing .openapi())")
            resolved_spec = app.openapi()
        elif base_url:
            url = base_url.rstrip("/") + openapi_url
            with httpx.Client(headers=headers, timeout=timeout, verify=verify, auth=auth) as sync_client:
                resp = sync_client.get(url)
                resp.raise_for_status()
                resolved_spec = resp.json()
        else:
            raise ValueError("You must provide either `app`, `base_url`, or an explicit `spec`.")

        # ---------- resolve Async client for tools ----------
        own_client = False
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
                verify=verify,
                auth=auth,
            )
            own_client = True
        elif base_url:
            tools_client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=timeout,
                verify=verify,
                auth=auth,
            )
            own_client = True
        else:
            raise ValueError("Unable to build AsyncClient: no `app`, no `base_url`, and no provided client.")

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

        async_cleanup = (tools_client.aclose if own_client else None)
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
            sm = m.session_manager or getattr(getattr(m.app, "state", None), "session_manager", None)

            # Skip when not required or when auto-mode found none
            if not m.require_manager:
                log.debug("[MCP] Mount '%s' does not require a session manager; skipping.", m.path)
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
        app.router.lifespan_context = self.lifespan

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

def import_object(module_path: str, *, attr: Optional[str] = None) -> Any:
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