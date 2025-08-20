from __future__ import annotations

import httpx
import contextlib
import importlib
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union, Callable, Awaitable

from ai_infra.mcp.server.openapi import _mcp_from_openapi
from ai_infra.mcp.server.tools import _mcp_from_tools, ToolDef, ToolFn

try:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except Exception:
    Starlette = None  # optional

log = logging.getLogger(__name__)


@dataclass
class MCPMount:
    path: str
    app: Any
    name: Optional[str] = None
    session_manager: Any | None = None
    require_manager: Optional[bool] = None
    async_cleanup: Optional[Callable[[], Awaitable[None]]] = None


class CoreMCPServer:
    def __init__(self, *, strict: bool = True, health_path: str = "/healthz") -> None:
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
            transport: str = "streamable_http",  # <— default, but configurable
            client: httpx.AsyncClient | None = None,
            client_factory: Callable[[], httpx.AsyncClient] | None = None,
            base_url: str | None = None,
            name: str | None = None,
    ) -> "CoreMCPServer":
        """
        Build an MCP server from an OpenAPI spec and mount it at `path` using the selected transport.
        """
        mcp = _mcp_from_openapi(
            spec,
            client=client,
            client_factory=client_factory,
            base_url=base_url,
        )
        # Reuse the same logic you already have for FastMCP mounting:
        return self.add_fastmcp(
            mcp,
            path,
            transport=transport,
            name=name,
            # require_manager stays auto-inferred in add_fastmcp/add_app
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
        mcp = _mcp_from_tools(name=name, tools=tools)
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
            # If you want to force providing the spec (e.g., for remote):
            spec: dict | str | Path | None = None,
            openapi_url: str = "/openapi.json",
            client: httpx.AsyncClient | None = None,
            client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> "CoreMCPServer":
        """
        Convert a FastAPI app (local) or a remote FastAPI service into an MCP server.

        - Same-process mode (recommended):
            pass `app` (FastAPI instance). We'll:
              * get spec via `app.openapi()` (dict)
              * call routes with httpx.ASGITransport(app=app) (no network)

        - Remote mode:
            pass `base_url` (e.g. "https://api.example.com").
            Provide `spec` (dict/filepath), or we'll fetch from `base_url + openapi_url`.

        Other params are forwarded to the OpenAPI->MCP builder and the MCP mounting.
        """
        # --- Resolve OpenAPI spec ---
        resolved_spec: dict
        if isinstance(spec, dict):
            resolved_spec = spec
        elif isinstance(spec, (str, Path)):
            # Let the OpenAPI builder normalize file/string; just pass through
            resolved_spec = spec  # type: ignore[assignment]
        elif app is not None:
            if not hasattr(app, "openapi"):
                raise TypeError("Provided `app` does not look like a FastAPI application (missing .openapi())")
            resolved_spec = app.openapi()
        else:
            # Remote FastAPI: need a base URL (from param or client.base_url)
            effective_base = base_url
            if not effective_base and client is not None:
                try:
                    effective_base = str(client.base_url) or None
                except Exception:
                    effective_base = None
            if not effective_base:
                raise ValueError("Remote FastAPI requires either `base_url` or a `client` with base_url.")
            # Fetch spec synchronously to avoid async-at-import issues
            url = effective_base.rstrip("/") + openapi_url
            with httpx.Client(timeout=30.0) as sync_client:
                resp = sync_client.get(url)
                resp.raise_for_status()
                resolved_spec = resp.json()

        # --- Resolve HTTP client for tools ---
        if client is not None:
            tools_client = client
            own_client = False
        elif client_factory is not None:
            tools_client = client_factory()
            own_client = True
        elif app is not None:
            transport_obj = httpx.ASGITransport(app=app)
            tools_client = httpx.AsyncClient(
                transport=transport_obj,
                base_url=base_url or "http://fastapi.local",
            )
            own_client = True
        else:
            # Remote mode without provided client: base_url is guaranteed above
            if not base_url:
                raise ValueError("Remote FastAPI mode requires `base_url` when no `client`/`client_factory` is given.")
            tools_client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
            own_client = True

        # Infer base URL for the tool builder:
        inferred_base = base_url
        if inferred_base is None:
            try:
                inferred_base = str(tools_client.base_url) or None  # httpx.URL('') -> ''
            except Exception:
                inferred_base = None

        # --- Build MCP server from spec ---
        mcp = _mcp_from_openapi(
            resolved_spec,
            client=tools_client,
            client_factory=None,
            base_url=inferred_base,  # allow spec.servers/_base_url to override per-call
        )

        # Ensure the async client we created is closed on shutdown
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