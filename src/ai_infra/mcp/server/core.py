from __future__ import annotations

import contextlib
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

try:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except Exception:
    Starlette = None  # optional

log = logging.getLogger(__name__)


@dataclass
class MCPMount:
    """
    path: base mount (parent of '/mcp'), e.g. '/openapi-app'
    app:  ASGI app (e.g., mcp.streamable_http_app(), mcp.sse_app())
    name: friendly label for logs
    session_manager: explicit override; if omitted, uses app.state.session_manager
    require_manager: None -> auto (run if a manager exists), True/False -> force
    """
    path: str
    app: Any
    name: Optional[str] = None
    session_manager: Any | None = None
    require_manager: Optional[bool] = None  # <-- changed default to None (auto)


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
            require_manager: Optional[bool] = None,   # None = auto
    ) -> "CoreMCPServer":
        m = MCPMount(
            path=normalize_mount(path),
            app=app,
            name=name,
            session_manager=session_manager,
            require_manager=require_manager,
        )
        # Auto-infer if not explicitly provided
        if m.require_manager is None:
            sm = m.session_manager or getattr(getattr(m.app, "state", None), "session_manager", None)
            m.require_manager = bool(sm)  # run if it has one, skip if it doesn't
        self._mounts.append(m)
        return self

    def add_fastmcp(
            self,
            mcp: Any,
            path: str,
            *,
            transport: str = "streamable_http",  # "streamable_http" | "sse" | "websocket"
            name: Optional[str] = None,
            require_manager: Optional[bool] = None,   # None = auto
    ) -> "CoreMCPServer":
        if transport == "streamable_http":
            sub_app = mcp.streamable_http_app()
        elif transport == "sse":
            sub_app = mcp.sse_app()
        elif transport == "websocket":
            sub_app = mcp.websocket_app()
        else:
            raise ValueError(f"Unknown transport: {transport}")

        sm = getattr(mcp, "session_manager", None)
        if sm and not getattr(getattr(sub_app, "state", object()), "session_manager", None):
            setattr(sub_app.state, "session_manager", sm)

        # Defer to add_app (which will auto-infer if require_manager is None)
        return self.add_app(path, sub_app, name=name, session_manager=sm, require_manager=require_manager)

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
        if transport and hasattr(obj, "streamable_http_app"):
            return self.add_fastmcp(obj, path, transport=transport, name=name, require_manager=require_manager)
        return self.add_app(path, obj, name=name, require_manager=require_manager)

    # ---------- mounting + lifespan ----------

    def mount_all(self, root_app: Any) -> None:
        for m in self._mounts:
            root_app.mount(m.path, m.app)
            label = (
                    m.name
                    or getattr(getattr(m.app, "state", object()), "mcp_name", None)
                    or "mcp"
            )
            log.info("Mounted MCP app '%s' at %s", label, m.path)

    def _iter_unique_session_managers(self) -> Iterable[tuple[str, Any]]:
        seen: set[int] = set()
        for m in self._mounts:
            sm = m.session_manager or getattr(getattr(m.app, "state", None), "session_manager", None)

            # Auto-skip if require_manager is False, or if it's auto and no manager exists
            if not m.require_manager:
                log.debug("[MCP] Mount '%s' does not require a session manager; skipping.", m.path)
                continue
            if m.require_manager and sm is None:
                msg = f"[MCP] Sub-app at '{m.path}' has no session_manager."
                if self._strict:
                    raise RuntimeError(msg)
                log.warning(msg + " Skipping.")
                continue

            # Dedup
            key = id(sm)
            if key in seen:
                continue
            seen.add(key)

            label = m.name or m.path
            yield label, sm

    @contextlib.asynccontextmanager
    async def lifespan(self, _app: Any):
        async with contextlib.AsyncExitStack() as stack:
            for label, sm in self._iter_unique_session_managers():
                log.info("Starting MCP session manager: %s", label)
                await stack.enter_async_context(sm.run())
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