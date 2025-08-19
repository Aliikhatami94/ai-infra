from __future__ import annotations

import contextlib
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

try:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except Exception:
    Starlette = None  # Starlette optional

log = logging.getLogger(__name__)


# ----------------------------
# small value object
# ----------------------------
@dataclass
class MCPMount:
    """
    Represents one mounted MCP sub-app.

    path: mount base (parent of '/mcp'), e.g. '/openapi-app'
    app:  ASGI app with optional app.state.session_manager (set by FastMCP apps)
    name: friendly label for logs
    session_manager: explicit override; if omitted we try app.state.session_manager
    """
    path: str
    app: Any
    name: Optional[str] = None
    session_manager: Any | None = None


# ----------------------------
# main orchestrator
# ----------------------------
class CoreMCPServer:
    """
    Minimal orchestrator for hosting multiple MCP servers.

    Examples
    --------
    server = CoreMCPServer(strict=True)
    server.add_app("/streamable-app", streamable_app, name="streamable")
    server.add_fastmcp(mcp_sse, "/sse-demo", transport="sse", name="sse")

    # FastAPI
    server.attach_to_fastapi(app)  # single line: mounts + wires lifespan

    # Standalone
    asgi = server.build_asgi_root()
    # uvicorn.run(asgi, host="0.0.0.0", port=8000)
    """

    def __init__(self, *, strict: bool = True, health_path: str = "/healthz") -> None:
        """
        strict=True  -> raise if a sub-app has no session_manager
        strict=False -> warn and skip that sub-app
        """
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
    ) -> "CoreMCPServer":
        """Register a prebuilt ASGI app (e.g., mcp.streamable_http_app())."""
        self._mounts.append(MCPMount(path=normalize_mount(path), app=app, name=name, session_manager=session_manager))
        return self

    def add_fastmcp(
            self,
            mcp: Any,
            path: str,
            *,
            transport: str = "streamable_http",  # "streamable_http" | "sse" | "websocket"
            name: Optional[str] = None,
    ) -> "CoreMCPServer":
        """Create the transport app from a FastMCP and add it."""
        if transport == "streamable_http":
            sub_app = mcp.streamable_http_app()
        elif transport == "sse":
            sub_app = mcp.sse_app()
        elif transport == "websocket":
            sub_app = mcp.websocket_app()
        else:
            raise ValueError(f"Unknown transport: {transport}")

        # ensure the app exposes the manager for clients that expect it at app.state.session_manager
        sm = getattr(mcp, "session_manager", None)
        if sm and not getattr(getattr(sub_app, "state", object()), "session_manager", None):
            setattr(sub_app.state, "session_manager", sm)

        return self.add_app(path, sub_app, name=name, session_manager=sm)

    def add_from_module(
            self,
            module_path: str,
            path: str,
            *,
            attr: Optional[str] = None,
            transport: Optional[str] = None,
            name: Optional[str] = None,
    ) -> "CoreMCPServer":
        """
        Load either a FastMCP object or an ASGI app from a module path.
        - If it's a FastMCP, you must pass transport (unless the object is already an app).
        - If it's already an ASGI app, transport is ignored.
        """
        obj = import_object(module_path, attr=attr)
        # If the object looks like a FastMCP (has .streamable_http_app etc.)
        if transport and hasattr(obj, "streamable_http_app"):
            return self.add_fastmcp(obj, path, transport=transport, name=name)

        # else assume it's an ASGI app
        return self.add_app(path, obj, name=name)

    # ---------- mounting + lifespan ----------

    def mount_all(self, root_app: Any) -> None:
        """Mount all registered sub-apps onto an existing ASGI root (FastAPI/Starlette)."""
        for m in self._mounts:
            root_app.mount(m.path, m.app)
            log.info("Mounted MCP app '%s' at %s", (m.name or "mcp"), m.path)

    def _iter_unique_session_managers(self) -> Iterable[tuple[str, Any]]:
        seen: set[int] = set()
        for m in self._mounts:
            sm = m.session_manager or getattr(getattr(m.app, "state", None), "session_manager", None)
            if sm is None:
                msg = f"[MCP] Sub-app at '{m.path}' has no session_manager."
                if self._strict:
                    raise RuntimeError(msg)
                log.warning(msg + " Skipping.")
                continue
            key = id(sm)
            if key in seen:
                continue
            seen.add(key)
            yield (m.name or m.path, sm)

    @contextlib.asynccontextmanager
    async def lifespan(self, _app: Any):
        async with contextlib.AsyncExitStack() as stack:
            for label, sm in self._iter_unique_session_managers():
                log.info("Starting MCP session manager: %s", label)
                await stack.enter_async_context(sm.run())
            yield  # close in reverse

    def attach_to_fastapi(self, app: Any) -> None:
        """One-liner FastAPI/Starlette integration: mount + wire lifespan."""
        self.mount_all(app)
        app.router.lifespan_context = self.lifespan

    # ---------- standalone root ----------

    def build_asgi_root(self) -> Any:
        """
        Build a minimal Starlette root that mounts all sub-apps
        and runs their session managers.
        """
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
        """For demos: run the standalone root via uvicorn."""
        import uvicorn
        uvicorn.run(self.build_asgi_root(), host=host, port=port, log_level=log_level)


# ----------------------------
# tiny utils
# ----------------------------
def normalize_mount(path: str) -> str:
    """
    The MCP client connects to '<mount>/mcp'. We mount at the *parent*.
    Accept '/foo', '/foo/mcp', 'foo', etc., and return '/foo'.
    """
    p = ("/" + path.strip("/")).rstrip("/")
    if p.endswith("/mcp"):
        p = p[:-4] or "/"
    return p or "/"

def import_object(module_path: str, *, attr: Optional[str] = None) -> Any:
    """
    Import object from 'pkg.mod' or 'pkg.mod:attr'. If attr not provided, try
    common names 'mcp' then 'app'.
    """
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