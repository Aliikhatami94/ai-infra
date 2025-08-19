from __future__ import annotations
from typing import Iterable, Tuple, Any
from fastapi import FastAPI
import contextlib


class CoreMCPServer:
    """
    Helper to mount one or more MCP Starlette apps onto FastAPI
    and wire their session managers into the main app lifespan.

    Usage:
        mcp_server = CoreMCPServer()
        mcp_server.mount(app, "/openapi-app", openapi_app)
        mcp_server.mount(app, "/streamable-app", streamable_app)
        app.router.lifespan_context = mcp_server.lifespan
    """

    def __init__(self):
        self._mounts: list[Tuple[str, Any]] = []

    def mount(self, root: FastAPI, path: str, sub_app: Any):
        """
        Mount a Starlette/ASGI MCP app under a path.
        We do *not* enter its lifespan here; we only mount and remember it.
        """
        root.mount(path, sub_app)
        self._mounts.append((path, sub_app))

    def _iter_session_managers(self, root_app: FastAPI):
        seen = set()
        # gather from our recorded mounts (safer than crawling routes)
        for _, sub in self._mounts:
            sm = getattr(getattr(sub, "state", None), "session_manager", None)
            if sm and id(sm) not in seen:
                seen.add(id(sm))
                yield sm

    @contextlib.asynccontextmanager
    async def lifespan(self, app: FastAPI):
        async with contextlib.AsyncExitStack() as stack:
            for sm in self._iter_session_managers(app):
                await stack.enter_async_context(sm.run())
            yield