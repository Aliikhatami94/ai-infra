import contextlib
from fastapi import FastAPI
from typing import List

from ai_infra.open_mcp.models import Server


def _servers_with_module(servers: List[Server]) -> List[Server]:
    """Return only servers that have a module."""
    return [s for s in servers if getattr(s, "module", None) is not None]

def make_lifespan(servers: List[Server]):
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with contextlib.AsyncExitStack() as stack:
            for server in _servers_with_module(servers):
                await stack.enter_async_context(server.module.session_manager.run())  # type: ignore[union-attr]
            yield
    return lifespan

def mount_mcps(app: FastAPI, servers: List[Server]) -> None:
    for server in _servers_with_module(servers):
        app.mount(
            server.config.url.removesuffix("/mcp"),
            server.module.streamable_http_app()
        )