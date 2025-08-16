import contextlib
from typing import List
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

from ai_infra.mcp.core import McpConfig


def make_lifespan(mcp_modules: List[FastMCP]):
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # The session_manager is initialized lazily by streamable_http_app().
        # add_mcp_to_fastapi calls mount_mcps() before startup, so we’re safe here.
        import contextlib as _ctx
        async with _ctx.AsyncExitStack() as stack:
            for module in mcp_modules:
                await stack.enter_async_context(module.session_manager.run())  # type: ignore[union-attr]
            yield
    return lifespan


def mount_mcps(app: FastAPI, mcp_config: McpConfig) -> None:
    for server in mcp_config.servers:
        app.mount(
            server.config.url.removesuffix("/mcp"),
            server.module.streamable_http_app()
        )