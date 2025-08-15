from fastapi import FastAPI
import contextlib
from typing import Iterable
from mcp.server.fastmcp import FastMCP

from .models import FastMcpConfig, FastMcpList


def make_lifespan(mcp_list: FastMcpList):
    mcp_list.ensure_all_bound()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # The session_manager is initialized lazily by streamable_http_app().
        # add_mcp_to_fastapi calls mount_mcps() before startup, so we’re safe here.
        import contextlib as _ctx
        async with _ctx.AsyncExitStack() as stack:
            for cfg in mcp_list.mcps:
                # Do not access the property earlier than this point.
                await stack.enter_async_context(cfg.module.session_manager.run())  # type: ignore[union-attr]
            yield
    return lifespan


def mount_mcps(app: FastAPI, mcp_list: FastMcpList) -> None:
    mcp_list.ensure_all_bound()
    for cfg in mcp_list.mcps:
        # This call initializes .session_manager on the FastMCP instance
        app.mount(cfg.path, cfg.module.streamable_http_app())  # type: ignore[union-attr]


def normalize_mcps(configs: Iterable[FastMcpConfig] | FastMcpList | dict[str, FastMCP]) -> FastMcpList:
    if isinstance(configs, FastMcpList):
        return configs
    if isinstance(configs, dict):
        return FastMcpList.from_mapping(configs)
    return FastMcpList(mcps=list(configs))  # type: ignore[arg-type]


def add_mcp_to_fastapi(app: FastAPI, configs: Iterable[FastMcpConfig] | FastMcpList | dict[str, FastMCP]) -> None:
    mcps = normalize_mcps(configs)
    # assign lifespan (not executed yet)
    app.router.lifespan_context = make_lifespan(mcps)
    # mount sub-apps now -> calls streamable_http_app() -> initializes session_manager
    mount_mcps(app, mcps)