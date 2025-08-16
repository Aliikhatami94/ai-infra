from fastapi import FastAPI

from ai_infra.open_mcp.core import McpConfig
from .utils import mount_mcps, make_lifespan


def add_mcp_to_fastapi(app: FastAPI, config: McpConfig) -> None:
    modules = [server.module for server in config.servers]
    app.router.lifespan_context = make_lifespan(modules)
    mount_mcps(app, config)