from fastapi import FastAPI

from ai_infra.open_mcp.core import McpConfig
from .utils import mount_mcps, make_lifespan


def add_mcp_to_fastapi(app: FastAPI, config: McpConfig) -> None:
    app.router.lifespan_context = make_lifespan(config)
    mount_mcps(app, config)