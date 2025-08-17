from fastapi import FastAPI

from ai_infra.mcp.fastapi.models import HostedMcp
from .utils import mount_mcps, make_lifespan


def add_mcp_to_fastapi(app: FastAPI, config: HostedMcp | dict) -> None:
    app.router.lifespan_context = make_lifespan(config.servers)
    mount_mcps(app, config.servers)