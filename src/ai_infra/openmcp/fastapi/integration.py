from fastapi import FastAPI

from ai_infra.openmcp.core import OpenMcp
from .utils import mount_mcps, make_lifespan


def add_mcp_to_fastapi(app: FastAPI, config: OpenMcp) -> None:
    app.router.lifespan_context = make_lifespan(config.servers)
    mount_mcps(app, config.servers)