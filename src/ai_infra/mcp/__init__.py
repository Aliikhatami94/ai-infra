from .models import (
    OpenMcp,
    RemoteServer,
    RemoteServerConfig,
)
from ai_infra.mcp.fastapi.models import (
    HostedMcp,
    HostedServer,
    HostedServerConfig
)
from ai_infra.mcp.fastapi.server import setup_mcp_server
from .fastapi import add_mcp_to_fastapi