from .models import (
    OpenMcp,
    RemoteServer,
    RemoteServerConfig,
    RemoteServerInfo,
)
from ai_infra.mcp.fastapi.models import (
    HostedMcp,
    HostedServerInfo,
    HostedServer,
    HostedServerConfig
)
from ai_infra.mcp.fastapi.server import setup_mcp_server
from .fastapi import add_mcp_to_fastapi