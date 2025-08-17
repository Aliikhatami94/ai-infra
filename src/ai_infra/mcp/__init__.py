from .models import (
    RemoteMcp,
    RemoteServer,
    RemoteServerConfig,
)
from ai_infra.mcp.hosting.models import (
    HostedMcp,
    HostedServer,
    HostedServerConfig
)
from ai_infra.mcp.core import CoreMCP
from ai_infra.mcp.hosting.server import setup_mcp_server
from .hosting import add_mcp_to_fastapi