import os

from dotenv import find_dotenv, load_dotenv

if not os.environ.get("AI_INFRA_ENV_LOADED"):
    load_dotenv(find_dotenv(usecwd=True))
    os.environ["AI_INFRA_ENV_LOADED"] = "1"

from ai_infra.graph.core import CoreGraph, Graph

# Backward-compatible deprecated aliases
# Re-export primary public API components (new names)
from ai_infra.llm.core import LLM, Agent, CoreAgent, CoreLLM
from ai_infra.llm.providers import Providers
from ai_infra.llm.providers.models import Models
from ai_infra.mcp.client.core import CoreMCPClient, MCPClient
from ai_infra.mcp.server.core import CoreMCPServer, MCPServer
from ai_infra.mcp.server.tools import mcp_from_functions

__all__ = [
    # New names (preferred)
    "LLM",
    "Agent",
    "Graph",
    "MCPServer",
    "MCPClient",
    "Models",
    "Providers",
    "mcp_from_functions",
    # Deprecated aliases (backward compatibility)
    "CoreLLM",
    "CoreAgent",
    "CoreGraph",
    "CoreMCPServer",
    "CoreMCPClient",
]
