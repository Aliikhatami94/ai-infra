import os

from dotenv import find_dotenv, load_dotenv

if not os.environ.get("AI_INFRA_ENV_LOADED"):
    load_dotenv(find_dotenv(usecwd=True))
    os.environ["AI_INFRA_ENV_LOADED"] = "1"

from ai_infra.graph import Graph
from ai_infra.llm import LLM, Agent
from ai_infra.llm.providers import Providers
from ai_infra.mcp.client.core import MCPClient
from ai_infra.mcp.server.core import MCPServer
from ai_infra.mcp.server.tools import mcp_from_functions

__all__ = [
    "LLM",
    "Agent",
    "Graph",
    "MCPServer",
    "MCPClient",
    "Providers",
    "mcp_from_functions",
]
