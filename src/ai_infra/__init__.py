import os

from dotenv import find_dotenv, load_dotenv

if not os.environ.get("AI_INFRA_ENV_LOADED"):
    load_dotenv(find_dotenv(usecwd=True))
    os.environ["AI_INFRA_ENV_LOADED"] = "1"

from ai_infra.callbacks import CallbackManager, Callbacks
from ai_infra.embeddings import Embeddings, VectorStore
from ai_infra.embeddings.vectorstore import Document, SearchResult

# Cross-cutting concerns
from ai_infra.errors import (
    AIInfraError,
    ConfigurationError,
    MCPError,
    OpenAPIError,
    ProviderError,
    ValidationError,
)
from ai_infra.graph import Graph
from ai_infra.llm import LLM, Agent
from ai_infra.llm.providers import Providers
from ai_infra.llm.tools.custom.retriever import create_retriever_tool, create_retriever_tool_async
from ai_infra.logging import configure_logging, get_logger
from ai_infra.mcp import MCPClient, MCPServer
from ai_infra.mcp.server.tools import mcp_from_functions
from ai_infra.retriever import Retriever
from ai_infra.tracing import TracingCallbacks, configure_tracing, get_tracer, trace

# Validation
from ai_infra.validation import (
    validate_llm_params,
    validate_output,
    validate_provider,
    validate_temperature,
)

__all__ = [
    # Core
    "LLM",
    "Agent",
    "Graph",
    "MCPServer",
    "MCPClient",
    "Providers",
    "mcp_from_functions",
    # Embeddings
    "Embeddings",
    "VectorStore",
    "Document",
    "SearchResult",
    # Retriever
    "Retriever",
    "create_retriever_tool",
    "create_retriever_tool_async",
    # Errors
    "AIInfraError",
    "ProviderError",
    "MCPError",
    "OpenAPIError",
    "ValidationError",
    "ConfigurationError",
    # Logging
    "configure_logging",
    "get_logger",
    # Callbacks
    "Callbacks",
    "CallbackManager",
    # Tracing
    "get_tracer",
    "configure_tracing",
    "trace",
    "TracingCallbacks",
    # Validation
    "validate_llm_params",
    "validate_output",
    "validate_provider",
    "validate_temperature",
]
