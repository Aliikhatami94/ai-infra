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
from ai_infra.imagegen import GeneratedImage, ImageGen, ImageGenProvider
from ai_infra.llm import (  # Realtime Voice API
    LLM,
    STT,
    TTS,
    Agent,
    AudioFormat,
    AudioOutput,
    AudioResponse,
    RealtimeConfig,
    RealtimeVoice,
    TranscriptionResult,
    VADMode,
    Voice,
    Workspace,
    realtime_voice,
    workspace,
)

# Phase 6.4 - Memory Management
from ai_infra.llm.memory import (
    MemoryItem,
    MemoryStore,
    SummarizationMiddleware,
    SummarizeResult,
    count_tokens,
    count_tokens_approximate,
    get_context_limit,
    summarize_messages,
    trim_messages,
)
from ai_infra.llm.personas import Persona
from ai_infra.llm.providers import Providers
from ai_infra.llm.tools.custom.retriever import create_retriever_tool, create_retriever_tool_async
from ai_infra.logging import configure_logging, get_logger
from ai_infra.mcp import MCPClient, MCPServer
from ai_infra.mcp.server.tools import mcp_from_functions

# Provider Registry (Phase 4.11)
from ai_infra.providers import (
    CapabilityConfig,
    ProviderCapability,
    ProviderConfig,
    ProviderRegistry,
    get_provider,
    is_provider_configured,
    list_providers,
    list_providers_for_capability,
)
from ai_infra.replay import MemoryStorage, SQLiteStorage, WorkflowRecorder, replay
from ai_infra.retriever import Retriever
from ai_infra.tools import (
    ProgressEvent,
    ProgressStream,
    progress,
    tools_from_models,
    tools_from_models_sql,
)
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
    # Provider Registry (Phase 4.11)
    "ProviderRegistry",
    "ProviderCapability",
    "ProviderConfig",
    "CapabilityConfig",
    "get_provider",
    "list_providers",
    "list_providers_for_capability",
    "is_provider_configured",
    # Multimodal (TTS, STT, Audio)
    "TTS",
    "STT",
    "AudioFormat",
    "AudioOutput",
    "AudioResponse",
    "TranscriptionResult",
    "Voice",
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
    # Image Generation
    "ImageGen",
    "GeneratedImage",
    "ImageGenProvider",
    # Phase 4.7 - Zero Friction Integrations
    "tools_from_models",
    "tools_from_models_sql",
    "Persona",
    "replay",
    "WorkflowRecorder",
    "MemoryStorage",
    "SQLiteStorage",
    "progress",
    "ProgressStream",
    "ProgressEvent",
    # Phase 4.8 - Unified Workspace Architecture
    "Workspace",
    "workspace",
    # Phase 4.10 - Realtime Voice API
    "RealtimeVoice",
    "realtime_voice",
    "RealtimeConfig",
    "VADMode",
    # Phase 6.4 - Memory Management
    "trim_messages",
    "count_tokens",
    "count_tokens_approximate",
    "get_context_limit",
    "summarize_messages",
    "SummarizeResult",
    "SummarizationMiddleware",
    "MemoryStore",
    "MemoryItem",
]
