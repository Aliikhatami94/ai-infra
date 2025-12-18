"""
Root conftest.py for ai-infra tests.

This file provides:
1. Common pytest markers for test categorization
2. Shared fixtures used across multiple test modules
3. Mock LLM/Embeddings providers
4. Common callback tracking utilities

Fixtures are organized by category:
- Environment fixtures (API key mocking)
- LLM fixtures (mock providers, responses)
- Callback fixtures (tracking, validation)
- Embeddings fixtures
- Storage/Backend fixtures
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        path = str(item.fspath)
        norm = path.replace("\\", "/")

        # Mark integration tests
        if "/tests/integration/" in norm:
            item.add_marker(pytest.mark.integration)

        # Mark by module name
        if "llm" in norm:
            item.add_marker(pytest.mark.llm)
        if "agent" in norm:
            item.add_marker(pytest.mark.agent)
        if "mcp" in norm:
            item.add_marker(pytest.mark.mcp)
        if "embedding" in norm:
            item.add_marker(pytest.mark.embeddings)
        if "retriever" in norm:
            item.add_marker(pytest.mark.retriever)
        if "callback" in norm:
            item.add_marker(pytest.mark.callbacks)
        if "streaming" in norm:
            item.add_marker(pytest.mark.streaming)
        if "imagegen" in norm:
            item.add_marker(pytest.mark.imagegen)
        if "tts" in norm or "stt" in norm or "realtime" in norm:
            item.add_marker(pytest.mark.multimodal)
        if "memory" in norm:
            item.add_marker(pytest.mark.memory)


def pytest_configure(config):
    """Register custom markers."""
    for name, desc in [
        ("integration", "Integration tests requiring API keys"),
        ("llm", "LLM/chat completion tests"),
        ("agent", "Agent framework tests"),
        ("mcp", "MCP client/server tests"),
        ("embeddings", "Embeddings provider tests"),
        ("retriever", "Retriever/RAG tests"),
        ("callbacks", "Callback system tests"),
        ("streaming", "Streaming response tests"),
        ("imagegen", "Image generation tests"),
        ("multimodal", "TTS/STT/realtime voice tests"),
        ("memory", "Memory/session management tests"),
        ("slow", "Slow-running tests"),
    ]:
        config.addinivalue_line("markers", f"{name}: {desc}")


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture
def mock_env_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only OpenAI key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)


@pytest.fixture
def mock_env_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only Anthropic key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)


@pytest.fixture
def mock_env_google(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only Google key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)


@pytest.fixture
def mock_env_all_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with all major provider keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")


@pytest.fixture
def mock_env_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with no API keys."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


# =============================================================================
# CALLBACK FIXTURES
# =============================================================================


class TrackingCallbacks:
    """Callbacks that track all events for testing.

    This is a reusable callback implementation that records all events
    for later assertion.

    Usage:
        callbacks, events = tracking_callbacks
        llm = LLM(callbacks=callbacks)
        # ... run operations ...
        assert len(events) > 0
        assert events[0][0] == "llm_start"
    """

    def __init__(self):
        self.events: list[tuple[str, Any]] = []

    def on_llm_start(self, event) -> None:
        self.events.append(("llm_start", event))

    def on_llm_end(self, event) -> None:
        self.events.append(("llm_end", event))

    def on_llm_error(self, event) -> None:
        self.events.append(("llm_error", event))

    def on_llm_token(self, event) -> None:
        self.events.append(("llm_token", event))

    def on_tool_start(self, event) -> None:
        self.events.append(("tool_start", event))

    def on_tool_end(self, event) -> None:
        self.events.append(("tool_end", event))

    def on_tool_error(self, event) -> None:
        self.events.append(("tool_error", event))

    def on_agent_start(self, event) -> None:
        self.events.append(("agent_start", event))

    def on_agent_end(self, event) -> None:
        self.events.append(("agent_end", event))

    def on_agent_action(self, event) -> None:
        self.events.append(("agent_action", event))

    def clear(self):
        """Clear all recorded events."""
        self.events.clear()


@pytest.fixture
def tracking_callbacks() -> tuple[TrackingCallbacks, list]:
    """Create callbacks that track all events.

    Returns a tuple of (callbacks, events_list) for testing.
    """
    callbacks = TrackingCallbacks()
    return callbacks, callbacks.events


# =============================================================================
# MOCK EMBEDDINGS
# =============================================================================


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Create mock embeddings with standard interface.

    Provides a mock that implements both sync and async embedding methods.
    """
    mock = MagicMock()
    mock.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    mock.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return mock


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Provide sample embedding vectors for testing."""
    return [
        [1.0, 0.0, 0.0],  # X axis
        [0.0, 1.0, 0.0],  # Y axis
        [0.0, 0.0, 1.0],  # Z axis
        [0.5, 0.5, 0.0],  # XY diagonal
        [0.5, 0.0, 0.5],  # XZ diagonal
    ]


# =============================================================================
# RETRIEVER FIXTURES
# =============================================================================


@pytest.fixture
def memory_backend():
    """Create a clean memory backend for testing."""
    from ai_infra.retriever.backends.memory import MemoryBackend

    return MemoryBackend()


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """Provide sample documents for retriever testing."""
    return [
        {"text": "Python is a programming language.", "metadata": {"source": "wiki"}},
        {"text": "Machine learning uses algorithms.", "metadata": {"source": "docs"}},
        {
            "text": "RAG combines retrieval with generation.",
            "metadata": {"source": "paper"},
        },
        {
            "text": "Embeddings represent text as vectors.",
            "metadata": {"source": "blog"},
        },
    ]


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is test content for the retriever.\n")
        f.write("It spans multiple lines.\n")
        f.write("Each line contains different information.\n")
        f.flush()
        yield f.name
    # Cleanup after test
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_directory():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        for i in range(3):
            filepath = os.path.join(tmpdir, f"doc_{i}.txt")
            with open(filepath, "w") as f:
                f.write(f"Document {i} content.\n")
                f.write(f"This is paragraph 2 of document {i}.\n")
        yield tmpdir


# =============================================================================
# LLM RESPONSE FIXTURES
# =============================================================================


@pytest.fixture
def mock_chat_response() -> dict[str, Any]:
    """Provide a mock chat completion response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


@pytest.fixture
def mock_streaming_chunks() -> list[dict[str, Any]]:
    """Provide mock streaming response chunks."""
    return [
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
        {"choices": [{"delta": {"content": "!"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Provide sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What can you help me with?"},
    ]


# =============================================================================
# TOOL FIXTURES
# =============================================================================


@pytest.fixture
def simple_tool():
    """Create a simple tool for testing."""

    def greet(name: str) -> str:
        """Greet a person by name."""
        return f"Hello, {name}!"

    return greet


@pytest.fixture
def async_tool():
    """Create an async tool for testing."""

    async def fetch_data(url: str) -> str:
        """Fetch data from a URL (mock)."""
        return f"Data from {url}"

    return fetch_data


@pytest.fixture
def sample_tools() -> list:
    """Provide a list of sample tools for agent testing."""

    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            result = eval(expression)  # In tests only, not production!
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def get_weather(city: str) -> str:
        """Get weather for a city (mock)."""
        return f"Weather in {city}: Sunny, 72°F"

    return [calculator, get_weather]


# =============================================================================
# MCP FIXTURES
# =============================================================================


@pytest.fixture
def mock_mcp_server():
    """Create a mock MCP server for testing."""
    server = Mock()
    server.list_tools = AsyncMock(return_value=[])
    server.call_tool = AsyncMock(return_value={"result": "success"})
    server.list_resources = AsyncMock(return_value=[])
    server.read_resource = AsyncMock(return_value={"content": "resource content"})
    return server


@pytest.fixture
def sample_mcp_tool_definition() -> dict[str, Any]:
    """Provide a sample MCP tool definition."""
    return {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }


# =============================================================================
# IMAGE GENERATION FIXTURES
# =============================================================================


@pytest.fixture
def mock_generated_image() -> dict[str, Any]:
    """Provide a mock generated image response."""
    return {
        "url": "https://example.com/generated/image.png",
        "b64_json": None,
        "revised_prompt": "A beautiful sunset over mountains",
    }


# =============================================================================
# SESSION/MEMORY FIXTURES
# =============================================================================


@pytest.fixture
def sample_session_data() -> dict[str, Any]:
    """Provide sample session data for memory testing."""
    return {
        "session_id": "test-session-123",
        "user_id": "user-456",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "metadata": {"created_at": "2024-01-01T00:00:00Z"},
    }


# =============================================================================
# HELPER FIXTURES
# =============================================================================


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client for HTTP testing."""
    client = Mock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset any global state between tests.

    Add cleanup for any module-level singletons or caches here.
    """
    yield
    # Cleanup after each test if needed
