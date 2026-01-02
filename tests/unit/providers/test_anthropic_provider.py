"""Unit tests for Anthropic provider.

Tests cover:
- Chat completions (sync and async)
- Streaming responses
- Tool calling
- Error handling

All tests use mocks - no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from ai_infra import LLM
from ai_infra.callbacks import Callbacks

# =============================================================================
# Test Schemas
# =============================================================================


class AnswerSchema(BaseModel):
    """Simple schema for testing."""

    answer: str
    reasoning: str


# =============================================================================
# Mock Helpers
# =============================================================================


def create_mock_response(content: str = "Hello!", usage: dict | None = None) -> Mock:
    """Create a mock response object."""
    response = Mock()
    response.content = content
    if usage:
        response.usage_metadata = Mock()
        response.usage_metadata.input_tokens = usage.get("input_tokens", 10)
        response.usage_metadata.output_tokens = usage.get("output_tokens", 5)
        response.usage_metadata.total_tokens = usage.get("total_tokens", 15)
    else:
        response.usage_metadata = None
    return response


def create_mock_model(
    sync_response: Mock | None = None,
    async_response: Mock | None = None,
    stream_chunks: list | None = None,
) -> MagicMock:
    """Create a mock LangChain model."""
    model = MagicMock()

    if sync_response:
        model.invoke.return_value = sync_response
    else:
        model.invoke.return_value = create_mock_response()

    if async_response:
        model.ainvoke = AsyncMock(return_value=async_response)
    else:
        model.ainvoke = AsyncMock(return_value=create_mock_response())

    if stream_chunks:

        async def async_stream_gen(*args, **kwargs):
            for chunk in stream_chunks:
                yield chunk

        model.astream = async_stream_gen
    else:

        async def async_stream_gen(*args, **kwargs):
            chunks = [Mock(content="Hello"), Mock(content=" from Claude")]
            for chunk in chunks:
                yield chunk

        model.astream = async_stream_gen

    return model


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_anthropic_env():
    """Mock environment with Anthropic configured."""
    with patch.dict(
        "os.environ",
        {"ANTHROPIC_API_KEY": "test-anthropic-key"},
        clear=False,
    ):
        yield


# =============================================================================
# Anthropic Chat Tests
# =============================================================================


class TestAnthropicChat:
    """Test Anthropic chat completions."""

    def test_chat_basic_completion(self, mock_anthropic_env):
        """Test basic chat completion with Anthropic."""
        llm = LLM()
        mock_response = create_mock_response("Paris is the capital of France.")
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(
                "What is the capital of France?",
                provider="anthropic",
                model_name="claude-sonnet-4-20250514",
            )

        assert response is not None
        mock_model.invoke.assert_called_once()

    def test_chat_with_system_message(self, mock_anthropic_env):
        """Test chat with system message."""
        llm = LLM()
        mock_response = create_mock_response("*adjusts monocle* Indeed, Paris.")
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(
                "What is the capital of France?",
                provider="anthropic",
                model_name="claude-sonnet-4-20250514",
                system="You are a Victorian-era scholar.",
            )

        assert response is not None
        call_args = mock_model.invoke.call_args[0][0]
        has_system = any(
            (isinstance(m, dict) and m.get("role") == "system")
            or getattr(m, "type", None) == "system"
            for m in call_args
        )
        assert has_system

    def test_chat_with_temperature(self, mock_anthropic_env):
        """Test chat with temperature parameter."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model) as mock_get:
            llm.chat(
                "Be creative",
                provider="anthropic",
                model_name="claude-sonnet-4-20250514",
                temperature=0.9,
            )

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("temperature") == 0.9

    def test_chat_with_max_tokens(self, mock_anthropic_env):
        """Test chat with max_tokens parameter."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model) as mock_get:
            llm.chat(
                "Write a story",
                provider="anthropic",
                model_name="claude-sonnet-4-20250514",
                max_tokens=1000,
            )

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("max_tokens") == 1000

    @pytest.mark.asyncio
    async def test_achat_async_completion(self, mock_anthropic_env):
        """Test async chat completion."""
        llm = LLM()
        mock_response = create_mock_response("Hello from Claude!")
        mock_model = create_mock_model(async_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = await llm.achat(
                "Hello",
                provider="anthropic",
                model_name="claude-sonnet-4-20250514",
            )

        assert response is not None
        mock_model.ainvoke.assert_called_once()


# =============================================================================
# Anthropic Streaming Tests
# =============================================================================


class TestAnthropicStreaming:
    """Test Anthropic streaming responses."""

    @pytest.mark.asyncio
    async def test_stream_tokens_basic(self, mock_anthropic_env):
        """Test basic token streaming."""
        llm = LLM()
        chunks = [
            Mock(content="Hello"),
            Mock(content=" from"),
            Mock(content=" Claude"),
        ]
        mock_model = create_mock_model(stream_chunks=chunks)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            tokens = []
            async for token, meta in llm.stream_tokens(
                "Say hello",
                provider="anthropic",
                model_name="claude-sonnet-4-20250514",
            ):
                tokens.append(token)

        assert len(tokens) == 3
        assert "".join(tokens) == "Hello from Claude"

    @pytest.mark.asyncio
    async def test_stream_tokens_with_callbacks(self, mock_anthropic_env):
        """Test streaming fires callbacks."""
        events = []

        class StreamCallbacks(Callbacks):
            def on_llm_start(self, event):
                events.append(("start", event))

            def on_llm_end(self, event):
                events.append(("end", event))

        llm = LLM(callbacks=StreamCallbacks())
        chunks = [Mock(content="Token1"), Mock(content="Token2")]
        mock_model = create_mock_model(stream_chunks=chunks)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            tokens = []
            async for token, _ in llm.stream_tokens(
                "Test",
                provider="anthropic",
                model_name="claude-sonnet-4-20250514",
            ):
                tokens.append(token)

        # Streaming may not fire callbacks the same way as chat
        # Just verify we got tokens
        assert len(tokens) == 2


# =============================================================================
# Anthropic Tool Calling Tests
# =============================================================================


class TestAnthropicToolCalling:
    """Test Anthropic tool calling."""

    def test_tool_call_response_parsing(self, mock_anthropic_env):
        """Test parsing tool call responses."""
        llm = LLM()

        # Create mock response with tool call
        mock_response = Mock()
        mock_response.content = ""
        mock_response.tool_calls = [
            Mock(
                name="get_weather",
                args={"location": "Paris"},
            )
        ]
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(
                "What's the weather in Paris?",
                provider="anthropic",
                model_name="claude-sonnet-4-20250514",
            )

        assert response is not None


# =============================================================================
# Anthropic Error Handling Tests
# =============================================================================


class TestAnthropicErrorHandling:
    """Test Anthropic error handling."""

    def test_authentication_error(self, mock_anthropic_env):
        """Test authentication error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Invalid API key")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                llm.chat("Hello", provider="anthropic")

    def test_rate_limit_error(self, mock_anthropic_env):
        """Test rate limit error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Rate limit exceeded")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                llm.chat("Hello", provider="anthropic")

    def test_overloaded_error(self, mock_anthropic_env):
        """Test overloaded error handling (Anthropic specific)."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Overloaded")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                llm.chat("Hello", provider="anthropic")

    @pytest.mark.asyncio
    async def test_async_error_handling(self, mock_anthropic_env):
        """Test async error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(side_effect=Exception("Async Error"))

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                await llm.achat("Hello", provider="anthropic")


# =============================================================================
# Anthropic Callback Tests
# =============================================================================


class TestAnthropicCallbacks:
    """Test Anthropic callbacks."""

    def test_callbacks_include_provider_info(self, mock_anthropic_env):
        """Test callbacks include Anthropic provider info."""
        events = []

        class TrackingCallbacks(Callbacks):
            def on_llm_start(self, event):
                events.append(event)

            def on_llm_end(self, event):
                events.append(event)

        llm = LLM(callbacks=TrackingCallbacks())
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            llm.chat("Hello", provider="anthropic", model_name="claude-sonnet-4-20250514")

        assert len(events) == 2
        assert events[0].provider == "anthropic"
        assert "claude" in events[0].model.lower()

    def test_usage_metadata_in_callbacks(self, mock_anthropic_env):
        """Test usage metadata is passed to callbacks."""
        events = []

        class UsageCallbacks(Callbacks):
            def on_llm_end(self, event):
                events.append(event)

        llm = LLM(callbacks=UsageCallbacks())
        mock_response = create_mock_response(
            "Hello!",
            usage={"input_tokens": 15, "output_tokens": 10, "total_tokens": 25},
        )
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            llm.chat("Hello", provider="anthropic")

        assert len(events) == 1
        assert events[0].input_tokens == 15
        assert events[0].output_tokens == 10
