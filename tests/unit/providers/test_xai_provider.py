"""Unit tests for xAI (Grok) provider.

Tests cover:
- Chat completions (sync and async)
- Streaming responses
- Error handling

All tests use mocks - no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from ai_infra import LLM
from ai_infra.callbacks import Callbacks

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
            chunks = [Mock(content="Hello"), Mock(content=" from Grok")]
            for chunk in chunks:
                yield chunk

        model.astream = async_stream_gen

    return model


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_xai_env():
    """Mock environment with xAI configured."""
    with patch.dict(
        "os.environ",
        {"XAI_API_KEY": "test-xai-key"},
        clear=False,
    ):
        yield


# =============================================================================
# xAI Chat Tests
# =============================================================================


class TestXAIChat:
    """Test xAI/Grok chat completions."""

    def test_chat_basic_completion(self, mock_xai_env):
        """Test basic chat completion with xAI."""
        llm = LLM()
        mock_response = create_mock_response("Paris is the capital of France.")
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(
                "What is the capital of France?",
                provider="xai",
                model_name="grok-beta",
            )

        assert response is not None
        mock_model.invoke.assert_called_once()

    def test_chat_with_system_message(self, mock_xai_env):
        """Test chat with system message."""
        llm = LLM()
        mock_response = create_mock_response("*sarcastic tone* It's Paris, obviously.")
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(
                "What is the capital of France?",
                provider="xai",
                model_name="grok-beta",
                system="You are sarcastic but helpful.",
            )

        assert response is not None
        call_args = mock_model.invoke.call_args[0][0]
        has_system = any(
            (isinstance(m, dict) and m.get("role") == "system")
            or getattr(m, "type", None) == "system"
            for m in call_args
        )
        assert has_system

    def test_chat_with_temperature(self, mock_xai_env):
        """Test chat with temperature parameter."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model) as mock_get:
            llm.chat(
                "Be creative",
                provider="xai",
                model_name="grok-beta",
                temperature=0.9,
            )

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("temperature") == 0.9

    @pytest.mark.asyncio
    async def test_achat_async_completion(self, mock_xai_env):
        """Test async chat completion."""
        llm = LLM()
        mock_response = create_mock_response("Hello from Grok!")
        mock_model = create_mock_model(async_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = await llm.achat(
                "Hello",
                provider="xai",
                model_name="grok-beta",
            )

        assert response is not None
        mock_model.ainvoke.assert_called_once()


# =============================================================================
# xAI Streaming Tests
# =============================================================================


class TestXAIStreaming:
    """Test xAI streaming responses."""

    @pytest.mark.asyncio
    async def test_stream_tokens_basic(self, mock_xai_env):
        """Test basic token streaming."""
        llm = LLM()
        chunks = [
            Mock(content="Hello"),
            Mock(content=" from"),
            Mock(content=" Grok"),
        ]
        mock_model = create_mock_model(stream_chunks=chunks)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            tokens = []
            async for token, meta in llm.stream_tokens(
                "Say hello",
                provider="xai",
                model_name="grok-beta",
            ):
                tokens.append(token)

        assert len(tokens) == 3
        assert "".join(tokens) == "Hello from Grok"

    @pytest.mark.asyncio
    async def test_stream_with_system(self, mock_xai_env):
        """Test streaming with system message."""
        llm = LLM()
        chunks = [Mock(content="Greetings"), Mock(content="!")]
        mock_model = create_mock_model(stream_chunks=chunks)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            tokens = []
            async for token, meta in llm.stream_tokens(
                "Hello",
                provider="xai",
                model_name="grok-beta",
                system="Be brief.",
            ):
                tokens.append(token)

        assert len(tokens) == 2


# =============================================================================
# xAI Error Handling Tests
# =============================================================================


class TestXAIErrorHandling:
    """Test xAI error handling."""

    def test_authentication_error(self, mock_xai_env):
        """Test authentication error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Invalid API key")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                llm.chat("Hello", provider="xai")

    def test_rate_limit_error(self, mock_xai_env):
        """Test rate limit error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Rate limit exceeded")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                llm.chat("Hello", provider="xai")

    @pytest.mark.asyncio
    async def test_async_error_handling(self, mock_xai_env):
        """Test async error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(side_effect=Exception("Async Error"))

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                await llm.achat("Hello", provider="xai")


# =============================================================================
# xAI Callback Tests
# =============================================================================


class TestXAICallbacks:
    """Test xAI callbacks."""

    def test_callbacks_include_provider_info(self, mock_xai_env):
        """Test callbacks include xAI provider info."""
        events = []

        class TrackingCallbacks(Callbacks):
            def on_llm_start(self, event):
                events.append(event)

            def on_llm_end(self, event):
                events.append(event)

        llm = LLM(callbacks=TrackingCallbacks())
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            llm.chat("Hello", provider="xai", model_name="grok-beta")

        assert len(events) == 2
        assert events[0].provider == "xai"
        assert "grok" in events[0].model.lower()
