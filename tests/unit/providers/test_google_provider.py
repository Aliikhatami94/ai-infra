"""Unit tests for Google (Gemini) provider.

Tests cover:
- Chat completions (sync and async)
- Streaming responses
- Structured output
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
    confidence: float


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
            chunks = [Mock(content="Hello"), Mock(content=" from Gemini")]
            for chunk in chunks:
                yield chunk

        model.astream = async_stream_gen

    return model


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_google_env():
    """Mock environment with Google configured."""
    with patch.dict(
        "os.environ",
        {"GOOGLE_API_KEY": "test-google-key"},
        clear=False,
    ):
        yield


# =============================================================================
# Google Chat Tests
# =============================================================================


class TestGoogleChat:
    """Test Google/Gemini chat completions."""

    def test_chat_basic_completion(self, mock_google_env):
        """Test basic chat completion with Google."""
        llm = LLM()
        mock_response = create_mock_response("Paris is the capital of France.")
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(
                "What is the capital of France?",
                provider="google_genai",
                model_name="gemini-2.0-flash",
            )

        assert response is not None
        mock_model.invoke.assert_called_once()

    def test_chat_with_system_message(self, mock_google_env):
        """Test chat with system message."""
        llm = LLM()
        mock_response = create_mock_response("In the style of a poet: Paris!")
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(
                "What is the capital of France?",
                provider="google_genai",
                model_name="gemini-2.0-flash",
                system="You are a poet. Answer in verse.",
            )

        assert response is not None
        call_args = mock_model.invoke.call_args[0][0]
        has_system = any(
            (isinstance(m, dict) and m.get("role") == "system")
            or getattr(m, "type", None) == "system"
            for m in call_args
        )
        assert has_system

    def test_chat_with_temperature(self, mock_google_env):
        """Test chat with temperature parameter."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model) as mock_get:
            llm.chat(
                "Be creative",
                provider="google_genai",
                model_name="gemini-2.0-flash",
                temperature=0.8,
            )

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("temperature") == 0.8

    @pytest.mark.asyncio
    async def test_achat_async_completion(self, mock_google_env):
        """Test async chat completion."""
        llm = LLM()
        mock_response = create_mock_response("Hello from Gemini!")
        mock_model = create_mock_model(async_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = await llm.achat(
                "Hello",
                provider="google_genai",
                model_name="gemini-2.0-flash",
            )

        assert response is not None
        mock_model.ainvoke.assert_called_once()


# =============================================================================
# Google Streaming Tests
# =============================================================================


class TestGoogleStreaming:
    """Test Google streaming responses."""

    @pytest.mark.asyncio
    async def test_stream_tokens_basic(self, mock_google_env):
        """Test basic token streaming."""
        llm = LLM()
        chunks = [
            Mock(content="Hello"),
            Mock(content=" from"),
            Mock(content=" Gemini"),
        ]
        mock_model = create_mock_model(stream_chunks=chunks)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            tokens = []
            async for token, meta in llm.stream_tokens(
                "Say hello",
                provider="google_genai",
                model_name="gemini-2.0-flash",
            ):
                tokens.append(token)

        assert len(tokens) == 3
        assert "".join(tokens) == "Hello from Gemini"

    @pytest.mark.asyncio
    async def test_stream_with_system(self, mock_google_env):
        """Test streaming with system message."""
        llm = LLM()
        chunks = [Mock(content="Greetings"), Mock(content=" human!")]
        mock_model = create_mock_model(stream_chunks=chunks)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            tokens = []
            async for token, meta in llm.stream_tokens(
                "Hello",
                provider="google_genai",
                model_name="gemini-2.0-flash",
                system="You are an alien.",
            ):
                tokens.append(token)

        assert len(tokens) == 2


# =============================================================================
# Google Structured Output Tests
# =============================================================================


class TestGoogleStructuredOutput:
    """Test Google structured output."""

    def test_structured_output_pydantic(self, mock_google_env):
        """Test structured output with Pydantic model."""
        llm = LLM()
        mock_response = create_mock_response('{"answer": "Paris", "confidence": 0.9}')
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            result = llm.chat(
                "What is the capital of France?",
                provider="google_genai",
                model_name="gemini-2.0-flash",
                output_schema=AnswerSchema,
            )

        assert result is not None


# =============================================================================
# Google Error Handling Tests
# =============================================================================


class TestGoogleErrorHandling:
    """Test Google error handling."""

    def test_quota_exceeded_error(self, mock_google_env):
        """Test quota exceeded error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Quota exceeded")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                llm.chat("Hello", provider="google_genai")

    def test_invalid_api_key_error(self, mock_google_env):
        """Test invalid API key error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Invalid API key")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                llm.chat("Hello", provider="google_genai")

    def test_safety_filter_error(self, mock_google_env):
        """Test safety filter blocking (Google specific)."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Content blocked by safety filters")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                llm.chat("Something blocked", provider="google_genai")

    @pytest.mark.asyncio
    async def test_async_error_handling(self, mock_google_env):
        """Test async error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(side_effect=Exception("Async Error"))

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                await llm.achat("Hello", provider="google_genai")


# =============================================================================
# Google Callback Tests
# =============================================================================


class TestGoogleCallbacks:
    """Test Google callbacks."""

    def test_callbacks_include_provider_info(self, mock_google_env):
        """Test callbacks include Google provider info."""
        events = []

        class TrackingCallbacks(Callbacks):
            def on_llm_start(self, event):
                events.append(event)

            def on_llm_end(self, event):
                events.append(event)

        llm = LLM(callbacks=TrackingCallbacks())
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            llm.chat("Hello", provider="google_genai", model_name="gemini-2.0-flash")

        assert len(events) == 2
        assert events[0].provider == "google_genai"
        assert "gemini" in events[0].model.lower()
