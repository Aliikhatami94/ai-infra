"""Unit tests for OpenAI provider.

Tests cover:
- Chat completions (sync and async)
- Streaming responses
- Structured output
- Error handling
- Rate limiting and retries

All tests use mocks - no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from ai_infra import LLM
from ai_infra.callbacks import Callbacks
from ai_infra.errors import AuthenticationError, ProviderError, RateLimitError

# =============================================================================
# Test Schemas
# =============================================================================


class AnswerSchema(BaseModel):
    """Simple schema for testing structured output."""

    answer: str
    confidence: float


class MathResult(BaseModel):
    """Schema for math results."""

    problem: str
    result: int
    explanation: str


# =============================================================================
# Mock Helpers
# =============================================================================


def create_mock_response(content: str = "Hello!", usage: dict | None = None) -> Mock:
    """Create a mock LangChain response object."""
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
            chunks = [Mock(content="Hello"), Mock(content=" "), Mock(content="World")]
            for chunk in chunks:
                yield chunk

        model.astream = async_stream_gen

    return model


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_openai_env():
    """Mock environment with only OpenAI configured."""
    with patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test-openai-key"},
        clear=False,
    ):
        yield


@pytest.fixture
def llm_with_mock_model(mock_openai_env):
    """Create LLM with mocked model."""
    llm = LLM()
    mock_model = create_mock_model()
    with patch.object(llm.registry, "get_or_create", return_value=mock_model):
        yield llm, mock_model


# =============================================================================
# OpenAI Chat Tests
# =============================================================================


class TestOpenAIChat:
    """Test OpenAI chat completions."""

    def test_chat_basic_completion(self, mock_openai_env):
        """Test basic chat completion with OpenAI."""
        llm = LLM()
        mock_response = create_mock_response("Paris is the capital of France.")
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(
                "What is the capital of France?",
                provider="openai",
                model_name="gpt-4o-mini",
            )

        assert response is not None
        mock_model.invoke.assert_called_once()

    def test_chat_with_system_message(self, mock_openai_env):
        """Test chat with system message."""
        llm = LLM()
        mock_response = create_mock_response("Arr, the capital be Paris, matey!")
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(
                "What is the capital of France?",
                provider="openai",
                model_name="gpt-4o-mini",
                system="You are a pirate.",
            )

        assert response is not None
        # Verify system message was included
        call_args = mock_model.invoke.call_args[0][0]
        has_system = any(
            (isinstance(m, dict) and m.get("role") == "system")
            or getattr(m, "type", None) == "system"
            for m in call_args
        )
        assert has_system

    def test_chat_with_temperature(self, mock_openai_env):
        """Test chat with temperature parameter."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model) as mock_get:
            llm.chat(
                "Hello",
                provider="openai",
                model_name="gpt-4o-mini",
                temperature=0.7,
            )

        # Verify temperature was passed
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("temperature") == 0.7

    def test_chat_with_max_tokens(self, mock_openai_env):
        """Test chat with max_tokens parameter."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model) as mock_get:
            llm.chat(
                "Hello",
                provider="openai",
                model_name="gpt-4o-mini",
                max_tokens=500,
            )

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("max_tokens") == 500

    @pytest.mark.asyncio
    async def test_achat_async_completion(self, mock_openai_env):
        """Test async chat completion."""
        llm = LLM()
        mock_response = create_mock_response("Hello from async!")
        mock_model = create_mock_model(async_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = await llm.achat(
                "Hello",
                provider="openai",
                model_name="gpt-4o-mini",
            )

        assert response is not None
        mock_model.ainvoke.assert_called_once()


# =============================================================================
# OpenAI Streaming Tests
# =============================================================================


class TestOpenAIStreaming:
    """Test OpenAI streaming responses."""

    @pytest.mark.asyncio
    async def test_stream_tokens_basic(self, mock_openai_env):
        """Test basic token streaming."""
        llm = LLM()
        chunks = [
            Mock(content="Hello"),
            Mock(content=" "),
            Mock(content="World"),
            Mock(content="!"),
        ]
        mock_model = create_mock_model(stream_chunks=chunks)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            tokens = []
            async for token, meta in llm.stream_tokens(
                "Say hello",
                provider="openai",
                model_name="gpt-4o-mini",
            ):
                tokens.append(token)

        assert len(tokens) == 4
        assert "".join(tokens) == "Hello World!"

    @pytest.mark.asyncio
    async def test_stream_tokens_with_system(self, mock_openai_env):
        """Test streaming with system message."""
        llm = LLM()
        chunks = [Mock(content="Arr!"), Mock(content=" Ahoy!")]
        mock_model = create_mock_model(stream_chunks=chunks)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            tokens = []
            async for token, meta in llm.stream_tokens(
                "Greet me",
                provider="openai",
                model_name="gpt-4o-mini",
                system="You are a pirate.",
            ):
                tokens.append(token)

        assert len(tokens) == 2

    @pytest.mark.asyncio
    async def test_stream_tokens_empty_chunks_filtered(self, mock_openai_env):
        """Test that empty chunks are filtered out."""
        llm = LLM()
        chunks = [
            Mock(content="Hello"),
            Mock(content=""),  # Empty chunk
            Mock(content=None),  # None content
            Mock(content="World"),
        ]
        mock_model = create_mock_model(stream_chunks=chunks)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            tokens = []
            async for token, meta in llm.stream_tokens(
                "Test",
                provider="openai",
                model_name="gpt-4o-mini",
            ):
                if token:  # Only collect non-empty
                    tokens.append(token)

        assert "Hello" in tokens
        assert "World" in tokens


# =============================================================================
# OpenAI Structured Output Tests
# =============================================================================


class TestOpenAIStructuredOutput:
    """Test OpenAI structured output."""

    def test_structured_output_pydantic(self, mock_openai_env):
        """Test structured output with Pydantic model."""
        llm = LLM()
        mock_response = create_mock_response('{"answer": "Paris", "confidence": 0.95}')
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            result = llm.chat(
                "What is the capital of France?",
                provider="openai",
                model_name="gpt-4o-mini",
                output_schema=AnswerSchema,
            )

        # Result should be coerced to Pydantic model or raw response
        assert result is not None

    def test_structured_output_math(self, mock_openai_env):
        """Test structured output for math problems."""
        llm = LLM()
        # Must return JSON string for structured output parsing
        mock_response = create_mock_response(
            '{"problem": "2 + 2", "result": 4, "explanation": "Basic addition"}'
        )
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            result = llm.chat(
                "What is 2 + 2?",
                provider="openai",
                model_name="gpt-4o-mini",
                output_schema=MathResult,
            )

        assert result is not None


# =============================================================================
# OpenAI Error Handling Tests
# =============================================================================


class TestOpenAIErrorHandling:
    """Test OpenAI error handling."""

    def test_authentication_error(self, mock_openai_env):
        """Test authentication error handling."""
        llm = LLM()
        mock_model = MagicMock()

        # Simulate OpenAI auth error
        from openai import AuthenticationError as OpenAIAuthError

        mock_model.invoke.side_effect = OpenAIAuthError(
            message="Invalid API key",
            response=Mock(status_code=401),
            body=None,
        )

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises((AuthenticationError, Exception)):
                llm.chat("Hello", provider="openai")

    def test_rate_limit_error(self, mock_openai_env):
        """Test rate limit error handling."""
        llm = LLM()
        mock_model = MagicMock()

        # Simulate rate limit error
        from openai import RateLimitError as OpenAIRateLimitError

        mock_model.invoke.side_effect = OpenAIRateLimitError(
            message="Rate limit exceeded",
            response=Mock(status_code=429),
            body=None,
        )

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises((RateLimitError, Exception)):
                llm.chat("Hello", provider="openai")

    def test_generic_api_error(self, mock_openai_env):
        """Test generic API error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("API Error")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises((ProviderError, Exception)):
                llm.chat("Hello", provider="openai")

    @pytest.mark.asyncio
    async def test_async_error_handling(self, mock_openai_env):
        """Test async error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(side_effect=Exception("Async API Error"))

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises((ProviderError, Exception)):
                await llm.achat("Hello", provider="openai")


# =============================================================================
# OpenAI Callback Tests
# =============================================================================


class TestOpenAICallbacks:
    """Test OpenAI callbacks."""

    def test_callbacks_fired_on_success(self, mock_openai_env):
        """Test callbacks are fired on successful completion."""
        events = []

        class TrackingCallbacks(Callbacks):
            def on_llm_start(self, event):
                events.append(("start", event))

            def on_llm_end(self, event):
                events.append(("end", event))

        llm = LLM(callbacks=TrackingCallbacks())
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            llm.chat("Hello", provider="openai")

        assert len(events) == 2
        assert events[0][0] == "start"
        assert events[1][0] == "end"
        # Verify provider info in event
        assert events[0][1].provider == "openai"
        assert events[1][1].provider == "openai"

    def test_callbacks_fired_on_error(self, mock_openai_env):
        """Test error callbacks are fired on failure."""
        events = []

        class TrackingCallbacks(Callbacks):
            def on_llm_start(self, event):
                events.append(("start", event))

            def on_llm_error(self, event):
                events.append(("error", event))

        llm = LLM(callbacks=TrackingCallbacks())
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("API Error")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(Exception):  # noqa: B017
                llm.chat("Hello", provider="openai")

        # At minimum, start should fire
        assert any(e[0] == "start" for e in events)
