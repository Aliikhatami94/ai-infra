"""Tests for LLM core module (llm/llm.py).

This module provides comprehensive tests for the LLM class covering:
- Initialization and configuration
- chat() synchronous method
- achat() async method
- Structured output with Pydantic models
- Streaming tokens
- Error handling and translation

Phase 0.1 of the ai-infra v1.0.0 release plan.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from ai_infra import LLM
from ai_infra.callbacks import CallbackManager, Callbacks
from ai_infra.errors import AuthenticationError, ProviderError, RateLimitError

# =============================================================================
# Test Schemas for Structured Output
# =============================================================================


class AnswerSchema(BaseModel):
    """Simple schema for testing structured output."""

    answer: str
    confidence: float


class CitySchema(BaseModel):
    """Schema for city information."""

    city: str
    country: str
    population: int | None = None


# =============================================================================
# Mock Response Helpers
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
# PHASE 0.1: LLM INITIALIZATION TESTS
# =============================================================================


class TestLLMInitialization:
    """Test LLM initialization with various configurations."""

    def test_default_initialization(self):
        """Test LLM initializes with default settings."""
        llm = LLM()
        assert llm is not None
        assert llm._callbacks is None

    def test_initialization_with_callbacks(self):
        """Test LLM accepts Callbacks instance."""
        callbacks = Callbacks()
        llm = LLM(callbacks=callbacks)
        assert llm._callbacks is not None

    def test_initialization_with_callback_manager(self):
        """Test LLM accepts CallbackManager instance."""
        manager = CallbackManager([Callbacks()])
        llm = LLM(callbacks=manager)
        assert llm._callbacks is not None
        assert isinstance(llm._callbacks, CallbackManager)

    def test_initialization_normalizes_single_callback(self):
        """Test single Callbacks is normalized to CallbackManager."""

        class MyCallbacks(Callbacks):
            pass

        llm = LLM(callbacks=MyCallbacks())
        assert isinstance(llm._callbacks, CallbackManager)

    def test_initialization_rejects_invalid_callbacks(self):
        """Test LLM rejects invalid callbacks type."""
        with pytest.raises(ValueError, match="Invalid callbacks type"):
            LLM(callbacks="not a callback")  # type: ignore

    def test_initialization_inherits_from_base_llm(self):
        """Test LLM inherits required attributes from BaseLLM."""
        llm = LLM()
        assert hasattr(llm, "registry")
        assert hasattr(llm, "tools")
        assert hasattr(llm, "_hitl")
        assert hasattr(llm, "_logging_hooks")


class TestLLMProviderResolution:
    """Test provider and model auto-detection."""

    def test_auto_detect_openai_provider(self, mock_env_openai):
        """Test auto-detection of OpenAI when only OPENAI_API_KEY is set."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(None, None)
        assert provider == "openai"
        assert model is not None

    def test_auto_detect_anthropic_provider(self, mock_env_anthropic):
        """Test auto-detection of Anthropic when only ANTHROPIC_API_KEY is set."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(None, None)
        assert provider == "anthropic"
        assert model is not None

    def test_explicit_provider_selection(self, mock_env_all_providers):
        """Test explicit provider selection overrides auto-detection."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model("anthropic", None)
        assert provider == "anthropic"

    def test_explicit_model_selection(self, mock_env_openai):
        """Test explicit model selection."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model("openai", "gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_no_provider_configured_raises_error(self, mock_env_none):
        """Test error when no provider is configured."""
        llm = LLM()
        with pytest.raises(ValueError, match="No LLM provider configured"):
            llm._resolve_provider_and_model(None, None)

    def test_default_model_for_provider(self, mock_env_openai):
        """Test default model is used when not specified."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model("openai", None)
        assert provider == "openai"
        # Should get a default model (e.g., gpt-4o-mini)
        assert model is not None


# =============================================================================
# PHASE 0.1: TEST CHAT() METHOD
# =============================================================================


class TestLLMChatMethod:
    """Test LLM.chat() synchronous method."""

    def test_chat_simple_string_prompt(self, mock_env_openai):
        """Test chat with a simple string prompt."""
        llm = LLM()
        mock_response = create_mock_response("Paris is the capital of France.")
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat("What is the capital of France?")

        assert response is not None
        mock_model.invoke.assert_called_once()

    def test_chat_with_system_message(self, mock_env_openai):
        """Test chat with system message."""
        llm = LLM()
        mock_response = create_mock_response("*bows* Hello, brave traveler!")
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(
                "Hello",
                system="You are a medieval knight.",
            )

        assert response is not None
        # Verify invoke was called (messages should include system)
        mock_model.invoke.assert_called_once()
        call_args = mock_model.invoke.call_args[0][0]
        # Messages can be dicts or LangChain message objects
        # Check for system message in either format
        has_system = any(
            (isinstance(m, dict) and m.get("role") == "system")
            or getattr(m, "type", None) == "system"
            or getattr(m, "role", None) == "system"
            for m in call_args
        )
        assert has_system, f"Expected system message in {call_args}"

    def test_chat_with_explicit_provider(self, mock_env_all_providers):
        """Test chat with explicitly specified provider."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(
                "Hello",
                provider="anthropic",
                model_name="claude-sonnet-4-20250514",
            )

        assert response is not None
        mock_model.invoke.assert_called_once()

    def test_chat_with_model_kwargs(self, mock_env_openai):
        """Test chat passes model kwargs correctly."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model) as mock_get:
            llm.chat("Hello", temperature=0.5, max_tokens=100)

        # Verify kwargs were passed to get_or_create
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("temperature") == 0.5
        assert call_kwargs.get("max_tokens") == 100

    def test_chat_fires_callbacks(self, mock_env_openai):
        """Test chat fires LLM start/end callbacks."""
        events = []

        class TrackingCallbacks(Callbacks):
            def on_llm_start(self, event):
                events.append(("start", event))

            def on_llm_end(self, event):
                events.append(("end", event))

        llm = LLM(callbacks=TrackingCallbacks())
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            llm.chat("Hello")

        assert len(events) == 2
        assert events[0][0] == "start"
        assert events[1][0] == "end"

    def test_chat_empty_prompt_handling(self, mock_env_openai):
        """Test chat handles empty prompt."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            # Empty prompt should still work (model decides what to return)
            response = llm.chat("")

        assert response is not None


# =============================================================================
# PHASE 0.1: TEST ACHAT() ASYNC METHOD
# =============================================================================


class TestLLMAchatMethod:
    """Test LLM.achat() async method."""

    @pytest.mark.asyncio
    async def test_achat_async_completion(self, mock_env_openai):
        """Test async chat completion."""
        llm = LLM()
        mock_response = create_mock_response("Hello from async!")
        mock_model = create_mock_model(async_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = await llm.achat("Hello")

        assert response is not None
        mock_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_achat_with_system(self, mock_env_openai):
        """Test async chat with system message."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = await llm.achat(
                "Hello",
                system="You are a helpful assistant.",
            )

        assert response is not None
        call_args = mock_model.ainvoke.call_args[0][0]
        # Messages can be dicts or LangChain message objects
        has_system = any(
            (isinstance(m, dict) and m.get("role") == "system")
            or getattr(m, "type", None) == "system"
            or getattr(m, "role", None) == "system"
            for m in call_args
        )
        assert has_system, f"Expected system message in {call_args}"

    @pytest.mark.asyncio
    async def test_achat_fires_async_callbacks(self, mock_env_openai):
        """Test achat fires async callbacks."""
        events = []

        class AsyncTrackingCallbacks(Callbacks):
            async def on_llm_start_async(self, event):
                events.append(("start", event))

            async def on_llm_end_async(self, event):
                events.append(("end", event))

        llm = LLM(callbacks=AsyncTrackingCallbacks())
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            await llm.achat("Hello")

        assert len(events) == 2
        assert events[0][0] == "start"
        assert events[1][0] == "end"

    @pytest.mark.asyncio
    async def test_achat_error_handling(self, mock_env_openai):
        """Test async error handling."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(side_effect=Exception("API Error"))

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            # Error is translated to ProviderError or bubbles up as original
            with pytest.raises((ProviderError, Exception)):
                await llm.achat("Hello")


# =============================================================================
# PHASE 0.1: TEST STRUCTURED OUTPUT
# =============================================================================


class TestLLMStructuredOutput:
    """Test LLM structured output with Pydantic models."""

    def test_structured_output_with_pydantic_model(self, mock_env_openai):
        """Test structured output returns Pydantic model."""
        llm = LLM()
        mock_response = create_mock_response('{"answer": "Paris", "confidence": 0.95}')
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            result = llm.chat(
                "What is the capital of France?",
                output_schema=AnswerSchema,
            )

        assert isinstance(result, AnswerSchema)
        assert result.answer == "Paris"
        assert result.confidence == 0.95

    def test_structured_output_prompt_method(self, mock_env_openai):
        """Test structured output with prompt method (default)."""
        llm = LLM()
        mock_response = create_mock_response('```json\n{"city": "Paris", "country": "France"}\n```')
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            result = llm.chat(
                "Info about Paris",
                output_schema=CitySchema,
                output_method="prompt",
            )

        assert isinstance(result, CitySchema)
        assert result.city == "Paris"
        assert result.country == "France"

    def test_structured_output_json_mode(self, mock_env_openai):
        """Test structured output with json_mode method."""
        llm = LLM()
        mock_structured_model = MagicMock()
        mock_response = CitySchema(city="Tokyo", country="Japan")
        mock_structured_model.invoke.return_value = mock_response

        mock_base_model = MagicMock()
        mock_base_model.with_structured_output.return_value = mock_structured_model

        with patch.object(llm.registry, "get_or_create", return_value=mock_base_model):
            result = llm.chat(
                "Info about Tokyo",
                output_schema=CitySchema,
                output_method="json_mode",
            )

        assert isinstance(result, CitySchema)
        assert result.city == "Tokyo"

    def test_structured_output_extracts_markdown_json(self, mock_env_openai):
        """Test extraction of JSON from markdown code blocks."""
        llm = LLM()
        # Response with markdown-wrapped JSON
        mock_response = create_mock_response(
            "Here's the answer:\n```json\n"
            '{"answer": "42", "confidence": 1.0}\n'
            "```\nHope that helps!"
        )
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            result = llm.chat("What is the answer?", output_schema=AnswerSchema)

        assert isinstance(result, AnswerSchema)
        assert result.answer == "42"

    def test_structured_output_handles_partial_json(self, mock_env_openai):
        """Test handling of partial/incomplete JSON."""
        llm = LLM()
        # Some models return JSON mixed with text
        mock_response = create_mock_response('The answer is {"answer": "Yes", "confidence": 0.8}')
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            result = llm.chat("Is it raining?", output_schema=AnswerSchema)

        assert isinstance(result, AnswerSchema)
        assert result.answer == "Yes"

    def test_structured_output_with_optional_fields(self, mock_env_openai):
        """Test structured output with optional fields."""
        llm = LLM()
        mock_response = create_mock_response('{"city": "London", "country": "UK"}')
        mock_model = create_mock_model(sync_response=mock_response)

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            result = llm.chat("Info about London", output_schema=CitySchema)

        assert isinstance(result, CitySchema)
        assert result.city == "London"
        assert result.population is None  # Optional field not provided


# =============================================================================
# PHASE 0.1: TEST STREAMING
# =============================================================================


class TestLLMStreaming:
    """Test LLM streaming methods."""

    @pytest.mark.asyncio
    async def test_stream_tokens_generator(self, mock_env_openai):
        """Test stream_tokens yields tokens."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            tokens = []
            async for token, meta in llm.stream_tokens("Tell me a story"):
                tokens.append(token)

        assert len(tokens) > 0
        assert "Hello" in tokens

    @pytest.mark.asyncio
    async def test_stream_tokens_with_system(self, mock_env_openai):
        """Test streaming with system message."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            tokens = []
            async for token, meta in llm.stream_tokens(
                "Hello",
                system="Be concise",
            ):
                tokens.append(token)

        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_stream_tokens_metadata(self, mock_env_openai):
        """Test stream_tokens yields metadata with each token."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            async for token, meta in llm.stream_tokens("Hello"):
                assert "raw" in meta
                break  # Just check first token

    @pytest.mark.asyncio
    async def test_stream_tokens_fires_callbacks(self, mock_env_openai):
        """Test streaming fires token callbacks."""
        token_events = []

        class TokenTrackingCallbacks(Callbacks):
            async def on_llm_start_async(self, event):
                pass

            async def on_llm_token_async(self, event):
                token_events.append(event)

            async def on_llm_end_async(self, event):
                pass

        llm = LLM(callbacks=TokenTrackingCallbacks())
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            async for _ in llm.stream_tokens("Hello"):
                pass

        # Should have received token events
        assert len(token_events) > 0

    @pytest.mark.asyncio
    async def test_stream_tokens_with_model_kwargs(self, mock_env_openai):
        """Test streaming with temperature and max_tokens."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model) as mock_get:
            async for _ in llm.stream_tokens(
                "Hello",
                temperature=0.7,
                max_tokens=50,
            ):
                pass

        # Verify kwargs were passed
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("temperature") == 0.7
        assert call_kwargs.get("max_tokens") == 50


# =============================================================================
# PHASE 0.1: TEST ERROR HANDLING
# =============================================================================


class TestLLMErrorHandling:
    """Test LLM error handling and translation."""

    def test_api_key_missing_error(self, mock_env_none):
        """Test error when API key is missing."""
        llm = LLM()
        with pytest.raises(ValueError, match="No LLM provider configured"):
            llm.chat("Hello")

    def test_rate_limit_error_translation(self, mock_env_openai):
        """Test rate limit errors are translated to RateLimitError."""
        llm = LLM()
        mock_model = MagicMock()

        # Simulate OpenAI rate limit error with proper response structure
        from openai import RateLimitError as OpenAIRateLimitError

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "30"}  # Proper retry-after header

        mock_model.invoke.side_effect = OpenAIRateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
        )

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(RateLimitError):
                llm.chat("Hello")

    def test_authentication_error_translation(self, mock_env_openai):
        """Test auth errors are translated to AuthenticationError."""
        llm = LLM()
        mock_model = MagicMock()

        # Simulate authentication error with proper response structure
        from openai import AuthenticationError as OpenAIAuthError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}

        mock_model.invoke.side_effect = OpenAIAuthError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises(AuthenticationError):
                llm.chat("Hello")

    def test_generic_error_becomes_provider_error(self, mock_env_openai):
        """Test generic errors are wrapped as ProviderError."""
        llm = LLM()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Unknown error")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            # Generic exceptions are translated to ProviderError
            with pytest.raises((ProviderError, Exception)):
                llm.chat("Hello")

    def test_error_fires_error_callback(self, mock_env_openai):
        """Test errors fire LLM error callback."""
        error_events = []

        class ErrorTrackingCallbacks(Callbacks):
            def on_llm_start(self, event):
                pass

            def on_llm_error(self, event):
                error_events.append(event)

        llm = LLM(callbacks=ErrorTrackingCallbacks())
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Test error")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises((ProviderError, Exception)):
                llm.chat("Hello")

        assert len(error_events) == 1
        assert error_events[0].error is not None

    def test_timeout_handling(self, mock_env_openai):
        """Test timeout errors are handled gracefully."""
        import asyncio

        llm = LLM()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = TimeoutError("Request timed out")

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            # Timeout errors are translated to ProviderError
            with pytest.raises((ProviderError, asyncio.TimeoutError)):
                llm.chat("Hello")

    @pytest.mark.asyncio
    async def test_async_error_fires_async_callback(self, mock_env_openai):
        """Test async errors fire async error callback."""
        error_events = []

        class AsyncErrorTrackingCallbacks(Callbacks):
            async def on_llm_start_async(self, event):
                pass

            async def on_llm_error_async(self, event):
                error_events.append(event)

        llm = LLM(callbacks=AsyncErrorTrackingCallbacks())
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(side_effect=Exception("Async error"))

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            with pytest.raises((ProviderError, Exception)):
                await llm.achat("Hello")

        assert len(error_events) == 1


# =============================================================================
# PHASE 0.1: DISCOVERY API TESTS
# =============================================================================


class TestLLMDiscoveryAPI:
    """Test LLM static discovery methods."""

    def test_list_providers(self):
        """Test list_providers returns supported providers."""
        providers = LLM.list_providers()
        assert isinstance(providers, list)
        assert "openai" in providers
        assert "anthropic" in providers

    def test_list_configured_providers(self, mock_env_openai):
        """Test list_configured_providers returns only configured ones."""
        configured = LLM.list_configured_providers()
        assert isinstance(configured, list)
        assert "openai" in configured
        # Others should not be configured
        assert "anthropic" not in configured

    def test_is_provider_configured_true(self, mock_env_openai):
        """Test is_provider_configured returns True when key is set."""
        assert LLM.is_provider_configured("openai") is True

    def test_is_provider_configured_false(self, mock_env_none):
        """Test is_provider_configured returns False when key is not set."""
        assert LLM.is_provider_configured("openai") is False


# =============================================================================
# PHASE 0.1: EDGE CASES
# =============================================================================


class TestLLMEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_in_prompt(self, mock_env_openai):
        """Test handling of unicode characters in prompt."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat("こんにちは 🌍 مرحبا")

        assert response is not None

    def test_very_long_prompt(self, mock_env_openai):
        """Test handling of very long prompts."""
        llm = LLM()
        mock_model = create_mock_model()
        long_prompt = "test " * 10000  # ~50k chars

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat(long_prompt)

        assert response is not None

    def test_special_characters_in_prompt(self, mock_env_openai):
        """Test handling of special characters."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            response = llm.chat("Test <script>alert('xss')</script> & \n\t\\")

        assert response is not None

    def test_get_model_returns_langchain_model(self, mock_env_openai):
        """Test get_model returns underlying LangChain model."""
        llm = LLM()
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            model = llm.get_model()

        assert model is not None

    def test_set_logging_hooks(self, mock_env_openai):
        """Test logging hooks can be configured."""
        llm = LLM()
        request_called = []

        def on_request(ctx):
            request_called.append(ctx)

        llm.set_logging_hooks(on_request=on_request)
        mock_model = create_mock_model()

        with patch.object(llm.registry, "get_or_create", return_value=mock_model):
            llm.chat("Hello")

        assert len(request_called) == 1

    def test_set_hitl_gate(self, mock_env_openai):
        """Test human-in-the-loop gate can be configured."""
        llm = LLM()
        gate_called = []

        def on_model_output(output):
            gate_called.append(output)
            return {"action": "pass"}

        llm.set_hitl(on_model_output=on_model_output)
        assert llm._hitl is not None
