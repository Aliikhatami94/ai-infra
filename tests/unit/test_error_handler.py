"""Tests for LLM error handling utilities."""

import pytest

from ai_infra.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from ai_infra.llm.utils.error_handler import (
    extract_retry_after,
    get_supported_kwargs,
    translate_provider_error,
    validate_kwargs,
    with_error_handling,
    with_error_handling_async,
)

# =============================================================================
# Test translate_provider_error
# =============================================================================


class TestTranslateProviderError:
    """Tests for translate_provider_error function."""

    def test_passes_through_existing_llm_error(self):
        """Should not re-wrap existing ai-infra errors."""
        original = AuthenticationError("test", provider="openai")
        result = translate_provider_error(original, provider="openai")
        assert result is original

    def test_translates_auth_error_by_status_code(self):
        """Should translate 401 to AuthenticationError."""

        class MockError(Exception):
            status_code = 401

        result = translate_provider_error(
            MockError("invalid key"),
            provider="openai",
        )
        assert isinstance(result, AuthenticationError)
        assert result.provider == "openai"

    def test_translates_rate_limit_by_status_code(self):
        """Should translate 429 to RateLimitError."""

        class MockError(Exception):
            status_code = 429

        result = translate_provider_error(
            MockError("too many requests"),
            provider="openai",
            model="gpt-4o",
        )
        assert isinstance(result, RateLimitError)
        assert result.provider == "openai"
        assert result.model == "gpt-4o"

    def test_translates_not_found_by_status_code(self):
        """Should translate 404 to ModelNotFoundError."""

        class MockError(Exception):
            status_code = 404

        result = translate_provider_error(
            MockError("model not found"),
            provider="openai",
            model="gpt-5-turbo",
        )
        assert isinstance(result, ModelNotFoundError)
        assert "gpt-5-turbo" in result.message

    def test_translates_auth_error_by_message_pattern(self):
        """Should translate auth errors by message pattern."""
        error = Exception("invalid api key provided")
        result = translate_provider_error(error, provider="openai")
        assert isinstance(result, AuthenticationError)

    def test_translates_rate_limit_by_message_pattern(self):
        """Should translate rate limit errors by message pattern."""
        error = Exception("rate limit exceeded, try again later")
        result = translate_provider_error(error, provider="openai")
        assert isinstance(result, RateLimitError)

    def test_translates_context_length_by_message_pattern(self):
        """Should translate context length errors by message pattern."""
        error = Exception("maximum context length exceeded, reduce input")
        result = translate_provider_error(error, provider="openai")
        assert isinstance(result, ContextLengthError)

    def test_translates_content_filter_by_message_pattern(self):
        """Should translate content filter errors by message pattern."""
        error = Exception("content blocked by safety filter")
        result = translate_provider_error(error, provider="anthropic")
        assert isinstance(result, ContentFilterError)

    def test_anthropic_specific_patterns(self):
        """Should match Anthropic-specific error patterns."""
        error = Exception("invalid x-api-key header")
        result = translate_provider_error(error, provider="anthropic")
        assert isinstance(result, AuthenticationError)

    def test_google_specific_patterns(self):
        """Should match Google-specific error patterns."""
        error = Exception("quota exceeded for API")
        result = translate_provider_error(error, provider="google_genai")
        assert isinstance(result, RateLimitError)

    def test_extracts_token_counts_from_context_error(self):
        """Should extract token counts from context length errors."""
        error = Exception("maximum context is 8192 tokens, requested 10000")
        result = translate_provider_error(error, provider="openai")
        assert isinstance(result, ContextLengthError)
        assert result.max_tokens == 8192
        assert result.requested_tokens == 10000

    def test_returns_generic_provider_error_for_other_status_codes(self):
        """Should return ProviderError for other status codes."""

        class MockError(Exception):
            status_code = 500

        result = translate_provider_error(
            MockError("internal server error"),
            provider="openai",
        )
        assert isinstance(result, ProviderError)
        assert result.status_code == 500

    def test_returns_original_for_unknown_errors(self):
        """Should return original error if not recognized."""
        error = Exception("some random error")
        result = translate_provider_error(error, provider="openai")
        assert result is error


# =============================================================================
# Test extract_retry_after
# =============================================================================


class TestExtractRetryAfter:
    """Tests for extract_retry_after function."""

    def test_extracts_from_retry_after_attribute(self):
        """Should extract from retry_after attribute."""

        class MockError(Exception):
            retry_after = 30

        result = extract_retry_after(MockError("rate limited"))
        assert result == 30.0

    def test_extracts_from_headers(self):
        """Should extract from headers."""

        class MockResponse:
            headers = {"Retry-After": "60"}

        class MockError(Exception):
            response = MockResponse()

        result = extract_retry_after(MockError("rate limited"))
        assert result == 60.0

    def test_extracts_from_message(self):
        """Should extract from error message."""
        error = Exception("rate limited, retry after 45 seconds")
        result = extract_retry_after(error)
        assert result == 45.0

    def test_returns_none_if_not_found(self):
        """Should return None if no retry-after info."""
        error = Exception("some error")
        result = extract_retry_after(error)
        assert result is None


# =============================================================================
# Test with_error_handling decorator
# =============================================================================


class TestWithErrorHandling:
    """Tests for with_error_handling decorator."""

    def test_decorator_passes_through_success(self):
        """Should pass through successful results."""

        @with_error_handling(provider="openai", model="gpt-4o")
        def my_func():
            return "success"

        result = my_func()
        assert result == "success"

    def test_decorator_translates_errors(self):
        """Should translate errors."""

        class MockError(Exception):
            status_code = 401

        @with_error_handling(provider="openai", model="gpt-4o")
        def my_func():
            raise MockError("bad key")

        with pytest.raises(AuthenticationError):
            my_func()

    def test_decorator_preserves_original_as_cause(self):
        """Should preserve original error as __cause__."""

        @with_error_handling(provider="openai", model="gpt-4o")
        def my_func():
            raise Exception("rate limit exceeded")

        with pytest.raises(RateLimitError) as exc_info:
            my_func()

        assert exc_info.value.__cause__ is not None


# =============================================================================
# Test with_error_handling_async decorator
# =============================================================================


class TestWithErrorHandlingAsync:
    """Tests for with_error_handling_async decorator."""

    @pytest.mark.asyncio
    async def test_async_decorator_passes_through_success(self):
        """Should pass through successful results."""

        @with_error_handling_async(provider="openai", model="gpt-4o")
        async def my_func():
            return "async success"

        result = await my_func()
        assert result == "async success"

    @pytest.mark.asyncio
    async def test_async_decorator_translates_errors(self):
        """Should translate errors."""

        class MockError(Exception):
            status_code = 429

        @with_error_handling_async(provider="anthropic", model="claude-3")
        async def my_func():
            raise MockError("rate limited")

        with pytest.raises(RateLimitError):
            await my_func()


# =============================================================================
# Test validate_kwargs
# =============================================================================


class TestValidateKwargs:
    """Tests for validate_kwargs function."""

    def test_no_warnings_for_common_kwargs(self):
        """Should not warn for common kwargs."""
        warnings = validate_kwargs(
            "openai",
            {"temperature": 0.7, "max_tokens": 100},
            warn=False,
        )
        assert warnings == []

    def test_no_warnings_for_provider_specific_kwargs(self):
        """Should not warn for correct provider-specific kwargs."""
        warnings = validate_kwargs(
            "openai",
            {"frequency_penalty": 0.5, "presence_penalty": 0.5},
            warn=False,
        )
        assert warnings == []

    def test_warns_for_wrong_provider_kwargs(self):
        """Should warn when using kwargs for wrong provider."""
        warnings = validate_kwargs(
            "anthropic",
            {"frequency_penalty": 0.5},  # OpenAI-specific
            warn=False,
        )
        assert len(warnings) == 1
        assert "frequency_penalty" in warnings[0]
        assert "openai" in warnings[0].lower()


# =============================================================================
# Test get_supported_kwargs
# =============================================================================


class TestGetSupportedKwargs:
    """Tests for get_supported_kwargs function."""

    def test_includes_common_kwargs(self):
        """Should include common kwargs."""
        kwargs = get_supported_kwargs("openai")
        assert "temperature" in kwargs
        assert "max_tokens" in kwargs
        assert "top_p" in kwargs

    def test_includes_provider_specific_kwargs(self):
        """Should include provider-specific kwargs."""
        openai_kwargs = get_supported_kwargs("openai")
        assert "frequency_penalty" in openai_kwargs
        assert "presence_penalty" in openai_kwargs

        anthropic_kwargs = get_supported_kwargs("anthropic")
        assert "top_k" in anthropic_kwargs
        assert "frequency_penalty" not in anthropic_kwargs

    def test_unknown_provider_gets_only_common(self):
        """Should return only common kwargs for unknown provider."""
        kwargs = get_supported_kwargs("unknown_provider")
        assert "temperature" in kwargs
        assert "frequency_penalty" not in kwargs
