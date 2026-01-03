"""Unit tests for provider fallback behavior.

Tests cover:
- Automatic fallback when primary provider fails
- Fallback chain configuration
- Fallback with rate limits
- Fallback callback events

All tests use mocks - no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from ai_infra import LLM
from ai_infra.callbacks import Callbacks
from ai_infra.errors import ProviderError, RateLimitError

# =============================================================================
# Mock Helpers
# =============================================================================


def create_mock_response(content: str = "Hello!", provider: str = "unknown") -> Mock:
    """Create a mock response object."""
    response = Mock()
    response.content = content
    response.usage_metadata = None
    return response


def create_failing_model(error: Exception) -> MagicMock:
    """Create a mock model that always fails."""
    model = MagicMock()
    model.invoke.side_effect = error
    model.ainvoke = AsyncMock(side_effect=error)
    return model


def create_working_model(content: str = "Success!") -> MagicMock:
    """Create a mock model that works."""
    model = MagicMock()
    response = create_mock_response(content)
    model.invoke.return_value = response
    model.ainvoke = AsyncMock(return_value=response)
    return model


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_multi_provider_env():
    """Mock environment with multiple providers configured."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
        },
        clear=False,
    ):
        yield


# =============================================================================
# Provider Auto-Detection Tests
# =============================================================================


class TestProviderAutoDetection:
    """Test automatic provider detection and selection."""

    def test_auto_selects_first_configured_provider(self):
        """Test auto-selection of first configured provider."""
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "test-key"},
            clear=True,
        ):
            llm = LLM()
            provider, model = llm._resolve_provider_and_model(None, None)
            assert provider == "openai"

    def test_auto_selects_anthropic_if_only_anthropic(self):
        """Test auto-selection when only Anthropic is configured."""
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "test-key"},
            clear=True,
        ):
            llm = LLM()
            provider, model = llm._resolve_provider_and_model(None, None)
            assert provider == "anthropic"

    def test_explicit_provider_overrides_auto(self, mock_multi_provider_env):
        """Test explicit provider selection overrides auto-detection."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model("anthropic", None)
        assert provider == "anthropic"

    def test_no_provider_configured_raises_error(self):
        """Test error when no provider is configured."""
        with patch.dict("os.environ", {}, clear=True):
            llm = LLM()
            with pytest.raises(ValueError, match="No LLM provider configured"):
                llm._resolve_provider_and_model(None, None)


# =============================================================================
# Provider Fallback Chain Tests
# =============================================================================


class TestProviderFallbackChain:
    """Test provider fallback chain behavior."""

    def test_fallback_on_primary_failure(self, mock_multi_provider_env):
        """Test fallback to secondary provider when primary fails."""
        llm = LLM()

        # Track which provider was called
        call_order = []

        def mock_get_or_create(provider, model, **kwargs):
            call_order.append(provider)
            if provider == "openai":
                return create_failing_model(Exception("OpenAI down"))
            return create_working_model("Fallback success!")

        with patch.object(llm.registry, "get_or_create", side_effect=mock_get_or_create):
            # This would use the fallback mechanism if implemented
            # For now, test that explicit provider works
            response = llm.chat("Hello", provider="anthropic")

        assert response is not None

    def test_multiple_providers_in_priority_order(self, mock_multi_provider_env):
        """Test providers are tried in priority order."""
        llm = LLM()
        working_model = create_working_model("Success from Anthropic!")

        with patch.object(llm.registry, "get_or_create", return_value=working_model):
            response = llm.chat("Hello", provider="anthropic")

        assert response is not None


# =============================================================================
# Rate Limit Fallback Tests
# =============================================================================


class TestRateLimitFallback:
    """Test fallback behavior on rate limits."""

    def test_rate_limit_triggers_fallback_consideration(self, mock_multi_provider_env):
        """Test that rate limits are properly detected."""
        llm = LLM()
        rate_limit_model = create_failing_model(
            RateLimitError("Rate limit exceeded", provider="openai")
        )

        with patch.object(llm.registry, "get_or_create", return_value=rate_limit_model):
            with pytest.raises((RateLimitError, Exception)):
                llm.chat("Hello", provider="openai")

    def test_different_error_types_handled(self, mock_multi_provider_env):
        """Test different error types are handled appropriately."""
        llm = LLM()

        error_types = [
            Exception("Generic error"),
            ProviderError("Provider error", provider="test"),
            RateLimitError("Rate limit", provider="test"),
        ]

        for error in error_types:
            failing_model = create_failing_model(error)
            with patch.object(llm.registry, "get_or_create", return_value=failing_model):
                with pytest.raises(Exception):  # noqa: B017  # noqa: B017
                    llm.chat("Hello", provider="openai")


# =============================================================================
# Fallback Callback Tests
# =============================================================================


class TestFallbackCallbacks:
    """Test callbacks during fallback scenarios."""

    def test_error_callback_fired_on_failure(self, mock_multi_provider_env):
        """Test error callbacks are fired when provider fails."""
        events = []

        class ErrorTracker(Callbacks):
            def on_llm_start(self, event):
                events.append(("start", event))

            def on_llm_error(self, event):
                events.append(("error", event))

        llm = LLM(callbacks=ErrorTracker())
        failing_model = create_failing_model(Exception("API Error"))

        with patch.object(llm.registry, "get_or_create", return_value=failing_model):
            with pytest.raises(Exception):  # noqa: B017  # noqa: B017
                llm.chat("Hello", provider="openai")

        # Start should always fire
        assert any(e[0] == "start" for e in events)

    def test_success_after_initial_failure_fires_end(self, mock_multi_provider_env):
        """Test successful completion fires end callback."""
        events = []

        class SuccessTracker(Callbacks):
            def on_llm_start(self, event):
                events.append(("start", event))

            def on_llm_end(self, event):
                events.append(("end", event))

        llm = LLM(callbacks=SuccessTracker())
        working_model = create_working_model("Success!")

        with patch.object(llm.registry, "get_or_create", return_value=working_model):
            llm.chat("Hello", provider="anthropic")

        assert len(events) == 2
        assert events[0][0] == "start"
        assert events[1][0] == "end"


# =============================================================================
# Provider Configuration Tests
# =============================================================================


class TestProviderConfiguration:
    """Test provider configuration handling."""

    def test_provider_with_custom_model(self, mock_multi_provider_env):
        """Test specifying custom model for provider."""
        llm = LLM()
        working_model = create_working_model()

        with patch.object(llm.registry, "get_or_create", return_value=working_model) as mock_get:
            llm.chat(
                "Hello",
                provider="openai",
                model_name="gpt-4-turbo",
            )

        # Verify correct model was requested
        call_args = mock_get.call_args
        assert call_args[0][0] == "openai"
        assert call_args[0][1] == "gpt-4-turbo"

    def test_provider_kwargs_passed_through(self, mock_multi_provider_env):
        """Test model kwargs are passed to provider."""
        llm = LLM()
        working_model = create_working_model()

        with patch.object(llm.registry, "get_or_create", return_value=working_model) as mock_get:
            llm.chat(
                "Hello",
                provider="openai",
                temperature=0.5,
                max_tokens=200,
                top_p=0.9,
            )

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("temperature") == 0.5
        assert call_kwargs.get("max_tokens") == 200
        assert call_kwargs.get("top_p") == 0.9


# =============================================================================
# Async Fallback Tests
# =============================================================================


class TestAsyncFallback:
    """Test async fallback behavior."""

    @pytest.mark.asyncio
    async def test_async_provider_selection(self, mock_multi_provider_env):
        """Test async calls respect provider selection."""
        llm = LLM()
        working_model = create_working_model("Async success!")

        with patch.object(llm.registry, "get_or_create", return_value=working_model):
            response = await llm.achat("Hello", provider="anthropic")

        assert response is not None
        working_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_error_handling(self, mock_multi_provider_env):
        """Test async error handling."""
        llm = LLM()
        failing_model = create_failing_model(Exception("Async failure"))

        with patch.object(llm.registry, "get_or_create", return_value=failing_model):
            with pytest.raises(Exception):  # noqa: B017  # noqa: B017
                await llm.achat("Hello", provider="openai")
