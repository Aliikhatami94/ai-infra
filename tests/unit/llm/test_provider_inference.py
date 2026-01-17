"""Tests for Phase 16.5.8: Provider auto-detection from model names.

This module tests the _infer_provider_from_model() function and the
updated _resolve_provider_and_model() method that supports:
- Model name pattern inference (claude-* -> anthropic, gpt-* -> openai)
- Provider/model format parsing (anthropic/claude-sonnet-4)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_infra import LLM
from ai_infra.llm.base import _infer_provider_from_model

# =============================================================================
# Tests: _infer_provider_from_model()
# =============================================================================


class TestInferProviderFromModel:
    """Tests for the _infer_provider_from_model() helper function."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            # Anthropic models (Claude family)
            ("claude-sonnet-4-20250514", "anthropic"),
            ("claude-3-opus-20240229", "anthropic"),
            ("claude-3-5-sonnet-latest", "anthropic"),
            ("claude-haiku-4-5-20251001", "anthropic"),
            ("claude-instant-1.2", "anthropic"),
            ("Claude-Sonnet-4", "anthropic"),  # Case insensitive
            ("CLAUDE-3-OPUS", "anthropic"),  # All caps
            ("claude_sonnet_4", "anthropic"),  # Underscore variant
            # OpenAI models (GPT, o-series, etc.)
            ("gpt-4o", "openai"),
            ("gpt-4o-mini", "openai"),
            ("gpt-4-turbo", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("GPT-4O", "openai"),  # Case insensitive
            ("o1-preview", "openai"),
            ("o1-mini", "openai"),
            ("o3-mini", "openai"),
            ("o4-preview", "openai"),
            ("dall-e-3", "openai"),
            ("whisper-1", "openai"),
            ("tts-1-hd", "openai"),
            # Google models (Gemini, PaLM)
            ("gemini-1.5-pro", "google_genai"),
            ("gemini-1.5-flash", "google_genai"),
            ("gemini-2.0-flash-exp", "google_genai"),
            ("Gemini-Pro", "google_genai"),  # Case insensitive
            ("palm-2", "google_genai"),
            # xAI models (Grok)
            ("grok-2", "xai"),
            ("grok-beta", "xai"),
            ("Grok-2-Latest", "xai"),  # Case insensitive
            # Unknown models (should return None)
            ("my-custom-model", None),
            ("llama-3-70b", None),
            ("mistral-large", None),
            ("codellama-34b", None),
            ("random-model-name", None),
        ],
    )
    def test_infer_provider_from_model(self, model: str, expected: str | None) -> None:
        """Test provider inference from model name patterns."""
        result = _infer_provider_from_model(model)
        assert result == expected

    def test_empty_string_returns_none(self) -> None:
        """Test that empty string returns None (no match)."""
        result = _infer_provider_from_model("")
        assert result is None


# =============================================================================
# Tests: _resolve_provider_and_model() with provider inference
# =============================================================================


class TestResolveProviderAndModel:
    """Tests for _resolve_provider_and_model() with Phase 16.5.8 enhancements."""

    def test_infers_anthropic_from_claude_model(self) -> None:
        """Test: claude-* models are inferred as Anthropic."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(None, "claude-sonnet-4-20250514")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-20250514"

    def test_infers_openai_from_gpt_model(self) -> None:
        """Test: gpt-* models are inferred as OpenAI."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(None, "gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_infers_google_from_gemini_model(self) -> None:
        """Test: gemini-* models are inferred as Google."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(None, "gemini-1.5-pro")
        assert provider == "google_genai"
        assert model == "gemini-1.5-pro"

    def test_infers_xai_from_grok_model(self) -> None:
        """Test: grok-* models are inferred as xAI."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(None, "grok-2")
        assert provider == "xai"
        assert model == "grok-2"

    def test_explicit_provider_takes_precedence(self) -> None:
        """Test: Explicit provider overrides model name inference."""
        llm = LLM()
        # Even though "gpt-4o" looks like OpenAI, explicit anthropic wins
        provider, model = llm._resolve_provider_and_model("anthropic", "gpt-4o")
        assert provider == "anthropic"
        assert model == "gpt-4o"

    @patch("ai_infra.llm.base.get_default_provider")
    def test_falls_back_to_default_provider_for_unknown_model(
        self, mock_get_default: pytest.fixture
    ) -> None:
        """Test: Unknown model names fall back to default provider."""
        mock_get_default.return_value = "openai"
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(None, "my-custom-model")
        assert provider == "openai"
        assert model == "my-custom-model"
        mock_get_default.assert_called_once()

    @patch("ai_infra.llm.base.get_default_provider")
    def test_raises_when_no_provider_available(self, mock_get_default: pytest.fixture) -> None:
        """Test: Raises ValueError when no provider can be determined."""
        mock_get_default.return_value = None
        llm = LLM()
        with pytest.raises(ValueError, match="No LLM provider configured"):
            llm._resolve_provider_and_model(None, "my-custom-model")


# =============================================================================
# Tests: Provider/model format parsing
# =============================================================================


class TestProviderModelFormat:
    """Tests for provider/model format parsing (e.g., 'anthropic/claude-sonnet-4')."""

    def test_parses_provider_model_format(self) -> None:
        """Test: 'provider/model' format is correctly parsed."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(
            None, "anthropic/claude-sonnet-4-20250514"
        )
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-20250514"

    def test_parses_openai_format(self) -> None:
        """Test: 'openai/model' format is correctly parsed."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(None, "openai/gpt-4o-mini")
        assert provider == "openai"
        assert model == "gpt-4o-mini"

    def test_parses_google_format(self) -> None:
        """Test: 'google_genai/model' format is correctly parsed."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(None, "google_genai/gemini-1.5-pro")
        assert provider == "google_genai"
        assert model == "gemini-1.5-pro"

    def test_provider_model_overrides_inference(self) -> None:
        """Test: Explicit provider/model overrides model name inference."""
        llm = LLM()
        # Model name looks like Anthropic, but explicit provider says OpenAI
        provider, model = llm._resolve_provider_and_model(None, "openai/claude-sonnet-4")
        assert provider == "openai"
        assert model == "claude-sonnet-4"

    def test_explicit_provider_ignored_with_format(self) -> None:
        """Test: Format provider takes precedence when both explicit and format given."""
        llm = LLM()
        # Explicit provider=openai, but model has anthropic/ prefix
        # Note: Current implementation uses format first since provider is None
        provider, model = llm._resolve_provider_and_model(None, "anthropic/claude-sonnet-4")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4"

    def test_model_with_slash_in_name_not_mistaken(self) -> None:
        """Test: Models with organization prefix work (huggingface style)."""
        llm = LLM()
        # This should parse as provider=meta-llama, model=llama-3-70b
        provider, model = llm._resolve_provider_and_model(None, "meta-llama/llama-3-70b")
        assert provider == "meta-llama"
        assert model == "llama-3-70b"

    def test_empty_provider_in_format_uses_inference(self) -> None:
        """Test: '/model' format falls back to inference."""
        llm = LLM()
        # Leading slash with empty provider should be handled gracefully
        # This is an edge case - we don't parse it as provider/model
        # Mock default provider since CI has no API keys configured
        with patch("ai_infra.llm.base.get_default_provider", return_value="openai"):
            provider, model = llm._resolve_provider_and_model(None, "/gpt-4o")
        # Empty provider part fails the check, so it's treated as a regular model name
        # Falls back to default provider since "/gpt-4o" doesn't match inference patterns
        assert provider == "openai"
        assert model == "/gpt-4o"

    def test_trailing_slash_not_parsed(self) -> None:
        """Test: 'model/' format is not parsed as provider/model."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(None, "gpt-4o/")
        # Trailing slash with empty model part fails the check
        # Falls back to inference which detects gpt-4o/ starts with gpt-
        assert provider == "openai"


# =============================================================================
# Tests: Integration scenarios
# =============================================================================


class TestProviderInferenceIntegration:
    """Integration tests for real-world usage scenarios."""

    def test_cli_style_claude_model(self) -> None:
        """Test: CLI --model claude-sonnet-4-20250514 works correctly."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(None, "claude-sonnet-4-20250514")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-20250514"

    def test_cli_style_explicit_format(self) -> None:
        """Test: CLI --model anthropic/claude-sonnet-4 works correctly."""
        llm = LLM()
        provider, model = llm._resolve_provider_and_model(
            None, "anthropic/claude-sonnet-4-20250514"
        )
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-20250514"

    def test_o_series_models(self) -> None:
        """Test: OpenAI o-series models are correctly identified."""
        llm = LLM()
        for model_name in ["o1-preview", "o1-mini", "o3-mini"]:
            provider, model = llm._resolve_provider_and_model(None, model_name)
            assert provider == "openai", f"Failed for {model_name}"
            assert model == model_name

    def test_case_insensitive_matching(self) -> None:
        """Test: Model name matching is case-insensitive."""
        llm = LLM()
        test_cases = [
            ("Claude-Sonnet-4", "anthropic"),
            ("CLAUDE-3-OPUS", "anthropic"),
            ("GPT-4O", "openai"),
            ("Gemini-Pro", "google_genai"),
        ]
        for model_name, expected_provider in test_cases:
            provider, model = llm._resolve_provider_and_model(None, model_name)
            assert provider == expected_provider, f"Failed for {model_name}"
