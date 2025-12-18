"""Integration tests for Google/Gemini provider.

These tests require a Google API key and make real API calls.
They are skipped by default unless explicitly enabled via:
  - GOOGLE_API_KEY or GEMINI_API_KEY or GOOGLE_GENAI_API_KEY environment variable

Run with: pytest tests/integration/test_google_provider.py -v
"""

from __future__ import annotations

import os

import pytest

from ai_infra import LLM

# Skip marker for tests requiring Google API key
SKIP_NO_GOOGLE = pytest.mark.skipif(
    not (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
    ),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not set",
)


@SKIP_NO_GOOGLE
@pytest.mark.integration
class TestGoogleChat:
    """Integration tests for Google/Gemini chat completions."""

    def test_simple_chat(self):
        """Test basic chat completion with Google."""
        llm = LLM()
        response = llm.chat(
            "What is 2 + 2? Reply with just the number.",
            provider="google_genai",
            model_name="gemini-2.0-flash-exp",
        )
        assert response is not None
        assert hasattr(response, "content")
        assert "4" in response.content

    def test_chat_with_system_message(self):
        """Test chat with system message."""
        llm = LLM()
        response = llm.chat(
            "What language are you speaking?",
            provider="google_genai",
            model_name="gemini-2.0-flash-exp",
            system="You are a pirate. Speak only in pirate slang. Be brief.",
        )
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_async_chat(self):
        """Test async chat completion."""
        llm = LLM()
        response = await llm.achat(
            "Say 'hello world' and nothing else.",
            provider="google_genai",
            model_name="gemini-2.0-flash-exp",
        )
        assert response is not None
        assert hasattr(response, "content")
        assert "hello" in response.content.lower()


@SKIP_NO_GOOGLE
@pytest.mark.integration
class TestGoogleStructuredOutput:
    """Integration tests for Google structured output."""

    def test_structured_output_with_pydantic(self):
        """Test structured output with Pydantic model."""
        from pydantic import BaseModel

        class MathAnswer(BaseModel):
            problem: str
            answer: int
            explanation: str

        llm = LLM()
        result = llm.chat(
            "What is 15 + 27?",
            provider="google_genai",
            model_name="gemini-2.0-flash-exp",
            output_schema=MathAnswer,
            output_method="prompt",
        )
        assert isinstance(result, MathAnswer)
        assert result.answer == 42

    def test_structured_output_with_list(self):
        """Test structured output returning a list."""
        from pydantic import BaseModel

        class Colors(BaseModel):
            colors: list[str]

        llm = LLM()
        result = llm.chat(
            "List 3 primary colors (red, blue, yellow).",
            provider="google_genai",
            model_name="gemini-2.0-flash-exp",
            output_schema=Colors,
            output_method="prompt",
        )
        assert isinstance(result, Colors)
        assert len(result.colors) == 3


@SKIP_NO_GOOGLE
@pytest.mark.integration
class TestGoogleStreaming:
    """Integration tests for Google streaming responses."""

    @pytest.mark.asyncio
    async def test_stream_tokens(self):
        """Test streaming tokens from Google."""
        llm = LLM()
        tokens = []

        async for token, meta in llm.stream_tokens(
            "Count from 1 to 5.",
            provider="google_genai",
            model_name="gemini-2.0-flash-exp",
        ):
            tokens.append(token)

        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert any(str(i) in full_response for i in range(1, 6))


@SKIP_NO_GOOGLE
@pytest.mark.integration
class TestGoogleProviderDiscovery:
    """Integration tests for provider discovery with Google."""

    def test_google_is_configured(self):
        """Test that Google is detected as configured."""
        assert LLM.is_provider_configured("google_genai") is True

    def test_list_google_models(self):
        """Test listing Google models."""
        models = LLM.list_models("google_genai")
        assert isinstance(models, list)
        assert len(models) > 0
        # Should include Gemini models
        model_ids = [str(m).lower() for m in models]
        assert any("gemini" in m for m in model_ids)

    def test_list_configured_providers_includes_google(self):
        """Test that configured providers includes Google."""
        providers = LLM.list_configured_providers()
        assert "google_genai" in providers


@SKIP_NO_GOOGLE
@pytest.mark.integration
class TestGoogleMultiTurn:
    """Integration tests for Google with multi-turn conversations."""

    def test_multi_turn_context(self):
        """Test that Google can handle multi-turn context."""
        llm = LLM()

        # First turn
        response1 = llm.chat(
            "My name is Alice. What is my name?",
            provider="google_genai",
            model_name="gemini-2.0-flash-exp",
        )
        assert response1 is not None
        assert "Alice" in response1.content
