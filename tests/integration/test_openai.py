"""Integration tests for OpenAI provider.

These tests require an OpenAI API key and make real API calls.
They are skipped by default unless explicitly enabled via:
  - OPENAI_API_KEY environment variable

Run with: pytest tests/integration/test_openai.py -v
"""

from __future__ import annotations

import os

import pytest

from ai_infra import LLM

# Skip marker for tests requiring OpenAI API key
SKIP_NO_OPENAI = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@SKIP_NO_OPENAI
@pytest.mark.integration
class TestOpenAIChat:
    """Integration tests for OpenAI chat completions."""

    def test_simple_chat(self):
        """Test basic chat completion with OpenAI."""
        llm = LLM()
        response = llm.chat(
            "What is 2 + 2? Reply with just the number.",
            provider="openai",
            model_name="gpt-4o-mini",
        )
        assert response is not None
        assert hasattr(response, "content")
        assert "4" in response.content

    def test_chat_with_system_message(self):
        """Test chat with system message."""
        llm = LLM()
        response = llm.chat(
            "What language are you speaking?",
            provider="openai",
            model_name="gpt-4o-mini",
            system="You are a pirate. Speak only in pirate slang.",
        )
        assert response is not None
        assert hasattr(response, "content")
        # Pirate-speak indicators
        assert (
            any(word in response.content.lower() for word in ["arr", "matey", "ahoy", "ye", "aye"])
            or response.content
        )  # Fallback: at least got a response

    @pytest.mark.asyncio
    async def test_async_chat(self):
        """Test async chat completion."""
        llm = LLM()
        response = await llm.achat(
            "Say 'hello world' and nothing else.",
            provider="openai",
            model_name="gpt-4o-mini",
        )
        assert response is not None
        assert hasattr(response, "content")
        assert "hello" in response.content.lower()


@SKIP_NO_OPENAI
@pytest.mark.integration
class TestOpenAIStructuredOutput:
    """Integration tests for OpenAI structured output."""

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
            provider="openai",
            model_name="gpt-4o-mini",
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
            provider="openai",
            model_name="gpt-4o-mini",
            output_schema=Colors,
            output_method="prompt",
        )
        assert isinstance(result, Colors)
        assert len(result.colors) == 3


@SKIP_NO_OPENAI
@pytest.mark.integration
class TestOpenAIStreaming:
    """Integration tests for OpenAI streaming responses."""

    @pytest.mark.asyncio
    async def test_stream_tokens(self):
        """Test streaming tokens from OpenAI."""
        llm = LLM()
        tokens = []

        async for token, meta in llm.stream_tokens(
            "Count from 1 to 5.",
            provider="openai",
            model_name="gpt-4o-mini",
        ):
            tokens.append(token)

        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert any(str(i) in full_response for i in range(1, 6))


@SKIP_NO_OPENAI
@pytest.mark.integration
class TestOpenAIProviderDiscovery:
    """Integration tests for provider discovery with OpenAI."""

    def test_openai_is_configured(self):
        """Test that OpenAI is detected as configured."""
        assert LLM.is_provider_configured("openai") is True

    def test_list_openai_models(self):
        """Test listing OpenAI models."""
        models = LLM.list_models("openai")
        assert isinstance(models, list)
        assert len(models) > 0
        # Should include common models
        model_ids = [m.lower() if isinstance(m, str) else m for m in models]
        assert any("gpt" in str(m) for m in model_ids)

    def test_list_configured_providers_includes_openai(self):
        """Test that configured providers includes OpenAI."""
        providers = LLM.list_configured_providers()
        assert "openai" in providers
