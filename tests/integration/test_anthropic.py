"""Integration tests for Anthropic provider.

These tests require an Anthropic API key and make real API calls.
They are skipped by default unless explicitly enabled via:
  - ANTHROPIC_API_KEY environment variable

Run with: pytest tests/integration/test_anthropic.py -v
"""

from __future__ import annotations

import os

import pytest

from ai_infra import LLM

# Skip marker for tests requiring Anthropic API key
SKIP_NO_ANTHROPIC = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@SKIP_NO_ANTHROPIC
@pytest.mark.integration
class TestAnthropicChat:
    """Integration tests for Anthropic chat completions."""

    def test_simple_chat(self):
        """Test basic chat completion with Anthropic."""
        llm = LLM()
        response = llm.chat(
            "What is 2 + 2? Reply with just the number.",
            provider="anthropic",
            model_name="claude-3-5-haiku-latest",
        )
        assert response is not None
        assert hasattr(response, "content")
        assert "4" in response.content

    def test_chat_with_system_message(self):
        """Test chat with system message."""
        llm = LLM()
        response = llm.chat(
            "What language are you speaking?",
            provider="anthropic",
            model_name="claude-3-5-haiku-latest",
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
            provider="anthropic",
            model_name="claude-3-5-haiku-latest",
        )
        assert response is not None
        assert hasattr(response, "content")
        assert "hello" in response.content.lower()


@SKIP_NO_ANTHROPIC
@pytest.mark.integration
class TestAnthropicStructuredOutput:
    """Integration tests for Anthropic structured output."""

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
            provider="anthropic",
            model_name="claude-3-5-haiku-latest",
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
            provider="anthropic",
            model_name="claude-3-5-haiku-latest",
            output_schema=Colors,
            output_method="prompt",
        )
        assert isinstance(result, Colors)
        assert len(result.colors) == 3


@SKIP_NO_ANTHROPIC
@pytest.mark.integration
class TestAnthropicStreaming:
    """Integration tests for Anthropic streaming responses."""

    @pytest.mark.asyncio
    async def test_stream_tokens(self):
        """Test streaming tokens from Anthropic."""
        llm = LLM()
        tokens = []

        async for token, meta in llm.stream_tokens(
            "Count from 1 to 5.",
            provider="anthropic",
            model_name="claude-3-5-haiku-latest",
        ):
            tokens.append(token)

        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert any(str(i) in full_response for i in range(1, 6))


@SKIP_NO_ANTHROPIC
@pytest.mark.integration
class TestAnthropicProviderDiscovery:
    """Integration tests for provider discovery with Anthropic."""

    def test_anthropic_is_configured(self):
        """Test that Anthropic is detected as configured."""
        assert LLM.is_provider_configured("anthropic") is True

    def test_list_configured_providers_includes_anthropic(self):
        """Test that configured providers includes Anthropic."""
        providers = LLM.list_configured_providers()
        assert "anthropic" in providers


@SKIP_NO_ANTHROPIC
@pytest.mark.integration
class TestAnthropicLongContext:
    """Integration tests for Anthropic with longer context."""

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation handling."""
        llm = LLM()

        # First turn
        response1 = llm.chat(
            "My name is Alice. Remember it.",
            provider="anthropic",
            model_name="claude-3-5-haiku-latest",
            system="You are a helpful assistant with excellent memory.",
        )
        assert response1 is not None

        # Note: This tests the LLM class, not conversation memory
        # Full conversation memory would use a different pattern
