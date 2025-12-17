"""Tests for fit_context() - the unified context management API."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_infra.llm.memory.context import (
    ContextResult,
    _format_messages,
    afit_context,
    fit_context,
)


class TestContextResult:
    """Tests for ContextResult dataclass."""

    def test_default_values(self):
        """Test that ContextResult has sensible defaults."""
        result = ContextResult(messages=[])
        assert result.messages == []
        assert result.summary is None
        assert result.tokens == 0
        assert result.action == "none"
        assert result.original_count == 0
        assert result.final_count == 0

    def test_all_fields_populated(self):
        """Test that all fields can be set."""
        msgs = [HumanMessage(content="Hi")]
        result = ContextResult(
            messages=msgs,
            summary="Test summary",
            tokens=100,
            action="summarized",
            original_count=50,
            final_count=5,
        )
        assert result.messages == msgs
        assert result.summary == "Test summary"
        assert result.tokens == 100
        assert result.action == "summarized"
        assert result.original_count == 50
        assert result.final_count == 5


class TestFitContextBasic:
    """Basic tests for fit_context() without summarization."""

    def test_empty_messages(self):
        """Test with empty message list."""
        result = fit_context([], max_tokens=4000)
        assert result.messages == []
        assert result.summary is None
        assert result.action == "none"
        assert result.original_count == 0
        assert result.final_count == 0

    def test_under_limit_returns_unchanged(self):
        """Test that messages under the limit are returned as-is."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        result = fit_context(messages, max_tokens=4000)
        assert len(result.messages) == 2
        assert result.action == "none"
        assert result.original_count == 2
        assert result.final_count == 2

    def test_over_limit_trims_by_default(self):
        """Test that messages over limit are trimmed when summarize=False."""
        # Create many messages to exceed a small token limit
        messages = [HumanMessage(content=f"Message {i} " * 20) for i in range(20)]
        result = fit_context(messages, max_tokens=200)
        assert result.action == "trimmed"
        assert result.final_count < result.original_count
        assert result.tokens <= 200

    def test_preserves_system_message_when_trimming(self):
        """Test that system message is preserved during trimming."""
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Message 1 " * 50),
            AIMessage(content="Response 1 " * 50),
            HumanMessage(content="Message 2 " * 50),
        ]
        result = fit_context(messages, max_tokens=200)
        assert result.action == "trimmed"
        # System message should be preserved
        assert isinstance(result.messages[0], SystemMessage)
        assert "helpful assistant" in result.messages[0].content

    def test_dict_message_format(self):
        """Test that dict message format is supported."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = fit_context(messages, max_tokens=4000)
        assert len(result.messages) == 2
        assert isinstance(result.messages[0], HumanMessage)
        assert isinstance(result.messages[1], AIMessage)

    def test_existing_summary_passed_through_when_no_action(self):
        """Test that existing summary is passed through when under limit."""
        messages = [HumanMessage(content="Hello")]
        result = fit_context(
            messages,
            max_tokens=4000,
            summary="Existing summary",
        )
        assert result.summary == "Existing summary"
        assert result.action == "none"

    def test_existing_summary_passed_through_when_trimming(self):
        """Test that existing summary is passed through when trimming."""
        messages = [HumanMessage(content=f"Message {i} " * 30) for i in range(10)]
        result = fit_context(
            messages,
            max_tokens=200,
            summary="Existing summary",
        )
        assert result.summary == "Existing summary"
        assert result.action == "trimmed"


class TestFitContextSummarization:
    """Tests for fit_context() with summarize=True."""

    def test_summarize_under_limit_no_action(self):
        """Test that summarize=True doesn't act when under limit."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!"),
        ]
        result = fit_context(messages, max_tokens=4000, summarize=True)
        assert result.action == "none"
        assert result.summary is None

    def test_summarize_creates_summary(self):
        """Test that summarize=True creates summary when over limit."""
        # Create messages that exceed the limit
        messages = [HumanMessage(content=f"Message {i} " * 30) for i in range(20)]

        # Mock LLM that returns a fixed summary
        class MockLLM:
            def chat(self, prompt):
                return AIMessage(content="This is a summary of the conversation.")

        result = fit_context(
            messages,
            max_tokens=200,
            summarize=True,
            llm=MockLLM(),
        )

        assert result.action == "summarized"
        assert result.summary is not None
        assert "summary" in result.summary.lower()

    def test_summarize_keeps_recent_messages(self):
        """Test that keep parameter is respected."""
        messages = [HumanMessage(content=f"Message {i} " * 30) for i in range(20)]

        class MockLLM:
            def chat(self, prompt):
                return AIMessage(content="Summary")

        result = fit_context(
            messages,
            max_tokens=200,
            summarize=True,
            keep=5,
            llm=MockLLM(),
        )

        assert result.action == "summarized"
        # Should have summary message + 5 kept messages
        # (excluding any system message handling)
        assert result.final_count <= 6  # summary + 5 kept

    def test_summarize_extends_existing_summary(self):
        """Test that existing summary is extended, not replaced."""
        messages = [HumanMessage(content=f"Message {i} " * 30) for i in range(10)]

        class MockLLM:
            def __init__(self):
                self.last_prompt = None

            def chat(self, prompt):
                self.last_prompt = prompt
                return AIMessage(content="Extended summary")

        mock_llm = MockLLM()
        result = fit_context(
            messages,
            max_tokens=200,
            summarize=True,
            summary="Previous summary content",
            llm=mock_llm,
        )

        assert result.action == "summarized"
        assert result.summary == "Extended summary"
        # Check that the extend prompt was used (contains existing summary)
        assert "Previous summary content" in mock_llm.last_prompt

    def test_summarize_preserves_system_message(self):
        """Test that system message is preserved when summarizing."""
        messages = [
            SystemMessage(content="You are a helpful assistant"),
        ] + [HumanMessage(content=f"Message {i} " * 30) for i in range(20)]

        class MockLLM:
            def chat(self, prompt):
                return AIMessage(content="Summary")

        result = fit_context(
            messages,
            max_tokens=200,
            summarize=True,
            llm=MockLLM(),
        )

        # First message should be original system message
        assert isinstance(result.messages[0], SystemMessage)
        assert "helpful assistant" in result.messages[0].content


class TestFitContextHelpers:
    """Tests for internal helper functions."""

    def test_format_messages(self):
        """Test message formatting for summarization."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            SystemMessage(content="Be helpful"),
        ]
        formatted = _format_messages(messages)
        assert "User: Hello" in formatted
        assert "Assistant: Hi there!" in formatted
        assert "System: Be helpful" in formatted


class TestFitContextAsync:
    """Tests for async afit_context()."""

    @pytest.mark.asyncio
    async def test_async_under_limit(self):
        """Test async version with messages under limit."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!"),
        ]
        result = await afit_context(messages, max_tokens=4000)
        assert result.action == "none"
        assert len(result.messages) == 2

    @pytest.mark.asyncio
    async def test_async_trim(self):
        """Test async version trims when over limit."""
        messages = [HumanMessage(content=f"Message {i} " * 30) for i in range(20)]
        result = await afit_context(messages, max_tokens=200)
        assert result.action == "trimmed"
        assert result.final_count < result.original_count

    @pytest.mark.asyncio
    async def test_async_summarize(self):
        """Test async version summarizes when requested."""
        messages = [HumanMessage(content=f"Message {i} " * 30) for i in range(20)]

        class MockLLM:
            async def achat(self, prompt):
                return AIMessage(content="Async summary")

        result = await afit_context(
            messages,
            max_tokens=200,
            summarize=True,
            llm=MockLLM(),
        )

        assert result.action == "summarized"
        assert result.summary == "Async summary"


class TestFitContextEdgeCases:
    """Edge case tests."""

    def test_keep_zero(self):
        """Test with keep=0 (don't keep any recent messages)."""
        messages = [HumanMessage(content=f"Message {i} " * 30) for i in range(10)]

        class MockLLM:
            def chat(self, prompt):
                return AIMessage(content="Summary of all")

        result = fit_context(
            messages,
            max_tokens=100,
            summarize=True,
            keep=0,
            llm=MockLLM(),
        )

        assert result.action == "summarized"
        # Should only have summary message
        assert result.final_count == 1
        assert "[Conversation summary]" in result.messages[0].content

    def test_max_tokens_very_small(self):
        """Test with very small max_tokens."""
        messages = [HumanMessage(content="Hello world")]
        result = fit_context(messages, max_tokens=10)
        # Should still return something (can't trim below content)
        assert result.messages is not None

    def test_single_message(self):
        """Test with single message."""
        messages = [HumanMessage(content="Hello")]
        result = fit_context(messages, max_tokens=4000)
        assert len(result.messages) == 1
        assert result.action == "none"
