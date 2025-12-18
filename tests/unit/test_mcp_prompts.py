"""Unit tests for MCP prompts support."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_infra.mcp.client.prompts import (
    PromptInfo,
    convert_mcp_prompt_to_message,
    list_mcp_prompts,
    load_mcp_prompt,
)

# ---------------------------------------------------------------------------
# Mock MCP types for testing
# ---------------------------------------------------------------------------


class MockTextContent:
    """Mock MCP TextContent."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockPromptMessage:
    """Mock MCP PromptMessage."""

    def __init__(self, role: str, text: str):
        self.role = role
        self.content = MockTextContent(text)


class MockPromptArgument:
    """Mock MCP PromptArgument."""

    def __init__(self, name: str, description: str = "", required: bool = False):
        self.name = name
        self.description = description
        self.required = required


class MockPrompt:
    """Mock MCP Prompt."""

    def __init__(
        self,
        name: str,
        description: str | None = None,
        arguments: list[MockPromptArgument] | None = None,
    ):
        self.name = name
        self.description = description
        self.arguments = arguments


class MockGetPromptResult:
    """Mock MCP GetPromptResult."""

    def __init__(self, messages: list[MockPromptMessage]):
        self.messages = messages


class MockListPromptsResult:
    """Mock MCP ListPromptsResult."""

    def __init__(self, prompts: list[MockPrompt]):
        self.prompts = prompts


# ---------------------------------------------------------------------------
# PromptInfo tests
# ---------------------------------------------------------------------------


class TestPromptInfo:
    """Tests for PromptInfo dataclass."""

    def test_create_prompt_info(self):
        """Test creating PromptInfo."""
        info = PromptInfo(
            name="test-prompt",
            description="A test prompt",
            arguments=[{"name": "arg1", "description": "First arg", "required": True}],
        )
        assert info.name == "test-prompt"
        assert info.description == "A test prompt"
        assert info.arguments == [{"name": "arg1", "description": "First arg", "required": True}]

    def test_from_mcp_prompt(self):
        """Test creating PromptInfo from MCP Prompt."""
        mcp_prompt = MockPrompt(
            name="code-review",
            description="Review code for issues",
            arguments=[
                MockPromptArgument("language", "Programming language", required=True),
                MockPromptArgument("code", "Code to review", required=True),
            ],
        )
        info = PromptInfo.from_mcp_prompt(mcp_prompt)

        assert info.name == "code-review"
        assert info.description == "Review code for issues"
        assert len(info.arguments) == 2
        assert info.arguments[0]["name"] == "language"
        assert info.arguments[0]["required"] is True

    def test_from_mcp_prompt_no_args(self):
        """Test creating PromptInfo from prompt without arguments."""
        mcp_prompt = MockPrompt(
            name="simple",
            description="Simple prompt",
            arguments=None,
        )
        info = PromptInfo.from_mcp_prompt(mcp_prompt)

        assert info.name == "simple"
        assert info.description == "Simple prompt"
        assert info.arguments is None

    def test_from_mcp_prompt_no_description(self):
        """Test creating PromptInfo from prompt without description."""
        mcp_prompt = MockPrompt(name="minimal", description=None, arguments=None)
        info = PromptInfo.from_mcp_prompt(mcp_prompt)

        assert info.name == "minimal"
        assert info.description is None


# ---------------------------------------------------------------------------
# convert_mcp_prompt_to_message tests
# ---------------------------------------------------------------------------


class TestConvertMcpPromptToMessage:
    """Tests for convert_mcp_prompt_to_message function."""

    def test_user_role_to_human_message(self):
        """Test user role converts to HumanMessage."""
        msg = MockPromptMessage(role="user", text="Hello, how are you?")
        result = convert_mcp_prompt_to_message(msg)

        assert isinstance(result, HumanMessage)
        assert result.content == "Hello, how are you?"

    def test_assistant_role_to_ai_message(self):
        """Test assistant role converts to AIMessage."""
        msg = MockPromptMessage(role="assistant", text="I'm doing well, thank you!")
        result = convert_mcp_prompt_to_message(msg)

        assert isinstance(result, AIMessage)
        assert result.content == "I'm doing well, thank you!"

    def test_system_role_to_system_message(self):
        """Test system role converts to SystemMessage."""
        msg = MockPromptMessage(role="system", text="You are a helpful assistant.")
        result = convert_mcp_prompt_to_message(msg)

        assert isinstance(result, SystemMessage)
        assert result.content == "You are a helpful assistant."

    def test_unsupported_role_raises(self):
        """Test unsupported role raises ValueError."""
        msg = MockPromptMessage(role="unknown", text="Some text")

        with pytest.raises(ValueError, match="Unsupported prompt role"):
            convert_mcp_prompt_to_message(msg)

    def test_string_content(self):
        """Test message with string content (not TextContent object)."""
        msg = MagicMock()
        msg.role = "user"
        msg.content = "Direct string content"

        result = convert_mcp_prompt_to_message(msg)
        assert isinstance(result, HumanMessage)
        assert result.content == "Direct string content"

    def test_unsupported_content_type_raises(self):
        """Test unsupported content type raises ValueError."""
        msg = MagicMock()
        msg.role = "user"
        msg.content = MagicMock()
        msg.content.type = "image"  # Not text

        with pytest.raises(ValueError, match="Unsupported prompt content type"):
            convert_mcp_prompt_to_message(msg)


# ---------------------------------------------------------------------------
# load_mcp_prompt tests
# ---------------------------------------------------------------------------


class TestLoadMcpPrompt:
    """Tests for load_mcp_prompt function."""

    @pytest.mark.asyncio
    async def test_load_simple_prompt(self):
        """Test loading a simple prompt."""
        session = AsyncMock()
        session.get_prompt.return_value = MockGetPromptResult(
            messages=[
                MockPromptMessage("system", "You are a helpful assistant."),
                MockPromptMessage("user", "Hello!"),
            ]
        )

        messages = await load_mcp_prompt(session, "greeting")

        session.get_prompt.assert_called_once_with("greeting", arguments=None)
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    @pytest.mark.asyncio
    async def test_load_prompt_with_arguments(self):
        """Test loading a prompt with arguments."""
        session = AsyncMock()
        session.get_prompt.return_value = MockGetPromptResult(
            messages=[MockPromptMessage("user", "Review this Python code: ...")]
        )

        messages = await load_mcp_prompt(
            session,
            "code-review",
            arguments={"language": "python", "code": "def foo(): pass"},
        )

        session.get_prompt.assert_called_once_with(
            "code-review",
            arguments={"language": "python", "code": "def foo(): pass"},
        )
        assert len(messages) == 1


# ---------------------------------------------------------------------------
# list_mcp_prompts tests
# ---------------------------------------------------------------------------


class TestListMcpPrompts:
    """Tests for list_mcp_prompts function."""

    @pytest.mark.asyncio
    async def test_list_prompts(self):
        """Test listing available prompts."""
        session = AsyncMock()
        session.list_prompts.return_value = MockListPromptsResult(
            prompts=[
                MockPrompt("greeting", "A greeting prompt"),
                MockPrompt(
                    "code-review",
                    "Review code",
                    arguments=[MockPromptArgument("code", "Code to review", True)],
                ),
            ]
        )

        prompts = await list_mcp_prompts(session)

        assert len(prompts) == 2
        assert prompts[0].name == "greeting"
        assert prompts[0].description == "A greeting prompt"
        assert prompts[1].name == "code-review"
        assert prompts[1].arguments is not None
        assert len(prompts[1].arguments) == 1

    @pytest.mark.asyncio
    async def test_list_prompts_empty(self):
        """Test listing prompts when none available."""
        session = AsyncMock()
        session.list_prompts.return_value = MockListPromptsResult(prompts=[])

        prompts = await list_mcp_prompts(session)

        assert prompts == []

    @pytest.mark.asyncio
    async def test_list_prompts_handles_none(self):
        """Test listing prompts handles None result."""
        session = AsyncMock()
        result = MagicMock()
        result.prompts = None
        session.list_prompts.return_value = result

        prompts = await list_mcp_prompts(session)

        assert prompts == []


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestPromptsIntegration:
    """Integration-style tests for prompts functionality."""

    @pytest.mark.asyncio
    async def test_full_prompt_workflow(self):
        """Test complete workflow: list, select, load prompt."""
        session = AsyncMock()

        # List prompts
        session.list_prompts.return_value = MockListPromptsResult(
            prompts=[
                MockPrompt(
                    "summarize",
                    "Summarize text",
                    arguments=[MockPromptArgument("text", "Text to summarize", True)],
                ),
            ]
        )

        prompts = await list_mcp_prompts(session)
        assert len(prompts) == 1
        assert prompts[0].name == "summarize"

        # Load the prompt
        session.get_prompt.return_value = MockGetPromptResult(
            messages=[
                MockPromptMessage("system", "You are a summarization expert."),
                MockPromptMessage("user", "Please summarize: Hello world"),
            ]
        )

        messages = await load_mcp_prompt(
            session,
            "summarize",
            arguments={"text": "Hello world"},
        )

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert "summarization expert" in messages[0].content
        assert isinstance(messages[1], HumanMessage)
        assert "Hello world" in messages[1].content

    def test_prompt_info_repr(self):
        """Test PromptInfo has useful string representation."""
        info = PromptInfo(name="test", description="Test prompt")
        repr_str = repr(info)

        assert "test" in repr_str
        assert "PromptInfo" in repr_str
