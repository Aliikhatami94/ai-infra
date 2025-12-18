"""Prompts support for MCP client.

This module provides functionality to fetch and convert MCP prompts
to LangChain messages, enabling reuse of prompts defined on MCP servers.

Example:
    ```python
    from ai_infra.mcp import MCPClient

    async with MCPClient([config]) as mcp:
        # List available prompts
        prompts = await mcp.list_prompts("my-server")
        print(prompts)

        # Get a prompt as LangChain messages
        messages = await mcp.get_prompt(
            "my-server",
            "code-review",
            arguments={"language": "python"},
        )
        # Use with LLM
        response = await llm.ainvoke(messages)
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import GetPromptResult, Prompt, PromptMessage


@dataclass
class PromptInfo:
    """Information about an available MCP prompt.

    Attributes:
        name: The prompt name/identifier.
        description: Optional description of the prompt.
        arguments: List of argument definitions (name, description, required).
    """

    name: str
    description: str | None = None
    arguments: list[dict[str, Any]] | None = None

    @classmethod
    def from_mcp_prompt(cls, prompt: Prompt) -> PromptInfo:
        """Create PromptInfo from MCP Prompt object.

        Args:
            prompt: The MCP Prompt object.

        Returns:
            PromptInfo with extracted fields.
        """
        args = None
        if prompt.arguments:
            args = [
                {
                    "name": arg.name,
                    "description": arg.description,
                    "required": arg.required if hasattr(arg, "required") else False,
                }
                for arg in prompt.arguments
            ]
        return cls(
            name=prompt.name,
            description=prompt.description,
            arguments=args,
        )


def convert_mcp_prompt_to_message(message: PromptMessage) -> BaseMessage:
    """Convert an MCP prompt message to a LangChain message.

    Supports user (HumanMessage), assistant (AIMessage), and system roles.
    Currently only text content is supported.

    Args:
        message: The MCP PromptMessage to convert.

    Returns:
        The corresponding LangChain BaseMessage.

    Raises:
        ValueError: If the content type or role is not supported.

    Example:
        ```python
        from mcp.types import PromptMessage, TextContent

        mcp_msg = PromptMessage(
            role="user",
            content=TextContent(type="text", text="Hello!")
        )
        lc_msg = convert_mcp_prompt_to_message(mcp_msg)
        # HumanMessage(content="Hello!")
        ```
    """
    # Handle text content
    content = message.content
    if hasattr(content, "type") and content.type == "text":
        text = content.text
    elif isinstance(content, str):
        text = content
    else:
        raise ValueError(
            f"Unsupported prompt content type: {type(content)}. "
            "Only text content is currently supported."
        )

    role = message.role
    if role == "user":
        return HumanMessage(content=text)
    if role == "assistant":
        return AIMessage(content=text)
    if role == "system":
        return SystemMessage(content=text)

    raise ValueError(f"Unsupported prompt role: {role}. Expected 'user', 'assistant', or 'system'.")


async def load_mcp_prompt(
    session: ClientSession,
    name: str,
    *,
    arguments: dict[str, Any] | None = None,
) -> list[BaseMessage]:
    """Load an MCP prompt and convert to LangChain messages.

    Args:
        session: The MCP ClientSession to use.
        name: The name of the prompt to load.
        arguments: Optional arguments to pass to the prompt template.

    Returns:
        List of LangChain messages from the prompt.

    Example:
        ```python
        async with client_session as session:
            messages = await load_mcp_prompt(
                session,
                "code-review",
                arguments={"language": "python", "code": "def foo(): pass"},
            )
        ```
    """
    response: GetPromptResult = await session.get_prompt(name, arguments=arguments)
    return [convert_mcp_prompt_to_message(m) for m in response.messages]


async def list_mcp_prompts(session: ClientSession) -> list[PromptInfo]:
    """List available prompts from an MCP server.

    Args:
        session: The MCP ClientSession to use.

    Returns:
        List of PromptInfo objects describing available prompts.

    Example:
        ```python
        async with client_session as session:
            prompts = await list_mcp_prompts(session)
            for p in prompts:
                print(f"{p.name}: {p.description}")
        ```
    """
    result = await session.list_prompts()
    prompts = getattr(result, "prompts", result) or []
    # Each p is expected to be a Prompt object
    return [PromptInfo.from_mcp_prompt(cast("Prompt", p)) for p in prompts]
