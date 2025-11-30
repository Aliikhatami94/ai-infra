"""Agent class for tool-using LLM agents.

This module provides the Agent class for running LLM agents with tools,
including support for sessions, human-in-the-loop approval, streaming,
and DeepAgents mode for autonomous multi-step task execution.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union

from ai_infra.llm.base import BaseLLM
from ai_infra.llm.session import (
    ResumeDecision,
    SessionConfig,
    SessionResult,
    SessionStorage,
    generate_session_id,
    get_pending_action,
    is_paused,
)
from ai_infra.llm.tools import (
    ApprovalConfig,
    ToolExecutionConfig,
    apply_output_gate,
    apply_output_gate_async,
    wrap_tool_for_approval,
    wrap_tool_for_hitl,
    wrap_tool_with_execution_config,
)
from ai_infra.llm.tools.approval import ApprovalHandler, AsyncApprovalHandler
from ai_infra.llm.tools.tool_controls import ToolCallControls
from ai_infra.llm.utils.error_handler import translate_provider_error
from ai_infra.llm.utils.runtime_bind import make_agent_with_context as rb_make_agent_with_context

from .utils import arun_with_fallbacks as _arun_fallbacks_util
from .utils import is_valid_response as _is_valid_response
from .utils import merge_overrides as _merge_overrides
from .utils import run_with_fallbacks as _run_fallbacks_util
from .utils import with_retry as _with_retry_util

# =============================================================================
# DeepAgents Types (re-exported for convenience)
# =============================================================================

try:
    from deepagents import CompiledSubAgent, FilesystemMiddleware, SubAgent, SubAgentMiddleware
    from deepagents import create_deep_agent as _create_deep_agent
    from langchain.agents.middleware.types import AgentMiddleware

    _HAS_DEEPAGENTS = True
except ImportError:
    _HAS_DEEPAGENTS = False

    # Define placeholders when deepagents is not installed
    def _missing_deepagents(*args, **kwargs):
        raise ImportError(
            "DeepAgents requires 'deepagents' package. " "Install with: pip install deepagents"
        )

    class SubAgent(dict):  # type: ignore[no-redef]
        """Placeholder for SubAgent when deepagents is not installed."""

        def __init__(self, *args, **kwargs):
            _missing_deepagents()

    class CompiledSubAgent:  # type: ignore[no-redef]
        """Placeholder for CompiledSubAgent when deepagents is not installed."""

        def __init__(self, *args, **kwargs):
            _missing_deepagents()

    class SubAgentMiddleware:  # type: ignore[no-redef]
        """Placeholder for SubAgentMiddleware when deepagents is not installed."""

        def __init__(self, *args, **kwargs):
            _missing_deepagents()

    class FilesystemMiddleware:  # type: ignore[no-redef]
        """Placeholder for FilesystemMiddleware when deepagents is not installed."""

        def __init__(self, *args, **kwargs):
            _missing_deepagents()

    AgentMiddleware = Any  # type: ignore[misc, assignment]

    def _create_deep_agent(*args, **kwargs):
        _missing_deepagents()


# Export DeepAgent types
__all__ = [
    "Agent",
    "SubAgent",
    "CompiledSubAgent",
    "SubAgentMiddleware",
    "FilesystemMiddleware",
]


class Agent(BaseLLM):
    """Agent-oriented interface (tool calling, streaming updates, fallbacks).

    The Agent class provides a simple API for running LLM agents with tools.
    Tools can be plain Python functions, LangChain tools, or MCP tools.

    Example - Basic usage:
        ```python
        def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"Weather in {city}: Sunny, 72°F"

        # Simple usage with tools
        agent = Agent(tools=[get_weather])
        result = agent.run("What's the weather in NYC?")
        ```

    Example - With session memory (conversations persist):
        ```python
        from ai_infra.llm.session import memory

        agent = Agent(tools=[...], session=memory())

        # Conversation 1 - remembered
        agent.run("I'm Bob", session_id="user-123")
        agent.run("What's my name?", session_id="user-123")  # Knows "Bob"

        # Different session - fresh start
        agent.run("What's my name?", session_id="user-456")  # Doesn't know
        ```

    Example - Pause and resume (HITL):
        ```python
        from ai_infra.llm.session import memory

        agent = Agent(
            tools=[dangerous_tool],
            session=memory(),
            pause_before=["dangerous_tool"],  # Pause before this tool
        )

        result = agent.run("Delete file.txt", session_id="task-1")

        if result.paused:
            # Show user what's pending, get approval
            print(result.pending_action)

            # Resume with decision
            result = agent.resume(session_id="task-1", approved=True)
        ```

    Example - Production with Postgres:
        ```python
        from ai_infra.llm.session import postgres

        agent = Agent(
            tools=[...],
            session=postgres("postgresql://..."),
        )
        # Sessions persist across restarts
        ```

    Example - Human approval (sync, per-request):
        ```python
        agent = Agent(
            tools=[dangerous_tool],
            require_approval=True,  # Console prompt for approval
        )
        ```

    Example - DeepAgents mode (autonomous multi-step tasks):
        ```python
        from ai_infra.llm import Agent
        from ai_infra.llm.session import memory

        # Define specialized agents
        researcher = Agent(
            name="researcher",
            description="Searches and analyzes code",
            system="You are a code research assistant.",
            tools=[search_codebase],
        )

        writer = Agent(
            name="writer",
            description="Writes and edits documentation",
            system="You are a technical writer.",
        )

        # Create a deep agent that can delegate to subagents
        agent = Agent(
            deep=True,
            session=memory(),
            subagents=[researcher, writer],  # Agents auto-convert to subagents
        )

        # The agent can now autonomously:
        # - Read/write/edit files
        # - Execute shell commands
        # - Delegate to subagents
        # - Maintain todo lists
        result = agent.run("Refactor the auth module to use JWT tokens")
        ```
    """

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        tools: Optional[List[Any]] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        *,
        # Agent identity (used when this Agent is passed as a subagent)
        name: Optional[str] = None,
        description: Optional[str] = None,
        system: Optional[str] = None,
        # Tool execution config
        on_tool_error: Literal["return_error", "retry", "abort"] = "return_error",
        tool_timeout: Optional[float] = None,
        validate_tool_results: bool = False,
        max_tool_retries: int = 1,
        # Approval config
        require_approval: Union[bool, List[str], Callable[[str, Dict[str, Any]], bool]] = False,
        approval_handler: Optional[Union[ApprovalHandler, AsyncApprovalHandler]] = None,
        # Session config (for persistence and pause/resume)
        session: Optional[SessionStorage] = None,
        pause_before: Optional[List[str]] = None,
        pause_after: Optional[List[str]] = None,
        # DeepAgents mode (autonomous multi-step task execution)
        deep: bool = False,
        subagents: Optional[List[Union["Agent", "SubAgent"]]] = None,
        middleware: Optional[Sequence["AgentMiddleware"]] = None,
        response_format: Optional[Any] = None,
        context_schema: Optional[Type[Any]] = None,
        use_longterm_memory: bool = False,
        **model_kwargs,
    ):
        """Initialize an Agent with optional tools and provider settings.

        Args:
            tools: List of tools (functions, LangChain tools, or MCP tools)
            provider: LLM provider (auto-detected if None)
            model_name: Model name (uses provider default if None)

            Agent Identity (for use as subagent):
                name: Agent name (required when used as a subagent)
                description: What this agent does (used by parent to decide delegation)
                system: System prompt / instructions for this agent

            Tool Execution:
                on_tool_error: How to handle tool execution errors:
                    - "return_error": Return error message to agent (default, allows recovery)
                    - "retry": Retry the tool call up to max_tool_retries times
                    - "abort": Re-raise the exception and stop execution
                tool_timeout: Timeout in seconds per tool call (None = no timeout)
                validate_tool_results: Validate tool results match return type annotations
                max_tool_retries: Max retry attempts when on_tool_error="retry" (default 1)

            Human Approval:
                require_approval: Tools that require human approval:
                    - False: No approval needed (default)
                    - True: All tools need approval
                    - List[str]: Only specified tools need approval
                    - Callable: Function(tool_name, args) -> bool for dynamic approval
                approval_handler: Custom approval handler function:
                    - If None and require_approval is True, uses console prompts
                    - Can be sync or async function taking ApprovalRequest

            Session & Persistence:
                session: Session storage backend for conversation memory and pause/resume.
                    Use memory() for development, postgres() for production.
                    Example: session=memory()
                pause_before: Tool names to pause before executing (requires session).
                    The agent will return a SessionResult with paused=True.
                pause_after: Tool names to pause after executing (requires session).

            DeepAgents Mode (autonomous multi-step tasks):
                deep: Enable DeepAgents mode for autonomous task execution.
                    When True, the agent has built-in tools for file operations
                    (ls, read_file, write_file, edit_file, glob, grep, execute),
                    todo management, and subagent orchestration.
                subagents: List of agents for delegation. Can be Agent instances
                    (automatically converted) or SubAgent dicts. Agent instances
                    must have name and description set.
                middleware: Additional middleware to apply to the deep agent.
                response_format: Structured output format for agent responses.
                context_schema: Schema for the deep agent context.
                use_longterm_memory: Enable long-term memory (requires session with store).

            **model_kwargs: Additional kwargs passed to the model
        """
        super().__init__()
        self._name = name
        self._description = description
        self._system = system
        self._default_provider = provider
        self._default_model_name = model_name
        self._default_model_kwargs = model_kwargs
        self._tool_execution_config = ToolExecutionConfig(
            on_error=on_tool_error,
            max_retries=max_tool_retries,
            timeout=tool_timeout,
            validate_results=validate_tool_results,
        )

        # DeepAgents mode config
        self._deep = deep
        self._subagents = self._convert_subagents(subagents) if subagents else None
        self._middleware = middleware
        self._response_format = response_format
        self._context_schema = context_schema
        self._use_longterm_memory = use_longterm_memory

        # Set up approval config
        self._approval_config: Optional[ApprovalConfig] = None
        if require_approval or approval_handler:
            # Determine if handler is async
            if approval_handler and asyncio.iscoroutinefunction(approval_handler):
                self._approval_config = ApprovalConfig(
                    require_approval=require_approval if require_approval else True,
                    approval_handler_async=approval_handler,
                )
            else:
                self._approval_config = ApprovalConfig(
                    require_approval=require_approval if require_approval else True,
                    approval_handler=approval_handler,  # type: ignore
                )

        # Set up session config for persistence and pause/resume
        self._session_config: Optional[SessionConfig] = None
        if session:
            self._session_config = SessionConfig(
                storage=session,
                pause_before=pause_before or [],
                pause_after=pause_after or [],
            )

        if tools:
            self.set_global_tools(tools)

    def run(
        self,
        prompt: str,
        *,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        system: Optional[str] = None,
        session_id: Optional[str] = None,
        **model_kwargs,
    ) -> Union[str, SessionResult]:
        """Run the agent with a simple prompt and return the response.

        Args:
            prompt: User prompt/message
            provider: Override provider (uses default if None)
            model_name: Override model (uses default if None)
            tools: Override tools (uses global tools if None)
            system: Optional system message
            session_id: Session ID for conversation persistence (requires session=...)
            **model_kwargs: Additional model kwargs

        Returns:
            str: The agent's final text response (if no session configured)
            SessionResult: Rich result with pause state (if session configured)

        Example - Basic:
            ```python
            agent = Agent(tools=[get_weather])
            result = agent.run("What's the weather in NYC?")
            print(result)  # "The weather in NYC is Sunny, 72°F"
            ```

        Example - With session:
            ```python
            from ai_infra.llm.session import memory

            agent = Agent(tools=[...], session=memory())
            result = agent.run("Hello", session_id="user-123")
            print(result.content)
            ```
        """
        # Resolve provider and model
        eff_provider = provider or self._default_provider
        eff_model = model_name or self._default_model_name
        eff_provider, eff_model = self._resolve_provider_and_model(eff_provider, eff_model)

        # Merge model kwargs
        eff_kwargs = {**self._default_model_kwargs, **model_kwargs}

        # Get config for session
        config = None
        eff_session_id = session_id or generate_session_id()
        if self._session_config:
            config = self._session_config.get_config(eff_session_id)

        # Use DeepAgent if deep=True
        if self._deep:
            deep_agent = self._build_deep_agent(
                provider=eff_provider,
                model_name=eff_model,
                tools=tools,
                system=system,
            )
            result = deep_agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
            )
        else:
            # Build messages
            messages: List[Dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Run agent
            result = self.run_agent(
                messages=messages,
                provider=eff_provider,
                model_name=eff_model,
                tools=tools,
                model_kwargs=eff_kwargs,
                config=config,
            )

        # If session is configured, return SessionResult
        if self._session_config:
            return self._make_session_result(result, eff_session_id)

        # Extract text content from result (legacy behavior)
        return self._extract_text_content(result)

    def _extract_text_content(self, result: Any) -> str:
        """Extract text content from agent result."""
        if hasattr(result, "get") and "messages" in result:
            # LangGraph agent output format
            msgs = result["messages"]
            if msgs:
                last_msg = msgs[-1]
                return getattr(last_msg, "content", str(last_msg))
        if hasattr(result, "content"):
            return result.content
        return str(result)

    def _convert_subagents(self, subagents: List[Union["Agent", Any]]) -> List[Any]:
        """Convert Agent instances to SubAgent format.

        This allows users to pass Agent instances directly to the subagents
        parameter, and they will be automatically converted to the SubAgent
        format expected by deepagents.

        Args:
            subagents: List of Agent instances or SubAgent dicts

        Returns:
            List of SubAgent dicts
        """
        converted = []
        for agent in subagents:
            if isinstance(agent, Agent):
                # Convert Agent to SubAgent format
                if not agent._name:
                    raise ValueError(
                        "Agent used as subagent must have 'name' set. "
                        "Example: Agent(name='researcher', description='...', ...)"
                    )
                if not agent._description:
                    raise ValueError(
                        "Agent used as subagent must have 'description' set. "
                        "Example: Agent(name='researcher', description='Researches topics', ...)"
                    )

                subagent_dict: Dict[str, Any] = {
                    "name": agent._name,
                    "description": agent._description,
                    "system_prompt": agent._system or "",
                    "tools": list(agent.tools) if agent.tools else [],
                }

                # Add optional model if specified
                if agent._default_provider or agent._default_model_name:
                    # Build model string or pass model kwargs
                    if agent._default_model_name:
                        subagent_dict["model"] = agent._default_model_name

                converted.append(subagent_dict)
            else:
                # Already a SubAgent dict, pass through
                converted.append(agent)
        return converted

    def _make_session_result(self, result: Any, session_id: str) -> SessionResult:
        """Convert agent result to SessionResult."""
        # Check if paused
        paused = is_paused(result)
        pending = get_pending_action(result) if paused else None

        # Extract messages
        messages = []
        if hasattr(result, "get") and "messages" in result:
            messages = result["messages"]

        # Extract content
        content = ""
        if not paused:
            content = self._extract_text_content(result)

        return SessionResult(
            content=content,
            paused=paused,
            pending_action=pending,
            session_id=session_id,
            messages=messages,
        )

    async def arun(
        self,
        prompt: str,
        *,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        system: Optional[str] = None,
        session_id: Optional[str] = None,
        **model_kwargs,
    ) -> Union[str, SessionResult]:
        """Async version of run().

        Args:
            prompt: User prompt/message
            provider: Override provider (uses default if None)
            model_name: Override model (uses default if None)
            tools: Override tools (uses global tools if None)
            system: Optional system message
            session_id: Session ID for conversation persistence (requires session=...)
            **model_kwargs: Additional model kwargs

        Returns:
            str: The agent's final text response (if no session configured)
            SessionResult: Rich result with pause state (if session configured)
        """
        # Resolve provider and model
        eff_provider = provider or self._default_provider
        eff_model = model_name or self._default_model_name
        eff_provider, eff_model = self._resolve_provider_and_model(eff_provider, eff_model)

        # Merge model kwargs
        eff_kwargs = {**self._default_model_kwargs, **model_kwargs}

        # Get config for session
        config = None
        eff_session_id = session_id or generate_session_id()
        if self._session_config:
            config = self._session_config.get_config(eff_session_id)

        # Use DeepAgent if deep=True
        if self._deep:
            deep_agent = self._build_deep_agent(
                provider=eff_provider,
                model_name=eff_model,
                tools=tools,
                system=system,
            )
            result = await deep_agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
            )
        else:
            # Build messages
            messages: List[Dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Run agent
            result = await self.arun_agent(
                messages=messages,
                provider=eff_provider,
                model_name=eff_model,
                tools=tools,
                model_kwargs=eff_kwargs,
                config=config,
            )

        # If session is configured, return SessionResult
        if self._session_config:
            return self._make_session_result(result, eff_session_id)

        # Extract text content from result (legacy behavior)
        return self._extract_text_content(result)

    def resume(
        self,
        session_id: str,
        *,
        approved: bool = True,
        modified_args: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Union[str, SessionResult]:
        """Resume a paused agent with a decision.

        Args:
            session_id: The session ID of the paused agent
            approved: Whether to approve the pending action
            modified_args: Modified arguments (optional, if approved)
            reason: Reason for the decision
            provider: Override provider
            model_name: Override model

        Returns:
            str or SessionResult depending on session configuration

        Example:
            ```python
            # Agent was paused
            result = agent.run("Delete file.txt", session_id="task-1")

            if result.paused:
                # Resume with approval
                result = agent.resume(session_id="task-1", approved=True)
            ```
        """
        if not self._session_config:
            raise ValueError("resume() requires session= to be configured")

        from langgraph.types import Command

        # Resolve provider and model
        eff_provider = provider or self._default_provider
        eff_model = model_name or self._default_model_name
        eff_provider, eff_model = self._resolve_provider_and_model(eff_provider, eff_model)

        # Build resume command
        decision = ResumeDecision(
            approved=approved,
            modified_args=modified_args,
            reason=reason,
        )

        config = self._session_config.get_config(session_id)

        # Get compiled agent
        agent, context = self._make_agent_with_context(
            eff_provider,
            eff_model,
            tools=None,  # Use global tools
            model_kwargs=self._default_model_kwargs,
        )

        # Resume with Command
        result = agent.invoke(
            Command(resume=decision.model_dump()),
            context=context,
            config=config,
        )

        return self._make_session_result(result, session_id)

    async def aresume(
        self,
        session_id: str,
        *,
        approved: bool = True,
        modified_args: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Union[str, SessionResult]:
        """Async version of resume().

        Args:
            session_id: The session ID of the paused agent
            approved: Whether to approve the pending action
            modified_args: Modified arguments (optional, if approved)
            reason: Reason for the decision
            provider: Override provider
            model_name: Override model

        Returns:
            str or SessionResult depending on session configuration
        """
        if not self._session_config:
            raise ValueError("aresume() requires session= to be configured")

        from langgraph.types import Command

        # Resolve provider and model
        eff_provider = provider or self._default_provider
        eff_model = model_name or self._default_model_name
        eff_provider, eff_model = self._resolve_provider_and_model(eff_provider, eff_model)

        # Build resume command
        decision = ResumeDecision(
            approved=approved,
            modified_args=modified_args,
            reason=reason,
        )

        config = self._session_config.get_config(session_id)

        # Get compiled agent
        agent, context = self._make_agent_with_context(
            eff_provider,
            eff_model,
            tools=None,  # Use global tools
            model_kwargs=self._default_model_kwargs,
        )

        # Resume with Command
        result = await agent.ainvoke(
            Command(resume=decision.model_dump()),
            context=context,
            config=config,
        )

        return self._make_session_result(result, session_id)

    def _make_agent_with_context(
        self,
        provider: str,
        model_name: str = None,
        tools: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,
    ) -> Tuple[Any, Any]:
        # Build a composite tool wrapper that applies execution config, approval, and HITL
        def _wrap_tool(t: Any) -> Any:
            # 1. Apply execution config (error handling, timeout, validation)
            wrapped = wrap_tool_with_execution_config(t, self._tool_execution_config)

            # 2. Apply new approval workflow if configured (recommended)
            if self._approval_config:
                wrapped = wrap_tool_for_approval(wrapped, self._approval_config)

            # 3. Apply legacy HITL if configured (for backward compatibility)
            if self._hitl.on_tool_call or self._hitl.on_tool_call_async:
                wrapped = wrap_tool_for_hitl(wrapped, self._hitl)

            return wrapped

        # Extract session config if available
        checkpointer = None
        store = None
        interrupt_before = None
        interrupt_after = None
        if self._session_config:
            checkpointer = self._session_config.storage.get_checkpointer()
            store = self._session_config.storage.get_store()
            interrupt_before = self._session_config.pause_before or None
            interrupt_after = self._session_config.pause_after or None

        return rb_make_agent_with_context(
            self.registry,
            provider=provider,
            model_name=model_name,
            tools=tools,
            extra=extra,
            model_kwargs=model_kwargs,
            tool_controls=tool_controls,
            require_explicit_tools=self.require_explicit_tools,
            global_tools=self.tools,
            # Apply execution config, approval, and HITL wrappers
            hitl_tool_wrapper=_wrap_tool,
            logger=self._logger,
            # Session/checkpoint config
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )

    def _build_deep_agent(
        self,
        provider: str,
        model_name: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        system: Optional[str] = None,
    ) -> Any:
        """Build a DeepAgents agent for autonomous multi-step task execution.

        This method creates a deep agent using LangChain's deepagents package,
        which provides built-in file tools (ls, read, write, edit, glob, grep, execute),
        todo management, and subagent orchestration.

        Args:
            provider: LLM provider
            model_name: Model name
            tools: Additional tools (added to built-in deep agent tools)
            system: System prompt / additional instructions

        Returns:
            Compiled DeepAgent graph
        """
        try:
            from deepagents import create_deep_agent
        except ImportError as e:
            raise ImportError(
                "DeepAgents mode requires 'deepagents' package. "
                "Install with: pip install deepagents"
            ) from e

        # Get model instance from registry
        model = self._get_model_for_deep_agent(provider, model_name)

        # Extract session config
        checkpointer = None
        store = None
        interrupt_on = None
        if self._session_config:
            checkpointer = self._session_config.storage.get_checkpointer()
            store = self._session_config.storage.get_store()
            # Convert pause_before/pause_after to interrupt_on dict
            if self._session_config.pause_before or self._session_config.pause_after:
                interrupt_on = {}
                for tool_name in self._session_config.pause_before or []:
                    interrupt_on[tool_name] = {"before": True}
                for tool_name in self._session_config.pause_after or []:
                    if tool_name in interrupt_on:
                        interrupt_on[tool_name]["after"] = True
                    else:
                        interrupt_on[tool_name] = {"after": True}

        # Merge global tools with provided tools
        all_tools = list(self.tools) if self.tools else []
        if tools:
            all_tools.extend(tools)

        return create_deep_agent(
            model=model,
            tools=all_tools if all_tools else None,
            system_prompt=system,
            middleware=tuple(self._middleware) if self._middleware else (),
            subagents=self._subagents,
            response_format=self._response_format,
            context_schema=self._context_schema,
            checkpointer=checkpointer,
            store=store,
            use_longterm_memory=self._use_longterm_memory,
            interrupt_on=interrupt_on,
        )

    def _get_model_for_deep_agent(self, provider: str, model_name: Optional[str] = None) -> Any:
        """Get a LangChain chat model instance for deep agent.

        Args:
            provider: LLM provider
            model_name: Model name

        Returns:
            BaseChatModel instance
        """
        # Resolve provider and model
        eff_provider, eff_model = self._resolve_provider_and_model(provider, model_name)

        # Get model from registry
        model_info = self.registry.get(eff_provider)
        if not model_info:
            raise ValueError(f"Unknown provider: {eff_provider}")

        chat_model_cls = model_info.get("chat_model")
        if not chat_model_cls:
            raise ValueError(f"Provider {eff_provider} does not support chat models")

        # Build model kwargs
        model_kwargs = {**self._default_model_kwargs}
        if eff_model:
            model_kwargs["model"] = eff_model

        return chat_model_cls(**model_kwargs)

    async def arun_agent(
        self,
        messages: List[Dict[str, Any]],
        provider: str,
        model_name: str = None,
        tools: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        agent, context = self._make_agent_with_context(
            provider, model_name, tools, extra, model_kwargs, tool_controls
        )

        async def _call():
            return await agent.ainvoke({"messages": messages}, context=context, config=config)

        try:
            retry_cfg = (extra or {}).get("retry") if extra else None
            if retry_cfg:
                res = await _with_retry_util(_call, **retry_cfg)
            else:
                res = await _call()
        except Exception as e:
            # Translate provider errors to ai-infra errors
            raise translate_provider_error(e, provider=provider, model=model_name) from e
        ai_msg = await apply_output_gate_async(res, self._hitl)
        return ai_msg

    def run_agent(
        self,
        messages: List[Dict[str, Any]],
        provider: str,
        model_name: str = None,
        tools: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        agent, context = self._make_agent_with_context(
            provider, model_name, tools, extra, model_kwargs, tool_controls
        )
        try:
            res = agent.invoke({"messages": messages}, context=context, config=config)
        except Exception as e:
            # Translate provider errors to ai-infra errors
            raise translate_provider_error(e, provider=provider, model=model_name) from e
        ai_msg = apply_output_gate(res, self._hitl)
        return ai_msg

    async def arun_agent_stream(
        self,
        messages: List[Dict[str, Any]],
        provider: str,
        model_name: str = None,
        tools: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        stream_mode: Union[str, Sequence[str]] = ("updates", "values"),
        tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        agent, context = self._make_agent_with_context(
            provider, model_name, tools, extra, model_kwargs, tool_controls
        )
        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)
        if modes == ["messages"]:
            async for token, meta in agent.astream(
                {"messages": messages}, context=context, config=config, stream_mode="messages"
            ):
                yield token, meta
            return
        last_values = None
        async for mode, chunk in agent.astream(
            {"messages": messages}, context=context, config=config, stream_mode=modes
        ):
            if mode == "values":
                last_values = chunk
                continue
            else:
                yield mode, chunk
        if last_values is not None:
            gated_values = await apply_output_gate_async(last_values, self._hitl)
            yield "values", gated_values

    async def astream_agent_tokens(
        self,
        messages: List[Dict[str, Any]],
        provider: str,
        model_name: str = None,
        tools: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        agent, context = self._make_agent_with_context(
            provider, model_name, tools, extra, model_kwargs, tool_controls
        )
        async for token, meta in agent.astream(
            {"messages": messages},
            context=context,
            config=config,
            stream_mode="messages",
        ):
            yield token, meta

    def agent(
        self,
        provider: str,
        model_name: str = None,
        tools: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        return self._make_agent_with_context(provider, model_name, tools, extra, model_kwargs)

    # ---------- fallbacks (sync) ----------
    def run_with_fallbacks(
        self,
        messages: List[Dict[str, Any]],
        candidates: List[Tuple[str, str]],
        tools: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        def _run_single(provider: str, model_name: str, overrides: Dict[str, Any]):
            eff_extra, eff_model_kwargs, eff_tools, eff_tool_controls = _merge_overrides(
                extra, model_kwargs, tools, tool_controls, overrides
            )
            return self.run_agent(
                messages=messages,
                provider=provider,
                model_name=model_name,
                tools=eff_tools,
                extra=eff_extra,
                model_kwargs=eff_model_kwargs,
                tool_controls=eff_tool_controls,
                config=config,
            )

        return _run_fallbacks_util(
            candidates=candidates,
            run_single=_run_single,
            validate=_is_valid_response,
        )

    # ---------- fallbacks (async) ----------
    async def arun_with_fallbacks(
        self,
        messages: List[Dict[str, Any]],
        candidates: List[Tuple[str, str]],
        tools: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        async def _run_single(provider: str, model_name: str, overrides: Dict[str, Any]):
            eff_extra, eff_model_kwargs, eff_tools, eff_tool_controls = _merge_overrides(
                extra, model_kwargs, tools, tool_controls, overrides
            )
            return await self.arun_agent(
                messages=messages,
                provider=provider,
                model_name=model_name,
                tools=eff_tools,
                extra=eff_extra,
                model_kwargs=eff_model_kwargs,
                tool_controls=eff_tool_controls,
                config=config,
            )

        return await _arun_fallbacks_util(
            candidates=candidates,
            run_single_async=_run_single,
            validate=_is_valid_response,
        )
