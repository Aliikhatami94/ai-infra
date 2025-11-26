from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from ai_infra.llm.defaults import DEFAULT_MODELS
from ai_infra.llm.providers.discovery import get_default_provider
from ai_infra.llm.tools.tool_controls import ToolCallControls
from ai_infra.llm.utils.logging_hooks import (
    ErrorContext,
    LoggingHooks,
    RequestContext,
    ResponseContext,
)
from ai_infra.llm.utils.runtime_bind import ModelRegistry
from ai_infra.llm.utils.runtime_bind import make_agent_with_context as rb_make_agent_with_context
from ai_infra.llm.utils.settings import ModelSettings
from ai_infra.llm.utils.structured import (
    build_structured_messages,
    coerce_from_text_or_fragment,
    coerce_structured_result,
    is_pydantic_schema,
    structured_mode_call_async,
    structured_mode_call_sync,
    validate_or_raise,
)

from .tools import HITLConfig, apply_output_gate, apply_output_gate_async, wrap_tool_for_hitl
from .utils import arun_with_fallbacks as _arun_fallbacks_util
from .utils import is_valid_response as _is_valid_response
from .utils import make_messages as _make_messages
from .utils import merge_overrides as _merge_overrides
from .utils import run_with_fallbacks as _run_fallbacks_util
from .utils import sanitize_model_kwargs
from .utils import with_retry as _with_retry_util


class BaseLLM:
    """Base class for LLM and Agent with shared configuration and utilities."""

    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.registry = ModelRegistry()
        self.tools: List[Any] = []
        self._hitl = HITLConfig()
        self._logging_hooks = LoggingHooks()
        self.require_explicit_tools: bool = False

    # shared configuration / policies
    def set_global_tools(self, tools: List[Any]):
        self.tools = tools or []

    def require_tools_explicit(self, required: bool = True):
        self.require_explicit_tools = required

    def set_logging_hooks(
        self,
        *,
        on_request=None,
        on_response=None,
        on_error=None,
        on_request_async=None,
        on_response_async=None,
        on_error_async=None,
    ):
        """Configure request/response logging hooks.

        Args:
            on_request: Callback(RequestContext) called before model invocation
            on_response: Callback(ResponseContext) called after successful response
            on_error: Callback(ErrorContext) called when an error occurs
            on_request_async: Async version of on_request
            on_response_async: Async version of on_response
            on_error_async: Async version of on_error

        Example:
            ```python
            import logging
            logger = logging.getLogger(__name__)

            llm = LLM()
            llm.set_logging_hooks(
                on_request=lambda ctx: logger.info("Request to %s/%s", ctx.provider, ctx.model_name),
                on_response=lambda ctx: logger.info("Response in %.2fms", ctx.duration_ms),
                on_error=lambda ctx: logger.error("Error: %s", ctx.error),
            )
            ```
        """
        self._logging_hooks.set(
            on_request=on_request,
            on_response=on_response,
            on_error=on_error,
            on_request_async=on_request_async,
            on_response_async=on_response_async,
            on_error_async=on_error_async,
        )
        return self

    def set_hitl(
        self,
        *,
        on_model_output=None,
        on_tool_call=None,
        on_model_output_async=None,
        on_tool_call_async=None,
    ):
        self._hitl.set(
            on_model_output=on_model_output,
            on_tool_call=on_tool_call,
            on_model_output_async=on_model_output_async,
            on_tool_call_async=on_tool_call_async,
        )

    @staticmethod
    def make_sys_gate(autoapprove: bool = False):
        def gate(tool_name: str, args: dict):
            if autoapprove:
                return {"action": "pass"}
            print(f"\nTool request: {tool_name}\nArgs: {args}")
            try:
                ans = input("Approve? [y]es / [b]lock: ").strip().lower()
            except EOFError:
                return {"action": "block", "replacement": "[auto-block: no input]"}
            if ans.startswith("y"):
                return {"action": "pass"}
            return {"action": "block", "replacement": "[blocked by user]"}

        return gate

    # model registry
    def set_model(self, provider: str, model_name: str, **kwargs):
        return self.registry.get_or_create(provider, model_name, **(kwargs or {}))

    def _get_or_create(self, provider: str, model_name: str, **kwargs):
        return self.registry.get_or_create(provider, model_name, **kwargs)

    def get_model(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **model_kwargs,
    ):
        """
        Get the underlying LangChain chat model for direct use.

        Provides access to the raw LangChain model for advanced use cases
        that require direct model interaction.

        Args:
            provider: Provider name (e.g., "openai", "anthropic"). Auto-detected if None.
            model_name: Model name (e.g., "gpt-4o"). Uses provider default if None.
            **model_kwargs: Additional kwargs passed to the model constructor.

        Returns:
            LangChain BaseChatModel instance.

        Raises:
            ValueError: If no provider is specified and none can be auto-detected.

        Example:
            >>> llm = LLM()
            >>> model = llm.get_model()  # Auto-detect provider
            >>> model = llm.get_model("anthropic", "claude-3-5-sonnet-latest")
            >>> # Use LangChain model directly
            >>> response = model.invoke([HumanMessage(content="Hello")])
        """
        # Resolve provider and model (auto-detect if not specified)
        resolved_provider, resolved_model = self._resolve_provider_and_model(provider, model_name)
        return self.registry.get_or_create(resolved_provider, resolved_model, **model_kwargs)

    def with_structured_output(
        self,
        provider: str,
        model_name: str,
        schema: Union[type[BaseModel], Dict[str, Any]],
        *,
        method: Literal["json_schema", "json_mode", "function_calling"] | None = "json_mode",
        **model_kwargs,
    ):
        model = self.registry.get_or_create(provider, model_name, **model_kwargs)
        try:
            # Pass method through if provided (LangChain 0.3 supports this)
            return model.with_structured_output(
                schema, **({} if method is None else {"method": method})
            )
        except Exception as e:  # pragma: no cover
            self._logger.warning(
                "[LLM] Structured output unavailable; provider=%s model=%s schema=%s error=%s",
                provider,
                model_name,
                getattr(schema, "__name__", type(schema)),
                e,
                exc_info=True,
            )
            return model

    def _run_with_retry_sync(self, fn, retry_cfg):
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            self._logger.warning(
                "[LLM] chat() retry config ignored due to running loop; use achat()."
            )
            return fn()

        async def _acall():
            return fn()

        return asyncio.run(_with_retry_util(_acall, **retry_cfg))

    # ========== PROMPT method helpers (shared by chat/achat) ==========
    def _prompt_structured_sync(
        self,
        *,
        user_msg: str,
        system: Optional[str],
        provider: str,
        model_name: str,
        schema: Union[type[BaseModel], Dict[str, Any]],
        extra: Optional[Dict[str, Any]],
        model_kwargs: Dict[str, Any],
    ) -> BaseModel:
        model = self.set_model(provider, model_name, **model_kwargs)
        messages: List[BaseMessage] = build_structured_messages(
            schema=schema, user_msg=user_msg, system_preamble=system
        )

        def _call():
            return model.invoke(messages)

        retry_cfg = (extra or {}).get("retry") if extra else None
        res = _call() if not retry_cfg else self._run_with_retry_sync(_call, retry_cfg)
        content = getattr(res, "content", None) or str(res)

        # Try direct/fragment validation
        coerced = coerce_from_text_or_fragment(schema, content)
        if coerced is not None:
            return coerced

        # Final fallback: provider structured mode (json_mode)
        try:
            return structured_mode_call_sync(
                self.with_structured_output,
                provider,
                model_name,
                schema,
                messages,
                model_kwargs,
            )
        except Exception:
            return validate_or_raise(schema, content)

    async def _prompt_structured_async(
        self,
        *,
        user_msg: str,
        system: Optional[str],
        provider: str,
        model_name: str,
        schema: Union[type[BaseModel], Dict[str, Any]],
        extra: Optional[Dict[str, Any]],
        model_kwargs: Dict[str, Any],
    ) -> BaseModel:
        """Async variant of prompt-only structured output with robust JSON fallback."""
        model = self.set_model(provider, model_name, **model_kwargs)
        messages: List[BaseMessage] = build_structured_messages(
            schema=schema, user_msg=user_msg, system_preamble=system
        )

        async def _call():
            return await model.ainvoke(messages)

        retry_cfg = (extra or {}).get("retry") if extra else None
        res = await (_with_retry_util(_call, **retry_cfg) if retry_cfg else _call())
        content = getattr(res, "content", None) or str(res)

        # Try direct/fragment validation
        coerced = coerce_from_text_or_fragment(schema, content)
        if coerced is not None:
            return coerced

        # Final fallback: provider structured mode (json_mode)
        try:
            return await structured_mode_call_async(
                self.with_structured_output,
                provider,
                model_name,
                schema,
                messages,
                model_kwargs,
            )
        except Exception:
            return validate_or_raise(schema, content)


class LLM(BaseLLM):
    """Direct model convenience interface (no agent graph)."""

    # =========================================================================
    # Discovery API - Static methods for provider/model discovery
    # =========================================================================

    @staticmethod
    def list_providers() -> List[str]:
        """
        List all supported provider names.

        Returns:
            List of provider names: ["openai", "anthropic", "google_genai", "xai"]

        Example:
            >>> LLM.list_providers()
            ['openai', 'anthropic', 'google_genai', 'xai']
        """
        from ai_infra.llm.providers.discovery import list_providers

        return list_providers()

    @staticmethod
    def list_configured_providers() -> List[str]:
        """
        List providers that have API keys configured.

        Returns:
            List of provider names with configured API keys.

        Example:
            >>> LLM.list_configured_providers()
            ['openai', 'anthropic']  # Only if these have API keys set
        """
        from ai_infra.llm.providers.discovery import list_configured_providers

        return list_configured_providers()

    @staticmethod
    def list_models(provider: str, *, refresh: bool = False) -> List[str]:
        """
        List available models for a specific provider.

        Fetches models dynamically from the provider's API.
        Results are cached for 1 hour by default.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            refresh: Force refresh from API, bypassing cache

        Returns:
            List of model IDs available from the provider.

        Raises:
            ValueError: If provider is not supported.
            RuntimeError: If provider is not configured (no API key).

        Example:
            >>> LLM.list_models("openai")
            ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', ...]
        """
        from ai_infra.llm.providers.discovery import list_models

        return list_models(provider, refresh=refresh)

    @staticmethod
    def list_all_models(*, refresh: bool = False) -> Dict[str, List[str]]:
        """
        List models for all configured providers.

        Args:
            refresh: Force refresh from API, bypassing cache

        Returns:
            Dict mapping provider name to list of model IDs.

        Example:
            >>> LLM.list_all_models()
            {
                'openai': ['gpt-4o', 'gpt-4o-mini', ...],
                'anthropic': ['claude-sonnet-4-20250514', ...],
            }
        """
        from ai_infra.llm.providers.discovery import list_all_models

        return list_all_models(refresh=refresh)

    @staticmethod
    def is_provider_configured(provider: str) -> bool:
        """
        Check if a provider has its API key configured.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")

        Returns:
            True if the provider's API key environment variable is set.

        Example:
            >>> LLM.is_provider_configured("openai")
            True
        """
        from ai_infra.llm.providers.discovery import is_provider_configured

        return is_provider_configured(provider)

    def _resolve_provider_and_model(
        self,
        provider: Optional[str],
        model_name: Optional[str],
    ) -> Tuple[str, str]:
        """
        Resolve provider and model, auto-detecting from environment if not specified.

        Args:
            provider: Provider name or None to auto-detect
            model_name: Model name or None to use provider's default

        Returns:
            Tuple of (provider, model_name)

        Raises:
            ValueError: If no provider is specified and none can be auto-detected
        """
        # Auto-detect provider if not specified
        if provider is None:
            provider = get_default_provider()
            if provider is None:
                raise ValueError(
                    "No LLM provider configured. Set one of these environment variables: "
                    "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or XAI_API_KEY. "
                    "Or explicitly pass provider='openai' (etc.) to the method."
                )

        # Use default model for provider if not specified
        if model_name is None:
            model_name = DEFAULT_MODELS.get(provider, "gpt-4o-mini")

        return provider, model_name

    # =========================================================================
    # Chat methods
    # =========================================================================

    def chat(
        self,
        user_msg: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        system: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        output_schema: Union[type[BaseModel], Dict[str, Any], None] = None,
        output_method: (
            Literal["json_schema", "json_mode", "function_calling", "prompt"] | None
        ) = "prompt",
        **model_kwargs,
    ):
        sanitize_model_kwargs(model_kwargs)

        # Resolve provider and model (auto-detect if not specified)
        provider, model_name = self._resolve_provider_and_model(provider, model_name)

        # Create request context for logging hooks
        request_ctx = RequestContext(
            user_msg=user_msg,
            system=system,
            provider=provider,
            model_name=model_name,
            model_kwargs=model_kwargs,
        )
        self._logging_hooks.call_request_sync(request_ctx)
        start_time = time.time()

        try:
            # PROMPT method uses shared helper
            if output_schema is not None and output_method == "prompt":
                res = self._prompt_structured_sync(
                    user_msg=user_msg,
                    system=system,
                    provider=provider,
                    model_name=model_name,
                    schema=output_schema,
                    extra=extra,
                    model_kwargs=model_kwargs,
                )
            else:
                # otherwise: existing structured (json_mode/function_calling/json_schema) or plain
                if output_schema is not None:
                    model = self.with_structured_output(
                        provider, model_name, output_schema, method=output_method, **model_kwargs
                    )
                else:
                    model = self.set_model(provider, model_name, **model_kwargs)

                messages = _make_messages(user_msg, system)

                def _call():
                    return model.invoke(messages)

                retry_cfg = (extra or {}).get("retry") if extra else None
                res = _call() if not retry_cfg else self._run_with_retry_sync(_call, retry_cfg)

            # Call response hook
            duration_ms = (time.time() - start_time) * 1000
            response_ctx = ResponseContext(
                request=request_ctx,
                response=res,
                duration_ms=duration_ms,
                token_usage=getattr(res, "usage_metadata", None),
            )
            self._logging_hooks.call_response_sync(response_ctx)

            if output_schema is not None and is_pydantic_schema(output_schema):
                return coerce_structured_result(output_schema, res)

            try:
                return apply_output_gate(res, self._hitl)
            except Exception:
                return res

        except Exception as e:
            # Call error hook
            duration_ms = (time.time() - start_time) * 1000
            error_ctx = ErrorContext(
                request=request_ctx,
                error=e,
                duration_ms=duration_ms,
            )
            self._logging_hooks.call_error_sync(error_ctx)
            raise

    async def achat(
        self,
        user_msg: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        system: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        output_schema: Union[type[BaseModel], Dict[str, Any], None] = None,
        output_method: (
            Literal["json_schema", "json_mode", "function_calling", "prompt"] | None
        ) = "prompt",
        **model_kwargs,
    ):
        sanitize_model_kwargs(model_kwargs)

        # Resolve provider and model (auto-detect if not specified)
        provider, model_name = self._resolve_provider_and_model(provider, model_name)

        # Create request context for logging hooks
        request_ctx = RequestContext(
            user_msg=user_msg,
            system=system,
            provider=provider,
            model_name=model_name,
            model_kwargs=model_kwargs,
        )
        await self._logging_hooks.call_request_async(request_ctx)
        start_time = time.time()

        try:
            if output_schema is not None and output_method == "prompt":
                res = await self._prompt_structured_async(
                    user_msg=user_msg,
                    system=system,
                    provider=provider,
                    model_name=model_name,
                    schema=output_schema,
                    extra=extra,
                    model_kwargs=model_kwargs,
                )
            else:
                if output_schema is not None:
                    model = self.with_structured_output(
                        provider, model_name, output_schema, method=output_method, **model_kwargs
                    )
                else:
                    model = self.set_model(provider, model_name, **model_kwargs)

                messages = _make_messages(user_msg, system)

                async def _call():
                    return await model.ainvoke(messages)

                retry_cfg = (extra or {}).get("retry") if extra else None
                res = await (_with_retry_util(_call, **retry_cfg) if retry_cfg else _call())

            # Call response hook
            duration_ms = (time.time() - start_time) * 1000
            response_ctx = ResponseContext(
                request=request_ctx,
                response=res,
                duration_ms=duration_ms,
                token_usage=getattr(res, "usage_metadata", None),
            )
            await self._logging_hooks.call_response_async(response_ctx)

            if output_schema is not None and is_pydantic_schema(output_schema):
                return coerce_structured_result(output_schema, res)

            try:
                return await apply_output_gate_async(res, self._hitl)
            except Exception:
                return res

        except Exception as e:
            # Call error hook
            duration_ms = (time.time() - start_time) * 1000
            error_ctx = ErrorContext(
                request=request_ctx,
                error=e,
                duration_ms=duration_ms,
            )
            await self._logging_hooks.call_error_async(error_ctx)
            raise

    async def stream_tokens(
        self,
        user_msg: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        system: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **model_kwargs,
    ):
        sanitize_model_kwargs(model_kwargs)

        # Resolve provider and model (auto-detect if not specified)
        provider, model_name = self._resolve_provider_and_model(provider, model_name)

        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if top_p is not None:
            model_kwargs["top_p"] = top_p
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = _make_messages(user_msg, system)
        async for event in model.astream(messages):
            text = getattr(event, "content", None)
            if text is None:
                text = getattr(event, "delta", None) or getattr(event, "text", None)
            if text is None:
                text = str(event)
            meta = {"raw": event}
            yield text, meta


class Agent(BaseLLM):
    """Agent-oriented interface (tool calling, streaming updates, fallbacks)."""

    def _make_agent_with_context(
        self,
        provider: str,
        model_name: str = None,
        tools: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,
    ) -> Tuple[Any, ModelSettings]:
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
            # Only provide a wrapper if HITL tool callback is active
            hitl_tool_wrapper=(
                (lambda t: wrap_tool_for_hitl(t, self._hitl))
                if (self._hitl.on_tool_call or self._hitl.on_tool_call_async)
                else None
            ),
            logger=self._logger,
        )

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

        retry_cfg = (extra or {}).get("retry") if extra else None
        if retry_cfg:
            res = await _with_retry_util(_call, **retry_cfg)
        else:
            res = await _call()
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
        res = agent.invoke({"messages": messages}, context=context, config=config)
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
            # on_attempt=lambda i, p, m: self._logger.info("Trying %s/%s (%d)", p, m, i),
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


# Backward-compatible aliases (deprecated)
def _deprecated_alias(name: str, new_class: type) -> type:
    """Create a deprecated alias that warns on instantiation."""

    class DeprecatedAlias(new_class):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                f"{name} is deprecated, use {new_class.__name__} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    DeprecatedAlias.__name__ = name
    DeprecatedAlias.__qualname__ = name
    return DeprecatedAlias


# NOTE: CoreLLM and CoreAgent aliases removed - use LLM and Agent directly
