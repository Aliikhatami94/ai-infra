from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Union, Sequence, Literal
import logging

from pydantic import BaseModel
from langchain_core.messages import BaseMessage

from ai_infra.llm.utils.settings import ModelSettings
from ai_infra.llm.utils.runtime_bind import ModelRegistry, make_agent_with_context as rb_make_agent_with_context
from ai_infra.llm.tools.tool_controls import ToolCallControls
from .tools import apply_output_gate, wrap_tool_for_hitl, HITLConfig, apply_output_gate_async
from .utils import (
    sanitize_model_kwargs,
    with_retry as _with_retry_util,
    run_with_fallbacks as _run_fallbacks_util,
    arun_with_fallbacks as _arun_fallbacks_util,
    is_valid_response as _is_valid_response,
    merge_overrides as _merge_overrides,
    make_messages as _make_messages,
)
from ai_infra.llm.utils.structured import (
    build_structured_messages,
    structured_mode_call_async,
    structured_mode_call_sync,
    validate_or_raise,
    is_pydantic_schema,
    coerce_from_text_or_fragment,
    coerce_structured_result,
)


class BaseLLMCore:
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.registry = ModelRegistry()
        self.tools: List[Any] = []
        self._hitl = HITLConfig()
        self.require_explicit_tools: bool = False

    # shared configuration / policies
    def set_global_tools(self, tools: List[Any]):
        self.tools = tools or []

    def require_tools_explicit(self, required: bool = True):
        self.require_explicit_tools = required

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
            return model.with_structured_output(schema, **({} if method is None else {"method": method}))
        except Exception as e:  # pragma: no cover
            self._logger.warning(
                "[CoreLLM] Structured output unavailable; provider=%s model=%s schema=%s error=%s",
                provider, model_name, getattr(schema, "__name__", type(schema)), e, exc_info=True,
            )
            return model

    def _run_with_retry_sync(self, fn, retry_cfg):
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            self._logger.warning("[CoreLLM] chat() retry config ignored due to running loop; use achat().")
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


class CoreLLM(BaseLLMCore):
    """Direct model convenience interface (no agent graph)."""

    def chat(
            self,
            user_msg: str,
            provider: str,
            model_name: str,
            system: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
            output_schema: Union[type[BaseModel], Dict[str, Any], None] = None,
            output_method: Literal["json_schema", "json_mode", "function_calling", "prompt"] | None = "prompt",
            **model_kwargs,
    ):
        sanitize_model_kwargs(model_kwargs)

        # PROMPT method uses shared helper
        if output_schema is not None and output_method == "prompt":
            return self._prompt_structured_sync(
                user_msg=user_msg,
                system=system,
                provider=provider,
                model_name=model_name,
                schema=output_schema,
                extra=extra,
                model_kwargs=model_kwargs,
            )

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

        if output_schema is not None and is_pydantic_schema(output_schema):
            return coerce_structured_result(output_schema, res)

        try:
            return apply_output_gate(res, self._hitl)
        except Exception:
            return res

    async def achat(
            self,
            user_msg: str,
            provider: str,
            model_name: str,
            system: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
            output_schema: Union[type[BaseModel], Dict[str, Any], None] = None,
            output_method: Literal["json_schema", "json_mode", "function_calling", "prompt"] | None = "prompt",
            **model_kwargs,
    ):
        sanitize_model_kwargs(model_kwargs)

        if output_schema is not None and output_method == "prompt":
            return await self._prompt_structured_async(
                user_msg=user_msg,
                system=system,
                provider=provider,
                model_name=model_name,
                schema=output_schema,
                extra=extra,
                model_kwargs=model_kwargs,
            )

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

        if output_schema is not None and is_pydantic_schema(output_schema):
            return coerce_structured_result(output_schema, res)

        try:
            return await apply_output_gate_async(res, self._hitl)
        except Exception:
            return res

    async def stream_tokens(
            self,
            user_msg: str,
            provider: str,
            model_name: str,
            system: Optional[str] = None,
            *,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **model_kwargs,
    ):
        sanitize_model_kwargs(model_kwargs)
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


class CoreAgent(BaseLLMCore):
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
                (lambda t: wrap_tool_for_hitl(t, self._hitl)) if (self._hitl.on_tool_call or self._hitl.on_tool_call_async) else None),
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
            config: Optional[Dict[str, Any]] = None
    ) -> Any:
        agent, context = self._make_agent_with_context(provider, model_name, tools, extra, model_kwargs, tool_controls)
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
            config: Optional[Dict[str, Any]] = None
    ) -> Any:
        agent, context = self._make_agent_with_context(provider, model_name, tools, extra, model_kwargs, tool_controls)
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
            config: Optional[Dict[str, Any]] = None
    ):
        agent, context = self._make_agent_with_context(provider, model_name, tools, extra, model_kwargs, tool_controls)
        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)
        if modes == ["messages"]:
            async for token, meta in agent.astream(
                    {"messages": messages},
                    context=context,
                    config=config,
                    stream_mode="messages"
            ):
                yield token, meta
            return
        last_values = None
        async for mode, chunk in agent.astream(
                {"messages": messages},
                context=context,
                config=config,
                stream_mode=modes
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