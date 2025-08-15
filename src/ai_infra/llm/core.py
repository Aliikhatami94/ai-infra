from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Union, Sequence
import logging

from pydantic import BaseModel

from .settings import ModelSettings
from .runtime_bind import ModelRegistry, make_agent_with_context as rb_make_agent_with_context
from .tool_controls import ToolCallControls
from .tools import apply_output_gate, wrap_tool_for_hitl, HITLConfig
from .utils import sanitize_model_kwargs, with_retry as _with_retry_util, run_with_fallbacks as _run_fallbacks_util


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

    def set_hitl(self, *, on_model_output=None, on_tool_call=None):
        self._hitl.set(on_model_output=on_model_output, on_tool_call=on_tool_call)

    @staticmethod
    def no_tools() -> Dict[str, Any]:
        return {"tool_controls": {"tool_choice": "none"}}

    @staticmethod
    def force_tool(name: str, *, once: bool = False, parallel: bool = False) -> Dict[str, Any]:
        return {"tool_controls": {"tool_choice": {"name": name}, "force_once": once, "parallel_tool_calls": parallel}}

    # model registry
    def set_model(self, provider: str, model_name: str, **kwargs):
        return self.registry.get_or_create(provider, model_name, **(kwargs or {}))

    def _get_or_create(self, provider: str, model_name: str, **kwargs):
        return self.registry.get_or_create(provider, model_name, **kwargs)

    # helpers
    @staticmethod
    def make_messages(user: str, system: Optional[str] = None, extras: Optional[List[Dict[str, Any]]] = None):
        msgs: List[Dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        if extras:
            msgs.extend(extras)
        return msgs

    def with_structured_output(
            self,
            provider: str,
            model_name: str,
            schema: Union[type[BaseModel], Dict[str, Any]],
            **model_kwargs,
    ):
        model = self.registry.get_or_create(provider, model_name, **model_kwargs)
        try:
            return model.with_structured_output(schema)
        except Exception as e:  # pragma: no cover
            self._logger.warning(
                "[CoreLLM] Structured output unavailable; provider=%s model=%s schema=%s error=%s",
                provider,
                model_name,
                getattr(schema, "__name__", type(schema)),
                e,
                exc_info=True,
            )
            return model


class CoreLLM(BaseLLMCore):
    """Direct model convenience interface (no agent graph)."""

    def chat(
            self,
            user_msg: str,
            provider: str,
            model_name: str,
            system: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
            **model_kwargs,
    ):
        sanitize_model_kwargs(model_kwargs)
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = self.make_messages(user_msg, system)
        def _call():
            return model.invoke(messages)
        retry_cfg = (extra or {}).get("retry") if extra else None
        if retry_cfg:
            import asyncio
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop and running_loop.is_running():
                self._logger.warning(
                    "[CoreLLM] chat() retry config ignored due to existing event loop; use achat() instead."
                )
                res = _call()
            else:
                async def _acall():
                    return _call()
                res = asyncio.run(_with_retry_util(_acall, **retry_cfg))
        else:
            res = _call()
        ai_msg = apply_output_gate(res, self._hitl)
        return ai_msg

    async def achat(
            self,
            user_msg: str,
            provider: str,
            model_name: str,
            system: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
            **model_kwargs,
    ):
        sanitize_model_kwargs(model_kwargs)
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = self.make_messages(user_msg, system)
        async def _call():
            return await model.ainvoke(messages)
        retry_cfg = (extra or {}).get("retry") if extra else None
        res = await (_with_retry_util(_call, **retry_cfg) if retry_cfg else _call())
        ai_msg = apply_output_gate(res, self._hitl)
        return ai_msg

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
        messages = self.make_messages(user_msg, system)
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
            model_name: str,
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
            hitl_tool_wrapper=(lambda t: wrap_tool_for_hitl(t, self._hitl)) if self._hitl.on_tool_call else None,
            logger=self._logger,
        )

    async def arun_agent(
            self,
            messages: List[Dict[str, Any]],
            provider: str,
            model_name: str,
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
        ai_msg = apply_output_gate(res, self._hitl)
        return ai_msg

    def run_agent(
            self,
            messages: List[Dict[str, Any]],
            provider: str,
            model_name: str,
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
            model_name: str,
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
            gated_values = apply_output_gate(last_values, self._hitl)
            yield "values", gated_values

    async def astream_agent_tokens(
            self,
            messages: List[Dict[str, Any]],
            provider: str,
            model_name: str,
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
            model_name: str,
            tools: Optional[List[Any]] = None,
            extra: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        return self._make_agent_with_context(provider, model_name, tools, extra, model_kwargs)

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
        """
        Try each (provider, model_name) in candidates in order until one succeeds.
        Uses run_agent under the hood, so HITL gating & tool policy still apply.

        Args:
            messages: chat state/messages
            candidates: ordered list of (provider, model_name)
            tools: per-call tools (pass [] to disable; None to use global tools if allowed)
            extra: misc runtime options (e.g., recursion_limit)
            model_kwargs: per-model kwargs (e.g., temperature)
            tool_controls: tool_choice/parallel/force_once controls
            config: graph/runtime config forwarded to agent.invoke

        Returns:
            The first successful agent result (already gated via apply_output_gate in run_agent).
        """
        def _single(provider: str, model_name: str):
            return self.run_agent(
                messages=messages,
                provider=provider,
                model_name=model_name,
                tools=tools,
                extra=extra,
                model_kwargs=model_kwargs,
                tool_controls=tool_controls,
                config=config,
            )

        # IMPORTANT: return the value from the fallback utility
        return _run_fallbacks_util(messages, candidates, _single)

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
        async def _single(provider: str, model_name: str):
            return await self.arun_agent(
                messages=messages,
                provider=provider,
                model_name=model_name,
                tools=tools,
                extra=extra,
                model_kwargs=model_kwargs,
                tool_controls=tool_controls,
                config=config,
            )
        # If your _run_fallbacks_util has no async version,
        # implement a simple loop here that awaits _single for each candidate.
        for prov, model in candidates:
            try:
                return await _single(prov, model)
            except Exception:
                continue
        raise RuntimeError("All fallback candidates failed.")