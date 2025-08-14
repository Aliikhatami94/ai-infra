from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Union, Sequence
import logging

from pydantic import BaseModel

from .settings import ModelSettings
from .runtime_bind import ModelRegistry, make_agent_with_context as rb_make_agent_with_context
from .tool_controls import ToolCallControls
from .tools import apply_output_gate, wrap_tool_for_hitl, HITLConfig
from .utils import sanitize_model_kwargs, with_retry as _with_retry_util, run_with_fallbacks as _run_fallbacks_util


class CoreLLM:
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.registry = ModelRegistry()
        self.tools: List[Any] = []
        self._hitl = HITLConfig()
        self.require_explicit_tools: bool = False

    def set_global_tools(self, tools: List[Any]):
        """Set global tools used when per-call tools not provided (unless explicit required)."""
        self.tools = tools or []

    def require_tools_explicit(self, required: bool = True):
        """If enabled, callers must pass tools each run (tools=[] to disable)."""
        self.require_explicit_tools = required

    def set_hitl(self, *, on_model_output=None, on_tool_call=None):
        """Register Human-In-The-Loop (HITL) callbacks for model outputs and tool calls.

        Parameters:
            on_model_output: Callable taking the final AI message (object or dict) and
                returning one of:
                  {"action": "pass"}
                  {"action": "modify", "replacement": str}
                  {"action": "block",  "replacement": str}
                - pass: leave message unchanged.
                - modify: replace final message content with replacement.
                - block: same as modify but semantically indicates rejection.
                Replacement is applied to ai_msg.content when present, or last
                messages[-1].content when the value is a state dict.

            on_tool_call: Callable invoked before an underlying tool executes.
                Signature: fn(tool_name: str, args: dict) -> decision dict
                Returns one of:
                  {"action": "pass"}
                  {"action": "modify", "args": {..new args..}}
                  {"action": "block",  "replacement": str}
                Semantics:
                  - pass: run tool with original args.
                  - modify: run tool with updated args ("args" key). If omitted,
                    original args are retained.
                  - block: tool is NOT executed; wrapper returns the replacement
                    string (or object). If the tool expected structured output and
                    replacement is JSON-parseable, we attempt json.loads(replacement).

        Notes:
            - Any exception inside callbacks defaults to action "pass".
            - Missing or unrecognized action defaults to "pass".
            - For streaming token modes (messages-only streaming), on_model_output
              modifications cannot be applied retroactively.
        """
        self._hitl.set(on_model_output=on_model_output, on_tool_call=on_tool_call)

    @staticmethod
    def no_tools() -> Dict[str, Any]:
        """Convenience: disable tool calling for a run."""
        return {"tool_controls": {"tool_choice": "none"}}

    @staticmethod
    def force_tool(name: str, *, once: bool = False, parallel: bool = False) -> Dict[str, Any]:
        """Convenience: force a specific tool (optionally only for first call)."""
        return {"tool_controls": {"tool_choice": {"name": name}, "force_once": once, "parallel_tool_calls": parallel}}

    def set_model(self, provider: str, model_name: str, **kwargs):
        # Delegate to registry (idempotent get/create)
        return self.registry.get_or_create(provider, model_name, **(kwargs or {}))

    def _get_or_create(self, provider: str, model_name: str, **kwargs):  # retained for backward compat
        return self.registry.get_or_create(provider, model_name, **kwargs)

    def _make_agent_with_context(
            self,
            provider: str,
            model_name: str,
            tools: Optional[List[Any]] = None,
            extra: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,
    ) -> Tuple[Any, ModelSettings]:
        """Delegate agent/context construction to runtime_bind.make_agent_with_context."""
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
            hitl_tool_wrapper=(lambda t: wrap_tool_for_hitl(t, self._hitl) if self._hitl.on_tool_call else None),
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
        ai_msg = apply_output_gate(res, self._hitl)  # replaced _apply_hitl
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
            gated_values = apply_output_gate(last_values, self._hitl)  # replaced manual gating
            yield "values", gated_values

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

    @staticmethod
    def make_messages(user: str, system: Optional[str] = None, extras: Optional[List[Dict[str, Any]]] = None):
        msgs: List[Dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        if extras:
            msgs.extend(extras)
        return msgs

    def run_with_fallbacks(
            self,
            messages: List[Dict[str, Any]],
            candidates: List[Tuple[str, str]],
            tools: Optional[List[Any]] = None,
            extra: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        def _single(provider: str, model_name: str):
            return self.run_agent(messages, provider, model_name, tools, extra, model_kwargs)
        return _run_fallbacks_util(messages, candidates, _single)

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
        """Return (agent, context) for custom wiring."""
        return self._make_agent_with_context(provider, model_name, tools, extra, model_kwargs)
