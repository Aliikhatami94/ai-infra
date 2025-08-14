from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Union, Sequence
import logging

from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime
from pydantic import BaseModel
from langchain_core.tools import BaseTool, tool as lc_tool, StructuredTool

from .settings import ModelSettings
from .utils import validate_provider_and_model, build_model_key, initialize_model
from .tool_controls import normalize_tool_controls, ToolCallControls
from .runtime_bind import ModelRegistry, make_agent_with_context as rb_make_agent_with_context


class CoreLLM:
    """
    Minimal, batteries-included LLM facade:
    - Model cache across providers
    - Agents from LangGraph prebuilt ReAct
    - Provider-normalized tool controls
    - Streaming (graph 'updates/values' and token 'messages')
    - Optional structured output
    """
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.registry = ModelRegistry()  # replaces prior self.models cache
        # self.models removed
        self.tools: List[Any] = []
        # Optional hooks
        self._hitl: Dict[str, Any] = {"on_model_output": None}
        self.require_explicit_tools: bool = False

    # ---------- Tool policy ----------
    def set_global_tools(self, tools: List[Any]):
        """Set global tools used when per-call tools not provided (unless explicit required)."""
        self.tools = tools or []

    def require_tools_explicit(self, required: bool = True):
        """If enabled, callers must pass tools each run (tools=[] to disable)."""
        self.require_explicit_tools = required

    # ---------- Hooks & helpers ----------
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
        self._hitl["on_model_output"] = on_model_output
        self._hitl["on_tool_call"] = on_tool_call

    @staticmethod
    def _maybe_await(result):
        """Resolve an awaitable in a sync context.
        - If result is a coroutine: run it with asyncio.run when no loop running.
        - If result is another awaitable (e.g. Future), wrap it in a coroutine first.
        - If an event loop is already running, we cannot synchronously wait safely; return None.
        """
        import inspect, asyncio, logging
        if not inspect.isawaitable(result):
            return result
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        # If a loop is running, avoid blocking; caller should supply sync callback instead.
        if loop and loop.is_running():
            logging.getLogger(__name__).warning(
                "_maybe_await: async HITL callback ignored (event loop active in sync pathway). Use async APIs for async callbacks."
            )
            return None
        # Ensure we have a coroutine object for asyncio.run
        if not asyncio.iscoroutine(result):
            async def _wrap(awaitable):
                return await awaitable
            result = _wrap(result)
        return asyncio.run(result)

    def _apply_hitl(self, ai_msg: Any) -> Any:
        on_out = self._hitl.get("on_model_output")
        if not on_out:
            return ai_msg
        try:
            decision = self._maybe_await(on_out(ai_msg))
            if isinstance(decision, dict) and decision.get("action") in ("modify", "block"):
                replacement = decision.get("replacement", "")
                # If ai_msg is a dict with a 'messages' list, update the last message's content
                if isinstance(ai_msg, dict) and isinstance(ai_msg.get("messages"), list) and ai_msg["messages"]:
                    last_msg = ai_msg["messages"][-1]
                    # Update content in-place if dict has content; otherwise ensure proper message dict
                    if isinstance(last_msg, dict) and "content" in last_msg:
                        last_msg["content"] = replacement
                    else:
                        ai_msg["messages"][-1] = {"role": "ai", "content": replacement}
                elif hasattr(ai_msg, "content"):
                    ai_msg.content = replacement
                else:
                    ai_msg = {"role": "ai", "content": replacement} if not isinstance(ai_msg, dict) else ai_msg
        except Exception:
            pass
        return ai_msg

    def _wrap_tool_for_hitl(self, tool_obj):
        on_tool = self._hitl.get("on_tool_call")
        if not on_tool:
            return tool_obj
    
        # Normalize to a BaseTool
        if isinstance(tool_obj, BaseTool):
            base = tool_obj
        elif callable(tool_obj):
            base = lc_tool(tool_obj)  # convert plain function into a BaseTool
        else:
            return tool_obj
    
        name = getattr(base, "name", getattr(tool_obj, "__name__", "tool"))
        description = getattr(base, "description", getattr(tool_obj, "__doc__", "")) or ""
        args_schema = getattr(base, "args_schema", None)
    
        def _impl(**kwargs):
            try:
                decision = self._maybe_await(on_tool(name, dict(kwargs) if kwargs else {}))
            except Exception:
                decision = {"action": "pass"}

            action = (decision or {}).get("action", "pass")
            if action == "block":
                # Return replacement verbatim; do not attempt schema/JSON coercion.
                return (decision or {}).get("replacement", "[blocked by reviewer]")
            if action == "modify":
                kwargs = (decision or {}).get("args", kwargs)
    
            return base.invoke(kwargs)
    
        try:
            _impl.__name__ = name
        except Exception:
            pass
    
        return StructuredTool.from_function(
            func=_impl,
            name=name,
            description=description,
            args_schema=args_schema,
            infer_schema=not bool(args_schema),
        )

    @staticmethod
    def no_tools() -> Dict[str, Any]:
        """Convenience: disable tool calling for a run."""
        return {"tool_controls": {"tool_choice": "none"}}

    @staticmethod
    def force_tool(name: str, *, once: bool = False, parallel: bool = False) -> Dict[str, Any]:
        """Convenience: force a specific tool (optionally only for first call)."""
        return {"tool_controls": {"tool_choice": {"name": name}, "force_once": once, "parallel_tool_calls": parallel}}

    # ---------- Models ----------
    def set_model(self, provider: str, model_name: str, **kwargs):
        # Delegate to registry (idempotent get/create)
        return self.registry.get_or_create(provider, model_name, **(kwargs or {}))

    def _get_or_create(self, provider: str, model_name: str, **kwargs):  # retained for backward compat
        return self.registry.get_or_create(provider, model_name, **kwargs)

    # ---------- Tool controls ----------
    @staticmethod
    def _tool_used(state: Any) -> bool:
        """Heuristic: did any AIMessage emit a tool call (or ToolMessage added)?"""
        msgs = state.get("messages", []) if isinstance(state, dict) else []
        for m in reversed(msgs):
            if getattr(m, "tool_calls", None):
                return True
            if getattr(m, "type", None) == "tool":  # ToolMessage
                return True
            if isinstance(m, dict) and (m.get("tool_calls") or m.get("type") == "tool"):
                return True
        return False

    # ---------- Agent ----------
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
            hitl_tool_wrapper=(self._wrap_tool_for_hitl if self._hitl.get("on_tool_call") else None),
            logger=self._logger,
        )

    # ---------- Public: Agent run ----------
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
            res = await self._with_retry(_call, **retry_cfg)
        else:
            res = await _call()
        # HITL final output gate
        ai_msg = self._apply_hitl(res)
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
        ai_msg = self._apply_hitl(res)
        return ai_msg

    # ---------- Public: Agent streaming ----------
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

        # NOTE: In "messages" token mode, chunks are emitted incrementally and can't be
        # post-edited after the fact. We pass them through; HITL applies only to the
        # *final* assembled message in other modes.
        if modes == ["messages"]:
            async for token, meta in agent.astream(
                    {"messages": messages},
                    context=context,
                    config=config,
                    stream_mode="messages"
            ):
                yield token, meta
            return

        # For ("updates","values") or either alone: stream updates ASAP,
        # but buffer the last "values" snapshot, then apply HITL before yielding it.
        last_values = None

        async for mode, chunk in agent.astream(
                {"messages": messages},
                context=context,
                config=config,
                stream_mode=modes
        ):
            if mode == "values":
                last_values = chunk  # buffer
                continue             # don't yield yet; we'll gate it first
            else:
                # Stream updates immediately (tool calls, intermediate agent steps, etc.)
                yield mode, chunk

        # --- NEW gating logic (preserve full values shape) ---
        if last_values is not None:
            decision = self._maybe_await(self._hitl.get("on_model_output")(last_values)) if self._hitl.get("on_model_output") else None
            gated_values = last_values
            if isinstance(decision, dict) and decision.get("action") in ("modify", "block"):
                replacement = decision.get("replacement", "")
                if isinstance(gated_values, dict):
                    msgs = gated_values.get("messages")
                    if isinstance(msgs, list) and msgs:
                        last = msgs[-1]
                        if hasattr(last, "content"):
                            last.content = replacement
                        elif isinstance(last, dict):
                            last["content"] = replacement
                        else:
                            msgs.append({"role": "ai", "content": replacement})
                    else:
                        gated_values["messages"] = [{"role": "ai", "content": replacement}]
            yield "values", gated_values

    # ---------- Direct model helpers (no agent) ----------
    def with_structured_output(
            self,
            provider: str,
            model_name: str,
            schema: Union[type[BaseModel], Dict[str, Any]],
            **model_kwargs,
    ):
        """Return model bound for structured output if supported; otherwise fallback with warning."""
        model = self.registry.get_or_create(provider, model_name, **model_kwargs)
        try:
            return model.with_structured_output(schema)
        except Exception as e:  # pragma: no cover - defensive
            self._logger.warning(
                "[CoreLLM] Structured output unavailable; falling back to raw model. provider=%s model=%s schema=%s error=%s",
                provider,
                model_name,
                getattr(schema, "__name__", type(schema)),
                e,
                exc_info=True,
            )
            return model

    # ---------- UX sugar ----------
    @staticmethod
    def make_messages(user: str, system: Optional[str] = None, extras: Optional[List[Dict[str, Any]]] = None):
        msgs: List[Dict[str, Any]] = []
        if system:
            # Use plain dict for system message to be consistent with other messages
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
        last_err = None
        for provider, model_name in candidates:
            try:
                return self.run_agent(messages, provider, model_name, tools, extra, model_kwargs)
            except Exception as e:
                last_err = e
        raise last_err or RuntimeError("All fallbacks failed")

    # === Convenience APIs ===
    def chat(
            self,
            user_msg: str,
            provider: str,
            model_name: str,
            system: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
            **model_kwargs,
    ):
        """One-liner chat without tools/agent graph (HITL + optional retry).

        Notes:
            - Retry logic is only applied if no event loop is currently running.
              If a loop is active (e.g. inside Jupyter/async context), the retry
              config is ignored (a warning is logged) to avoid nested loop errors.
              Use `await achat(...)` for reliable retry behavior in async contexts.
        """
        # Drop stray agent/tool kwargs that shouldn't reach raw model init
        for bad in ("tools", "tool_choice", "parallel_tool_calls", "force_once"):
            model_kwargs.pop(bad, None)
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
                # Cannot safely call asyncio.run(); fall back to single attempt.
                self._logger.warning(
                    "[CoreLLM] chat() retry config ignored because an event loop is already running; use achat() for retries."
                )
                res = _call()
            else:
                async def _acall():
                    return _call()
                res = asyncio.run(self._with_retry(_acall, **retry_cfg))
        else:
            res = _call()

        ai_msg = self._apply_hitl(res)
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
        """Async one-liner chat (HITL and retry)."""
        # Drop stray agent/tool kwargs that shouldn't reach raw model init
        for bad in ("tools", "tool_choice", "parallel_tool_calls", "force_once"):
            model_kwargs.pop(bad, None)
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = self.make_messages(user_msg, system)

        async def _call():
            return await model.ainvoke(messages)

        retry_cfg = (extra or {}).get("retry") if extra else None
        res = await (self._with_retry(_call, **retry_cfg) if retry_cfg else _call())

        ai_msg = self._apply_hitl(res)
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
        # drop keys that belong to agent/tooling, not base LLMs
        for bad in ("tools", "tool_choice", "parallel_tool_calls", "force_once"):
            model_kwargs.pop(bad, None)
        # Only add explicitly listed kwargs if not None
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if top_p is not None:
            model_kwargs["top_p"] = top_p
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = self.make_messages(user_msg, system)

        # New: astream yields one object per tick (e.g., AIMessageChunk), not (token, meta)
        async for event in model.astream(messages):
            # Try to extract just the text delta; fall back to str(event)
            text = getattr(event, "content", None)
            if text is None:
                # Some providers chunk via .additional_kwargs or .delta etc.
                text = getattr(event, "delta", None) or getattr(event, "text", None)
            if text is None:
                text = str(event)

            # Minimal metadata for callers that still want a "meta"
            meta = {"raw": event}
            yield text, meta

    # inside class CoreLLM

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
        """
        Stream ONLY the agent's token deltas (no 'updates' / no 'values').

        Notes:
        - Tool HITL is enforced because _make_agent_with_context() wraps tools
          via _wrap_tool_for_hitl when on_tool_call is set.
        - Final-output HITL (on_model_output) is NOT applied here, since token
          chunks are emitted incrementally and can't be post-edited.
        """
        agent, context = self._make_agent_with_context(
            provider, model_name, tools, extra, model_kwargs, tool_controls
        )

        async for token, meta in agent.astream(
                {"messages": messages},
                context=context,
                config=config,
                stream_mode="messages",   # only LLM token deltas
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

    async def _with_retry(self, afn, *, max_tries=3, base=0.5, jitter=0.2):
        """Exponential backoff retry for transient errors around an awaited call."""
        import asyncio, random
        last = None
        for i in range(max_tries):
            try:
                return await afn()
            except Exception as e:
                last = e
                await asyncio.sleep(base * (2 ** i) + random.random() * jitter)
        raise last
