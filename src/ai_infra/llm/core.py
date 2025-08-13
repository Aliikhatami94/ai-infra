from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Union
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime
from langchain_core.messages import SystemMessage
from pydantic import BaseModel as PydanticModel
from langchain_core.tools import tool as lc_tool, StructuredTool

from .settings import ModelSettings
from .utils import validate_provider_and_model, build_model_key, initialize_model
from .tool_controls import normalize_tool_controls, ToolCallControls

load_dotenv()

class CoreLLM:
    """
    Minimal, batteries-included LLM facade:
    - Model cache across providers
    - Agents from LangGraph prebuilt ReAct
    - Provider-normalized tool controls
    - Streaming (graph 'updates/values' and token 'messages')
    - Optional structured output
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tools: List[Any] = []
        # Optional hooks
        self._hitl: Dict[str, Any] = {"on_model_output": None}
        self._metrics_cb: Optional[Any] = None

    # ---------- Hooks & helpers ----------
    def set_hitl(self, *, on_model_output=None, on_tool_call=None):
        """Register HITL callbacks.
        on_model_output(ai_msg) -> {"action": "pass"|"modify"|"block", "replacement": str}
        on_tool_call(name:str, args:dict) -> {"action": "pass"|"modify"|"block", "args": dict, "replacement": str}
        """
        self._hitl["on_model_output"] = on_model_output
        self._hitl["on_tool_call"] = on_tool_call

    def _apply_hitl(self, ai_msg: Any) -> Any:
        on_out = self._hitl.get("on_model_output")
        if not on_out:
            return ai_msg
        try:
            decision = on_out(ai_msg)
            if isinstance(decision, dict) and decision.get("action") in ("modify", "block"):
                replacement = decision.get("replacement", "")
                if hasattr(ai_msg, "content"):
                    ai_msg.content = replacement
                else:
                    ai_msg = replacement
        except Exception:
            pass
        return ai_msg

    def _wrap_tool_for_hitl(self, tool_obj):
        """
        Wrap a tool with a human-approval gate, returning a proper LangChain Tool.
        - If `tool_obj` is a plain function, convert it to a tool first (so we get schema).
        - If it's already a BaseTool, preserve its metadata & schema.
        - Returns a StructuredTool accepted by create_react_agent.
        """
        on_tool = self._hitl.get("on_tool_call")
        if not on_tool:
            return tool_obj  # No HITL gating requested

        # 1) Normalize to a BaseTool so we have .name/.description/.args_schema
        #    a) If it's already a BaseTool, keep it
        #    b) If it's a plain callable, convert it using @tool
        if isinstance(tool_obj, PydanticModel):
            base = tool_obj
        elif callable(tool_obj):
            base = lc_tool(tool_obj)  # makes a BaseTool (with inferred schema from signature)
        else:
            # Unknown shape -> return as-is (safer than breaking)
            return tool_obj

        name = getattr(base, "name", getattr(tool_obj, "__name__", "tool"))
        description = getattr(base, "description", getattr(tool_obj, "__doc__", "")) or ""
        args_schema = getattr(base, "args_schema", None)

        # 2) Build an approval-gated function we’ll wrap into a StructuredTool
        #    Keep the function name so LangChain’s converter is happy.
        def _impl(**kwargs):
            try:
                decision = on_tool(name, dict(kwargs) if kwargs else {})
            except Exception:
                decision = {"action": "pass"}

            action = (decision or {}).get("action", "pass")
            if action == "block":
                return (decision or {}).get("replacement", "[blocked by reviewer]")
            if action == "modify":
                kwargs = (decision or {}).get("args", kwargs)

            # Delegate to the original tool’s .invoke()
            return base.invoke(kwargs)

        # ensure it looks like a normal function for LC internals
        try:
            _impl.__name__ = name  # gives convert.tool something it recognizes
        except Exception:
            pass

        # 3) Return a proper StructuredTool so prebuilt agent accepts it
        return StructuredTool.from_function(
            func=_impl,
            name=name,
            description=description,
            args_schema=args_schema,          # preserve exact schema if it exists
            infer_schema=not bool(args_schema)  # infer if none was provided
        )

    def set_metrics(self, on_metrics):
        """Register a callback receiving usage/cost/latency info per run."""
        self._metrics_cb = on_metrics

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
        validate_provider_and_model(provider, model_name)
        key = build_model_key(provider, model_name)
        if key not in self.models:
            self.models[key] = initialize_model(key, provider, **(kwargs or {}))
        return self.models[key]

    def _get_or_create(self, provider: str, model_name: str, **kwargs):
        return self.set_model(provider, model_name, **kwargs)

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

    def _select_model(self, state: Any, runtime: Runtime[ModelSettings]) -> Any:
        ctx = runtime.context
        key = build_model_key(ctx.provider, ctx.model_name)
        if key not in self.models:
            self.set_model(ctx.provider, ctx.model_name, **(ctx.extra.get("model_kwargs", {}) if ctx.extra else {}))

        model = self.models[key]
        tools = ctx.tools if ctx.tools is not None else self.tools
        extra = ctx.extra or {}

        tool_choice, parallel_tool_calls, force_once = normalize_tool_controls(
            ctx.provider, extra.get("tool_controls")
        )
        if force_once and self._tool_used(state):
            tool_choice = None

        return model.bind_tools(
            tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

    # ---------- Agent ----------
    def _make_agent_with_context(
            self,
            provider: str,
            model_name: str,
            tools: Optional[List[Any]] = None,
            extra: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,  # NEW
    ) -> Tuple[Any, ModelSettings]:
        model_kwargs = model_kwargs or {}
        self.set_model(provider, model_name, **model_kwargs)

        # merge tool_controls into extra for runtime.context
        if tool_controls is not None:
            from dataclasses import is_dataclass, asdict
            if is_dataclass(tool_controls):
                tool_controls = asdict(tool_controls)
            extra = {**(extra or {}), "tool_controls": tool_controls}

        context = ModelSettings(
            provider=provider,
            model_name=model_name,
            tools=tools,
            extra={"model_kwargs": model_kwargs, **(extra or {})},
        )

        effective_tools = tools or self.tools
        if self._hitl.get("on_tool_call"):
            effective_tools = [self._wrap_tool_for_hitl(t) for t in effective_tools]

        agent = create_react_agent(model=self._select_model, tools=effective_tools)
        return agent, context

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
        started = __import__("time").time() * 1000
        async def _call():
            return await agent.ainvoke({"messages": messages}, context=context, config=config)
        retry_cfg = (extra or {}).get("retry") if extra else None
        if retry_cfg:
            res = await self._with_retry(_call, **retry_cfg)
        else:
            res = await _call()
        # HITL final output gate
        on_out = self._hitl.get("on_model_output")
        if on_out:
            try:
                decision = on_out(res)
                if isinstance(decision, dict) and decision.get("action") in ("modify", "block"):
                    replacement = decision.get("replacement", "")
                    # Try to replace content on a copy-like object
                    if hasattr(res, "content"):
                        res.content = replacement
                    else:
                        res = replacement
            except Exception:
                pass
        # metrics
        self._emit_metrics(provider, model_name, res, started_at_ms=started)
        return res

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
        import time
        started = time.time() * 1000
        res = agent.invoke({"messages": messages}, context=context, config=config)
        on_out = self._hitl.get("on_model_output")
        if on_out:
            try:
                decision = on_out(res)
                if isinstance(decision, dict) and decision.get("action") in ("modify", "block"):
                    replacement = decision.get("replacement", "")
                    if hasattr(res, "content"):
                        res.content = replacement
                    else:
                        res = replacement
            except Exception:
                pass
        self._emit_metrics(provider, model_name, res, started_at_ms=started)
        return res


    # ---------- Public: Agent streaming ----------
    async def arun_agent_stream(
            self,
            messages: List[Dict[str, Any]],
            provider: str,
            model_name: str,
            tools: Optional[List[Any]] = None,
            extra: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            stream_mode: Union[str, List[str]] = ("updates", "values"),
            tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,
            config: Optional[Dict[str, Any]] = None
    ):
        agent, context = self._make_agent_with_context(provider, model_name, tools, extra, model_kwargs, tool_controls)
        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)
        import time
        started = time.time() * 1000
        last_value = None

        if modes == ["messages"]:
            async for token, meta in agent.astream({"messages": messages}, context=context, config=config, stream_mode="messages"):
                yield token, meta
            # token mode has no final AIMessage; emit bare metrics
            self._emit_metrics(provider, model_name, ai_msg=None, started_at_ms=started)
            return

        async for mode, chunk in agent.astream({"messages": messages}, context=context, config=config, stream_mode=modes):
            # track last value for optional HITL
            if mode == "values":
                last_value = chunk
            yield mode, chunk

        # post-stream hooks
        if last_value is not None:
            on_out = self._hitl.get("on_model_output")
            if on_out:
                try:
                    decision = on_out(last_value)
                    if isinstance(decision, dict) and decision.get("action") in ("modify", "block"):
                        replacement = decision.get("replacement", "")
                        if hasattr(last_value, "content"):
                            last_value.content = replacement
                        else:
                            last_value = replacement
                except Exception:
                    pass
            self._emit_metrics(provider, model_name, last_value, started_at_ms=started)


    # ---------- Direct model helpers (no agent) ----------
    def with_structured_output(
            self,
            provider: str,
            model_name: str,
            schema: Union[type[PydanticModel], Dict[str, Any]],
            **model_kwargs,
    ):
        model = self._get_or_create(provider, model_name, **model_kwargs)
        try:
            return model.with_structured_output(schema)
        except Exception:
            return model

    # ---------- UX sugar ----------
    @staticmethod
    def make_messages(user: str, system: Optional[str] = None, extras: Optional[List[Dict[str, Any]]] = None):
        msgs: List[Dict[str, Any]] = []
        if system:
            msgs.append(SystemMessage(content=system).model_dump())
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
        """One-liner chat without tools/agent graph (now with HITL, metrics, retry)."""
        import time
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = self.make_messages(user_msg, system)
        started = time.time() * 1000

        def _call():
            return model.invoke(messages)

        # optional sync retry config: extra={"retry":{"max_tries":3,"base":0.5,"jitter":0.2}}
        retry_cfg = (extra or {}).get("retry") if extra else None
        if retry_cfg:
            # reuse async retry in a sync wrapper
            import asyncio
            async def _acall(): return _call()
            res = asyncio.run(self._with_retry(_acall, **retry_cfg))
        else:
            res = _call()

        res = self._apply_hitl(res)
        self._emit_metrics(provider, model_name, res, started_at_ms=started)
        return res

    async def achat(
            self,
            user_msg: str,
            provider: str,
            model_name: str,
            system: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
            **model_kwargs,
    ):
        """Async one-liner chat (HITL, metrics, retry)."""
        import time
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = self.make_messages(user_msg, system)
        started = time.time() * 1000

        async def _call():
            return await model.ainvoke(messages)

        retry_cfg = (extra or {}).get("retry") if extra else None
        res = await (self._with_retry(_call, **retry_cfg) if retry_cfg else _call())

        res = self._apply_hitl(res)
        self._emit_metrics(provider, model_name, res, started_at_ms=started)
        return res

    async def stream_tokens(
            self,
            user_msg: str,
            provider: str,
            model_name: str,
            system: Optional[str] = None,
            **model_kwargs,
    ):
        """Token stream (LLM raw streaming), no agent graph. Emits bare metrics at the end."""
        import time
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = self.make_messages(user_msg, system)
        started = time.time() * 1000

        async for token, meta in model.astream(messages):
            yield token, meta

        # We don't have a final AIMessage here; just emit latency.
        self._emit_metrics(provider, model_name, ai_msg=None, started_at_ms=started)

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
        import asyncio, random, time
        last = None
        for i in range(max_tries):
            try:
                return await afn()
            except Exception as e:
                last = e
                await asyncio.sleep(base * (2 ** i) + random.random() * jitter)
        raise last

    def _emit_metrics(self, provider: str, model_name: str, ai_msg: Any, started_at_ms: Optional[float] = None):
        """Call user metrics hook with tokens/cost/latency if available."""
        if not self._metrics_cb:
            return
        usage = getattr(ai_msg, "usage_metadata", None) or getattr(ai_msg, "response_metadata", {}).get("token_usage", None)
        latency_ms = None
        if started_at_ms is not None:
            import time
            latency_ms = int((time.time() * 1000) - started_at_ms)
        payload = {
            "provider": provider,
            "model": model_name,
            "usage": usage,
            "latency_ms": latency_ms,
        }
        try:
            self._metrics_cb(payload)
        except Exception:
            pass