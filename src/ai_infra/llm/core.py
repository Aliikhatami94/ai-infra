from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime
from langchain_core.messages import SystemMessage
from pydantic import BaseModel as PydanticModel

from .settings import ModelSettings
from .utils import validate_provider_and_model, build_model_key, initialize_model
from .tool_controls import normalize_tool_controls

load_dotenv()

@dataclass
class ToolCallControls:
    tool_choice: Optional[Dict[str, Any]] = None     # e.g. {"name":"my_tool"} | "none" | "auto" | "any"
    parallel_tool_calls: bool = True
    force_once: bool = False                         # Only enforce tool_choice for the first call in a run


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
    ) -> Tuple[Any, ModelSettings]:
        model_kwargs = model_kwargs or {}
        self.set_model(provider, model_name, **model_kwargs)
        context = ModelSettings(
            provider=provider,
            model_name=model_name,
            tools=tools,
            extra={"model_kwargs": model_kwargs, **(extra or {})},
        )
        agent = create_react_agent(model=self._select_model, tools=tools or self.tools)
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
    ) -> Any:
        agent, context = self._make_agent_with_context(provider, model_name, tools, extra, model_kwargs)
        return await agent.ainvoke({"messages": messages}, context=context)

    def run_agent(
            self,
            messages: List[Dict[str, Any]],
            provider: str,
            model_name: str,
            tools: Optional[List[Any]] = None,
            extra: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        agent, context = self._make_agent_with_context(provider, model_name, tools, extra, model_kwargs)
        return agent.invoke({"messages": messages}, context=context)

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
    ):
        agent, context = self._make_agent_with_context(provider, model_name, tools, extra, model_kwargs)
        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)

        if modes == ["messages"]:
            async for token, meta in agent.astream({"messages": messages}, context=context, stream_mode="messages"):
                yield token, meta
            return

        async for mode, chunk in agent.astream({"messages": messages}, context=context, stream_mode=modes):
            yield mode, chunk

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
            msgs.append(SystemMessage(content=system).dict())
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
            **model_kwargs,
    ):
        """One-liner chat without tools/agent graph."""
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = self.make_messages(user_msg, system)
        return model.invoke(messages)

    async def achat(
            self,
            user_msg: str,
            provider: str,
            model_name: str,
            system: Optional[str] = None,
            **model_kwargs,
    ):
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = self.make_messages(user_msg, system)
        return await model.ainvoke(messages)

    async def stream_tokens(
            self,
            user_msg: str,
            provider: str,
            model_name: str,
            system: Optional[str] = None,
            **model_kwargs,
    ):
        """Token stream (LLM raw streaming), no agent graph."""
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = self.make_messages(user_msg, system)
        async for token, meta in model.astream(messages):
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