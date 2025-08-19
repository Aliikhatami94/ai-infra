from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import logging
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime
from langchain_core.tools import BaseTool, tool as lc_tool

from .settings import ModelSettings
from ai_infra.llm.tools.tool_controls import normalize_tool_controls, ToolCallControls
from ai_infra.llm.utils import (
    validate_provider_and_model,
    build_model_key,
    initialize_model,
)

_logger = logging.getLogger(__name__)

class ModelRegistry:
    """Lightweight model cache / registry per provider+model key."""
    def __init__(self):
        self._models: Dict[str, Any] = {}

    def get_or_create(self, provider: str, model_name: str, **kwargs) -> Any:
        validate_provider_and_model(provider, model_name)
        key = build_model_key(provider, model_name)
        if key not in self._models:
            self._models[key] = initialize_model(key, provider, **(kwargs or {}))
        return self._models[key]

    def get(self, provider: str, model_name: str) -> Any:
        key = build_model_key(provider, model_name)
        return self._models.get(key)


def tool_used(state: Any) -> bool:
    """Heuristic: did the agent already emit a tool call (or a tool message)?"""
    msgs = state.get("messages", []) if isinstance(state, dict) else []
    for m in reversed(msgs):
        if getattr(m, "tool_calls", None):
            return True
        if getattr(m, "type", None) == "tool":  # ToolMessage
            return True
        if isinstance(m, dict) and (m.get("tool_calls") or m.get("type") == "tool"):
            return True
    return False


def bind_model_with_tools(
    state: Any,
    runtime: Runtime[ModelSettings],
    registry: ModelRegistry,
    *,
    global_tools: Optional[List[Any]] = None,
) -> Any:
    """Select (or lazily init) the model and bind tools according to controls.

    This mirrors the prior CoreLLM._select_model method but is factored out for reuse.
    """
    ctx = runtime.context
    key_model_kwargs = (ctx.extra.get("model_kwargs", {}) if ctx.extra else {})
    model = registry.get_or_create(ctx.provider, ctx.model_name, **key_model_kwargs)

    tools = ctx.tools if ctx.tools is not None else (global_tools or [])
    extra = ctx.extra or {}

    tool_choice, parallel_tool_calls, force_once = normalize_tool_controls(
        ctx.provider, extra.get("tool_controls")
    )

    # Gemini special-case: do not send explicit tool_choice if no tools are bound.
    if ctx.provider == "google_genai" and not tools:
        tool_choice = None

    if force_once and tool_used(state):
        tool_choice = None

    return model.bind_tools(
        tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )


def make_agent_with_context(
    registry: ModelRegistry,
    *,
    provider: str,
    model_name: str,
    tools: Optional[List[Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    tool_controls: Optional[ToolCallControls | Dict[str, Any]] = None,
    require_explicit_tools: bool = False,
    global_tools: Optional[List[Any]] = None,
    hitl_tool_wrapper = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, ModelSettings]:
    """Construct an agent (LangGraph ReAct) and its runtime context.

    Handles:
      - model warm-up via registry
      - optional tool control merging
      - implicit global tools policy
      - HITL tool wrapping
      - agent graph creation with deferred model binding
    """
    model_kwargs = model_kwargs or {}
    registry.get_or_create(provider, model_name, **model_kwargs)
    if tool_controls is not None:
        from dataclasses import is_dataclass, asdict
        if is_dataclass(tool_controls):
            tool_controls = asdict(tool_controls)
        extra = {**(extra or {}), "tool_controls": tool_controls}

    # Effective tools resolution
    effective_tools = global_tools or []
    if tools is not None:
        effective_tools = tools
    else:
        if (global_tools and len(global_tools) > 0) and require_explicit_tools:
            raise ValueError(
                "Implicit global tools use forbidden (require_tools_explicit=True). "
                "Pass tools=[] to run without tools or tools=[...] to specify explicitly."
            )
        if global_tools and len(global_tools) > 0 and logger:
            logger.info(
                "[CoreLLM] Using global self.tools (%d). Pass tools=[] to suppress or set require_tools_explicit(True) to forbid implicit use.",
                len(global_tools)
            )

    effective_tools = [nt for nt in (_normalize_tool(t) for t in effective_tools) if nt is not None]

    if hitl_tool_wrapper is not None:
        wrapped_tools: List[Any] = []
        for t in effective_tools:
            try:
                w = hitl_tool_wrapper(t)
                wrapped_tools.append(w if w is not None else t)  # fallback to original tool
            except Exception:
                wrapped_tools.append(t)  # defensive fallback
        effective_tools = wrapped_tools

    if not effective_tools and logger:
        logger.warning("No tools bound; agent will not call tools.")

    context = ModelSettings(
        provider=provider,
        model_name=model_name,
        tools=effective_tools,
        extra={"model_kwargs": model_kwargs or {}, **(extra or {})},
    )
    def _selector(state, rt: Runtime[ModelSettings]):
        return bind_model_with_tools(state, rt, registry, global_tools=context.tools)
    agent = create_react_agent(model=_selector, tools=effective_tools)
    return agent, context


def _normalize_tool(t):
    if t is None:
        return None
    if isinstance(t, BaseTool):
        return t
    if callable(t):
        return lc_tool(t)
    if isinstance(t, dict):  # leave dict (ignored by ToolNode) but log
        _logger.warning("Dict-shaped tool provided and will be ignored by ToolNode: keys=%s", list(t.keys()))
        return None
    _logger.warning("Unsupported tool type ignored: %r", type(t))
    return None
