from __future__ import annotations

import asyncio

"""
Centralized HITL + tool policy utilities.

This module consolidates logic that previously lived ad‑hoc in CoreLLM / runtime_bind:
  - HITLConfig: stores callbacks and provides a .set API
  - maybe_await: safe sync resolver for (possibly) async callbacks
  - apply_output_gate: applies model output moderation / modification gate
  - wrap_tool_for_hitl: wraps a tool with pre‑execution HITL policy
  - ToolPolicy: configuration holder for tool selection policy
  - compute_effective_tools: merges per‑call tools with global tools under policy

None of these functions mutate global state; they are pure / side‑effect free
(except logging) and can be composed by higher‑level orchestration code.
"""
from typing import Any, Callable, Dict, List, Optional, Sequence
import logging
import inspect

from langchain_core.tools import BaseTool, tool as lc_tool, StructuredTool  # type: ignore

__all__ = [
    "HITLConfig",
    "maybe_await",
    "apply_output_gate",
    "apply_output_gate_async",
    "wrap_tool_for_hitl",
    "ToolPolicy",
    "compute_effective_tools",
]

logger = logging.getLogger(__name__)


# ---------- HITL Configuration ----------
class HITLConfig:
    """Container for Human-In-The-Loop (HITL) callbacks.

    on_model_output(ai_msg) -> decision dict or None
        decision: {action: pass|modify|block, replacement: str}

    on_tool_call(name: str, args: dict) -> decision dict or None
        decision: {action: pass|modify|block, args: {...}, replacement: any}
    """
    def __init__(
            self,
            *,
            on_model_output: Optional[Callable[..., Any]] = None,
            on_tool_call: Optional[Callable[..., Any]] = None,
            on_model_output_async: Optional[Callable[..., Any]] = None,
            on_tool_call_async: Optional[Callable[..., Any]] = None,
    ):
        self.on_model_output = on_model_output
        self.on_tool_call = on_tool_call
        self.on_model_output_async = on_model_output_async
        self.on_tool_call_async = on_tool_call_async

    def set(
            self,
            *,
            on_model_output: Optional[Callable[..., Any]] = None,
            on_tool_call: Optional[Callable[..., Any]] = None,
            on_model_output_async: Optional[Callable[..., Any]] = None,
            on_tool_call_async: Optional[Callable[..., Any]] = None,
    ):
        if on_model_output is not None:
            self.on_model_output = on_model_output
        if on_tool_call is not None:
            self.on_tool_call = on_tool_call
        if on_model_output_async is not None:
            self.on_model_output_async = on_model_output_async
        if on_tool_call_async is not None:
            self.on_tool_call_async = on_tool_call_async
        return self

    async def call_model_output(self, ai_msg: Any):
        if self.on_model_output_async:
            return await self.on_model_output_async(ai_msg)
        if self.on_model_output:
            return await asyncio.to_thread(self.on_model_output, ai_msg)
        return None

    async def call_tool(self, name: str, args: Dict[str, Any]):
        if self.on_tool_call_async:
            return await self.on_tool_call_async(name, args)
        if self.on_tool_call:
            return await asyncio.to_thread(self.on_tool_call, name, args)
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "on_model_output": self.on_model_output,
            "on_tool_call": self.on_tool_call,
            "on_model_output_async": self.on_model_output_async,
            "on_tool_call_async": self.on_tool_call_async,
        }


class _HITLWrappedTool(BaseTool):
    """Async-first wrapper around a BaseTool enforcing async execution."""

    def __init__(self, base: BaseTool, hitl: HITLConfig):
        super().__init__(name=getattr(base, "name", "tool"), description=getattr(base, "description", "") or "")
        self._base = base
        self._hitl = hitl
        # preserve args schema if present
        if hasattr(base, "args_schema") and base.args_schema is not None:
            self.args_schema = base.args_schema  # type: ignore[attr-defined]

    # Disallow sync path to avoid StructuredTool sync errors
    def _run(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("HITL-wrapped tools are async-only. Use ainvoke/_arun.")

    async def _arun(self, *args, **kwargs):  # type: ignore[override]
        args_dict = dict(kwargs) if kwargs else {}
        try:
            decision = await self._hitl.call_tool(self.name, args_dict)
        except Exception:
            decision = {"action": "pass"}

        action = (decision or {}).get("action", "pass")
        if action == "block":
            return (decision or {}).get("replacement", "[blocked by reviewer]")
        if action == "modify":
            args_dict = (decision or {}).get("args", args_dict)

        # prefer async on the base tool
        if hasattr(self._base, "ainvoke"):
            return await self._base.ainvoke(args_dict)
        # fallback: run sync tool in a thread
        return await asyncio.to_thread(self._base.invoke, args_dict)



# ---------- Async helper ----------
def maybe_await(result: Any) -> Any:
    """Resolve an awaitable in a sync context safely.

    Behavior mirrors CoreLLM._maybe_await:
      - If result is not awaitable, return as-is.
      - If an event loop is running, log warning and return None (cannot block).
      - If coroutine / awaitable and no loop, run it to completion.
    """
    if not inspect.isawaitable(result):
        return result
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        logger.warning(
            "maybe_await: async callback ignored (event loop active in sync pathway). Use async APIs for async callbacks."  # noqa: E501
        )
        return None
    if not asyncio.iscoroutine(result):
        async def _wrap(awaitable):
            return await awaitable
        result = _wrap(result)
    return asyncio.run(result)


# ---------- Output gating ----------
def apply_output_gate(ai_msg: Any, hitl: Optional[HITLConfig | Dict[str, Any]]) -> Any:
    """Apply HITL on_model_output gate to a model/agent final output.

    ai_msg can be:
      - a LangChain AIMessage (has .content)
      - a dict state with messages: [...]
      - any other object (left unchanged unless replaced)
    """
    if not hitl:
        return ai_msg
    on_out = hitl.on_model_output if isinstance(hitl, HITLConfig) else hitl.get("on_model_output")
    if not on_out:
        return ai_msg
    try:
        decision = maybe_await(on_out(ai_msg))
        if isinstance(decision, dict) and decision.get("action") in ("modify", "block"):
            replacement = decision.get("replacement", "")
            if isinstance(ai_msg, dict) and isinstance(ai_msg.get("messages"), list) and ai_msg["messages"]:
                last_msg = ai_msg["messages"][-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    last_msg["content"] = replacement
                elif hasattr(last_msg, "content"):
                    last_msg.content = replacement  # type: ignore[attr-defined]
                else:
                    ai_msg["messages"][-1] = {"role": "ai", "content": replacement}
            elif hasattr(ai_msg, "content"):
                ai_msg.content = replacement  # type: ignore[attr-defined]
            else:
                ai_msg = {"role": "ai", "content": replacement} if not isinstance(ai_msg, dict) else ai_msg
    except Exception:  # pragma: no cover - defensive
        pass
    return ai_msg

async def apply_output_gate_async(ai_msg: Any, hitl: Optional[HITLConfig]) -> Any:
    if not hitl:
        return ai_msg
    try:
        decision = await hitl.call_model_output(ai_msg)
        if isinstance(decision, dict) and decision.get("action") in ("modify", "block"):
            replacement = decision.get("replacement", "")
            # mirror your existing mutation logic:
            if isinstance(ai_msg, dict) and isinstance(ai_msg.get("messages"), list) and ai_msg["messages"]:
                last_msg = ai_msg["messages"][-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    last_msg["content"] = replacement
                elif hasattr(last_msg, "content"):
                    last_msg.content = replacement  # type: ignore
                else:
                    ai_msg["messages"][-1] = {"role": "ai", "content": replacement}
            elif hasattr(ai_msg, "content"):
                ai_msg.content = replacement  # type: ignore
            else:
                ai_msg = {"role": "ai", "content": replacement} if not isinstance(ai_msg, dict) else ai_msg
    except Exception:
        pass
    return ai_msg


# ---------- Tool wrapping ----------
def wrap_tool_for_hitl(tool_obj: Any, hitl: Optional[HITLConfig]):
    """Return an async-first HITL-wrapped tool when a tool_call gate is present."""
    if not hitl or not (hitl.on_tool_call or hitl.on_tool_call_async):
        return tool_obj

    # Normalize to BaseTool
    if isinstance(tool_obj, BaseTool):
        base = tool_obj
    elif callable(tool_obj):
        base = lc_tool(tool_obj)  # wraps function into a BaseTool (supports ainvoke when func is async)
    else:
        return tool_obj

    return _HITLWrappedTool(base, hitl)


# ---------- Tool policy ----------
class ToolPolicy:
    """Holds configuration flags controlling tool resolution.

    Attributes:
        require_explicit (bool): If True, implicit use of global tools is forbidden.
    """
    def __init__(self, *, require_explicit: bool = False):
        self.require_explicit = require_explicit

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ToolPolicy(require_explicit={self.require_explicit})"


def compute_effective_tools(
    call_tools: Optional[Sequence[Any]],
    global_tools: Optional[Sequence[Any]],
    policy: ToolPolicy,
    *,
    logger_: Optional[logging.Logger] = None,
) -> List[Any]:
    """Compute effective tools for a call given per-call list & global list under policy.

    Logic mirrors the inline section previously in make_agent_with_context:
      - If call_tools is provided (even empty list), use it directly.
      - Else if global tools exist and policy requires explicit: raise.
      - Else use global tools (may be empty) and optionally log.
    """
    global_tools = list(global_tools or [])
    if call_tools is not None:
        return list(call_tools)  # explicit override (including empty list to suppress)

    if global_tools and policy.require_explicit:
        raise ValueError(
            "Implicit global tools use forbidden (require_explicit=True). "
            "Pass tools=[] to run without tools or tools=[...] to specify explicitly."
        )
    if global_tools and logger_:
        logger_.info(
            "[ToolPolicy] Using implicit global tools (%d). Pass tools=[] to suppress or enable require_explicit to forbid.",
            len(global_tools),
        )
    return global_tools

