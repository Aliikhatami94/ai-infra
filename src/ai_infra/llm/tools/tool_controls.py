from __future__ import annotations
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass

from ai_infra.llm.providers import Providers


@dataclass
class ToolCallControls:
    tool_choice: Optional[Dict[str, Any]] = None     # e.g. {"name":"my_tool"} | "none" | "auto" | "any"
    parallel_tool_calls: bool = True
    force_once: bool = False                         # Only enforce tool_choice for the first call in a run


def _ensure_dict(obj: Any) -> Dict[str, Any] | None:
    if not obj:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return None


def _extract_name(tool_choice: Any) -> Optional[str]:
    if not isinstance(tool_choice, dict):
        return None
    return tool_choice.get("name") or (tool_choice.get("function") or {}).get("name")

def no_tools() -> Dict[str, Any]:
    return {"tool_controls": {"tool_choice": "none"}}

def force_tool(name: str, *, once: bool = False, parallel: bool = False) -> Dict[str, Any]:
    return {"tool_controls": {
        "tool_choice": {"name": name},
        "force_once": once,
        "parallel_tool_calls": parallel,
    }}

def normalize_tool_controls(
        provider: str,
        controls: Any
) -> Tuple[Any, bool, bool]:
    """
    Return (tool_choice, parallel_tool_calls, force_once) in the exact shape
    required by the *LangChain provider adapters*.

    - Accepts neutral user inputs like:
        {"tool_choice": "none" | "auto" | "any" | {"name":"..."} | {"function":{"name":"..."}}}
        {"parallel_tool_calls": bool}
        {"force_once": bool}
    - Converts to provider-specific wire formats.
    """
    # defaults
    tool_choice: Any = None
    parallel_tool_calls: bool = True
    force_once: bool = False

    d = _ensure_dict(controls)
    if not d:
        return tool_choice, parallel_tool_calls, force_once

    tool_choice = d.get("tool_choice")
    parallel_tool_calls = d.get("parallel_tool_calls", True)
    force_once = bool(d.get("force_once", False))

    # Strings “none|auto|any” can pass through for all except Gemini (we’ll map those).
    if isinstance(tool_choice, str):
        if provider != Providers.google_genai:
            return tool_choice, parallel_tool_calls, force_once

    # Provider-specific mapping
    name = _extract_name(tool_choice)

    if provider in (Providers.openai, Providers.xai):
        # OpenAI/XAI accept string tags or OpenAI-style function routing.
        if isinstance(tool_choice, str):
            return tool_choice, parallel_tool_calls, force_once
        if name:
            tool_choice = {"type": "function", "function": {"name": name}}

    elif provider == Providers.anthropic:
        # Anthropic accepts "none|auto|any" or {"type":"tool","name":...}
        if isinstance(tool_choice, str):
            return tool_choice, parallel_tool_calls, force_once
        if name:
            tool_choice = {"type": "tool", "name": name}

    elif provider == Providers.google_genai:
        # Gemini wants FunctionCallingConfig
        def gg(mode: str, names: Optional[List[str]] = None):
            cfg = {"function_calling_config": {"mode": mode}}
            if names:
                cfg["function_calling_config"]["allowed_function_names"] = names
            return cfg

        if isinstance(tool_choice, str):
            s = tool_choice.lower()
            if s == "none":
                tool_choice = gg("NONE")
            elif s == "auto":
                tool_choice = gg("AUTO")
            elif s == "any":
                tool_choice = gg("ANY")
            else:
                tool_choice = gg("AUTO")
        elif name:
            tool_choice = gg("ANY", [name])
        else:
            tool_choice = None

    # Others pass-through
    return tool_choice, parallel_tool_calls, force_once