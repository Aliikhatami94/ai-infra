import json
from typing import Any, List
from ai_infra.llm import CoreLLM
from .states import PlannerState, ActionStep


def _gather_msgs(state, roles=("user", "human")) -> str:
    msgs = state.get("messages") or []
    parts: list[str] = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") in roles:
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
    return "\n".join(parts)

def _gather_sys(state) -> str:
    msgs = state.get("messages") or []
    parts: list[str] = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
    return "\n".join(parts)

def _compose_system(base: str, state: PlannerState) -> str:
    inherited = _gather_sys(state)
    return base if not inherited else f"{base}\n\n{inherited}"

def _icon(step: ActionStep) -> str:
    k = getattr(step, "kind", "")
    return {"reason":"💡","tool":"🔧","assert":"✅","ask":"❓"}.get(k, "•")

def _render_presentation_md(state: PlannerState) -> str:
    lines = ["## Proposed plan"]
    steps = state.get("plan") or []
    if steps:
        for i, s in enumerate(steps, 1):
            k = s.kind
            if k == "reason":
                title = f" — {s.title}" if getattr(s, "title", None) else ""
                lines.append(f"{i}. { _icon(s) } {s.text}{title}")
            elif k == "tool":
                title = f" — {s.title}" if getattr(s, "title", None) else ""
                lines.append(f"{i}. { _icon(s) } {s.rationale}{title}")
                lines.append(f"   - tool: `{s.tool}`")
                lines.append(f"   - args: `{json.dumps(s.args, ensure_ascii=False)}`")
            elif k == "assert":
                lines.append(f"{i}. { _icon(s) } Assert: {s.condition}")
                if s.on_fail_hint:
                    lines.append(f"   - if false: {s.on_fail_hint}")
            elif k == "ask":
                lines.append(f"{i}. { _icon(s) } Ask: {s.question}")
            else:
                lines.append(f"{i}. • (unknown step)")
    else:
        lines.append("_(no steps)_")

    qs = state.get("questions") or []
    if qs:
        lines.append("\n**Open questions:**")
        for q in qs:
            lines.append(f"- {q}")
    return "\n".join(lines)

async def call_structured(
        state: PlannerState,
        output_schema: Any,
        *,
        base_sys: str,
        user: str,
) -> Any:
    llm = CoreLLM()
    sys = _compose_system(base_sys, state)
    return await llm.achat(
        user_msg=user,
        system=sys,
        provider=state["provider"],
        model_name=state["model_name"],
        output_schema=output_schema,
    )

# Utility: concise one-liners for tool catalog (token-lean)
def _summarize_tool(t: Any) -> str:
    desc = " ".join((getattr(t, "description", "") or "").split())
    req: List[str] = []
    opt: List[str] = []
    schema = getattr(t, "args_schema", None)
    if schema and hasattr(schema, "get"):
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])
        for arg in props.keys():
            (req if arg in required else opt).append(arg)
    segs = [getattr(t, "name", ""), desc]
    if req: segs.append("req=" + ",".join(req))
    if opt: segs.append("opt=" + ",".join(opt))
    return " | ".join(segs)