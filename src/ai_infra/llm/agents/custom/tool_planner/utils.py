import json
from typing import Any, Dict, List

from ai_infra import CoreLLM
from ai_infra.llm.agents.custom.tool_planner.states import PlannerState, PlanDraft


def _gather_msgs(state, roles=("user", "human")) -> str:
    msgs = state.get("messages") or []
    parts: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in roles:
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
    return "\n".join(parts)

def _gather_sys(state) -> str:
    msgs = state.get("messages") or []
    parts: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
    return "\n".join(parts)

def _render_presentation_md(state: PlannerState) -> str:
    lines = ["### Proposed plan"]
    steps = state.get("plan") or []
    if steps:
        for i, s in enumerate(steps, 1):
            lines.append(f"{i}. {s.get('rationale','(no logic)')}")
            lines.append(f"   - tool: `{s.get('tool')}`")
            lines.append(f"   - args: `{json.dumps(s.get('args', {}), ensure_ascii=False)}`")
    else:
        lines.append("_(no steps)_")
    qs = state.get("questions") or []
    if qs:
        lines.append("\n**Open questions:**")
        for q in qs:
            lines.append(f"- {q}")
    return "\n".join(lines)

def _compose_system(base: str, state: PlannerState) -> str:
    """Append any inherited system messages to a base system string."""
    inherited = _gather_sys(state)
    return base if not inherited else f"{base}\n\n{inherited}"


def _safe_json_loads(raw: str, fallback_questions_msg: str) -> Dict[str, Any]:
    """Parse raw JSON from the LLM; on failure, return a minimal fallback object."""
    try:
        return json.loads(raw)
    except Exception:
        return {"plan": [], "questions": [fallback_questions_msg]}


def _normalize_plan(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize plan steps ensuring a flat string tool name and defaulted fields."""
    norm: List[Dict[str, Any]] = []
    for step in steps or []:
        tool = step.get("tool")
        if isinstance(tool, dict):
            tool = tool.get("name") or ""
        norm.append({
            "title": step.get("title") or "",
            "tool": tool or "",
            "args": step.get("args") or {},
            "rationale": step.get("rationale") or "",
        })
    return norm


def _summarize_tool(t: Any) -> str:
    """One-line, token-lean summary: name | desc | req= | opt=."""
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
    if req:
        segs.append("req=" + ",".join(req))
    if opt:
        segs.append("opt=" + ",".join(opt))
    return " | ".join(segs)

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