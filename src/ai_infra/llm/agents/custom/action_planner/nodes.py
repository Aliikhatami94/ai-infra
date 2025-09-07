from __future__ import annotations

import asyncio, json

from .states import PlannerState, ComplexityAssessment, PlanDraft
from .utils import _gather_msgs, _render_presentation_md, _summarize_tool, call_structured


async def assess_complexity(state: PlannerState) -> PlannerState:
    base_sys = (
        "ROLE=ActionPlanner-ComplexityRater\n"
        "Rate the user's request and whether *action planning* is needed.\n"
        "Definitions:\n"
        "- trivial: one-liner answer or single action; no orchestration.\n"
        "- simple: ≤2 steps; minimal branching; args obvious.\n"
        "- moderate: 3–5 steps or needs clarification/sanity checks.\n"
        "- complex: >5 steps, dependencies, or multi-tool/multi-decision workflow.\n"
        "Return JSON: {complexity, reason, skip_planning}.\n"
        "Set skip_planning=true for trivial/simple when no missing config."
    )
    user_msg = _gather_msgs(state)
    assess = await call_structured(state, ComplexityAssessment, base_sys=base_sys, user=user_msg)

    state["meta_complexity"] = assess.complexity
    state["meta_reason"] = assess.reason
    state["skipped"] = bool(assess.skip_planning or assess.complexity in ("trivial", "simple"))

    if state["skipped"]:
        state["plan"] = []
        state["questions"] = []
        state["presentation_md"] = (
            "### No plan needed\n"
            f"- Complexity: **{assess.complexity}**\n"
            f"- Reason: {assess.reason}\n"
            "_Proceed directly without planning._"
        )
        state["approved"] = True
        state["awaiting_approval"] = False
        state["aborted"] = False
    return state

async def analyze(state: PlannerState) -> PlannerState:
    # token-lean catalog for the LLM (keeps drafts grounded & cheap)
    lines: list[str] = [_summarize_tool(t) for t in state.get("tools", [])]
    state["tool_summary"] = "\n".join(lines)
    return state

async def draft_plan(state: PlannerState) -> PlannerState:
    """
    Produce a *general action plan* (steps can be reasoning, checks, asks, or tool calls).
    Commands are not special: use the 'run_cli' tool with a `command` arg.
    Keep it concise. Use steps only if they add value.
    """
    base_sys = (
        "ROLE=ActionPlanner\n"
        "Draft a concise action plan as an ordered list of steps. Allowed step kinds:\n"
        "- reason: {kind:'reason', text, title?}\n"
        "- assert: {kind:'assert', condition, on_fail_hint?}\n"
        "- ask:    {kind:'ask', question, key?}\n"
        "- tool:   {kind:'tool', tool, args, rationale, title?}\n"
        "Rules:\n"
        "1) Prefer 3–7 steps for moderate/complex; fewer if simple.\n"
        "2) Use the provided tools only. Shell commands must use tool 'run_cli' with an exact `command` string.\n"
        "3) Add 'assert' before risky/dangerous steps; include minimal 'on_fail_hint'.\n"
        "4) If configuration is missing, add 'ask' steps and/or include `questions`.\n"
        "5) Be terse; no fluff."
    )
    user_msg = _gather_msgs(state)
    tools_md = state.get("tool_summary") or ""
    user = f"{user_msg}\n\nAvailable tools:\n{tools_md}\n"
    draft = await call_structured(state, PlanDraft, base_sys=base_sys, user=user)

    state["plan"] = [s for s in draft.plan]         # Pydantic instances kept (for rendering)
    state["questions"] = draft.questions or []
    return state

async def present_for_hitl(state: PlannerState) -> PlannerState:
    mode = state.get("io_mode") or "terminal"
    presentation_md = _render_presentation_md(state)
    state["presentation_md"] = presentation_md

    if mode == "api":
        state["awaiting_approval"] = True
        state["approved"] = False
        state["aborted"] = False
        state["feedback"] = state.get("feedback", "")
        return state

    print("\n" + presentation_md)
    print("\nPlease review the plan. You may:")
    print("- type 'y' to approve,")
    print("- type 'r: <feedback>' to request changes,")
    print("- type anything else to reject.")
    ans = (await asyncio.to_thread(input, "\nApprove plan? [y / r:<feedback> / n]: ")).strip()
    ans_l = ans.lower()

    if ans_l == "y":
        state["approved"] = True
        state["feedback"] = ""
        state["aborted"] = False
        state["awaiting_approval"] = False
        return state

    if ans_l.startswith("r:"):
        state["approved"] = False
        state["feedback"] = ans[2:].strip()
        state["aborted"] = False
        state["awaiting_approval"] = False
        return state

    state["approved"] = False
    state["feedback"] = ""
    state["aborted"] = True
    state["awaiting_approval"] = False
    return state

async def replan(state: PlannerState) -> PlannerState:
    base_sys = (
        "ROLE=ActionPlanner-Revision\n"
        "Revise the plan using the same schema and constraints. Keep it concise."
    )
    user_msg = _gather_msgs(state)
    tools_md = state.get("tool_summary") or ""
    feedback = state.get("feedback", "")

    user = (
        f"Original request:\n{user_msg}\n\n"
        f"Available tools:\n{tools_md}\n\n"
        f"Current plan:\n{json.dumps([s.model_dump() for s in state.get('plan', [])], ensure_ascii=False)}\n\n"
        f"Feedback:\n{feedback}"
    )
    draft = await call_structured(state, PlanDraft, base_sys=base_sys, user=user)
    state["plan"] = [s for s in draft.plan]
    state["questions"] = draft.questions or []
    return state