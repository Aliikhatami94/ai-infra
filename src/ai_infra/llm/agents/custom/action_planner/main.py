from typing import Dict, Any, Literal
from langgraph.graph import END, START

from ai_infra import CoreGraph
from ai_infra.graph import ConditionalEdge, Edge
from ai_infra.llm.agents.custom.action_planner.states import PlannerState
from ai_infra.llm.agents.custom.action_planner.nodes import (
    assess_complexity,
    analyze,
    draft_plan,
    present_for_hitl,
    replan,
)
from ai_infra.llm import PROVIDER, MODEL


ActionPlannerGraph = CoreGraph(
    state_type=PlannerState,
    node_definitions=[assess_complexity, analyze, draft_plan, present_for_hitl, replan],
    edges=[
        Edge(start=START, end="assess_complexity"),
        ConditionalEdge(
            start="assess_complexity",
            router_fn=lambda s: (END if bool(s.get("skipped")) else "analyze"),
            targets=["analyze", END],
        ),
        Edge(start="analyze", end="draft_plan"),
        Edge(start="draft_plan", end="present_for_hitl"),
        ConditionalEdge(
            start="present_for_hitl",
            router_fn=lambda s: (
                END if bool(s.get("awaiting_approval"))
                else END if bool(s.get("approved"))
                else END if bool(s.get("aborted"))
                else "replan"
            ),
            targets=["replan", END],
        ),
        Edge(start="replan", end="present_for_hitl"),
    ],
)

async def run_action_planner(
        *,
        messages,
        tools: list,
        io_mode: Literal["terminal", "api"] = "terminal",
        provider: str = PROVIDER,
        model_name: str = MODEL,
) -> Dict[str, Any]:
    initial: PlannerState = {
        "messages": messages,
        "provider": provider,
        "model_name": model_name,
        "tools": tools,
        "io_mode": io_mode,
        "approved": False,
        "aborted": False,
        "awaiting_approval": False,
        "feedback": "",
    }
    result = await ActionPlannerGraph.arun(initial)
    # Convert plan (Pydantic) to plain dicts for downstream consumers
    plan_dicts = [s.model_dump() for s in (result.get("plan") or [])]
    return {
        "plan": plan_dicts,
        "questions": result.get("questions", []),
        "presentation_md": result.get("presentation_md", ""),
        "approved": bool(result.get("approved")),
        "aborted": bool(result.get("aborted")),
        "awaiting_approval": bool(result.get("awaiting_approval")),
        # optional metadata if you want it downstream
        "meta_complexity": result.get("meta_complexity"),
        "meta_reason": result.get("meta_reason"),
        "skipped": result.get("skipped", False),
    }