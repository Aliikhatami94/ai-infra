from typing import Dict, Any, Literal
from langgraph.graph import END, START

from ai_infra import CoreGraph, Providers, Models
from ai_infra.graph import ConditionalEdge, Edge
from ai_infra.llm.agents.custom.tool_planner.states import PlannerState
from ai_infra.llm.agents.custom.tool_planner.nodes import (
    assess_complexity,
    analyze,
    draft_plan,
    present_for_hitl,
    replan,
)


PROVIDER = Providers.openai
MODEL_NAME = Models.openai.default.value

PlannerGraph = CoreGraph(
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

async def tool_planner(
        *,
        messages,
        tools: list,
        io_mode: Literal["terminal", "api"] = "terminal",
        provider: str = PROVIDER,
        model_name: str = MODEL_NAME,
) -> Dict[str, Any]:
    """
    Plans a sequence of tool calls based on input messages and available tools.
    Supports human-in-the-loop approval in terminal or API modes.
    """
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
    result = await PlannerGraph.arun(initial)
    return {
        "plan": result.get("plan", []),
        "questions": result.get("questions", []),
        "presentation_md": result.get("presentation_md", ""),
        "approved": bool(result.get("approved")),
        "aborted": bool(result.get("aborted")),
        "awaiting_approval": bool(result.get("awaiting_approval")),
    }