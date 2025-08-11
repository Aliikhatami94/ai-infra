from ai_infra.graph.models import Edge, ConditionalEdge
from ai_infra.graph.core import CoreGraph

__all__ = [
    "CoreGraph",
    "Edge",
    "ConditionalEdge",
]

from typing import TypedDict
from langgraph.constants import START, END

class State(TypedDict):
    value: int
    messages: list[str]  # for reducer demo

def inc(state: State) -> State:
    s = dict(state)
    s["value"] += 1
    return s

def times2(state: State) -> State:
    s = dict(state)
    s["value"] *= 2
    return s

def over10(state: State) -> str:
    # choose next node based on value
    if state["value"] < 0:
        return "inc"
    if state["value"] < 10:
        return "times2"
    else:
        return END

math_graph = CoreGraph(
    state_type=State,
    node_definitions=[inc, times2],
    edges=[
        Edge(start="inc", end="times2"),
        ConditionalEdge(
            from_node="times2",
            router_fn=over10,
            targets=["inc", "times2", END],
        ),
    ]
)

print(math_graph.graph.get_graph().draw_mermaid())
