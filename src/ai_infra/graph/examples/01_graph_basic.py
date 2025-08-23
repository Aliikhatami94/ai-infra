"""01_graph_basic: Basic graph with conditional looping.
Usage: python -m quickstart.run graph_basic
"""
from typing_extensions import TypedDict
from langgraph.graph import END
from ai_infra.graph.core import CoreGraph
from ai_infra.graph.models import Edge, ConditionalEdge

MAX_VALUE = 40

class MyState(TypedDict):
    value: int

def inc(state: MyState) -> MyState:
    """Increment value."""
    state["value"] += 1
    return state

def mul(state: MyState) -> MyState:
    """Double value."""
    state["value"] *= 2
    return state


def _trace(node_name, state, event):  # type: ignore[override]
    print(f"{event.upper()} node={node_name} state={state}")


graph = CoreGraph(
    state_type=MyState,
    node_definitions=[inc, mul],
    edges=[
        Edge(start="inc", end="mul"),
        ConditionalEdge(
            start="mul",
            router_fn=lambda s: "inc" if s["value"] < MAX_VALUE else END,
            targets=["inc", END],
        ),
    ],
)


def main():
    result = graph.run({"value": 1}, trace=_trace)
    print("Final:", result)