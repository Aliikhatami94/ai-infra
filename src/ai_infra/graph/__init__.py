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

if __name__ == '__main__':

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
        if state["value"] < 10:
            return "inc"
        if state["value"] < 20:
            return "times2"
        else:
            return END

    math_graph = CoreGraph(
        state_type=State,
        node_definitions=[inc, times2],
        edges=[
            Edge(start="inc", end="times2"),
            ConditionalEdge(
                start="times2",
                router_fn=over10,
                targets=["inc", "times2", END],
            ),
        ]
    )

    def my_trace(node, state, event):
        print(f"{event.upper()} {node}: {state}")

    res = math_graph.run(value=1, messages=[], trace=my_trace)
    print(res)
