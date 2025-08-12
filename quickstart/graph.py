from typing_extensions import TypedDict
from langgraph.graph import END

class MyState(TypedDict):
    value: int

def inc(s):
    """Increment the value in the state."""
    s['value'] += 1
    return s

def mul(s):
    """Multiply the value in the state by 2."""
    s['value'] *= 2
    return s

from ai_infra.graph.core import CoreGraph
from ai_infra.graph.models import Edge, ConditionalEdge

def my_trace(node_name, state, event):
    print(f"{event.upper()} node: {node_name}, state: {state}")

graph = CoreGraph(
    state_type=MyState,
    node_definitions=[inc, mul],
    edges=[
        Edge(start="inc", end="mul"),
        ConditionalEdge(
            start="mul",
            router_fn=lambda s: "inc" if s['value'] < 40 else END,
            targets=["inc", END]
        )
    ],
)

if __name__ == '__main__':
    res = graph.run(value=1, trace=my_trace)
    print(res)
