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

initial_state = {"value": 1}

def run_graph():
    """Run the graph with the initial state."""
    result = graph.run(initial_state)
    print(result)

def stream_updates():
    for mode, chunk in graph.stream({"value": 1}):
        print(mode, chunk)

def stream_values():
    """Stream values from the graph."""
    for r in graph.stream_values(initial_state):
        print(r)

if __name__ == '__main__':
    stream_updates()
