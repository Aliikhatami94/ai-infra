"""02_graph_stream_values: Stream only state value snapshots.
Usage: python -m quickstart.run graph_stream_values
"""
from typing_extensions import TypedDict
from ai_infra.graph.core import CoreGraph
from ai_infra.graph.models import Edge

class MyState(TypedDict):
    value: int

def inc(state: MyState) -> MyState:
    state["value"] += 1
    return state

def main():
    graph = CoreGraph(
        state_type=MyState,
        node_definitions=[inc],
        edges=[Edge(start="inc", end="inc")],  # simple loop; rely on user to break (example)
    )
    # For demonstration, manually break after 5 iterations
    iterations = 0
    for snapshot in graph.stream_values({"value": 0}):
        print(snapshot)
        iterations += 1
        if iterations >= 5:
            break