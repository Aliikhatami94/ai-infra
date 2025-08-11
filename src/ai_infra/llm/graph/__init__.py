from typing import Type, TypedDict, Callable, Optional
from langgraph.graph import StateGraph


class CoreGraph:
    def __init__(
            self,
            *,
            state_type: Type[TypedDict],
            node_definitions: list[Callable],
            edges: list[tuple[str, str]],
            conditional_edges: Optional[list[tuple[str, Callable, dict]]] = None,
            memory_store=None
    ):
        self.state_type = state_type
        self.node_definitions = [(fn.__name__, fn) for fn in node_definitions]
        self.edges = edges
        self.conditional_edges = conditional_edges

        self.graph = self.build_graph().compile(checkpointer=memory_store)

    def build_graph(self) -> StateGraph:
        wf = StateGraph(self.state_type)

        # Add all nodes
        for name, fn in self.node_definitions:
            wf.add_node(name, fn)

        # Add conditional edges if defined
        if self.conditional_edges:
            for from_node, router_fn, path_map in self.conditional_edges:
                wf.add_conditional_edges(from_node, router_fn, path_map)

        # Add linear edges
        for start, end in self.edges:
            wf.add_edge(start, end)

        return wf