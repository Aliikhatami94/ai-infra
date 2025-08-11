from typing import Type, TypedDict, Callable, Optional, Dict, List, Any
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, ValidationError, Field


class GraphStructure(BaseModel):
    """Pydantic model representing the graph's structural information."""
    state_type_name: str
    state_schema: Dict[str, str]
    node_count: int
    nodes: List[str]
    edge_count: int
    edges: List[tuple[str, str]]
    conditional_edge_count: int
    conditional_edges: Optional[List[Dict[str, Any]]] = None
    entry_points: List[str]
    exit_points: List[str]
    has_memory: bool


class CoreGraphConfig(BaseModel):
    node_definitions: list[Callable]
    edges: list[tuple[str, str]]
    conditional_edges: Optional[list[tuple[str, Callable, dict]]] = None
    memory_store: Optional[object] = None


class CoreGraph:
    def __init__(
            self,
            *,
            state_type: type,
            node_definitions: list[Callable],
            edges: list[tuple[str, str]],
            conditional_edges: Optional[list[tuple[str, Callable, dict]]] = None,
            memory_store=None
    ):
        # Manual validation for state_type
        if not isinstance(state_type, type) or not hasattr(state_type, '__annotations__'):
            raise ValueError("state_type must be a TypedDict class")
        self.state_type = state_type
        # Validate other config with Pydantic
        try:
            self.config = CoreGraphConfig(
                node_definitions=node_definitions,
                edges=edges,
                conditional_edges=conditional_edges,
                memory_store=memory_store
            )
        except ValidationError as e:
            raise ValueError(f"Invalid CoreGraph configuration: {e}")
        self.node_definitions = [(fn.__name__, fn) for fn in self.config.node_definitions]
        self.edges = self.config.edges
        self.conditional_edges = self.config.conditional_edges

        self.graph = self.build_graph().compile(checkpointer=self.config.memory_store)

    def build_graph(self) -> StateGraph:
        """Constructs and returns a StateGraph based on the current configuration."""
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

    def analyze(self) -> GraphStructure:
        """Return a structured analysis of the graph using Pydantic models."""
        nodes = [name for name, _ in self.node_definitions]

        # Analyze entry and exit points
        start_nodes = [edge[1] for edge in self.edges if edge[0] in ['__start__', START]]
        end_nodes = [edge[0] for edge in self.edges if edge[1] in ['__end__', END]]

        # Process conditional edges
        conditional_edges_data = None
        if self.conditional_edges:
            conditional_edges_data = [
                {
                    "from_node": from_node,
                    "router_function": getattr(router_fn, '__name__', str(router_fn)),
                    "path_options": list(path_map.keys())
                }
                for from_node, router_fn, path_map in self.conditional_edges
            ]

        # Get state schema as string representations
        state_schema = {}
        if hasattr(self.state_type, '__annotations__'):
            state_schema = {
                key: getattr(value, '__name__', str(value))
                for key, value in self.state_type.__annotations__.items()
            }

        return GraphStructure(
            state_type_name=self.state_type.__name__,
            state_schema=state_schema,
            node_count=len(nodes),
            nodes=nodes,
            edge_count=len(self.edges),
            edges=self.edges,
            conditional_edge_count=len(self.conditional_edges) if self.conditional_edges else 0,
            conditional_edges=conditional_edges_data,
            entry_points=start_nodes,
            exit_points=end_nodes,
            has_memory=bool(self.config.memory_store)
        )

    def describe(self) -> Dict:
        """Return the graph structure as a dictionary for programmatic use."""
        return self.analyze().model_dump()

if __name__ == "__main__":
    from typing import TypedDict
    import json

    class MyState(TypedDict):
        value: int

    def node_a(state: MyState) -> MyState:
        """Increment value by 1."""
        state = state.copy()
        state["value"] += 1
        return state

    def node_b(state: MyState) -> MyState:
        """Multiply value by 2."""
        state = state.copy()
        state["value"] *= 2
        return state

    # Define nodes and edges
    nodes = [node_a, node_b]
    edges = [(START, "node_a"), ("node_a", "node_b"), ("node_b", END)]

    # Create the graph
    graph = CoreGraph(
        state_type=MyState,
        node_definitions=nodes,
        edges=edges
    )

    graph_dict = graph.analyze()
    print(graph_dict)
