import pprint
from typing import Type, TypedDict, Callable, Optional, Dict, List, Any, Protocol, runtime_checkable, TypeVar, Mapping, Union
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, ValidationError, Field, ConfigDict

S = TypeVar("S", bound=Mapping)

@runtime_checkable
class NodeFn(Protocol[S]):
    def __call__(self, state: S) -> S: ...


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
    unreachable: Optional[List[str]] = None


class CoreGraphConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    node_definitions: list[Any]
    edges: list[tuple[str, str]]
    conditional_edges: Optional[list[tuple[str, Any, dict]]] = None
    memory_store: Optional[object] = None


class CoreGraph:
    def __init__(
        self,
        *,
        state_type: type,
        node_definitions: Union[list[NodeFn], dict[str, NodeFn]],
        edges: Union[list[tuple[str, str]], str],
        conditional_edges: Optional[list[tuple[str, NodeFn, dict]]] = None,
        memory_store=None
    ):
        # Accept TypedDict or dict as state_type
        if not (isinstance(state_type, type) and (issubclass(state_type, dict) or hasattr(state_type, '__annotations__'))):
            raise ValueError("state_type must be a TypedDict or dict subclass")
        self.state_type = state_type

        # Accept node_definitions as list[fn] or dict[name, fn]
        if isinstance(node_definitions, dict):
            node_map = node_definitions.copy()
        elif isinstance(node_definitions, list):
            node_map = {fn.__name__: fn for fn in node_definitions}
        else:
            raise ValueError("node_definitions must be a list of functions or a dict of name: function")
        node_names = list(node_map.keys())
        if len(set(node_names)) != len(node_names):
            raise ValueError("Node names must be unique")
        all_nodes = set(node_names)

        # Auto-wire linear graphs
        if edges == "linear":
            if len(node_names) < 2:
                raise ValueError("At least two nodes required for linear wiring")
            edges = [(node_names[i], node_names[i+1]) for i in range(len(node_names)-1)]
        elif not (isinstance(edges, list) and all(isinstance(e, tuple) and len(e) == 2 for e in edges)):
            raise ValueError("edges must be a list of (start, end) tuples or 'linear'")

        # Inference of START/END
        if edges and edges[0][0] != START:
            edges = [(START, edges[0][0])] + edges
        if edges and edges[-1][1] != END:
            edges = edges + [(edges[-1][1], END)]

        # Validate edge endpoints
        for start, end in edges:
            for endpoint in (start, end):
                if endpoint not in all_nodes and endpoint not in (START, END):
                    raise ValueError(f"Edge endpoint '{endpoint}' is not a known node or START/END")
        # Validate conditional path maps
        if conditional_edges:
            for from_node, _, path_map in conditional_edges:
                for target in path_map.keys():
                    if target not in all_nodes and target not in (START, END):
                        raise ValueError(f"Conditional path target '{target}' is not a known node or START/END")
        config = CoreGraphConfig(
            node_definitions=list(node_map.values()),
            edges=edges,
            conditional_edges=conditional_edges,
            memory_store=memory_store
        )
        self._config = config
        self.node_definitions = list(node_map.items())
        self.edges = config.edges
        self.conditional_edges = config.conditional_edges
        self.graph = self.build_graph().compile(checkpointer=config.memory_store)

    def run(self, initial_state, *, config=None, on_enter=None, on_exit=None):
        """
        Run the compiled graph with optional tracing hooks.
        on_enter(node_name, state) and on_exit(node_name, state) are called before/after each node.
        """
        def trace_wrapper(node_name, fn):
            def wrapped(state):
                if on_enter:
                    on_enter(node_name, state)
                result = fn(state)
                if on_exit:
                    on_exit(node_name, result)
                return result
            return wrapped

        # Patch nodes with tracing if hooks are provided
        if on_enter or on_exit:
            patched_nodes = {}
            for name, fn in self.node_definitions:
                patched_nodes[name] = trace_wrapper(name, fn)
            # Rebuild graph with patched nodes
            wf = StateGraph(self.state_type)
            for name, fn in patched_nodes.items():
                wf.add_node(name, fn)
            if self.conditional_edges:
                for from_node, router_fn, path_map in self.conditional_edges:
                    wf.add_conditional_edges(from_node, router_fn, path_map)
            for start, end in self.edges:
                wf.add_edge(start, end)
            compiled = wf.compile(checkpointer=self._config.memory_store)
        else:
            compiled = self.graph
        return compiled.invoke(initial_state, config=config) if config is not None else compiled.invoke(initial_state)

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
        # Compute entry and exit points using START/END constants only
        entry_points = [end for start, end in self.edges if start == START]
        exit_points = [start for start, end in self.edges if end == END]
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
        state_schema = {}
        if hasattr(self.state_type, '__annotations__'):
            state_schema = {
                key: getattr(value, '__name__', str(value))
                for key, value in self.state_type.__annotations__.items()
            }
        # Compute reachable nodes
        reachable = set(entry_points)
        edges_map = {start: [] for start, _ in self.edges}
        for start, end in self.edges:
            edges_map.setdefault(start, []).append(end)
        # BFS to find all reachable nodes
        queue = list(entry_points)
        while queue:
            node = queue.pop(0)
            for neighbor in edges_map.get(node, []):
                if neighbor not in reachable and neighbor not in (START, END):
                    reachable.add(neighbor)
                    queue.append(neighbor)
        unreachable = [n for n in nodes if n not in reachable]
        return GraphStructure(
            state_type_name=self.state_type.__name__,
            state_schema=state_schema,
            node_count=len(nodes),
            nodes=nodes,
            edge_count=len(self.edges),
            edges=self.edges,
            conditional_edge_count=len(self.conditional_edges) if self.conditional_edges else 0,
            conditional_edges=conditional_edges_data,
            entry_points=entry_points,
            exit_points=exit_points,
            has_memory=bool(self._config.memory_store),
            # Add unreachable nodes to the output for debugging
            unreachable=unreachable
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

    def node_c(state: MyState) -> MyState:
        """Multiply value by 2."""
        state = state.copy()
        state["value"] *= 10
        return state

    # Define nodes and edges
    nodes = [node_a, node_b, node_c]
    edges = [(START, "node_a"), ("node_a", "node_b"), ("node_b", END)]

    # Create the graph
    graph = CoreGraph(
        state_type=MyState,
        node_definitions=nodes,
        edges=edges
    )

    graph_dict = graph.describe()
    pprint.pprint(graph_dict)

    def trace_enter(node, state):
        print(f"Entering {node}: {state}")

    def trace_exit(node, state):
        print(f"Exiting {node}: {state}")

    result = graph.run({"value": 1}, on_enter=trace_enter, on_exit=trace_exit)
