import pprint
import inspect
import asyncio
from typing import Any, Protocol, runtime_checkable, TypeVar, Mapping, Union, Sequence, Awaitable, Dict, List, Optional
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, ConfigDict

S = TypeVar("S", bound=Mapping)

@runtime_checkable
class NodeFn(Protocol[S]):
    def __call__(self, state: S) -> S | Awaitable[S]: ...

@runtime_checkable
class RouterFn(Protocol[S]):
    def __call__(self, state: S) -> str | Awaitable[str]: ...


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
    node_definitions: Sequence[Any]
    edges: Sequence[tuple[str, str]]
    conditional_edges: Optional[Sequence[tuple[str, Any, dict]]] = None
    memory_store: Optional[object] = None


class CoreGraph:
    def __init__(
        self,
        *,
        state_type: type,
        node_definitions: Union[Sequence[NodeFn], dict[str, NodeFn]],
        edges: Union[Sequence[tuple[str, str]], str],
        conditional_edges: Optional[Sequence[tuple[str, RouterFn, dict]]] = None,
        memory_store=None
    ):
        # Accept TypedDict or dict as state_type
        if not (isinstance(state_type, type) and (issubclass(state_type, dict) or hasattr(state_type, '__annotations__'))):
            raise ValueError("state_type must be a TypedDict or dict subclass")
        self.state_type = state_type

        # Accept node_definitions as Sequence[fn] or dict[name, fn]
        if isinstance(node_definitions, dict):
            node_map = node_definitions.copy()
        elif isinstance(node_definitions, Sequence):
            node_map = {fn.__name__: fn for fn in node_definitions}
        else:
            raise ValueError("node_definitions must be a sequence of functions or a dict of name: function")
        node_names = list(node_map.keys())
        if len(set(node_names)) != len(node_names):
            raise ValueError("Node names must be unique")
        all_nodes = set(node_names)

        # Auto-wire linear graphs
        if edges == "linear":
            if len(node_names) < 2:
                raise ValueError("At least two nodes required for linear wiring")
            edges = [(node_names[i], node_names[i+1]) for i in range(len(node_names)-1)]
        elif not (isinstance(edges, Sequence) and all(isinstance(e, tuple) and len(e) == 2 for e in edges)):
            raise ValueError("edges must be a sequence of (start, end) tuples or 'linear'")

        # Inference of START/END
        has_start = any(start == START for start, _ in edges)
        has_end = any(end == END for _, end in edges)
        if edges and not has_start:
            edges = [(START, edges[0][0])] + list(edges)
        if edges and not has_end:
            edges = list(edges) + [(edges[-1][1], END)]

        # Validate edge endpoints
        for start, end in edges:
            for endpoint in (start, end):
                if endpoint not in all_nodes and endpoint not in (START, END):
                    raise ValueError(f"Edge endpoint '{endpoint}' is not a known node or START/END")
        # Validate conditional path maps (validate from_node and values)
        if conditional_edges:
            for from_node, router_fn, path_map in conditional_edges:
                if from_node not in all_nodes and from_node not in (START, END):
                    raise ValueError(f"Conditional edge from_node '{from_node}' is not a known node or START/END")
                for target in path_map.values():
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
        # Always build the graph with async-wrapped nodes/routers
        self.graph = self._build_graph_with_nodes().compile(checkpointer=config.memory_store)

    def _wrap_async(self, fn):
        if inspect.iscoroutinefunction(fn):
            return fn
        async def async_wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return async_wrapper

    def _build_graph_with_nodes(self, node_items=None):
        wf = StateGraph(self.state_type)
        node_items = node_items or self.node_definitions
        for name, fn in node_items:
            wf.add_node(name, self._wrap_async(fn))
        if self.conditional_edges:
            for from_node, router_fn, path_map in self.conditional_edges:
                wf.add_conditional_edges(from_node, self._wrap_async(router_fn), path_map)
        for start, end in self.edges:
            wf.add_edge(start, end)
        return wf

    async def run_async(self, initial_state, *, config=None, on_enter=None, on_exit=None):
        """
        Async run: supports both sync and async nodes and tracing hooks.
        on_enter(node_name, state) and on_exit(node_name, state) can be sync or async.
        """
        def make_tracer(hook):
            if not hook:
                return None
            if inspect.iscoroutinefunction(hook):
                return hook
            async def async_hook(*args, **kwargs):
                return hook(*args, **kwargs)
            return async_hook
        on_enter_async = make_tracer(on_enter)
        on_exit_async = make_tracer(on_exit)

        def trace_wrapper(node_name, fn):
            async def wrapped(state):
                if on_enter_async:
                    await on_enter_async(node_name, state)
                result = await fn(state)
                if on_exit_async:
                    await on_exit_async(node_name, result)
                return result
            return wrapped
        if on_enter or on_exit:
            patched_nodes = [(name, trace_wrapper(name, self._wrap_async(fn))) for name, fn in self.node_definitions]
            wf = StateGraph(self.state_type)
            for name, fn in patched_nodes:
                wf.add_node(name, fn)
            if self.conditional_edges:
                for from_node, router_fn, path_map in self.conditional_edges:
                    wf.add_conditional_edges(from_node, self._wrap_async(router_fn), path_map)
            for start, end in self.edges:
                wf.add_edge(start, end)
            compiled = wf.compile(checkpointer=self._config.memory_store)
            return await compiled.invoke(initial_state, config=config) if config is not None else await compiled.invoke(initial_state)
        else:
            return await self.graph.invoke(initial_state, config=config) if config is not None else await self.graph.invoke(initial_state)

    def run(self, initial_state, *, config=None, on_enter=None, on_exit=None):
        """
        Synchronous wrapper for async run. Use this for sync code.
        """
        return asyncio.run(self.run_async(initial_state, config=config, on_enter=on_enter, on_exit=on_exit))

    def build_graph(self) -> StateGraph:
        """Constructs and returns a StateGraph based on the current configuration."""
        return self._build_graph_with_nodes()

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

    result = graph.run_async({"value": 1}, on_enter=trace_enter, on_exit=trace_exit)
    print(result)
