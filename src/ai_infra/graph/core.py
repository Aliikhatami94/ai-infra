import inspect
import asyncio
from typing import Any, Sequence, Union, Dict
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from ai_infra.graph.models import GraphStructure, Edge, ConditionalEdge, EdgeType
from ai_infra.graph.utils import (
    normalize_node_definitions, normalize_initial_state, validate_edges, validate_conditional_edges,
    make_router_wrapper, make_hook, make_trace_fn, make_trace_wrapper
)

class CoreGraph:
    def __init__(self, *, state_type: type, node_definitions: Union[Sequence, dict], edges: Sequence[EdgeType], memory_store=None):
        if not (isinstance(state_type, type) and (issubclass(state_type, dict) or hasattr(state_type, '__annotations__'))):
            raise ValueError("state_type must be a TypedDict or dict subclass")
        self.state_type = state_type
        node_definitions = normalize_node_definitions(node_definitions)
        node_names, all_nodes = list(node_definitions.keys()), set(node_definitions.keys())
        regular_edges, conditional_edges = [], []
        for edge in edges:
            if isinstance(edge, Edge):
                regular_edges.append((edge.start, edge.end))
            elif isinstance(edge, ConditionalEdge):
                for target in edge.targets:
                    if target not in all_nodes and target not in (START, END):
                        raise ValueError(f"ConditionalEdge target '{target}' is not a known node or START/END")
                conditional_edges.append((edge.start, make_router_wrapper(edge.router_fn, edge.targets), {t: t for t in edge.targets}))
            else:
                raise ValueError(f"Unknown edge type: {edge}")
        if regular_edges and not any(start == START for start, _ in regular_edges):
            regular_edges = [(START, regular_edges[0][0])] + list(regular_edges)
        if regular_edges and not any(end == END for _, end in regular_edges):
            regular_edges = list(regular_edges) + [(regular_edges[-1][1], END)]
        validate_edges(regular_edges, all_nodes)
        validate_conditional_edges(conditional_edges, all_nodes)
        self.node_definitions = list(node_definitions.items())
        self.edges = regular_edges
        self.conditional_edges = conditional_edges
        self._memory_store = memory_store
        self.graph = self._build_graph().compile(checkpointer=self._memory_store)

    def _wrap(self, fn, sync):
        if sync:
            if not inspect.iscoroutinefunction(fn):
                return fn
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(fn(*args, **kwargs))
            return sync_wrapper
        else:
            if inspect.iscoroutinefunction(fn):
                return fn
            async def async_wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return async_wrapper

    def _build_graph(self, node_items=None, sync=False):
        wf = StateGraph(self.state_type)
        node_items = node_items or self.node_definitions
        for name, fn in node_items:
            wf.add_node(name, self._wrap(fn, sync))
        for start, router_fn, path_map in self.conditional_edges:
            wf.add_conditional_edges(start, self._wrap(router_fn, sync), path_map)
        for start, end in self.edges:
            wf.add_edge(start, end)
        return wf

    def _prepare_run(self, initial_state=None, *, config=None, on_enter=None, on_exit=None, trace=None, sync=False, **kwargs):
        initial_state = normalize_initial_state(initial_state, kwargs)
        on_enter_fn = make_hook(on_enter, sync=sync)
        on_exit_fn = make_hook(on_exit, sync=sync)
        trace_fn = make_trace_fn(trace, sync=sync)
        if on_enter or on_exit or trace:
            patched_nodes = [
                (name, make_trace_wrapper(name, self._wrap(fn, sync), on_enter_fn, on_exit_fn, trace_fn, sync))
                for name, fn in self.node_definitions
            ]
            wf = self._build_graph(node_items=patched_nodes, sync=sync)
            compiled = wf.compile(checkpointer=self._memory_store)
        else:
            compiled = self._build_graph(sync=sync).compile(checkpointer=self._memory_store) if sync else self.graph
        return compiled, initial_state, config

    async def arun(self, initial_state=None, *, config=None, on_enter=None, on_exit=None, trace=None, **kwargs):
        compiled, initial_state, config = self._prepare_run(initial_state, config=config, on_enter=on_enter, on_exit=on_exit, trace=trace, sync=False, **kwargs)
        return await compiled.ainvoke(initial_state, config=config) if config is not None else await compiled.ainvoke(initial_state)

    def run(self, initial_state=None, *, config=None, on_enter=None, on_exit=None, trace=None, **kwargs):
        compiled, initial_state, config = self._prepare_run(initial_state, config=config, on_enter=on_enter, on_exit=on_exit, trace=trace, sync=True, **kwargs)
        return compiled.invoke(initial_state, config=config) if config is not None else compiled.invoke(initial_state)

    def analyze(self) -> GraphStructure:
        nodes = [name for name, _ in self.node_definitions]
        entry_points = [end for start, end in self.edges if start == START] or nodes[:1]
        exit_points = [start for start, end in self.edges if end == END]
        conditional_edges_data = [
            {"start": start, "router_function": getattr(router_fn, '__name__', str(router_fn)), "path_options": list(path_map.keys())}
            for start, router_fn, path_map in self.conditional_edges
        ] if self.conditional_edges else None
        state_schema = {key: getattr(value, '__name__', str(value)) for key, value in getattr(self.state_type, '__annotations__', {}).items()}
        reachable = set(entry_points)
        edges_map = {start: [] for start, _ in self.edges}
        for start, end in self.edges:
            edges_map.setdefault(start, []).append(end)
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
            has_memory=False,
            unreachable=unreachable
        )

    def describe(self) -> Dict:
        return self.analyze().model_dump()
