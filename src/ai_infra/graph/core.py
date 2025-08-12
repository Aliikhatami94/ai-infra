import inspect
import asyncio
from typing import Any, Sequence, Union, Optional, Mapping, Awaitable, Dict
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from ai_infra.graph.models import GraphStructure, Edge, ConditionalEdge, EdgeType
from ai_infra.graph.protocols import NodeFn, RouterFn

class CoreGraph:
    def __init__(
        self,
        *,
        state_type: type,
        node_definitions: Union[Sequence[NodeFn], dict[str, NodeFn]],
        edges: Sequence[EdgeType],
        memory_store=None
    ):
        # Accept TypedDict or dict as state_type
        if not (isinstance(state_type, type) and (issubclass(state_type, dict) or hasattr(state_type, '__annotations__'))):
            raise ValueError("state_type must be a TypedDict or dict subclass")
        self.state_type = state_type

        # Accept node_definitions as Sequence[fn] or dict[name, fn]
        node_definitions = self._normalize_node_definitions(node_definitions)
        node_names, all_nodes = self._validate_node_names(node_definitions)

        # Separate regular and conditional edges
        regular_edges = []
        conditional_edges = []
        for edge in edges:
            if isinstance(edge, Edge):
                regular_edges.append((edge.start, edge.end))
            elif isinstance(edge, ConditionalEdge):
                # Validate targets are known nodes
                for target in edge.targets:
                    if target not in all_nodes and target not in (START, END):
                        raise ValueError(f"ConditionalEdge target '{target}' is not a known node or START/END")
                # Wrap router_fn to enforce it returns a valid target
                def make_router_wrapper(fn, valid_targets):
                    async def wrapper(state):
                        result = await fn(state) if inspect.iscoroutinefunction(fn) else fn(state)
                        if result not in valid_targets:
                            raise ValueError(f"Router function returned '{result}', which is not in targets {valid_targets}")
                        return result
                    return wrapper
                conditional_edges.append((edge.start, make_router_wrapper(edge.router_fn, edge.targets), {t: t for t in edge.targets}))
            else:
                raise ValueError(f"Unknown edge type: {edge}")

        # Inference of START/END for regular edges
        has_start = any(start == START for start, _ in regular_edges)
        has_end = any(end == END for _, end in regular_edges)
        if regular_edges and not has_start:
            regular_edges = [(START, regular_edges[0][0])] + list(regular_edges)
        if regular_edges and not has_end:
            regular_edges = list(regular_edges) + [(regular_edges[-1][1], END)]

        # Validate edge endpoints
        self._validate_edges(regular_edges, all_nodes)
        # Validate conditional path maps (validate start and values)
        self._validate_conditional_edges(conditional_edges, all_nodes)
        self.node_definitions = list(node_definitions.items())
        self.edges = regular_edges
        self.conditional_edges = conditional_edges
        self._config = None  # No longer using CoreGraphConfig
        self._memory_store = memory_store
        self.graph = self._build_graph_with_nodes().compile(checkpointer=self._memory_store)

    def _wrap_async(self, fn):
        if inspect.iscoroutinefunction(fn):
            return fn
        async def async_wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return async_wrapper

    def _wrap_sync(self, fn):
        if not inspect.iscoroutinefunction(fn):
            return fn
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(fn(*args, **kwargs))
        return sync_wrapper

    def _build_graph_with_nodes(self, node_items=None, sync_mode=False):
        wf = StateGraph(self.state_type)
        node_items = node_items or self.node_definitions
        wrap = self._wrap_sync if sync_mode else self._wrap_async
        for name, fn in node_items:
            wf.add_node(name, wrap(fn))
        if self.conditional_edges:
            for start, router_fn, path_map in self.conditional_edges:
                wf.add_conditional_edges(start, wrap(router_fn), path_map)
        for start, end in self.edges:
            wf.add_edge(start, end)
        return wf

    def _make_tracer(self, hook, event=None, sync_mode=False):
        if not hook:
            return None
        if inspect.iscoroutinefunction(hook):
            if sync_mode:
                def sync_hook(node, state):
                    if event is None:
                        return asyncio.run(hook(node, state))
                    else:
                        return asyncio.run(hook(node, state, event))
                return sync_hook
            else:
                return lambda node, state: hook(node, state) if event is None else hook(node, state, event)
        async def async_hook(node, state):
            if event is None:
                return hook(node, state)
            else:
                return hook(node, state, event)
        return async_hook

    def _make_trace_fn(self, trace, sync_mode=False):
        if not trace:
            return None
        if sync_mode:
            def trace_sync_fn(node, state, event):
                if inspect.iscoroutinefunction(trace):
                    return asyncio.run(trace(node, state, event))
                else:
                    return trace(node, state, event)
            return trace_sync_fn
        else:
            async def trace_async_fn(node, state, event):
                if inspect.iscoroutinefunction(trace):
                    await trace(node, state, event)
                else:
                    trace(node, state, event)
            return trace_async_fn

    def _make_trace_wrapper(self, node_name, fn, on_enter_fn, on_exit_fn, trace_fn, sync_mode=False):
        if sync_mode:
            def wrapped(state):
                if on_enter_fn:
                    on_enter_fn(node_name, state)
                if trace_fn:
                    trace_fn(node_name, state, "enter")
                result = fn(state)
                if on_exit_fn:
                    on_exit_fn(node_name, result)
                if trace_fn:
                    trace_fn(node_name, result, "exit")
                return result
            return wrapped
        else:
            async def wrapped(state):
                if on_enter_fn:
                    await on_enter_fn(node_name, state)
                if trace_fn:
                    await trace_fn(node_name, state, "enter")
                result = await fn(state)
                if on_exit_fn:
                    await on_exit_fn(node_name, result)
                if trace_fn:
                    await trace_fn(node_name, result, "exit")
                return result
            return wrapped

    def _prepare_run(self, initial_state=None, *, config=None, on_enter=None, on_exit=None, trace=None, sync_mode=False, **kwargs):
        """
        Prepares the compiled graph and initial state, handling hooks and node patching.
        Returns (compiled_graph, initial_state, config)
        """
        initial_state = self._normalize_initial_state(initial_state, kwargs)
        on_enter_fn = self._make_tracer(on_enter, sync_mode=sync_mode)
        on_exit_fn = self._make_tracer(on_exit, sync_mode=sync_mode)
        trace_fn = self._make_trace_fn(trace, sync_mode=sync_mode)
        wrap = self._wrap_sync if sync_mode else self._wrap_async
        if on_enter or on_exit or trace:
            patched_nodes = [
                (name, self._make_trace_wrapper(name, wrap(fn), on_enter_fn, on_exit_fn, trace_fn, sync_mode=sync_mode))
                for name, fn in self.node_definitions
            ]
            wf = self._build_graph_with_nodes(node_items=patched_nodes, sync_mode=sync_mode)
            compiled = wf.compile(checkpointer=self._memory_store)
        else:
            compiled = self._build_graph_with_nodes(sync_mode=sync_mode).compile(checkpointer=self._memory_store) if sync_mode else self.graph
        return compiled, initial_state, config

    async def run_async(self, initial_state=None, *, config=None, on_enter=None, on_exit=None, trace=None, **kwargs):
        """
        Async version of run. Uses ainvoke on the compiled graph.
        """
        compiled, initial_state, config = self._prepare_run(initial_state, config=config, on_enter=on_enter, on_exit=on_exit, trace=trace, sync_mode=False, **kwargs)
        if config is not None:
            return await compiled.ainvoke(initial_state, config=config)
        else:
            return await compiled.ainvoke(initial_state)

    def run(self, initial_state=None, *, config=None, on_enter=None, on_exit=None, trace=None, **kwargs):
        """
        Synchronous version of run_async. Uses invoke on the compiled graph.
        """
        compiled, initial_state, config = self._prepare_run(initial_state, config=config, on_enter=on_enter, on_exit=on_exit, trace=trace, sync_mode=True, **kwargs)
        if config is not None:
            return compiled.invoke(initial_state, config=config)
        else:
            return compiled.invoke(initial_state)

    def build_graph(self) -> StateGraph:
        """Constructs and returns a StateGraph based on the current configuration."""
        return self._build_graph_with_nodes()

    def analyze(self) -> GraphStructure:
        """Return a structured analysis of the graph using Pydantic models."""
        nodes = [name for name, _ in self.node_definitions]
        # Compute entry and exit points using START/END constants only
        entry_points = [end for start, end in self.edges if start == START] or nodes[:1]
        exit_points = [start for start, end in self.edges if end == END]
        conditional_edges_data = None
        if self.conditional_edges:
            conditional_edges_data = [
                {
                    "start": start,
                    "router_function": getattr(router_fn, '__name__', str(router_fn)),
                    "path_options": list(path_map.keys())
                }
                for start, router_fn, path_map in self.conditional_edges
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
            has_memory=False,  # self._config is always None, so just use False
            unreachable=unreachable
        )

    def describe(self) -> Dict:
        """Return the graph structure as a dictionary for programmatic use."""
        return self.analyze().model_dump()

    def _normalize_node_definitions(self, node_definitions):
        """
        Always return a dict of node_name: fn, inferring names from function __name__ if needed.
        """
        if isinstance(node_definitions, dict):
            return node_definitions.copy()
        elif isinstance(node_definitions, Sequence):
            return {fn.__name__: fn for fn in node_definitions}
        else:
            raise ValueError("node_definitions must be a sequence of functions or a dict of name: function")

    def _normalize_initial_state(self, initial_state, kwargs):
        if initial_state is None:
            return kwargs
        elif kwargs:
            raise ValueError("Provide either initial_state or keyword arguments, not both.")
        return initial_state

    def _validate_node_names(self, node_map):
        node_names = list(node_map.keys())
        if len(set(node_names)) != len(node_names):
            raise ValueError("Node names must be unique")
        return node_names, set(node_names)

    def _validate_edges(self, edges, all_nodes):
        for start, end in edges:
            for endpoint in (start, end):
                if endpoint not in all_nodes and endpoint not in (START, END):
                    raise ValueError(f"Edge endpoint '{endpoint}' is not a known node or START/END")

    def _validate_conditional_edges(self, conditional_edges, all_nodes):
        for start, router_fn, path_map in conditional_edges:
            if start not in all_nodes and start not in (START, END):
                raise ValueError(f"Conditional edge start '{start}' is not a known node or START/END")
            for target in path_map.values():
                if target not in all_nodes and target not in (START, END):
                    raise ValueError(f"Conditional path target '{target}' is not a known node or START/END")
