from collections.abc import AsyncIterator, Iterator
from typing import Any, Sequence, Union, Dict
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from ai_infra.graph.models import GraphStructure, EdgeType
from ai_infra.graph.utils import (
    normalize_node_definitions, normalize_initial_state,
    build_edges, wrap_node, normalize_stream_mode,
    make_hook, make_trace_fn, make_trace_wrapper
)

class CoreGraph:
    def __init__(
            self,
            *,
            state_type: type,
            node_definitions: Union[Sequence, dict],
            edges: Sequence[EdgeType],
            checkpointer=None,
            store=None
    ):
        if not (isinstance(state_type, type) and (issubclass(state_type, dict) or hasattr(state_type, '__annotations__'))):
            raise ValueError("state_type must be a TypedDict or dict subclass")
        self.state_type = state_type

        node_definitions = normalize_node_definitions(node_definitions)
        self.node_definitions = list(node_definitions.items())

        # centralize edge building/validation + START/END
        regular_edges, conditional_edges = build_edges(list(node_definitions.keys()), edges)
        self.edges = regular_edges
        self.conditional_edges = conditional_edges

        self._checkpointer = checkpointer
        self._store = store
        self.graph = self._build_graph().compile(checkpointer=self._checkpointer, store=self._store)

    def _build_graph(self, node_items=None, sync: bool=False) -> StateGraph:
        wf = StateGraph(self.state_type)
        node_items = node_items or self.node_definitions
        for name, fn in node_items:
            wf.add_node(name, wrap_node(fn, sync))
        for start, router_fn, path_map in self.conditional_edges:
            wf.add_conditional_edges(start, wrap_node(router_fn, sync), path_map)
        for start, end in self.edges:
            wf.add_edge(start, end)
        return wf

    def _prepare_run(
            self,
            initial_state=None,
            *,
            config=None,
            on_enter=None,
            on_exit=None,
            trace=None,
            sync: bool=False,
            **kwargs
    ):
        initial_state = normalize_initial_state(initial_state, kwargs)
        # fast path: no hooks → use cached compiled graph
        if not (on_enter or on_exit or trace):
            compiled = self._build_graph(sync=sync).compile(checkpointer=self._checkpointer, store=self._store) if sync else self.graph
            return compiled, initial_state, config

        # else, patch & recompile
        on_enter_fn = make_hook(on_enter, sync=sync)
        on_exit_fn = make_hook(on_exit, sync=sync)
        trace_fn = make_trace_fn(trace, sync=sync)
        patched_nodes = [
            (name, make_trace_wrapper(name, wrap_node(fn, sync), on_enter_fn, on_exit_fn, trace_fn, sync))
            for name, fn in self.node_definitions
        ]
        compiled = self._build_graph(node_items=patched_nodes, sync=sync).compile(checkpointer=self._checkpointer, store=self._store)
        return compiled, initial_state, config

    # ---- invoke -----------------------------------------------------------------
    async def arun(self, initial_state=None, *, config=None, on_enter=None, on_exit=None, trace=None, **kwargs) -> Any:
        compiled, initial_state, config = self._prepare_run(initial_state, config=config, on_enter=on_enter, on_exit=on_exit, trace=trace, sync=False, **kwargs)
        return await compiled.ainvoke(initial_state, config=config) if config is not None else await compiled.ainvoke(initial_state)

    def run(self, initial_state=None, *, config=None, on_enter=None, on_exit=None, trace=None, **kwargs) -> Any:
        compiled, initial_state, config = self._prepare_run(initial_state, config=config, on_enter=on_enter, on_exit=on_exit, trace=trace, sync=True, **kwargs)
        return compiled.invoke(initial_state, config=config) if config is not None else compiled.invoke(initial_state)

    # ---- streaming ---------------------------------------------------------------
    async def astream(self, initial_state=None, *, config=None, stream_mode=("updates","values")) -> AsyncIterator[tuple[str, Any]]:
        stream_mode = normalize_stream_mode(stream_mode)
        compiled, initial_state, config = self._prepare_run(initial_state, config=config, sync=False)
        async for mode, chunk in compiled.astream(initial_state, config=config, stream_mode=stream_mode):
            yield mode, chunk

    def stream(self, initial_state=None, *, config=None, stream_mode=("updates","values")) -> Iterator[tuple[str, Any]]:
        stream_mode = normalize_stream_mode(stream_mode)
        compiled, initial_state, config = self._prepare_run(initial_state, config=config, sync=True)
        for mode, chunk in compiled.stream(initial_state, config=config, stream_mode=stream_mode):
            yield mode, chunk

    async def astream_values(self, initial_state=None, *, config=None):
        async for _, chunk in self.astream(initial_state, config=config, stream_mode="values"):
            yield chunk

    def stream_values(self, initial_state=None, *, config=None):
        for _, chunk in self.stream(initial_state, config=config, stream_mode="values"):
            yield chunk

    # ---- analysis / debug --------------------------------------------------------
    def analyze(self) -> GraphStructure:
        nodes = [name for name, _ in self.node_definitions]
        entry_points = [end for start, end in self.edges if start == START] or nodes[:1]
        exit_points = [start for start, end in self.edges if end == END]
        conditional_edges_data = [
            {"start": start, "router_function": getattr(router_fn, '__name__', str(router_fn)), "path_options": list(path_map.keys())}
            for start, router_fn, path_map in self.conditional_edges
        ] if self.conditional_edges else None
        state_schema = {key: getattr(value, '__name__', str(value)) for key, value in getattr(self.state_type, '__annotations__', {}).items()}
        # reachability
        reachable = set(entry_points)
        edges_map = {start: [] for start, _ in self.edges}
        for start, end in self.edges:
            edges_map.setdefault(start, []).append(end)
        queue = list(entry_points)
        while queue:
            node = queue.pop(0)
            for nbr in edges_map.get(node, []):
                if nbr not in reachable and nbr not in (START, END):
                    reachable.add(nbr); queue.append(nbr)
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
            has_memory=self._checkpointer is not None,
            unreachable=unreachable
        )

    def describe(self) -> Dict:
        return self.analyze().model_dump()

    def get_state(self, config):
        return self.graph.get_state(config)

    def get_state_history(self, config):
        return list(self.graph.get_state_history(config))

    def get_arch_diagram(self) -> str:
        return self.graph.get_graph().draw_mermaid()