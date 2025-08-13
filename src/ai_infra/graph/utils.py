import inspect
import asyncio
from typing import Sequence, Any
from langgraph.constants import START, END
from ai_infra.graph.models import Edge, ConditionalEdge

def normalize_node_definitions(node_definitions):
    if isinstance(node_definitions, dict):
        return node_definitions.copy()
    return {fn.__name__: fn for fn in node_definitions}

def normalize_initial_state(initial_state, kwargs):
    if initial_state is None:
        return kwargs
    if kwargs:
        raise ValueError("Provide either initial_state or keyword arguments, not both.")
    return initial_state

def validate_edges(edges, all_nodes):
    for start, end in edges:
        for endpoint in (start, end):
            if endpoint not in all_nodes and endpoint not in (START, END):
                raise ValueError(f"Edge endpoint '{endpoint}' is not a known node or START/END")

def validate_conditional_edges(conditional_edges, all_nodes):
    for start, router_fn, path_map in conditional_edges:
        if start not in all_nodes and start not in (START, END):
            raise ValueError(f"Conditional edge start '{start}' is not a known node or START/END")
        for target in path_map.values():
            if target not in all_nodes and target not in (START, END):
                raise ValueError(f"Conditional path target '{target}' is not a known node or START/END")

def make_router_wrapper(fn, valid_targets):
    async def wrapper(state):
        result = await fn(state) if inspect.iscoroutinefunction(fn) else fn(state)
        if result not in valid_targets:
            raise ValueError(f"Router function returned '{result}', which is not in targets {valid_targets}")
        return result
    return wrapper

def make_hook(hook, event=None, sync=False):
    if not hook:
        return None
    if inspect.iscoroutinefunction(hook):
        if sync:
            def sync_hook(node, state):
                return asyncio.run(hook(node, state) if event is None else hook(node, state, event))
            return sync_hook
        return lambda node, state: hook(node, state) if event is None else hook(node, state, event)
    async def async_hook(node, state):
        return hook(node, state) if event is None else hook(node, state, event)
    return async_hook

def make_trace_fn(trace, sync=False):
    if not trace:
        return None
    if sync:
        def trace_sync(node, state, event):
            return asyncio.run(trace(node, state, event)) if inspect.iscoroutinefunction(trace) else trace(node, state, event)
        return trace_sync
    async def trace_async(node, state, event):
        if inspect.iscoroutinefunction(trace):
            await trace(node, state, event)
        else:
            trace(node, state, event)
    return trace_async

def make_trace_wrapper(name, fn, on_enter, on_exit, trace, sync):
    if sync:
        def wrapped(state):
            if on_enter: on_enter(name, state)
            if trace: trace(name, state, "enter")
            result = fn(state)
            if on_exit: on_exit(name, result)
            if trace: trace(name, result, "exit")
            return result
        return wrapped
    async def wrapped(state):
        if on_enter: await on_enter(name, state)
        if trace: await trace(name, state, "enter")
        result = await fn(state)
        if on_exit: await on_exit(name, result)
        if trace: await trace(name, result, "exit")
        return result
    return wrapped

# ---- new helpers ---------------------------------------------------------------

def normalize_stream_mode(stream_mode):
    if stream_mode is None:
        return ["updates"]
    if isinstance(stream_mode, str):
        return [stream_mode]
    return list(stream_mode)

def wrap_node(fn, sync: bool):
    if sync:
        if not inspect.iscoroutinefunction(fn):
            return fn
        def sync_wrapper(*args, **kwargs):
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    raise RuntimeError(
                        "CoreGraph.run/stream cannot execute async nodes inside a running event loop. "
                        "Use arun/astream instead."
                    )
            except RuntimeError:
                # no running loop; safe to asyncio.run
                pass
            return asyncio.run(fn(*args, **kwargs))
        return sync_wrapper
    if inspect.iscoroutinefunction(fn):
        return fn
    async def async_wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return async_wrapper

def build_edges(node_names: Sequence[str], edges: Sequence[Any]):
    """Return (regular_edges, conditional_edges) normalized + auto START/END guarded."""
    all_nodes = set(node_names)
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
    if regular_edges and not any(s == START for s, _ in regular_edges):
        regular_edges = [(START, regular_edges[0][0]), *regular_edges]
    if regular_edges and not any(e == END for _, e in regular_edges):
        regular_edges = [*regular_edges, (regular_edges[-1][1], END)]
    validate_edges(regular_edges, all_nodes)
    validate_conditional_edges(conditional_edges, all_nodes)
    return regular_edges, conditional_edges