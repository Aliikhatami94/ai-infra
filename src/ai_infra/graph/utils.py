import inspect
import asyncio
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
        else:
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
    else:
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
    else:
        async def wrapped(state):
            if on_enter: await on_enter(name, state)
            if trace: await trace(name, state, "enter")
            result = await fn(state)
            if on_exit: await on_exit(name, result)
            if trace: await trace(name, result, "exit")
            return result
        return wrapped
