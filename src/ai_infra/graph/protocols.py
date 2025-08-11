from typing import Any, Protocol, runtime_checkable, TypeVar, Mapping, Awaitable

S = TypeVar("S", bound=Mapping[str, Any])

@runtime_checkable
class NodeFn(Protocol[S]):
    def __call__(self, state: S) -> S | Awaitable[S]: ...

@runtime_checkable
class RouterFn(Protocol[S]):
    def __call__(self, state: S) -> str | Awaitable[str]: ...

