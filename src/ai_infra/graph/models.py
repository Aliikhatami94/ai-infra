from typing import Any, Optional
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict


class GraphStructure(BaseModel):
    """Pydantic model representing the graph's structural information."""

    state_type_name: str
    state_schema: dict[str, str]
    node_count: int
    nodes: list[str]
    edge_count: int
    edges: list[tuple[str, str]]
    conditional_edge_count: int
    conditional_edges: Optional[list[dict[str, Any]]] = None
    entry_points: list[str]
    exit_points: list[str]
    has_memory: bool
    unreachable: Optional[list[str]] = None


class GraphConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    node_definitions: Sequence[Any]
    edges: Sequence[tuple[str, str]]
    conditional_edges: Optional[Sequence[tuple[str, Any, dict]]] = None
    memory_store: Optional[object] = None


class Edge(BaseModel):
    start: str
    end: str


class ConditionalEdge(BaseModel):
    start: str
    router_fn: Callable
    targets: list[str]


EdgeType = Edge | ConditionalEdge
