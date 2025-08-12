from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, ConfigDict

class GraphStructure(BaseModel):
    """Pydantic model representing the graph's structural information."""
    state_type_name: str
    state_schema: Dict[str, str]
    node_count: int
    nodes: List[str]
    edge_count: int
    edges: List[Tuple[str, str]]
    conditional_edge_count: int
    conditional_edges: Optional[List[Dict[str, Any]]] = None
    entry_points: List[str]
    exit_points: List[str]
    has_memory: bool
    unreachable: Optional[List[str]] = None

class CoreGraphConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    node_definitions: Sequence[Any]
    edges: Sequence[Tuple[str, str]]
    conditional_edges: Optional[Sequence[Tuple[str, Any, dict]]] = None
    memory_store: Optional[object] = None

class Edge(BaseModel):
    start: str
    end: str

class ConditionalEdge(BaseModel):
    start: str
    router_fn: Callable
    targets: list[str]

EdgeType = Union[Edge, ConditionalEdge]
