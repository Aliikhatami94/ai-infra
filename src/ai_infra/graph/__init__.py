# Backward-compatible deprecated alias
from ai_infra.graph.core import CoreGraph, Graph
from ai_infra.graph.models import ConditionalEdge, Edge

__all__ = [
    # New name (preferred)
    "Graph",
    "Edge",
    "ConditionalEdge",
    # Deprecated alias (backward compatibility)
    "CoreGraph",
]
