from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

__all__ = ["OperationContext"]

class OperationContext(BaseModel):
    """Metadata for a single OpenAPI operation used to register an MCP tool.

    Uses Pydantic for future validation/serialization needs.
    """
    name: str
    description: str
    method: str
    path: str
    path_params: List[Dict[str, Any]] = Field(default_factory=list)
    query_params: List[Dict[str, Any]] = Field(default_factory=list)
    header_params: List[Dict[str, Any]] = Field(default_factory=list)
    wants_body: bool = False
    body_content_type: Optional[str] = None
    body_required: bool = False

    def full_description(self) -> str:
        return self.description

