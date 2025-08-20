from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

__all__ = ["OpenAPISpec", "OperationContext", "Operation"]

OpenAPISpec = Dict[str, Any]
Operation = Dict[str, Any]

class OperationContext(BaseModel):
    name: str
    description: str
    method: str
    path: str
    path_params: List[Dict[str, Any]] = Field(default_factory=list)
    query_params: List[Dict[str, Any]] = Field(default_factory=list)
    header_params: List[Dict[str, Any]] = Field(default_factory=list)
    cookie_params: List[Dict[str, Any]] = Field(default_factory=list)
    wants_body: bool = False
    body_content_type: Optional[str] = None
    body_required: bool = False

    def full_description(self) -> str:
        return self.description