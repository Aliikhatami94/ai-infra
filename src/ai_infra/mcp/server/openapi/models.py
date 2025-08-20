from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

__all__ = ["OpenAPISpec", "OperationContext", "Operation", "OpReport", "BuildReport"]

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

@dataclass
class OpReport:
    operation_id: Optional[str]
    tool_name: str
    method: str
    path: str
    base_url: str
    base_url_source: str                 # override | operation | path | root | none
    has_body: bool
    body_content_type: Optional[str]
    body_required: bool
    params: Dict[str, int]
    security: Dict[str, Any]
    input_model_fields: int = 0          # number of input fields
    media_types_seen: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class BuildReport:
    title: str
    total_ops: int = 0
    registered_tools: int = 0
    skipped_ops: int = 0
    warnings: List[str] = field(default_factory=list)
    ops: List[OpReport] = field(default_factory=list)

    def to_json(self) -> str:
        def _default(o):
            if isinstance(o, (BuildReport, OpReport)):
                return o.__dict__
            return str(o)
        return json.dumps(self, default=_default, indent=2)