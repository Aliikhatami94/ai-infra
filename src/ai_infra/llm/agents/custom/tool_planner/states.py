from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Literal

from langgraph.graph import MessagesState

from pydantic import BaseModel, Field, ConfigDict, field_validator


class ComplexityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    complexity: Literal["trivial", "simple", "moderate", "complex"] = Field(
        ..., description="Overall complexity rating."
    )
    reason: str = Field(..., description="Why this rating was chosen (1-2 lines).")
    skip_planning: bool = Field(
        ..., description="True if planning should be skipped and we can act directly."
    )

class Tool(BaseModel):
    name: str
    description: str
    args_schema: Optional[Any]

class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str = Field(..., description="Why this step is needed.")
    tool: str = Field(..., description="Name of the tool to use.")
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool."
    )

class PlanDraft(BaseModel):
    # Required fields (no defaults). Add min_length if you want non-empty lists.
    model_config = ConfigDict(extra="forbid")
    plan: List[PlanStep] = Field(..., min_length=1, description="Ordered steps to plan the action.")
    questions: List[str] = Field(..., description="Questions needing user input.")
    # If you also want to reject blank questions:
    @field_validator("questions")
    @classmethod
    def no_blank_questions(cls, v: List[str]) -> List[str]:
        if any((q is None) or (not str(q).strip()) for q in v):
            raise ValueError("Questions must be non-empty strings.")
        return v

class PlannerState(TypedDict, total=False):
    messages: MessagesState
    provider: str
    model_name: str
    tools: List[Tool]
    tool_summary: str
    plan: List[PlanStep]
    questions: List[str]

    meta_complexity: Literal["trivial", "simple", "moderate", "complex"]
    meta_reason: str
    skipped: bool   # True if planning was skipped by gate

    # HITL control
    io_mode: Literal["terminal", "api"]
    awaiting_approval: bool
    approved: bool
    aborted: bool
    feedback: str

    # For API mode: return a presentation string
    presentation_md: str