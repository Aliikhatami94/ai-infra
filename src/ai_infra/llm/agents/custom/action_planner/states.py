from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Literal, Union, Annotated
from pydantic import BaseModel, Field, ConfigDict, field_validator

from langgraph.graph import MessagesState


class ComplexityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    complexity: Literal["trivial", "simple", "moderate", "complex"]
    reason: str
    skip_planning: bool

# ---- Step types (discriminated by `kind`) ----

class ReasonStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["reason"] = "reason"
    text: str
    title: str | None = None  # optional subtitle

class ToolStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["tool"] = "tool"
    tool: str                          # e.g. "run_cli" | "file_read" | "project_scan"
    args: Dict[str, Any] = Field(default_factory=dict)
    rationale: str                     # why this call is needed
    title: str | None = None

class AssertStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["assert"] = "assert"
    condition: str                     # human-readable guard, e.g. "Poetry >= 1.6 is installed"
    on_fail_hint: str | None = None    # short fix suggestion

class AskStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["ask"] = "ask"
    question: str
    key: str | None = None             # optional variable name to store the answer

ActionStep = Annotated[Union[ReasonStep, ToolStep, AssertStep, AskStep], Field(discriminator="kind")]

class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan: List[ActionStep] = Field(..., min_length=1)
    questions: List[str] = Field(default_factory=list)

    @field_validator("questions")
    @classmethod
    def no_blank_questions(cls, v: List[str]) -> List[str]:
        if any((q is None) or (not str(q).strip()) for q in v):
            raise ValueError("Questions must be non-empty strings.")
        return v

# ---- Planner state ----

class Tool(BaseModel):
    name: str
    description: str
    args_schema: Optional[Any]

class PlannerState(TypedDict, total=False):
    messages: MessagesState
    provider: str
    model_name: str
    tools: List[Tool]
    tool_summary: str
    plan: List[ActionStep]
    questions: List[str]

    meta_complexity: Literal["trivial", "simple", "moderate", "complex"]
    meta_reason: str
    skipped: bool

    io_mode: Literal["terminal", "api"]
    awaiting_approval: bool
    approved: bool
    aborted: bool
    feedback: str

    presentation_md: str