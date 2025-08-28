# ai_infra/llm/core_structured_utils.py
from __future__ import annotations
from typing import List, Type
import json
from pydantic import BaseModel, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser



def build_structured_messages(
        *,
        schema: Type[BaseModel],
        user_msg: str,
        system_preamble: str | None = None,
        forbid_prose: bool = True,
):
    parser = PydanticOutputParser(pydantic_object=schema)
    fmt = parser.get_format_instructions()

    sys_lines: List[str] = []
    if system_preamble:
        sys_lines.append(system_preamble.strip())
    sys_lines.append("Return ONLY a single JSON object that matches the schema below.")
    if forbid_prose:
        sys_lines.append("Do NOT include any prose, markdown, or extra keys. JSON only.")
    sys_lines.append(fmt)
    messages = [
        SystemMessage(content="\n\n".join(sys_lines)),
        HumanMessage(content=user_msg)
    ]
    return messages

def validate_or_raise(schema: type[BaseModel], raw_json: str) -> BaseModel:
    try:
        return schema.model_validate_json(raw_json)
    except ValidationError:
        # Try parsing then validating as python dict (sometimes minor fixups happen upstream)
        obj = json.loads(raw_json)
        return schema.model_validate(obj)