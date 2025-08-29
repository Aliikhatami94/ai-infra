from __future__ import annotations

from typing import List, TypeVar, Any, Type
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


T = TypeVar("T", bound=BaseModel)

def coerce_structured_result(schema: Type[T], res: Any) -> T:
    """Normalize a model result into a validated Pydantic object of type `schema`."""
    if isinstance(res, schema):
        return res
    if isinstance(res, dict):
        return schema.model_validate(res)

    # AIMessage-like object?
    content = getattr(res, "content", None)
    if isinstance(content, str) and content.strip():
        # try direct JSON
        try:
            return schema.model_validate_json(content)
        except Exception:
            # maybe it's text + JSON → attempt json.loads
            return schema.model_validate(json.loads(content))

    # last resort: stringify
    text = str(res)
    try:
        return schema.model_validate_json(text)
    except Exception:
        try:
            return schema.model_validate(json.loads(text))
        except Exception as e:
            raise ValueError(
                f"Could not coerce model output into {schema.__name__}: {type(res)} / {text[:200]} ..."
            ) from e


def is_pydantic_schema(obj) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)