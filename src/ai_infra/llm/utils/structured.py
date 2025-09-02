from __future__ import annotations

import re, json

from typing import List, TypeVar, Any, Type
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

def extract_json_candidate(text: str) -> Any | None:
    """
    Best-effort: pull the first balanced JSON object/array from a free-form reply.
    Handles code fences and minor trailing-commas.
    """
    if not text:
        return None
    # strip code fences if present
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"```$", "", t)
    # find first '{' or '['
    i1, i2 = t.find("{"), t.find("[")
    starts = [i for i in (i1, i2) if i != -1]
    if not starts:
        return None
    start = min(starts)

    # scan for a balanced close, honoring quotes/escapes
    stack = []
    in_str = False
    esc = False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            top = stack.pop()
            if (top == "{" and ch != "}") or (top == "[" and ch != "]"):
                continue
            if not stack:
                end = i + 1
                snippet = t[start:end]
                try:
                    return json.loads(snippet)
                except Exception:
                    # try a light cleanup for trailing commas
                    cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
                    try:
                        return json.loads(cleaned)
                    except Exception:
                        return None
    return None