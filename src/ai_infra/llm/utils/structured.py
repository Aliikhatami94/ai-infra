from __future__ import annotations

import re, json

from typing import List, TypeVar, Any, Type, Callable
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

def is_pydantic_schema(obj) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)

T = TypeVar("T", bound=BaseModel)

def coerce_from_text_or_fragment(schema, text: str):
    """
    Try strict schema validation first; on failure, try extracting a JSON fragment
    and validate that. Return a validated object or None if both attempts fail.
    """
    # Strict: raw content
    try:
        return validate_or_raise(schema, text)
    except Exception:
        pass

    # Fragment: first JSON-looking snippet
    try:
        cand = _extract_json_candidate(text)
        if cand is None:
            return None
        if is_pydantic_schema(schema):
            return schema.model_validate(cand)
        return validate_or_raise(schema, json.dumps(cand))
    except Exception:
        return None

def coerce_structured_result(schema: Type[T], res: Any) -> T:
    """Normalize arbitrary model output into a validated Pydantic object of type `schema`."""
    # Already the right type
    if isinstance(res, schema):
        return res
    # Plain dict
    if isinstance(res, dict):
        return schema.model_validate(res)

    # AIMessage-like or str: prefer robust text path
    content = getattr(res, "content", None)
    if isinstance(content, str) and content.strip():
        obj = coerce_from_text_or_fragment(schema, content)
        if obj is not None:
            return obj
        # Fall through to hard-fail with context

    if isinstance(res, str):
        obj = coerce_from_text_or_fragment(schema, res)
        if obj is not None:
            return obj
        # Fall through to hard-fail with context

    # Last resorts: stringify and try again with the text pipeline
    text = str(res)
    obj = coerce_from_text_or_fragment(schema, text)
    if obj is not None:
        return obj

    # Make failure explicit and helpful
    preview = (content if isinstance(content, str) and content.strip() else text)[:200]
    raise ValueError(
        f"Could not coerce model output into {schema.__name__}: {type(res)} / {preview} ..."
    )

def _extract_json_candidate(text: str) -> Any | None:
    """
    Best-effort: pull the first balanced JSON object/array from a free-form reply.
    Handles code fences and minor trailing commas.
    """
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"```$", "", t)

    i1, i2 = t.find("{"), t.find("[")
    starts = [i for i in (i1, i2) if i != -1]
    if not starts:
        return None
    start = min(starts)

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
                    cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
                    try:
                        return json.loads(cleaned)
                    except Exception:
                        return None
    return None

def structured_mode_call_sync(
        with_structured_output_fn: Callable[..., Any],
        provider: str,
        model_name: str,
        schema,
        messages,
        model_kwargs,
):
    """
    Single retry using provider's native structured mode ('json_mode') to coerce output.
    Raises if validation still fails.
    """
    model2 = with_structured_output_fn(provider, model_name, schema, method="json_mode", **model_kwargs)
    res2 = model2.invoke(messages)
    content2 = getattr(res2, "content", None) or str(res2)
    return validate_or_raise(schema, content2)

async def structured_mode_call_async(
        with_structured_output_fn: Callable[..., Any],
        provider: str,
        model_name: str,
        schema,
        messages,
        model_kwargs,
):
    """
    Async counterpart of structured_mode_call_sync.
    """
    model2 = with_structured_output_fn(provider, model_name, schema, method="json_mode", **model_kwargs)
    res2 = await model2.ainvoke(messages)
    content2 = getattr(res2, "content", None) or str(res2)
    return validate_or_raise(schema, content2)