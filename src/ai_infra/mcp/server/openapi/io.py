from __future__ import annotations
import json
from pathlib import Path
from typing import Union
import yaml
from .models import OpenAPISpec

__all__ = ["load_openapi", "load_spec"]

def load_openapi(path_or_str: Union[str, Path]) -> OpenAPISpec:
    p = Path(str(path_or_str))
    text = p.read_text(encoding="utf-8") if p.exists() else str(path_or_str)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)

def load_spec(path_or_str: Union[str, Path]) -> OpenAPISpec:
    return load_openapi(path_or_str)