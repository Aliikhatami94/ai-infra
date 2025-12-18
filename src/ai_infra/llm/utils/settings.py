from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelSettings:
    provider: str
    model_name: str
    tools: Optional[list[Any]] = None
    extra: Optional[dict[str, Any]] = None
