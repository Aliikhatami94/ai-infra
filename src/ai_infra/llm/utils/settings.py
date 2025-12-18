from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSettings:
    provider: str
    model_name: str
    tools: list[Any] | None = None
    extra: dict[str, Any] | None = None
