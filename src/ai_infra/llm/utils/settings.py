from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelSettings:
    provider: str
    model_name: str
    tools: Optional[List[Any]] = None
    extra: Optional[Dict[str, Any]] = None
