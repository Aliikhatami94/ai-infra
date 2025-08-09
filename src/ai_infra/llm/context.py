from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class LLMContext:
    provider: str
    model_name: str
    tools: Optional[List[Any]] = None
    extra: Optional[Dict[str, Any]] = None