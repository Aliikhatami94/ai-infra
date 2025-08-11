from .providers import Providers
from .models import Models

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

@dataclass
class LLMContext:
    provider: str  # Should be one of Providers.<provider>
    model_name: str  # Should be one of Models.<provider>.<model>.value
    tools: Optional[List[Any]] = None
    extra: Optional[Dict[str, Any]] = None