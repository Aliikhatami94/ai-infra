from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional


@dataclass
class MCPMount:
    path: str
    app: Any
    name: Optional[str] = None
    session_manager: Any | None = None
    require_manager: Optional[bool] = None
    async_cleanup: Optional[Callable[[], Awaitable[None]]] = None