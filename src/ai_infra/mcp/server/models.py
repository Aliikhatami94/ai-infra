from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class MCPMount:
    path: str
    app: Any
    name: str | None = None
    session_manager: Any | None = None
    require_manager: bool | None = None
    async_cleanup: Callable[[], Awaitable[None]] | None = None
