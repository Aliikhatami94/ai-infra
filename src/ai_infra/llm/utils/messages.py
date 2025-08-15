from __future__ import annotations
from typing import Any, Dict, List, Optional

def make_messages(user: str, system: Optional[str] = None, extras: Optional[List[Dict[str, Any]]] = None):
    msgs: List[Dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    if extras:
        msgs.extend(extras)
    return msgs

def is_valid_response(res: Any) -> bool:
    """Generic 'did we get something usable?' check."""
    content = getattr(res, "content", None)
    if content is not None:
        return str(content).strip() != ""
    if isinstance(res, dict) and isinstance(res.get("messages"), list) and res["messages"]:
        last = res["messages"][-1]
        if hasattr(last, "content"):
            return str(getattr(last, "content", "")).strip() != ""
        if isinstance(last, dict):
            return str(last.get("content", "")).strip() != ""
    return res is not None