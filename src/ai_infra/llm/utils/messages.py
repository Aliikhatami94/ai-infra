from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

ImageInput = Union[str, bytes, Path]


def make_messages(
    user: str,
    system: Optional[str] = None,
    extras: Optional[List[Dict[str, Any]]] = None,
    images: Optional[List[ImageInput]] = None,
    provider: Optional[str] = None,  # Kept for backwards compatibility, but ignored
):
    """Create a list of messages for LLM chat.

    Args:
        user: The user message.
        system: Optional system message.
        extras: Optional additional messages.
        images: Optional list of images (URLs, bytes, or file paths).
        provider: Deprecated - no longer needed. Kept for backwards compatibility.

    Returns:
        List of message dicts.
    """
    msgs: List[Dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})

    # Handle images in user message
    if images:
        from ai_infra.llm.multimodal.vision import build_vision_content

        content = build_vision_content(user, images)
        msgs.append({"role": "user", "content": content})
    else:
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
