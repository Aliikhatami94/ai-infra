from .chat_cmds import register as register_chat
from .discovery_cmds import register as register_discovery
from .help import _HELP
from .imagegen_cmds import register as register_imagegen
from .mcp_cmds import register as register_mcp
from .multimodal_cmds import register as register_multimodal
from .stdio_publisher_cmds import register as register_stdio_publisher

__all__ = [
    "_HELP",
    "register_chat",
    "register_discovery",
    "register_imagegen",
    "register_mcp",
    "register_multimodal",
    "register_stdio_publisher",
]
