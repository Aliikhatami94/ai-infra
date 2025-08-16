import contextlib
from fastapi import FastAPI
from typing import List, Tuple
import importlib
from urllib.parse import urlparse

from ai_infra.open_mcp.models import Server


def _is_hosted(s: Server) -> bool:
    return getattr(s.info, "type", None) == "hosted"

def _parse_module_path(path: str) -> Tuple[str, str | None]:
    """
    Supports:
      - "pkg.module"                  -> import module, then auto-detect attr ("mcp" then "app")
      - "pkg.module:attr"             -> import module and get `attr`
    """
    if ":" in path:
        mod, attr = path.split(":", 1)
        return mod.strip(), attr.strip() or None
    return path.strip(), None

def _load_hosted_mcp(module_path: str):
    """
    Import the MCP server object from `module_path`.
    - If attr is provided (pkg.mod:attr), return that attribute.
    - Else try common names in order: "mcp", "app".
    Raises clear errors if not found or wrong type.
    """
    mod_name, attr = _parse_module_path(module_path)
    module = importlib.import_module(mod_name)

    if attr:
        obj = getattr(module, attr, None)
        if obj is None:
            raise ImportError(f"Attribute '{attr}' not found in '{mod_name}'")
        return obj

    # Auto-detect common names
    for candidate in ("mcp", "app"):
        if hasattr(module, candidate):
            return getattr(module, candidate)

    raise ImportError(
        f"No MCP server object found in '{mod_name}'. "
        "Provide an attribute explicitly (e.g., 'pkg.module:mcp')."
    )

def _hosted_servers(servers: List[Server]) -> List[Server]:
    """Return only servers that are hosted."""
    return [s for s in servers if _is_hosted(s)]

def make_lifespan(servers: List[Server]):
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with contextlib.AsyncExitStack() as stack:
            for server in _hosted_servers(servers):
                mcp_obj = _load_hosted_mcp(server.info.module_path)  # type: ignore[attr-defined]
                # Expect a FastMCP instance with a session_manager
                if not hasattr(mcp_obj, "session_manager"):
                    raise TypeError(
                        f"Loaded object from '{server.info.module_path}' "
                        "does not expose 'session_manager'."
                    )
                await stack.enter_async_context(mcp_obj.session_manager.run())
            yield
    return lifespan

def _extract_mount_base(url_or_path: str) -> str:
    if url_or_path.startswith("/"):
        return url_or_path.rstrip("/").removesuffix("/mcp") or "/mcp"
    parsed = urlparse(url_or_path)
    path = (parsed.path or "/mcp").rstrip("/").removesuffix("/mcp")
    return path or "/mcp"

def mount_mcps(app: FastAPI, servers: List[Server]) -> None:
    for server in _hosted_servers(servers):
        mcp_obj = _load_hosted_mcp(server.info.module_path)  # type: ignore[attr-defined]
        base = _extract_mount_base(server.config.url or "/mcp")
        app.mount(base, mcp_obj.streamable_http_app())