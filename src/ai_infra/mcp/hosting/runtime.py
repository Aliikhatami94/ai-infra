import contextlib
from fastapi import FastAPI
from typing import List, Tuple
import importlib
from urllib.parse import urlparse

from ai_infra.mcp.hosting.models import HostedServer


# ---------- imports ----------

def _split_module_path(path: str) -> Tuple[str, str | None]:
    """Accept 'pkg.mod' or 'pkg.mod:attr'."""
    if ":" in path:
        mod, attr = path.split(":", 1)
        return mod.strip(), (attr.strip() or None)
    return path.strip(), None

def load_mcp_object(module_path: str):
    """
    Import a FastMCP-like object from module_path.
    If no attr provided, try common names: 'mcp', then 'app'.
    """
    mod_name, attr = _split_module_path(module_path)
    module = importlib.import_module(mod_name)

    if attr:
        obj = getattr(module, attr, None)
        if obj is None:
            raise ImportError(f"Attribute '{attr}' not found in module '{mod_name}'")
        return obj

    for candidate in ("mcp", "app"):
        if hasattr(module, candidate):
            return getattr(module, candidate)

    raise ImportError(
        f"No MCP server object found in '{mod_name}'. "
        "Provide an attribute explicitly (e.g., 'pkg.module:mcp')."
    )


# ---------- lifespan ----------

def make_lifespan_manager(servers: List[HostedServer]):
    """
    Start/stop all hosted MCP session managers with the app lifespan.
    """
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with contextlib.AsyncExitStack() as stack:
            for server in servers:
                mcp_obj = load_mcp_object(server.module_path)  # type: ignore[attr-defined]
                if not hasattr(mcp_obj, "session_manager"):
                    raise TypeError(
                        f"Loaded object from '{server.module_path}' has no 'session_manager'."
                    )
                await stack.enter_async_context(mcp_obj.session_manager.run())
            yield
    return lifespan


# ---------- mounting ----------

def _mount_base_from_url(url_or_path: str) -> str:
    """
    Normalize mount base; accept absolute path or full URL.
    Ensures we mount at the parent of '/mcp' if present.
    """
    if not url_or_path:
        return "/mcp"
    if url_or_path.startswith("/"):
        return url_or_path.rstrip("/").removesuffix("/mcp") or "/mcp"
    parsed = urlparse(url_or_path)
    path = (parsed.path or "/mcp").rstrip("/").removesuffix("/mcp")
    return path or "/mcp"

def mount_hosted_servers(app: FastAPI, servers: List[HostedServer]) -> None:
    """
    Mount each hosted MCP's ASGI app at the configured path.
    """
    for server in servers:
        mcp_obj = load_mcp_object(server.module_path)  # type: ignore[attr-defined]
        base = _mount_base_from_url(server.config.url or "/mcp")
        app.mount(base, mcp_obj.streamable_http_app())