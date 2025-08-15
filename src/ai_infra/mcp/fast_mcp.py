from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Iterable, List
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI
import contextlib


class FastMcpConfig(BaseModel):
    path: str
    module: FastMCP | None = None

    # 👇 this line fixes the error
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_module(self):
        if self.module is not None:
            missing = []
            if not hasattr(self.module, "session_manager"):
                missing.append("session_manager")
            if not hasattr(self.module, "streamable_http_app"):
                missing.append("streamable_http_app()")
            if missing:
                raise ValueError(f"FastMCP at path='{self.path}' missing: {', '.join(missing)}")
        return self

class FastMcpList(BaseModel):
    mcps: List[FastMcpConfig] = Field(default_factory=list)

    def ensure_all_bound(self) -> None:
        """Raise if any config has no module bound."""
        unbound = [c.path for c in self.mcps if c.module is None]
        if unbound:
            raise ValueError(
                f"MCP modules not bound for paths: {', '.join(unbound)}"
            )

    @classmethod
    def from_mapping(cls, mapping: dict[str, FastMCP]) -> "FastMcpList":
        """
        Convenience: build from { '/path': fast_mcp_instance, ... }.
        """
        return cls(
            mcps=[FastMcpConfig(path=path, module=mod) for path, mod in mapping.items()]
        )

def make_lifespan(mcp_list: FastMcpList):
    """
    Build a FastAPI lifespan function that enters each MCP's async session context.

    Usage:
        app.router.lifespan_context = make_lifespan(mcps)
    """
    mcp_list.ensure_all_bound()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with contextlib.AsyncExitStack() as stack:
            for cfg in mcp_list.mcps:
                # Each FastMCP exposes an async context manager at .session_manager.run()
                await stack.enter_async_context(cfg.module.session_manager.run())  # type: ignore[union-attr]
            yield

    return lifespan


def mount_mcps(app: FastAPI, mcp_list: FastMcpList) -> None:
    """
    Mount each MCP’s HTTP app at its configured path.

    Usage:
        mount_mcps(app, mcps)
    """
    mcp_list.ensure_all_bound()
    for cfg in mcp_list.mcps:
        app.mount(cfg.path, cfg.module.streamable_http_app())  # type: ignore[union-attr]

def normalize_mcps(
        configs: Iterable[FastMcpConfig] | FastMcpList | dict[str, FastMCP]
) -> FastMcpList:
    """
    Accept multiple shapes and return a McpList:
      - McpList
      - iterable of McpConfig
      - dict[str, FastMCP] where keys are mount paths
    """
    if isinstance(configs, FastMcpList):
        return configs
    if isinstance(configs, dict):
        return FastMcpList.from_mapping(configs)
    # Assume iterable of McpConfig
    return FastMcpList(mcps=list(configs))  # type: ignore[arg-type]


def add_mcp_to_fastapi(
        app: FastAPI,
        configs: Iterable[FastMcpConfig] | FastMcpList | dict[str, FastMCP],
) -> None:
    """
    One-shot helper:
      - normalizes configs
      - sets app.router.lifespan_context
      - mounts subapps

    Returns (lifespan_fn, None) for convenience / testing.
    """
    mcps = normalize_mcps(configs)
    lifespan = make_lifespan(mcps)
    app.router.lifespan_context = lifespan
    mount_mcps(app, mcps)
    return