from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Iterable, List
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI
import contextlib


class FastMcpConfig(BaseModel):
    path: str
    module: FastMCP | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_module(self):
        mod = self.module
        if mod is None:
            return self
        missing = []
        # Check attributes on the CLASS to avoid triggering lazy properties
        cls = type(mod)
        if not (hasattr(cls, "session_manager") or "session_manager" in dir(cls)):
            missing.append("session_manager")
        if not hasattr(mod, "streamable_http_app"):
            missing.append("streamable_http_app()")
        if missing:
            raise ValueError(f"FastMCP at path='{self.path}' missing: {', '.join(missing)}")
        return self

class FastMcpList(BaseModel):
    mcps: List[FastMcpConfig] = Field(default_factory=list)

    def ensure_all_bound(self) -> None:
        unbound = [c.path for c in self.mcps if c.module is None]
        if unbound:
            raise ValueError(f"MCP modules not bound for paths: {', '.join(unbound)}")

    @classmethod
    def from_mapping(cls, mapping: dict[str, FastMCP]) -> "FastMcpList":
        return cls(mcps=[FastMcpConfig(path=path, module=mod) for path, mod in mapping.items()])


def make_lifespan(mcp_list: FastMcpList):
    mcp_list.ensure_all_bound()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # The session_manager is initialized lazily by streamable_http_app().
        # add_mcp_to_fastapi calls mount_mcps() before startup, so we’re safe here.
        import contextlib as _ctx
        async with _ctx.AsyncExitStack() as stack:
            for cfg in mcp_list.mcps:
                # Do not access the property earlier than this point.
                await stack.enter_async_context(cfg.module.session_manager.run())  # type: ignore[union-attr]
            yield
    return lifespan


def mount_mcps(app: FastAPI, mcp_list: FastMcpList) -> None:
    mcp_list.ensure_all_bound()
    for cfg in mcp_list.mcps:
        # This call initializes .session_manager on the FastMCP instance
        app.mount(cfg.path, cfg.module.streamable_http_app())  # type: ignore[union-attr]


def normalize_mcps(configs: Iterable[FastMcpConfig] | FastMcpList | dict[str, FastMCP]) -> FastMcpList:
    if isinstance(configs, FastMcpList):
        return configs
    if isinstance(configs, dict):
        return FastMcpList.from_mapping(configs)
    return FastMcpList(mcps=list(configs))  # type: ignore[arg-type]


def add_mcp_to_fastapi(app: FastAPI, configs: Iterable[FastMcpConfig] | FastMcpList | dict[str, FastMCP]) -> None:
    mcps = normalize_mcps(configs)
    # assign lifespan (not executed yet)
    app.router.lifespan_context = make_lifespan(mcps)
    # mount sub-apps now -> calls streamable_http_app() -> initializes session_manager
    mount_mcps(app, mcps)