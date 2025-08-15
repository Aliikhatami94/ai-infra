from typing import List
from pydantic import BaseModel, Field, model_validator, ConfigDict
from mcp.server.fastmcp import FastMCP


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