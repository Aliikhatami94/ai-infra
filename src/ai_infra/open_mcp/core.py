from typing import List, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools

from .models import McpConfig, ToolDef, Prompt
from .utils import (
    resolve_arg_path as _resolve_arg_path,
    make_system_messages as _make_system_messages
)


class CoreMCP:
    """
    Main entry point for interacting with the MCP system.

    Args:
        config (dict | McpConfig): Configuration dictionary or McpConfig model.
    """
    def __init__(self, config: dict | McpConfig):
        self.config = config if isinstance(config, McpConfig) else McpConfig(**config)

    @staticmethod
    def _process_config_dict(cfg, host: Optional[str], resolve_arg_path) -> dict:
        config_dict = cfg.dict(exclude_unset=True, exclude_none=True)
        if config_dict.get("args"):
            config_dict["args"] = [resolve_arg_path(arg) for arg in config_dict["args"]]
        url = config_dict.get("url")
        if url and host and not url.startswith("http"):
            config_dict["url"] = host.rstrip("/") + "/" + url.lstrip("/")
        return {k: v for k, v in config_dict.items() if v is not None}

    async def get_metadata(self):
        client = await self.get_client()

        for server in self.config.servers:
            server_key = server.name
            async with client.session(server_key) as session:
                tools = await load_mcp_tools(session)

            server.tools = [
                ToolDef(name=t.name, description=t.description)
                for t in tools
            ]

        return self.config.model_dump()

    async def get_server_prompt(self, additional_context: List[Prompt] = None) -> List[SystemMessage]:
        return _make_system_messages(self.config.prompts or [], additional_context)

    async def get_server_setup(self) -> dict:
        host = getattr(self.config, "host", None)
        servers = self.config.servers
        server_setup = {}
        for server in servers:
            cfg = server.config
            server_setup[server.metadata.name] = self._process_config_dict(cfg, host, _resolve_arg_path)
        return server_setup

    async def get_client(self):
        server_setup = await self.get_server_setup()
        return MultiServerMCPClient(server_setup)

    async def get_tools(self):
        client = await self.get_client()
        tools = await client.get_tools()
        return tools