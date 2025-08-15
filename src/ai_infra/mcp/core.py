from typing import Optional, List, Dict, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
import inspect
import os
from .models import McpConfig
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools

class CoreMCP:
    """
    Main entry point for interacting with the MCP system.

    Args:
        config (dict | McpConfig): Configuration dictionary or McpConfig model.
    """
    def __init__(self, config: dict | McpConfig):
        self.config = config if isinstance(config, McpConfig) else McpConfig(**config)

    @staticmethod
    def _make_system_messages(prompts: Dict[str, Any], additional_context: List[str]) -> List[SystemMessage]:
        base = [SystemMessage(content="\n\n".join(v) if isinstance(v, list) else v)
                for v in prompts.values() if v]
        additional = [SystemMessage(content=p) for p in additional_context if p]
        return base + additional

    async def get_metadata(self):
        servers = self.config.servers
        client = await self.get_client()
        for server_key, server_obj in servers.items():
            async with client.session(server_key) as session:
                tools = await load_mcp_tools(session)
            servers[server_key]["tools"] = [
                {"name": tool.name, "description": tool.description}
                for tool in tools
            ]
        return self.config.dict()

    async def get_server_prompt(self, additional_context: List[str]) -> List[SystemMessage]:
        return self._make_system_messages(self.config.prompts or {}, additional_context)

    def _resolve_arg_path(self, filename: str) -> str:
        for frame_info in inspect.stack():
            if os.path.abspath(frame_info.filename) != os.path.abspath(__file__):
                caller_file = frame_info.filename
                break
        else:
            caller_file = __file__
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename
        rel_path = os.path.abspath(os.path.join(caller_dir, filename))
        if os.path.exists(rel_path):
            return rel_path
        for root, dirs, files in os.walk(caller_dir):
            if os.path.basename(filename) in files:
                return os.path.abspath(os.path.join(root, os.path.basename(filename)))
        raise FileNotFoundError(f"Could not find file: {filename} (checked as absolute, relative to {caller_dir}, and recursively)")

    @staticmethod
    def _process_config_dict(cfg, host: Optional[str], resolve_arg_path) -> dict:
        config_dict = cfg.dict(exclude_unset=True, exclude_none=True)
        if config_dict.get("args"):
            config_dict["args"] = [resolve_arg_path(arg) for arg in config_dict["args"]]
        url = config_dict.get("url")
        if url and host and not url.startswith("http"):
            config_dict["url"] = host.rstrip("/") + "/" + url.lstrip("/")
        return {k: v for k, v in config_dict.items() if v is not None}

    async def get_server_setup(self) -> dict:
        host = getattr(self.config, "host", None)
        servers = self.config.servers
        server_setup = {}
        for server in servers:
            cfg = server.config
            server_setup[server.name] = self._process_config_dict(cfg, host, self._resolve_arg_path)
        return server_setup

    async def get_client(self):
        server_setup = await self.get_server_setup()
        return MultiServerMCPClient(server_setup)

    async def get_tools(self):
        client = await self.get_client()
        tools = await client.get_tools()
        return tools