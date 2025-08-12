from typing import Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
import inspect
import os

from .models import McpConfig

class CoreMCP:
    """
    CoreMCP is the main entry point for interacting with the MCP system.

    Args:
        config (dict): Configuration dictionary following the McpConfig model.
        base_url (Optional[str]): Optional base URL for the MCP server. If provided, relative server URLs in the config will be resolved against this base URL.
    """
    def __init__(self, config: dict | McpConfig):
        # Validate and store config as McpConfig
        if isinstance(config, McpConfig):
            self.config = config
        else:
            self.config = McpConfig(**config)

    async def get_metadata(self):
        # Use self.config (McpConfig instance) directly
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

    async def get_server_prompt(self, additional_context: list[str]) -> list:
        """
        Constructs a list of SystemMessage prompts for the server, combining base prompts from the configuration
        with any additional context provided.

        Args:
            additional_context (list[str]): Additional context strings to be included as prompt messages.

        Returns:
            list: A list of SystemMessage objects representing the full prompt sequence.
        """
        prompts = self.config.prompts or {}
        base_prompts = []
        for key, value in prompts.items():
            if isinstance(value, list):
                base_prompts.append(SystemMessage(content="\n\n".join(value)))
            elif isinstance(value, str):
                base_prompts.append(SystemMessage(content=value))

        additional_prompts = [
            SystemMessage(content=prompt)
            for prompt in additional_context if prompt
        ]

        return base_prompts + additional_prompts

    def _resolve_arg_path(self, filename: str) -> str:
        # Dynamically determine the parent directory of the frame where CoreMCP is instantiated (not where this method is defined)
        for frame_info in inspect.stack():
            # Skip frames from this file/module
            if os.path.abspath(frame_info.filename) != os.path.abspath(__file__):
                caller_file = frame_info.filename
                break
        else:
            caller_file = __file__
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        # 1. Absolute path
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename
        # 2. Relative to caller
        rel_path = os.path.abspath(os.path.join(caller_dir, filename))
        if os.path.exists(rel_path):
            return rel_path
        # 3. Walk the tree as fallback
        for root, dirs, files in os.walk(caller_dir):
            if os.path.basename(filename) in files:
                return os.path.abspath(os.path.join(root, os.path.basename(filename)))
        raise FileNotFoundError(f"Could not find file: {filename} (checked as absolute, relative to {caller_dir}, and recursively)")

    async def get_server_setup(self) -> dict:
        """
        Returns a dictionary of server configurations in the format expected by MultiServerMCPClient.
        Uses the Pydantic models directly and only transforms fields as needed (e.g., resolves relative URLs and argument paths).
        """
        host = getattr(self.config, "host", None)
        servers = self.config.servers
        server_setup = {}
        for name, server in servers.items():
            cfg = server.config
            config_dict = cfg.dict(exclude_unset=True, exclude_none=True)
            # Resolve args if present
            if config_dict.get("args"):
                config_dict["args"] = [self._resolve_arg_path(arg) for arg in config_dict["args"]]
            # Resolve URL if needed
            url = config_dict.get("url")
            if url and host and not url.startswith("http"):
                config_dict["url"] = host.rstrip("/") + "/" + url.lstrip("/")
            # Remove fields not needed by MultiServerMCPClient (if any)
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
            server_setup[name] = config_dict
        return server_setup

    async def get_server(self):
        server_setup = await self.get_server_setup()
        return MultiServerMCPClient(server_setup)