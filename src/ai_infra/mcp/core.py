from typing import Optional
from langchain_mcp_adapters.client import MultiServerMCPClient

from .models import McpConfig

class CoreMCP:
    """
    CoreMCP is the main entry point for interacting with the MCP system.

    Args:
        config (dict): Configuration dictionary following the McpConfig model.
        base_url (Optional[str]): Optional base URL for the MCP server. If provided, relative server URLs in the config will be resolved against this base URL.
    """
    def __init__(self, config: dict):
        # Validate and store config as McpConfig
        self.config = McpConfig(**config)

    async def get_metadata(self):
        # Use self.config if config is not provided
        if config is None:
            config = self.config.dict()
        servers = config.get("servers", {})

        client = await self.get_mcp_client()
        for server_key, server_info in servers.items():
            async with client.session(server_key) as session:
                tools = await load_mcp_tools(session)

            servers[server_key]["tools"] = [{
                "name": tool.name,
                "description": tool.description
            } for tool in tools]

        return config

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
        search_root = self.config_path.parent
        path = next(search_root.rglob(filename), None)
        if not path:
            raise FileNotFoundError(f"Could not find file: {filename}")
        return str(path.resolve())

    async def get_server_setup(self, config: Optional[dict] = None) -> dict:
        if config is None:
            config = self.config.dict()
        server_config = {}
        servers = config.get("servers", {})
        host = self.config.host if hasattr(self.config, "host") else None

        for name, server in servers.items():
            config = server.get('config', {})
            resolved_args = [
                self._resolve_arg_path(arg)
                for arg in config.get("args", [])
            ]
            server_config[name] = {}

            if "command" in config:
                server_config[name]["command"] = config["command"]
            if resolved_args:
                server_config[name]["command"] = resolved_args
            # Prepend host from config if set and url is not absolute
            if "url" in config:
                url = config["url"]
                if host and not url.startswith("http"):
                    url = host.rstrip("/") + "/" + url.lstrip("/")
                server_config[name]["url"] = url
            if "transport" in config:
                server_config[name]["transport"] = config["transport"]

        return server_config

    async def get_client(self):
        server_config = await self.get_server_setup()
        return MultiServerMCPClient(server_config)

    async def get_tools(self):
        client = await self.get_client()
        return await client.get_tools()