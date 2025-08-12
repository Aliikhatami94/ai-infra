class MCP:
    def __init__(self, config):
        self.config = None

    async def get_agent_metadata(self, config: Optional[dict] = None):
        servers = config.get("servers", {})

        client = await self.get_mcp_client()
        for server_key, server_info in servers.items():
            async with client.session(server_key) as session:
                tools = await load_mcp_tools(session)

            servers[server_key]["tools"] = [{
                "name": tool.name,
                "description": tool.description
            } for tool in tools]

        config["available_models"] = BaseMcp.get_available_models()
        return config

    async def get_server_prompt(self, additional_context: list[str]):
        metadata = await self.get_agent_metadata()
        prompts = metadata.get('prompts', {})

        base_prompts = [
            SystemMessage(content="\n\n".join(prompts.get(key)))
            for key, _ in prompts.items()
        ]
        additional_prompts = [
            SystemMessage(content=prompt)
            for prompt in additional_context if prompt
        ]

        return base_prompts + additional_prompts

    def resolve_arg_path(self, filename: str) -> str:
        search_root = self.config_path.parent
        path = next(search_root.rglob(filename), None)
        if not path:
            raise FileNotFoundError(f"Could not find file: {filename}")
        return str(path.resolve())

    async def get_mcp_config(self, config: Optional[dict] = None) -> dict:
        server_config = {}
        servers = server_config_raw.get("servers", {})

        for name, server in servers.items():
            config = server.get('config', {})
            resolved_args = [
                self.resolve_arg_path(arg)
                for arg in config.get("args", [])
            ]
            server_config[name] = {}

            if "command" in config:
                server_config[name]["command"] = config["command"]
            if resolved_args:
                server_config[name]["command"] = resolved_args
            if "url" in config:
                server_config[name]["url"] = mcp_host + config["url"]
            if "transport" in config:
                server_config[name]["transport"] = config["transport"]

        return server_config

    async def get_mcp_client(self):
        server_config = await self.get_mcp_config()
        return MultiServerMCPClient(server_config)

    async def get_mcp_tools(self):
        client = await self.get_mcp_client()
        return await client.get_tools()