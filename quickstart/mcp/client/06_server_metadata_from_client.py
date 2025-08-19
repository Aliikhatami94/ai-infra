from ai_infra.mcp.client.core import CoreMCPClient

cfg = [
    {
        "transport": "streamable_http",
        "url": "http://0.0.0.0:8000/streamable-app/mcp",
    },
    {
        "transport": "sse",
        "url": "http://0.0.0.0:8000/sse-demo/sse",
    },
]

client = CoreMCPClient(cfg)

async def main():
    # 1) Discover server names from their MCP handshake
    await client.discover()
    print("Discovered:", client.server_names())   # e.g. ['demo-streamable']

    # 2) Use the discovered name to open a session
    name = client.server_names()[0]
    async with client.get_client(name) as session:
        info = getattr(session, "mcp_server_info", {}) or {}
        print("Connected to:", info)

    # 3) LangChain adapter spanning all discovered servers
    ms = await client.list_clients()
    tools = await ms.get_tools()
    print("Tool count:", len(tools))