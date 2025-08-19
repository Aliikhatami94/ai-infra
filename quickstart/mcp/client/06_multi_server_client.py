from ai_infra.mcp.client.core import CoreMCPClient

async def main():
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
    tools = await client.list_tools()
    print(tools)