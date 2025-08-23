from ai_infra.mcp.client.core import CoreMCPClient

async def main():
    cfg = [
        {
            "transport": "streamable_http",
            "url": "http://0.0.0.0:8000/raw-mount/mcp",
        },
        {
            "transport": "sse",
            "url": "http://0.0.0.0:8000/from-code-sse/sse",
        },
    ]
    client = CoreMCPClient(cfg)
    tools = await client.list_tools()
    print(tools)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())