from ai_infra.mcp.client import MCPClient


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
    client = MCPClient(cfg)
    tools = await client.list_tools()
    print(tools)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
