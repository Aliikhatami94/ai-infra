import asyncio

from ai_infra.mcp.client.core import CoreMCPClient

cfg = [
    {
        "name": "streamable-app",
        "config": {
            "transport": "streamable_http",
            "url": "http://0.0.0.0:8000/streamable-app/mcp",
        }
    }
]

client = CoreMCPClient(cfg)

async def main():
    async with client.get_client("streamable-app") as session:
        # session is already initialized by get_client()
        info = getattr(session, "mcp_server_info", {}) or {}
        print(info)

if __name__ == "__main__":
    asyncio.run(main())