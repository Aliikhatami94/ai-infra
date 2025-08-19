import asyncio, sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_path = (Path(__file__).resolve().parents[1] / "server" / "02_stdio.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],   # direct file path
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(tools)

if __name__ == "__main__":
    asyncio.run(main())