import asyncio
from ai_infra.mcp.client.core import CoreMCPClient

client = CoreMCPClient([
    {"transport": "stdio", "command": "npx", "args": ["-y", "wikipedia-mcp"]},
])
doc = asyncio.run(client.get_metadata())
print(doc)
