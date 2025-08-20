import asyncio
from ai_infra.mcp.client.core import CoreMCPClient

client = CoreMCPClient([
    {"transport": "stdio", "command": "npx", "args": ["-y", "wikipedia-mcp"]},
    {"transport": "streamable_http", "url": "http://0.0.0.0:8000/api/mcp"},
])
doc = asyncio.run(client.get_openmcp())
print(doc)
