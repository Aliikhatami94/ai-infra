import asyncio
from ai_infra.mcp.client.core import CoreMCPClient

client = CoreMCPClient([
    {"transport": "streamable_http", "url": "http://0.0.0.0:8000/api/mcp"},
])
tools = asyncio.run(client.list_tools())
docs = asyncio.run(client.get_openmcp("apiframeworks_api"))
print(docs)