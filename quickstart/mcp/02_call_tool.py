import asyncio
from ai_infra.mcp.core import CoreMCP
from ai_infra.llm import CoreAgent

from quickstart.mcp.configs.config_importer import openmcp_config

agent = CoreAgent()

async def _call_mcp_tool(config):
    mcp = CoreMCP(config=config)
    resp = await mcp.call_tool(
        server_name="test_server",
        tool_name="get_weather",
        arguments={"city": "New York City"}
    )
    print(resp)

def main():
    asyncio.run(_call_mcp_tool(openmcp_config))