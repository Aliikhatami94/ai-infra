import asyncio
from ai_infra.mcp.core import CoreMCP
from ai_infra.llm import Models, Providers, CoreAgent

from quickstart.mcp.configs.config_importer import (
    json_config,
    yaml_config,
    openmcp_config
)


agent = CoreAgent()

async def _ask_agent(config):
    mcp = CoreMCP(config=config)
    tools = await mcp.get_tools()
    sys_msg = await mcp.get_server_prompt()
    res = await agent.arun_agent(
        messages=[
            *sys_msg,
            {"role": "user", "content": "Tell me your name and then tell me the capital of France (search wikipedia). Then tell me the weather in New York City."},
        ],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4_1_mini.value,
        tools=tools,
    )
    print(res)

def main(config_type):
    if config_type == "json":
        asyncio.run(_ask_agent(json_config))
    elif config_type == "yaml":
        asyncio.run(_ask_agent(yaml_config))
    else:
        asyncio.run(_ask_agent(openmcp_config))
