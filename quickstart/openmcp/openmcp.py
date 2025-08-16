from src.ai_infra.openmcp.core import CoreMCP
from ai_infra.llm import Models, Providers, CoreAgent

from quickstart.openmcp.json_config.config import mcp_json_config
from quickstart.openmcp.yaml_config.config import mcp_yaml_config
from quickstart.openmcp.default_config.config import config as default_config

agent = CoreAgent()

async def ask_agent(config = default_config):
    mcp = CoreMCP(config=config)
    tools = await mcp.get_tools()
    sys_msg = await mcp.get_server_prompt()
    res = await agent.arun_agent(
        messages=[
            *sys_msg,
            {"role": "user", "content": "Tell me your name and then tell me What is the capital of France? search wikipedia. then tell me the weather in New York City."}
        ],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4_1_mini.value,
        tools=tools
    )
    print(res)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(ask_agent())
#     asyncio.run(ask_agent(config=mcp_json_config))
#     asyncio.run(ask_agent(config=mcp_yaml_config))