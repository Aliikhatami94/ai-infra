"""06_hitl: Human-in-the-loop tool approval.
Usage: python -m quickstart.run llm_hitl
If HITL_MODE=interactive the script will prompt for each tool call.
Otherwise it auto-approves tool execution.
"""
import os
from langchain_core.tools import tool
from ai_infra.llm import CoreAgent, Providers, Models


def main():
    agent = CoreAgent()

    @tool
    def get_weather(city: str) -> str:
        """Return fake weather for city."""
        return f"Weather in {city}: sunny 80F"

    agent.set_hitl(on_tool_call=agent.make_sys_gate())
    resp = agent.run_agent(
        messages=[{"role": "user", "content": "Use a tool to get weather for Tokyo."}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4_1_mini.value,
        tools=[get_weather],
    )
    print(getattr(resp, "content", resp))

if __name__ == '__main__':
    main()