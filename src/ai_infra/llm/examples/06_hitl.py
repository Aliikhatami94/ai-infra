"""06_hitl: Human-in-the-loop tool approval.
Usage: python -m quickstart.run llm_hitl
If HITL_MODE=interactive the script will prompt for each tool call.
Otherwise it auto-approves tool execution.
"""
import os
from langchain_core.tools import tool
from ai_infra.llm import CoreAgent, Providers, Models


def _make_gate():
    interactive = os.environ.get("HITL_MODE", "").lower() == "interactive"
    def gate(tool_name: str, args: dict):
        if not interactive:
            # auto approve in non-interactive environments
            return {"action": "pass"}
        print(f"Tool request: {tool_name} with args={args}")
        try:
            ans = input("Approve? [y]es / [b]lock:").strip().lower()
        except EOFError:
            return {"action": "block", "replacement": "[auto-block: no input]"}
        if ans.startswith("b"):
            return {"action": "block", "replacement": "[blocked by user]"}
        return {"action": "pass"}
    return gate


def main():
    agent = CoreAgent()

    @tool
    def get_weather(city: str) -> str:
        """Return fake weather for city."""
        return f"Weather in {city}: sunny 80F"

    agent.set_hitl(on_tool_call=_make_gate())
    resp = agent.run_agent(
        messages=[{"role": "user", "content": "Use a tool to get weather for Tokyo."}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4_1_mini.value,
        tools=[get_weather],
    )
    print(getattr(resp, "content", resp))
