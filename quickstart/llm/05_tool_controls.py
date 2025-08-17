"""05_tool_controls: Tool choice & control example.
Usage: python -m quickstart.run llm_tool_controls
Demonstrates forcing a specific tool call and disabling parallel calls.
"""
from langchain_core.tools import tool
from ai_infra.llm import CoreAgent, Providers, Models
from ai_infra.llm.tools.tool_controls import ToolCallControls


def main():
    agent = CoreAgent()

    @tool
    def weather_a(city: str) -> str:
        """Return a sunny weather stub."""
        return f"Weather in {city}: sunny 75F"

    @tool
    def weather_b(city: str) -> str:
        """Return a rainy weather stub."""
        return f"Weather in {city}: rainy 60F"

    resp = agent.run_agent(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
        tools=[weather_a, weather_b],
        tool_controls=ToolCallControls(
            tool_choice={"name": "weather_b"},  # force specific tool
            parallel_tool_calls=False,
            force_once=True,
        ),
    )
    print(getattr(resp, "content", resp))
