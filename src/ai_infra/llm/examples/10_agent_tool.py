"""10_agent_tool: Minimal agent example with tool run.
Usage: python -m quickstart.run llm_agent_basic
What you learn: constructing an agent and running a simple user prompt.
"""

from ai_infra.llm import Agent, Providers
from ai_infra.llm.tools.custom.cli import run_cli


def main():
    agent = Agent()
    resp = agent.run_agent(
        messages=[{"role": "user", "content": "What is the current directory?"}],
        provider=Providers.openai,
        model_name="gpt-4o",
        tools=[run_cli],
        model_kwargs={"temperature": 0.7},
    )
    print("Response:\n", getattr(resp, "content", resp))
