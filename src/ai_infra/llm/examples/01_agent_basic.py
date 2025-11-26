"""01_agent_basic: Minimal agent example.
Usage: python -m quickstart.run llm_agent_basic
What you learn: constructing an agent and running a simple user prompt.
"""

from ai_infra.llm import Agent, Providers


def main():
    agent = Agent()
    resp = agent.run_agent(
        messages=[{"role": "user", "content": "Introduce yourself in one sentence."}],
        provider=Providers.openai,
        model_name="gpt-4o",
        model_kwargs={"temperature": 0.7},
    )
    print("Response:\n", getattr(resp, "content", resp))
