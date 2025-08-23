"""01_agent_basic: Minimal agent example.
Usage: python -m quickstart.run llm_agent_basic
What you learn: constructing an agent and running a simple user prompt.
"""
from ai_infra.llm import CoreAgent, Providers, Models


def main():
    agent = CoreAgent()
    resp = agent.run_agent(
        messages=[{"role": "user", "content": "Introduce yourself in one sentence."}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
        model_kwargs={"temperature": 0.7},
    )
    print("Response:\n", getattr(resp, "content", resp))