"""08_agent_stream_tokens: Stream raw token deltas.
Usage: python -m quickstart.run llm_agent_stream_tokens
Focus: astream_agent_tokens helper.
"""
import asyncio
from ai_infra.llm import CoreAgent, Providers, Models


def main():
    agent = CoreAgent()

    async def _run():
        print(">>> Token deltas:\n")
        async for token, meta in agent.astream_agent_tokens(
            messages=[{"role": "user", "content": "List three colors separated by commas."}],
            provider=Providers.openai,
            model_name=Models.openai.gpt_4_1_mini.value,
        ):
            print(token, end="", flush=True)
        print("\n--- end tokens ---")

    asyncio.run(_run())
