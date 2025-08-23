"""07_retry: Retry & backoff example.
Usage: python -m quickstart.run llm_retry
Shows configuring retries when the model/tool call may transiently fail.
(This example will still succeed normally; adjust prompt or keys to test retries.)
"""
import asyncio
from ai_infra.llm import CoreAgent, Providers, Models
from ai_infra.llm.tools.tool_controls import no_tools


def main():
    agent = CoreAgent()

    async def _run():
        extra = {
            **no_tools(),
            "retry": {"max_tries": 3, "base": 0.5, "jitter": 0.2},
        }
        resp = await agent.arun_agent(
            messages=[{"role": "user", "content": "Say a short motivational quote."}],
            provider=Providers.openai,
            model_name=Models.openai.gpt_4_1_mini.value,
            extra=extra,
        )
        print(getattr(resp, "content", resp))

    asyncio.run(_run())
