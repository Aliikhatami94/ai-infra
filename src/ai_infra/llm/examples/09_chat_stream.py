"""09_chat_stream: Plain LLM token streaming.
Usage: python -m quickstart.run llm_chat_stream
Shows collecting tokens from a streaming chat completion.
"""
import asyncio
from ai_infra.llm import CoreLLM, Providers, Models


def main():
    llm = CoreLLM()

    async def _run():
        stream = llm.stream_tokens(
            user_msg="Explain quantum tunneling in 2 short sentences.",
            system="You are a concise educational assistant.",
            provider=Providers.openai,
            model_name=Models.openai.gpt_4o.value,
        )
        collected = []
        async for token, meta in stream:
            collected.append(token)
            print(token, end="", flush=True)
        print("\n--- end stream ---\nFull text:\n" + "".join(collected))

    asyncio.run(_run())
