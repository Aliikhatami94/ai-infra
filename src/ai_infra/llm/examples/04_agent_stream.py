"""04_agent_stream: Streaming agent response example.
Usage: python -m quickstart.run llm_agent_stream
Streams message chunks (mode = 'messages').
"""
import asyncio
from ai_infra.llm import CoreAgent, Providers, Models


def _print_stream(async_iter):
    async def _run():
        async for chunk, meta in async_iter:
            # chunk may be a message-like object; print its content if present
            content = getattr(chunk, "content", chunk)
            if content:
                print(content, end="", flush=True)
        print("\n--- end stream ---")
    asyncio.run(_run())


def main():
    agent = CoreAgent()
    stream = agent.arun_agent_stream(
        messages=[{"role": "user", "content": "Write two short lines about Mars."}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4_1_mini.value,
        model_kwargs={"temperature": 0.5},
        stream_mode="messages",
    )
    _print_stream(stream)