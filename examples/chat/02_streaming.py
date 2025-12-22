#!/usr/bin/env python
"""Streaming Responses Example.

This example demonstrates:
- Streaming tokens as they're generated
- Real-time output display
- Token metadata access
- Async streaming patterns

Required API Keys (at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
"""

import asyncio

from ai_infra import LLM


async def main():
    """Stream tokens from the LLM."""
    llm = LLM()

    print("=" * 60)
    print("Streaming Tokens Example")
    print("=" * 60)

    print("\nQuestion: Tell me a short story about a robot learning to paint.")
    print("\nStreaming response:")
    print("-" * 40)

    # Stream tokens - yields (token, metadata) tuples
    async for token, meta in llm.stream_tokens(
        "Tell me a short story about a robot learning to paint. Keep it under 100 words."
    ):
        print(token, end="", flush=True)

    print("\n" + "-" * 40)

    # With system message
    print("\n" + "=" * 60)
    print("Streaming with System Message")
    print("=" * 60)

    print("\nSystem: You are a pirate. Speak like one!")
    print("Question: What's the weather like today?")
    print("\nStreaming response:")
    print("-" * 40)

    async for token, _ in llm.stream_tokens(
        "What's the weather like today?",
        system="You are a pirate. Speak like one! Keep it brief.",
    ):
        print(token, end="", flush=True)

    print("\n" + "-" * 40)

    # With parameters
    print("\n" + "=" * 60)
    print("Streaming with Parameters")
    print("=" * 60)

    print("\nUsing low temperature for factual response:")
    print("-" * 40)

    async for token, _ in llm.stream_tokens(
        "What is the boiling point of water?",
        temperature=0.0,
        max_tokens=50,
    ):
        print(token, end="", flush=True)

    print("\n" + "-" * 40)


async def collect_streamed_response():
    """Example: Collect all tokens into a string."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Collecting Streamed Response")
    print("=" * 60)

    tokens = []
    async for token, _ in llm.stream_tokens("Count from 1 to 5."):
        tokens.append(token)
        print(f"Received token: {token!r}")

    full_response = "".join(tokens)
    print(f"\nFull response: {full_response}")


async def streaming_with_progress():
    """Example: Show progress during streaming."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Streaming with Progress Indicator")
    print("=" * 60)

    token_count = 0
    print("\nGenerating... ", end="", flush=True)

    async for token, _ in llm.stream_tokens("List 5 programming languages.", max_tokens=100):
        token_count += 1
        # Update progress every 10 tokens
        if token_count % 10 == 0:
            print(".", end="", flush=True)

    print(f" Done! ({token_count} tokens)")


async def streaming_to_buffer():
    """Example: Stream to a buffer for processing."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Streaming to Buffer")
    print("=" * 60)

    buffer = []
    word_count = 0

    async for token, _ in llm.stream_tokens("Write a haiku about programming.", max_tokens=50):
        buffer.append(token)
        # Count words (rough estimate based on spaces)
        if " " in token or "\n" in token:
            word_count += 1

    result = "".join(buffer)
    print(f"\nHaiku:\n{result}")
    print(f"\nApproximate word count: {word_count}")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(collect_streamed_response())
    asyncio.run(streaming_with_progress())
    asyncio.run(streaming_to_buffer())
