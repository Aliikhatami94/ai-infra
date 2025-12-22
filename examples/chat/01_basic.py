#!/usr/bin/env python
"""Basic Chat Completion Example.

This example demonstrates:
- Creating an LLM instance
- Sending a simple chat message
- Getting a response
- Auto-detection of configured providers

Required API Keys (at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
- XAI_API_KEY
"""

from ai_infra import LLM


def main():
    # Create LLM instance - provider auto-detected from available API keys
    llm = LLM()

    # Simple chat - the most basic usage
    print("=" * 60)
    print("Basic Chat Example")
    print("=" * 60)

    response = llm.chat("What is the capital of France?")
    print("\nQuestion: What is the capital of France?")
    print(f"Answer: {response.content}")

    # With a system message for context
    print("\n" + "=" * 60)
    print("Chat with System Message")
    print("=" * 60)

    response = llm.chat(
        "What's the best way to learn programming?",
        system="You are a helpful programming tutor. Keep responses concise.",
    )
    print("\nQuestion: What's the best way to learn programming?")
    print(f"Answer: {response.content}")

    # Specify provider and model explicitly
    print("\n" + "=" * 60)
    print("Chat with Explicit Provider/Model")
    print("=" * 60)

    # Check what providers are configured
    configured = LLM.list_configured_providers()
    print(f"\nConfigured providers: {configured}")

    if "openai" in configured:
        response = llm.chat(
            "Say hello in French!",
            provider="openai",
            model_name="gpt-4o-mini",
        )
        print("\nUsing OpenAI gpt-4o-mini:")
        print(f"Answer: {response.content}")
    elif configured:
        # Use first configured provider
        provider = configured[0]
        response = llm.chat(
            "Say hello in French!",
            provider=provider,
        )
        print(f"\nUsing {provider}:")
        print(f"Answer: {response.content}")

    # With model parameters
    print("\n" + "=" * 60)
    print("Chat with Temperature Control")
    print("=" * 60)

    # Low temperature = more deterministic
    response = llm.chat(
        "Write a one-sentence description of Python.",
        temperature=0.0,
    )
    print("\nTemperature 0.0 (deterministic):")
    print(f"Answer: {response.content}")

    # Higher temperature = more creative
    response = llm.chat(
        "Write a one-sentence description of Python.",
        temperature=0.9,
    )
    print("\nTemperature 0.9 (creative):")
    print(f"Answer: {response.content}")


def async_example():
    """Async version of basic chat."""
    import asyncio

    async def run():
        llm = LLM()

        # Async chat
        response = await llm.achat("What is 2 + 2?")
        print(f"\nAsync Answer: {response.content}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
    print("\n" + "=" * 60)
    print("Async Example")
    print("=" * 60)
    async_example()
