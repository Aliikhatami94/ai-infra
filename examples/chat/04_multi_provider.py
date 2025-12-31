#!/usr/bin/env python
"""Multi-Provider Example.

This example demonstrates:
- Provider discovery and configuration checking
- Switching between providers dynamically
- Comparing responses across providers
- Fallback strategies

Required API Keys (multiple recommended):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
- XAI_API_KEY
"""

import asyncio

from ai_infra import LLM


def main():
    llm = LLM()

    # Discover available providers
    print("=" * 60)
    print("Provider Discovery")
    print("=" * 60)

    all_providers = LLM.list_providers()
    print(f"\nAll supported providers: {all_providers}")

    configured_providers = LLM.list_configured_providers()
    print(f"Configured providers (with API keys): {configured_providers}")

    # Check specific provider
    for provider in all_providers:
        is_configured = LLM.is_provider_configured(provider)
        status = "[OK] Configured" if is_configured else "[X] Not configured"
        print(f"  {provider}: {status}")

    if not configured_providers:
        print("\n[!]  No providers configured! Set at least one API key.")
        return

    # List models for configured providers
    print("\n" + "=" * 60)
    print("Available Models")
    print("=" * 60)

    for provider in configured_providers[:2]:  # Limit to first 2 for brevity
        try:
            models = LLM.list_models(provider)
            print(f"\n{provider} models (first 5):")
            for model in models[:5]:
                print(f"  - {model}")
        except Exception as e:
            print(f"\n{provider}: Could not list models - {e}")

    # Use specific providers
    print("\n" + "=" * 60)
    print("Using Specific Providers")
    print("=" * 60)

    question = "What is 2 + 2? Answer with just the number."

    for provider in configured_providers:
        try:
            response = llm.chat(
                question,
                provider=provider,
            )
            print(f"\n{provider}: {response.content.strip()}")
        except Exception as e:
            print(f"\n{provider}: Error - {e}")


def compare_providers():
    """Compare responses across providers."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Comparing Provider Responses")
    print("=" * 60)

    configured = LLM.list_configured_providers()
    if len(configured) < 2:
        print("Need at least 2 configured providers for comparison.")
        return

    prompt = "Explain what an API is in one sentence."
    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    for provider in configured:
        try:
            response = llm.chat(prompt, provider=provider)
            print(f"\n{provider}:")
            print(f"  {response.content}")
        except Exception as e:
            print(f"\n{provider}: Error - {e}")


def provider_specific_models():
    """Use specific models from different providers."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Provider-Specific Models")
    print("=" * 60)

    # Define provider/model pairs to test
    models_to_test = [
        ("openai", "gpt-4o-mini"),
        ("openai", "gpt-4o"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("anthropic", "claude-3-5-haiku-20241022"),
        ("google_genai", "gemini-2.0-flash"),
        ("google_genai", "gemini-1.5-flash"),
        ("xai", "grok-2"),
    ]

    prompt = "Say 'Hello' in exactly 5 different languages, one per line."

    for provider, model in models_to_test:
        if not LLM.is_provider_configured(provider):
            print(f"\n{provider}/{model}: ⏭  Skipped (not configured)")
            continue

        try:
            response = llm.chat(
                prompt,
                provider=provider,
                model_name=model,
                temperature=0.0,
            )
            print(f"\n{provider}/{model}:")
            # Show first 3 lines of response
            lines = response.content.strip().split("\n")[:3]
            for line in lines:
                print(f"  {line}")
            if len(response.content.strip().split("\n")) > 3:
                print("  ...")
        except Exception as e:
            print(f"\n{provider}/{model}: Error - {e}")


def fallback_strategy():
    """Demonstrate fallback between providers."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Provider Fallback Strategy")
    print("=" * 60)

    configured = LLM.list_configured_providers()
    prompt = "What is the meaning of life? Answer briefly."

    print(f"\nConfigured providers: {configured}")
    print("Trying providers in order until one succeeds...\n")

    for provider in configured:
        try:
            response = llm.chat(prompt, provider=provider, max_tokens=50)
            print(f"[OK] {provider} succeeded:")
            print(f"  {response.content.strip()}")
            break  # Stop at first success
        except Exception as e:
            print(f"[X] {provider} failed: {e}")
            continue
    else:
        print("All providers failed!")


async def async_multi_provider():
    """Async multi-provider example."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Async Multi-Provider Comparison")
    print("=" * 60)

    configured = LLM.list_configured_providers()
    if not configured:
        print("No providers configured!")
        return

    prompt = "What's your favorite color? Answer in one word."

    # Run multiple providers concurrently
    async def query_provider(provider: str):
        try:
            response = await llm.achat(prompt, provider=provider)
            return provider, response.content.strip()
        except Exception as e:
            return provider, f"Error: {e}"

    tasks = [query_provider(p) for p in configured]
    results = await asyncio.gather(*tasks)

    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    for provider, response in results:
        print(f"{provider}: {response}")


if __name__ == "__main__":
    main()
    compare_providers()
    provider_specific_models()
    fallback_strategy()
    asyncio.run(async_multi_provider())
