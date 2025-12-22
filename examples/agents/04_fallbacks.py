#!/usr/bin/env python
"""Provider Fallback Chain Example.

This example demonstrates:
- Automatic fallback between providers when one fails
- Configuring fallback chains
- Retry logic with provider switching
- Graceful degradation strategies

Required API Keys (multiple recommended):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
- XAI_API_KEY
"""

import asyncio

from ai_infra import LLM, Agent

# =============================================================================
# Sample Tools
# =============================================================================


def get_stock_price(symbol: str) -> str:
    """Get the current stock price.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, GOOGL).

    Returns:
        Current stock price information.
    """
    prices = {
        "AAPL": "$178.50",
        "GOOGL": "$142.30",
        "MSFT": "$378.90",
        "AMZN": "$178.25",
        "TSLA": "$248.60",
    }
    return prices.get(symbol.upper(), f"No data for {symbol}")


def analyze_trend(symbol: str, days: int = 30) -> str:
    """Analyze stock price trend.

    Args:
        symbol: Stock ticker symbol.
        days: Number of days to analyze.

    Returns:
        Trend analysis summary.
    """
    trends = {
        "AAPL": "Bullish - up 5% over period",
        "GOOGL": "Neutral - stable with slight volatility",
        "MSFT": "Bullish - consistent growth",
        "AMZN": "Mixed - recovering from dip",
        "TSLA": "Volatile - high swings both directions",
    }
    return f"{symbol} ({days}d): {trends.get(symbol.upper(), 'No data')}"


# =============================================================================
# Fallback Examples
# =============================================================================


def basic_fallback_chain():
    """Basic provider fallback chain."""
    print("=" * 60)
    print("Basic Provider Fallback Chain")
    print("=" * 60)

    configured = LLM.list_configured_providers()
    print(f"\nConfigured providers: {configured}")

    if len(configured) < 2:
        print("Need at least 2 providers for fallback demo.")
        print("Set multiple API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.")
        return

    # Build fallback candidates - list of (provider, model) tuples
    # Will try each in order until one succeeds
    candidates = []
    for provider in configured:
        if provider == "openai":
            candidates.append(("openai", "gpt-4o-mini"))
        elif provider == "anthropic":
            candidates.append(("anthropic", "claude-3-5-haiku-20241022"))
        elif provider == "google_genai":
            candidates.append(("google_genai", "gemini-2.0-flash"))
        elif provider == "xai":
            candidates.append(("xai", "grok-2"))

    print(f"Fallback chain: {candidates}")

    agent = Agent(tools=[get_stock_price, analyze_trend])

    # Using run_with_fallbacks
    result = agent.run_with_fallbacks(
        messages=[{"role": "user", "content": "What's Apple's stock price?"}],
        candidates=candidates,
        tools=[get_stock_price],
    )

    print(f"\nResult: {result}")


async def async_fallback_chain():
    """Async provider fallback chain."""
    print("\n" + "=" * 60)
    print("Async Provider Fallback Chain")
    print("=" * 60)

    configured = LLM.list_configured_providers()

    if len(configured) < 2:
        print("Need at least 2 providers for fallback demo.")
        return

    # Build candidates
    candidates = []
    for provider in configured[:3]:  # Use up to 3 providers
        if provider == "openai":
            candidates.append(("openai", "gpt-4o-mini"))
        elif provider == "anthropic":
            candidates.append(("anthropic", "claude-3-5-haiku-20241022"))
        elif provider == "google_genai":
            candidates.append(("google_genai", "gemini-2.0-flash"))

    print(f"Async fallback chain: {candidates}")

    agent = Agent(tools=[get_stock_price, analyze_trend])

    result = await agent.arun_with_fallbacks(
        messages=[
            {"role": "user", "content": "Analyze the trend for Microsoft stock over 60 days"}
        ],
        candidates=candidates,
        tools=[analyze_trend],
    )

    print(f"\nResult: {result}")


def manual_fallback_pattern():
    """Manual fallback pattern for more control."""
    print("\n" + "=" * 60)
    print("Manual Fallback Pattern")
    print("=" * 60)

    configured = LLM.list_configured_providers()

    if not configured:
        print("No providers configured!")
        return

    agent = Agent(tools=[get_stock_price])

    prompt = "What's the current price of Tesla stock?"
    print(f"\nPrompt: {prompt}")

    # Try each provider in order
    last_error = None
    for provider in configured:
        try:
            print(f"\n  Trying {provider}...")
            result = agent.run(prompt, provider=provider)
            print(f"  ✓ Success with {provider}")
            print(f"\nResult: {result}")
            return
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            last_error = e
            continue

    print(f"\nAll providers failed! Last error: {last_error}")


def fallback_with_different_models():
    """Fallback with different models per provider."""
    print("\n" + "=" * 60)
    print("Fallback with Different Models")
    print("=" * 60)

    configured = LLM.list_configured_providers()

    # Define multiple models per provider for fallback
    candidates = []

    if "openai" in configured:
        # Try faster model first, then more capable
        candidates.extend(
            [
                ("openai", "gpt-4o-mini"),
                ("openai", "gpt-4o"),
            ]
        )

    if "anthropic" in configured:
        candidates.extend(
            [
                ("anthropic", "claude-3-5-haiku-20241022"),
                ("anthropic", "claude-sonnet-4-20250514"),
            ]
        )

    if "google_genai" in configured:
        candidates.extend(
            [
                ("google_genai", "gemini-2.0-flash"),
                ("google_genai", "gemini-1.5-pro"),
            ]
        )

    if not candidates:
        print("No providers configured!")
        return

    print("Model fallback chain:")
    for provider, model in candidates:
        print(f"  {provider}/{model}")

    agent = Agent(tools=[get_stock_price, analyze_trend])

    result = agent.run_with_fallbacks(
        messages=[{"role": "user", "content": "Get the price and 30-day trend for GOOGL"}],
        candidates=candidates,
        tools=[get_stock_price, analyze_trend],
    )

    print(f"\nResult: {result}")


def cost_aware_fallback():
    """Fallback chain ordered by cost (cheapest first)."""
    print("\n" + "=" * 60)
    print("Cost-Aware Fallback (Cheapest First)")
    print("=" * 60)

    configured = LLM.list_configured_providers()

    # Models ordered by approximate cost (cheapest first)
    cost_ordered = [
        ("google_genai", "gemini-2.0-flash"),  # Cheapest/free tier
        ("openai", "gpt-4o-mini"),  # Very cheap
        ("anthropic", "claude-3-5-haiku-20241022"),  # Cheap
        ("xai", "grok-2"),  # Medium
        ("openai", "gpt-4o"),  # More expensive
        ("anthropic", "claude-sonnet-4-20250514"),  # Most expensive
    ]

    # Filter to only configured providers
    candidates = [(provider, model) for provider, model in cost_ordered if provider in configured]

    if not candidates:
        print("No providers configured!")
        return

    print("Cost-ordered fallback chain:")
    for provider, model in candidates:
        print(f"  {provider}/{model}")

    agent = Agent(tools=[get_stock_price])

    result = agent.run_with_fallbacks(
        messages=[{"role": "user", "content": "What's Amazon's stock price?"}],
        candidates=candidates,
    )

    print(f"\nResult: {result}")


def latency_optimized_fallback():
    """Fallback with latency measurement."""
    print("\n" + "=" * 60)
    print("Latency-Optimized Fallback")
    print("=" * 60)

    import time

    configured = LLM.list_configured_providers()

    if not configured:
        print("No providers configured!")
        return

    agent = Agent(tools=[get_stock_price])
    prompt = "Get Apple stock price"

    # Measure latency for each provider
    latencies = {}

    for provider in configured[:3]:  # Test up to 3
        try:
            start = time.time()
            agent.run(prompt, provider=provider)
            elapsed = time.time() - start
            latencies[provider] = elapsed
            print(f"  {provider}: {elapsed:.2f}s")
        except Exception as e:
            latencies[provider] = float("inf")
            print(f"  {provider}: Failed ({e})")

    # Sort by latency
    sorted_providers = sorted(latencies.keys(), key=lambda p: latencies[p])
    print(f"\nOptimal order by latency: {sorted_providers}")

    # Use fastest provider
    if sorted_providers and latencies[sorted_providers[0]] < float("inf"):
        fastest = sorted_providers[0]
        print(f"\nUsing fastest provider: {fastest}")
        result = agent.run("Get Tesla stock price and analyze trend", provider=fastest)
        print(f"Result: {result}")


def retry_with_fallback():
    """Combine retry logic with fallback."""
    print("\n" + "=" * 60)
    print("Retry + Fallback Combined")
    print("=" * 60)

    configured = LLM.list_configured_providers()

    if not configured:
        print("No providers configured!")
        return

    # Create agent with retry config
    agent = Agent(
        tools=[get_stock_price],
        on_tool_error="retry",  # Retry failed tools
        max_tool_retries=2,  # Up to 2 retries per tool
    )

    candidates = []
    for provider in configured[:2]:
        if provider == "openai":
            candidates.append(("openai", "gpt-4o-mini"))
        elif provider == "anthropic":
            candidates.append(("anthropic", "claude-3-5-haiku-20241022"))
        elif provider == "google_genai":
            candidates.append(("google_genai", "gemini-2.0-flash"))

    if not candidates:
        candidates = [(configured[0], None)]  # Default model

    print(f"Fallback chain with retry: {candidates}")
    print("Each provider will retry tools up to 2 times before falling back.\n")

    result = agent.run_with_fallbacks(
        messages=[{"role": "user", "content": "What's the price of AAPL?"}],
        candidates=candidates,
    )

    print(f"Result: {result}")


if __name__ == "__main__":
    basic_fallback_chain()
    asyncio.run(async_fallback_chain())
    manual_fallback_pattern()
    fallback_with_different_models()
    cost_aware_fallback()
    latency_optimized_fallback()
    retry_with_fallback()
