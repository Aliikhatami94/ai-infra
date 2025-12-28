#!/usr/bin/env python
"""Basic Embeddings Example.

This example demonstrates:
- Creating embeddings from text
- Auto-detection of available providers
- Using free local embeddings (HuggingFace)
- Using cloud providers (OpenAI, Google, Voyage, Cohere)
- Calculating similarity between texts

Embeddings are vector representations of text that capture semantic meaning.
Texts with similar meanings have similar vectors.

Providers (auto-detected from environment):
- huggingface/local: Free, runs locally (no API key needed)
- openai: OpenAI embeddings (requires OPENAI_API_KEY)
- google_genai: Google embeddings (requires GOOGLE_API_KEY)
- voyage: Voyage AI embeddings (requires VOYAGE_API_KEY)
- cohere: Cohere embeddings (requires COHERE_API_KEY)
"""

from ai_infra import Embeddings

# =============================================================================
# Example 1: Zero-Config Embeddings
# =============================================================================


def zero_config_embeddings():
    """Create embeddings with auto-detection.

    ai-infra automatically:
    1. Checks for API keys in environment
    2. Falls back to free local embeddings (HuggingFace)
    """
    print("=" * 60)
    print("1. Zero-Config Embeddings (Auto-Detection)")
    print("=" * 60)

    # Auto-detects provider from environment
    # Falls back to free huggingface if no API keys found
    embeddings = Embeddings()

    # Get provider info
    print(f"\nProvider: {embeddings.provider}")
    print(f"Model: {embeddings.model}")

    # Generate embedding
    text = "Hello, world!"
    vector = embeddings.embed(text)

    print(f"\nText: '{text}'")
    print(f"Embedding dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")


# =============================================================================
# Example 2: Free Local Embeddings
# =============================================================================


def free_local_embeddings():
    """Use free local embeddings (no API key needed).

    HuggingFace sentence-transformers run entirely on your machine.
    Great for development, testing, and cost-sensitive applications.
    """
    print("\n" + "=" * 60)
    print("2. Free Local Embeddings (No API Key)")
    print("=" * 60)

    # Explicit local embeddings
    embeddings = Embeddings(provider="huggingface")
    # or: Embeddings(provider="local")  # alias

    print(f"\nProvider: {embeddings.provider}")
    print(f"Model: {embeddings.model}")
    print("(Running locally, no API calls!)")

    # Generate embeddings
    texts = [
        "Machine learning is fascinating",
        "Deep learning uses neural networks",
        "The weather is nice today",
    ]

    print("\nGenerating embeddings for 3 texts...")
    for text in texts:
        vector = embeddings.embed(text)
        print(f"  '{text[:40]}...' → {len(vector)} dims")


# =============================================================================
# Example 3: Specific Provider
# =============================================================================


def specific_provider():
    """Use a specific embedding provider.

    Each provider has different models and characteristics.
    """
    print("\n" + "=" * 60)
    print("3. Specific Provider Selection")
    print("=" * 60)

    print("\nSupported providers:")
    print("  - huggingface: Free local (sentence-transformers)")
    print("  - openai: OpenAI (text-embedding-3-small/large)")
    print("  - google_genai: Google (embedding-001)")
    print("  - voyage: Voyage AI (voyage-2, voyage-large-2)")
    print("  - cohere: Cohere (embed-english-v3.0)")

    # Example with OpenAI (if available)
    import os

    if os.getenv("OPENAI_API_KEY"):
        embeddings = Embeddings(provider="openai")
        print(f"\n✓ OpenAI provider: {embeddings.model}")
        vector = embeddings.embed("Test")
        print(f"  Dimensions: {len(vector)}")
    else:
        print("\n(Set OPENAI_API_KEY to test OpenAI embeddings)")

    # Always available: local
    embeddings = Embeddings(provider="huggingface")
    print(f"\n✓ HuggingFace provider: {embeddings.model}")
    vector = embeddings.embed("Test")
    print(f"  Dimensions: {len(vector)}")


# =============================================================================
# Example 4: Custom Model
# =============================================================================


def custom_model():
    """Use a specific model within a provider."""
    print("\n" + "=" * 60)
    print("4. Custom Model Selection")
    print("=" * 60)

    # HuggingFace with different model
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",  # 384 dims, fast
        # "sentence-transformers/all-mpnet-base-v2",  # 768 dims, better quality
        # "BAAI/bge-small-en-v1.5",  # 384 dims, excellent quality
    ]

    for model in models:
        print(f"\nModel: {model}")
        try:
            embeddings = Embeddings(provider="huggingface", model=model)
            vector = embeddings.embed("Test sentence")
            print(f"  Dimensions: {len(vector)}")
        except Exception as e:
            print(f"  Error (install with pip): {e}")


# =============================================================================
# Example 5: Similarity Calculation
# =============================================================================


def similarity_calculation():
    """Calculate similarity between texts.

    Cosine similarity measures how similar two vectors are.
    Score ranges from -1 (opposite) to 1 (identical).
    """
    print("\n" + "=" * 60)
    print("5. Similarity Calculation")
    print("=" * 60)

    embeddings = Embeddings()

    # Define text pairs
    pairs = [
        ("I love dogs", "I love cats"),  # Similar - both about loving pets
        ("I love dogs", "The stock market crashed"),  # Different topics
        ("Machine learning is fun", "AI is interesting"),  # Similar - tech
        ("Hello", "Hi there"),  # Similar - greetings
    ]

    print(f"\nUsing provider: {embeddings.provider}")
    print("\nSimilarity scores (0-1, higher = more similar):\n")

    for text1, text2 in pairs:
        score = embeddings.similarity(text1, text2)
        bar = "█" * int(score * 20)
        print(f"  {score:.3f} {bar}")
        print(f"    '{text1}'")
        print(f"    '{text2}'")
        print()


# =============================================================================
# Example 6: Embedding Properties
# =============================================================================


def embedding_properties():
    """Explore embedding vector properties."""
    print("\n" + "=" * 60)
    print("6. Embedding Vector Properties")
    print("=" * 60)

    embeddings = Embeddings()

    text = "Artificial intelligence is transforming technology"
    vector = embeddings.embed(text)

    print(f"\nText: '{text}'")
    print("\nVector properties:")
    print(f"  Dimensions: {len(vector)}")
    print(f"  Type: {type(vector[0])}")
    print(f"  Min value: {min(vector):.4f}")
    print(f"  Max value: {max(vector):.4f}")
    print(f"  Mean value: {sum(vector) / len(vector):.4f}")

    # Vectors are normalized (length ~= 1)
    import math

    magnitude = math.sqrt(sum(x * x for x in vector))
    print(f"  Magnitude: {magnitude:.4f} (normalized ≈ 1.0)")


# =============================================================================
# Example 7: Comparing Providers
# =============================================================================


def compare_providers():
    """Compare results from different providers."""
    print("\n" + "=" * 60)
    print("7. Provider Comparison")
    print("=" * 60)

    import os

    text = "Natural language processing enables computers to understand text"

    # Collect available providers
    available = []

    # Always have HuggingFace
    available.append(("huggingface", Embeddings(provider="huggingface")))

    # Check for API keys
    if os.getenv("OPENAI_API_KEY"):
        available.append(("openai", Embeddings(provider="openai")))
    if os.getenv("VOYAGE_API_KEY"):
        available.append(("voyage", Embeddings(provider="voyage")))
    if os.getenv("GOOGLE_API_KEY"):
        available.append(("google_genai", Embeddings(provider="google_genai")))

    print(f"\nText: '{text[:50]}...'")
    print(f"\nAvailable providers: {len(available)}")
    print()

    for name, emb in available:
        vector = emb.embed(text)
        print(f"  {name}:")
        print(f"    Model: {emb.model}")
        print(f"    Dimensions: {len(vector)}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Embeddings Examples")
    print("=" * 60)
    print("\nEmbeddings convert text to vectors that capture meaning.")
    print("ai-infra auto-detects providers and falls back to free local.\n")

    zero_config_embeddings()
    free_local_embeddings()
    specific_provider()
    custom_model()
    similarity_calculation()
    embedding_properties()
    compare_providers()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Zero-config works out of the box (auto-detects)")
    print("  2. Free local embeddings available (no API key)")
    print("  3. Multiple providers supported")
    print("  4. similarity() for comparing texts")


if __name__ == "__main__":
    main()
