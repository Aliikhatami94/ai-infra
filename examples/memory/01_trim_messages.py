#!/usr/bin/env python
"""Trim Messages Example.

This example demonstrates:
- Trimming messages by count (last/first N)
- Trimming messages by token limit
- Preserving system messages
- Custom token counters
- Integration with LLM chat

Use trim_messages when you need to fit a conversation
into a model's context window without summarization.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_infra.llm.memory import count_tokens_approximate, trim_messages

# =============================================================================
# Example 1: Keep Last N Messages
# =============================================================================


def trim_last_n():
    """Keep the last N messages (most recent)."""
    print("=" * 60)
    print("1. Keep Last N Messages")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a programming language."),
        HumanMessage(content="Is it easy to learn?"),
        AIMessage(content="Yes, Python has simple syntax."),
        HumanMessage(content="What about performance?"),
        AIMessage(content="It's slower than compiled languages."),
        HumanMessage(content="Can I use it for AI?"),
        AIMessage(content="Yes! It's the top language for AI/ML."),
    ]

    print(f"\n  Original: {len(messages)} messages")

    # Keep last 4 messages (preserves system by default)
    trimmed = trim_messages(
        messages,
        strategy="last",
        max_messages=4,
    )

    print(f"  After trim: {len(trimmed)} messages")
    print("\n  Kept messages:")
    for msg in trimmed:
        role = msg.__class__.__name__.replace("Message", "")
        content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"    [{role}] {content}")


# =============================================================================
# Example 2: Keep First N Messages
# =============================================================================


def trim_first_n():
    """Keep the first N messages (oldest)."""
    print("\n" + "=" * 60)
    print("2. Keep First N Messages")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a programming language."),
        HumanMessage(content="Tell me more."),
        AIMessage(content="It was created by Guido van Rossum."),
        HumanMessage(content="When?"),
        AIMessage(content="In the late 1980s."),
    ]

    print(f"\n  Original: {len(messages)} messages")

    # Keep first 3 messages (rare use case)
    trimmed = trim_messages(
        messages,
        strategy="first",
        max_messages=3,
    )

    print(f"  After trim: {len(trimmed)} messages")
    print("\n  Kept messages:")
    for msg in trimmed:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"    [{role}] {msg.content[:50]}")


# =============================================================================
# Example 3: Trim by Token Limit
# =============================================================================


def trim_by_tokens():
    """Keep messages that fit within a token budget."""
    print("\n" + "=" * 60)
    print("3. Trim by Token Limit")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a helpful coding assistant."),
        HumanMessage(content="Explain Python decorators in detail."),
        AIMessage(
            content="""Decorators are a powerful Python feature.
They allow you to modify function behavior without changing code.
A decorator is a function that takes another function and extends it.
The @syntax is just syntactic sugar for wrapper functions.
Common uses: logging, timing, authentication, caching."""
        ),
        HumanMessage(content="Can you show me an example?"),
        AIMessage(
            content="""Here's a timing decorator:
```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Took {time.time()-start:.2f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
```"""
        ),
        HumanMessage(content="How do I add arguments to decorators?"),
    ]

    print(f"\n  Original: {len(messages)} messages")
    print(f"  Original tokens: ~{count_tokens_approximate(messages)}")

    # Trim to fit in 200 tokens
    trimmed = trim_messages(
        messages,
        strategy="token",
        max_tokens=200,
        preserve_system=True,
    )

    print(f"\n  After trim: {len(trimmed)} messages")
    print(f"  Trimmed tokens: ~{count_tokens_approximate(trimmed)}")


# =============================================================================
# Example 4: Preserve System Message
# =============================================================================


def preserve_system():
    """System message is always preserved by default."""
    print("\n" + "=" * 60)
    print("4. Preserve System Message")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a Python expert. Be concise."),
        HumanMessage(content="What is a list?"),
        AIMessage(content="A mutable ordered collection."),
        HumanMessage(content="What is a tuple?"),
        AIMessage(content="An immutable ordered collection."),
        HumanMessage(content="What is a dict?"),
        AIMessage(content="A key-value mapping."),
    ]

    # Keep only last 2 messages, but system is always preserved
    trimmed = trim_messages(
        messages,
        strategy="last",
        max_messages=2,
        preserve_system=True,  # Default
    )

    print("\n  Result (system preserved + last 2):")
    for msg in trimmed:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"    [{role}] {msg.content}")

    # Optionally, don't preserve system
    trimmed_no_system = trim_messages(
        messages,
        strategy="last",
        max_messages=2,
        preserve_system=False,
    )

    print("\n  Without preserving system:")
    for msg in trimmed_no_system:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"    [{role}] {msg.content}")


# =============================================================================
# Example 5: Custom Token Counter
# =============================================================================


def custom_token_counter():
    """Use a custom token counter function."""
    print("\n" + "=" * 60)
    print("5. Custom Token Counter")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi there! How can I help you today?"),
        HumanMessage(content="Tell me about Python."),
        AIMessage(content="Python is a versatile programming language."),
    ]

    # Simple word-based counter (for demonstration)
    def word_counter(msgs):
        """Count words instead of tokens."""
        total = 0
        for m in msgs:
            total += len(m.content.split())
        return total

    print(f"\n  Approximate tokens: {count_tokens_approximate(messages)}")
    print(f"  Word count: {word_counter(messages)}")

    # Trim using word counter
    trimmed = trim_messages(
        messages,
        strategy="token",
        max_tokens=20,  # 20 words
        token_counter=word_counter,
    )

    print(f"\n  After trim (max 20 words): {len(trimmed)} messages")
    print(f"  Word count after: {word_counter(trimmed)}")


# =============================================================================
# Example 6: Dict Format Messages
# =============================================================================


def dict_messages():
    """trim_messages also works with dict-format messages."""
    print("\n" + "=" * 60)
    print("6. Dict Format Messages")
    print("=" * 60)

    # Dict format (OpenAI-style)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What is 3+3?"},
        {"role": "assistant", "content": "3+3 equals 6."},
    ]

    trimmed = trim_messages(
        messages,
        strategy="last",
        max_messages=2,
    )

    print("\n  Input: dict format")
    print(f"  Output: {len(trimmed)} BaseMessage objects")
    for msg in trimmed:
        print(f"    {type(msg).__name__}: {msg.content}")


# =============================================================================
# Example 7: Integration with LLM
# =============================================================================


def llm_integration():
    """Use trimmed messages with LLM.chat()."""
    print("\n" + "=" * 60)
    print("7. Integration with LLM")
    print("=" * 60)

    print("\n  Pattern for managing context in chat:")
    print("""
    from ai_infra import LLM
    from ai_infra.memory import trim_messages

    llm = LLM()
    conversation = []

    while True:
        user_input = input("You: ")
        conversation.append(HumanMessage(content=user_input))

        # Trim before sending to LLM
        trimmed = trim_messages(
            conversation,
            strategy="token",
            max_tokens=3000,  # Leave room for response
        )

        response = llm.chat(trimmed)
        print(f"AI: {response}")

        conversation.append(AIMessage(content=response))
""")


# =============================================================================
# Example 8: Empty Edge Case
# =============================================================================


def edge_cases():
    """Handle edge cases gracefully."""
    print("\n" + "=" * 60)
    print("8. Edge Cases")
    print("=" * 60)

    # Empty list
    result = trim_messages([], strategy="last", max_messages=5)
    print(f"\n  Empty list: {result}")

    # max_messages=0
    messages = [HumanMessage(content="Hello")]
    result = trim_messages(messages, strategy="last", max_messages=0)
    print(f"  max_messages=0: {result}")

    # Single message
    messages = [HumanMessage(content="Hello")]
    result = trim_messages(messages, strategy="last", max_messages=10)
    print(f"  Single message, max=10: {len(result)} messages")


# =============================================================================
# Example 9: Token Counting Utilities
# =============================================================================


def token_counting():
    """Demonstrate token counting utilities."""
    print("\n" + "=" * 60)
    print("9. Token Counting Utilities")
    print("=" * 60)

    from ai_infra.memory import count_tokens_approximate, get_context_limit

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Tell me about Python programming language."),
        AIMessage(content="Python is a high-level, interpreted language."),
    ]

    # Approximate count (fast, no dependencies)
    approx = count_tokens_approximate(messages)
    print(f"\n  Approximate tokens: {approx}")

    # Context limits for different models
    print("\n  Model context limits:")
    models = ["gpt-4", "gpt-4-turbo", "gpt-4o", "claude-3-opus"]
    for model in models:
        try:
            limit = get_context_limit(model)
            print(f"    {model}: {limit:,} tokens")
        except ValueError:
            print(f"    {model}: (unknown)")


# =============================================================================
# Example 10: Practical Chat Loop
# =============================================================================


def practical_example():
    """Complete example of context management in a chat loop."""
    print("\n" + "=" * 60)
    print("10. Practical Chat Loop")
    print("=" * 60)

    print("\n  Complete context management pattern:")
    print("""
    from ai_infra import LLM
    from ai_infra.memory import trim_messages, count_tokens_approximate

    def chat_with_context_management():
        llm = LLM()
        messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        MAX_CONTEXT = 4000  # Token budget

        while True:
            # Get user input
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break

            messages.append(HumanMessage(content=user_input))

            # Check if we need to trim
            tokens = count_tokens_approximate(messages)
            if tokens > MAX_CONTEXT:
                print(f"  [Trimming from {len(messages)} messages...]")
                messages = trim_messages(
                    messages,
                    strategy="token",
                    max_tokens=MAX_CONTEXT,
                    preserve_system=True,
                )
                print(f"  [Trimmed to {len(messages)} messages]")

            # Get response
            response = llm.chat(messages)
            print(f"AI: {response}")

            messages.append(AIMessage(content=response))
""")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Trim Messages Examples")
    print("=" * 60)
    print("\nFit conversations into context windows without summarization.\n")

    trim_last_n()
    trim_first_n()
    trim_by_tokens()
    preserve_system()
    custom_token_counter()
    dict_messages()
    llm_integration()
    edge_cases()
    token_counting()
    practical_example()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print('  1. strategy="last" keeps most recent messages (most common)')
    print('  2. strategy="token" respects exact token budgets')
    print("  3. System messages are preserved by default")
    print("  4. Works with both BaseMessage and dict formats")
    print("  5. Use count_tokens_approximate() for fast estimates")


if __name__ == "__main__":
    main()
