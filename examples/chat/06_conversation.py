#!/usr/bin/env python
"""Multi-turn Conversation Example.

This example demonstrates:
- Building conversation history manually
- Context management with system messages
- Conversation patterns and best practices
- Memory-efficient conversation handling

Required API Keys (at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
"""

import asyncio
from dataclasses import dataclass, field

from ai_infra import LLM


@dataclass
class Conversation:
    """Simple conversation manager for multi-turn chat."""

    llm: LLM
    system: str | None = None
    history: list[dict] = field(default_factory=list)
    max_turns: int = 20  # Limit history to prevent context overflow

    def chat(self, message: str, **kwargs) -> str:
        """Send a message and get a response, maintaining history."""
        # Build full prompt with history context
        context = self._build_context(message)

        # Get response
        response = self.llm.chat(
            context,
            system=self.system,
            **kwargs,
        )

        # Extract content
        content = response.content

        # Update history
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": content})

        # Trim history if too long
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2) :]

        return content

    async def achat(self, message: str, **kwargs) -> str:
        """Async version of chat."""
        context = self._build_context(message)
        response = await self.llm.achat(context, system=self.system, **kwargs)
        content = response.content

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": content})

        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2) :]

        return content

    def _build_context(self, current_message: str) -> str:
        """Build context string from history and current message."""
        if not self.history:
            return current_message

        # Format history as context
        context_parts = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")

        context_parts.append(f"User: {current_message}")

        return "\n\n".join(context_parts)

    def clear(self):
        """Clear conversation history."""
        self.history = []


def main():
    llm = LLM()

    # Basic multi-turn conversation
    print("=" * 60)
    print("Multi-turn Conversation")
    print("=" * 60)

    conv = Conversation(
        llm=llm,
        system="You are a helpful math tutor. Be patient and explain concepts clearly.",
    )

    # First turn
    print("\nUser: What is a prime number?")
    response = conv.chat("What is a prime number?")
    print(f"Assistant: {response}")

    # Follow-up (model remembers context)
    print("\nUser: Can you give me an example?")
    response = conv.chat("Can you give me an example?")
    print(f"Assistant: {response}")

    # Another follow-up
    print("\nUser: Is 15 a prime number?")
    response = conv.chat("Is 15 a prime number?")
    print(f"Assistant: {response}")


def conversation_with_persona():
    """Conversation with a specific persona."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Conversation with Persona")
    print("=" * 60)

    conv = Conversation(
        llm=llm,
        system=(
            "You are Captain Jack, a friendly pirate who loves to share stories "
            "about your adventures on the high seas. You always speak in a pirate accent "
            "and occasionally mention your ship, 'The Golden Parrot'. Keep responses brief."
        ),
    )

    exchanges = [
        "Hello! Who are you?",
        "What's the most exciting adventure you've had?",
        "Do you have any treasure?",
    ]

    for user_msg in exchanges:
        print(f"\nUser: {user_msg}")
        response = conv.chat(user_msg)
        print(f"Captain Jack: {response}")


def conversation_branching():
    """Demonstrate conversation branching and topic changes."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Conversation with Topic Changes")
    print("=" * 60)

    conv = Conversation(
        llm=llm,
        system="You are a knowledgeable assistant. Answer concisely.",
    )

    # Start with one topic
    print("\nUser: Tell me about Python programming.")
    response = conv.chat("Tell me about Python programming.")
    print(f"Assistant: {response[:200]}..." if len(response) > 200 else f"Assistant: {response}")

    # Change topic but reference previous
    print("\nUser: Is it easier than JavaScript?")
    response = conv.chat("Is it easier than JavaScript?")
    print(f"Assistant: {response[:200]}..." if len(response) > 200 else f"Assistant: {response}")

    # Completely new topic
    print("\nUser: Now tell me about cooking pasta.")
    response = conv.chat("Now tell me about cooking pasta.")
    print(f"Assistant: {response[:200]}..." if len(response) > 200 else f"Assistant: {response}")


def context_length_management():
    """Demonstrate handling long conversations."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Context Length Management")
    print("=" * 60)

    # Short history limit for demo
    conv = Conversation(llm=llm, system="Be very brief.", max_turns=3)

    print(f"\nMax turns in history: {conv.max_turns}")

    # Generate several turns
    for i in range(5):
        user_msg = f"Count to {i + 1}"
        print(f"\nTurn {i + 1} - User: {user_msg}")
        response = conv.chat(user_msg)
        print(f"Assistant: {response}")
        print(f"History length: {len(conv.history)} messages")


def structured_conversation():
    """Conversation with structured output."""
    from pydantic import BaseModel, Field

    llm = LLM()

    print("\n" + "=" * 60)
    print("Structured Conversation Output")
    print("=" * 60)

    class Response(BaseModel):
        answer: str = Field(description="The actual answer")
        confidence: str = Field(description="high, medium, or low")
        follow_up_questions: list[str] = Field(description="Suggested follow-up questions")

    # Note: For structured output, we handle history differently
    history = []

    def structured_chat(user_msg: str) -> Response:
        # Build context with history
        context = ""
        if history:
            for msg in history[-4:]:  # Last 4 messages
                role = "User" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content']}\n\n"

        context += f"User: {user_msg}"

        result: Response = llm.chat(
            context,
            system="You are helpful. Provide structured answers.",
            output_schema=Response,
        )

        # Update history
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": result.answer})

        return result

    # First question
    print("\nUser: What is machine learning?")
    response = structured_chat("What is machine learning?")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Follow-ups: {response.follow_up_questions}")

    # Follow-up
    if response.follow_up_questions:
        follow_up = response.follow_up_questions[0]
        print(f"\nUser: {follow_up}")
        response = structured_chat(follow_up)
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")


async def async_conversation():
    """Async multi-turn conversation."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Async Conversation")
    print("=" * 60)

    conv = Conversation(
        llm=llm,
        system="You are a friendly chat companion. Keep responses under 50 words.",
    )

    exchanges = [
        "Hi there!",
        "What's your favorite thing to discuss?",
        "That sounds interesting!",
    ]

    for user_msg in exchanges:
        print(f"\nUser: {user_msg}")
        response = await conv.achat(user_msg)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
    conversation_with_persona()
    conversation_branching()
    context_length_management()
    structured_conversation()
    asyncio.run(async_conversation())
