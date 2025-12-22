#!/usr/bin/env python
"""Structured Output Example.

This example demonstrates:
- Pydantic models for structured output
- Different output methods (prompt, json_schema, function_calling)
- Type-safe responses
- Complex nested schemas

Required API Keys (at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
"""

import asyncio
from enum import Enum

from pydantic import BaseModel, Field

from ai_infra import LLM


# Simple structured output
class Answer(BaseModel):
    """A simple answer with explanation."""

    value: str = Field(description="The answer to the question")
    explanation: str = Field(description="Why this is the answer")
    confidence: float = Field(description="Confidence level from 0 to 1", ge=0.0, le=1.0)


# More complex nested structure
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TodoItem(BaseModel):
    """A todo item with details."""

    title: str = Field(description="Brief title of the task")
    description: str = Field(description="Detailed description")
    priority: Priority = Field(description="Task priority level")
    estimated_hours: float = Field(description="Estimated hours to complete", ge=0)
    tags: list[str] = Field(description="Relevant tags for categorization")


class TodoList(BaseModel):
    """A list of todo items for a project."""

    project_name: str = Field(description="Name of the project")
    items: list[TodoItem] = Field(description="List of todo items")
    total_estimated_hours: float = Field(description="Total hours for all items")


# Extraction example
class Person(BaseModel):
    """Information about a person."""

    name: str = Field(description="Full name")
    age: int | None = Field(description="Age if mentioned", default=None)
    occupation: str | None = Field(description="Job or occupation if mentioned", default=None)
    location: str | None = Field(description="Location if mentioned", default=None)


class ExtractedPeople(BaseModel):
    """People extracted from text."""

    people: list[Person] = Field(description="List of people mentioned")
    context: str = Field(description="Brief summary of the context")


def main():
    llm = LLM()

    # Basic structured output
    print("=" * 60)
    print("Basic Structured Output")
    print("=" * 60)

    result: Answer = llm.chat(
        "What is the capital of France?",
        output_schema=Answer,
    )

    print("\nQuestion: What is the capital of France?")
    print(f"Value: {result.value}")
    print(f"Explanation: {result.explanation}")
    print(f"Confidence: {result.confidence}")

    # Complex nested structure
    print("\n" + "=" * 60)
    print("Complex Nested Structure")
    print("=" * 60)

    todo_result: TodoList = llm.chat(
        "Create a todo list for building a personal website. Include 3-4 tasks.",
        system="You are a project planning assistant. Create detailed, actionable tasks.",
        output_schema=TodoList,
    )

    print(f"\nProject: {todo_result.project_name}")
    print(f"Total Estimated Hours: {todo_result.total_estimated_hours}")
    print("\nTasks:")
    for i, item in enumerate(todo_result.items, 1):
        print(f"\n  {i}. {item.title}")
        print(f"     Priority: {item.priority.value}")
        print(f"     Estimated: {item.estimated_hours}h")
        print(f"     Tags: {', '.join(item.tags)}")
        print(f"     Description: {item.description}")

    # Information extraction
    print("\n" + "=" * 60)
    print("Information Extraction")
    print("=" * 60)

    text = """
    Last week, I met with Sarah Johnson, a 34-year-old software engineer from
    San Francisco. She introduced me to her colleague, Mike Chen, who works as
    a data scientist. We also briefly spoke with the CEO, Amanda Williams.
    """

    extracted: ExtractedPeople = llm.chat(
        f"Extract information about people from this text:\n\n{text}",
        output_schema=ExtractedPeople,
    )

    print(f"\nInput text: {text.strip()}")
    print(f"\nContext: {extracted.context}")
    print(f"\nExtracted {len(extracted.people)} people:")
    for person in extracted.people:
        print(f"\n  - Name: {person.name}")
        if person.age:
            print(f"    Age: {person.age}")
        if person.occupation:
            print(f"    Occupation: {person.occupation}")
        if person.location:
            print(f"    Location: {person.location}")


def output_methods_example():
    """Demonstrate different output methods."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Different Output Methods")
    print("=" * 60)

    class SimpleAnswer(BaseModel):
        answer: str
        is_correct: bool

    question = "Is the sky blue on a clear day?"

    # Method 1: prompt (default) - works with any provider
    print("\n1. Using 'prompt' method (default, most compatible):")
    result = llm.chat(
        question,
        output_schema=SimpleAnswer,
        output_method="prompt",
    )
    print(f"   Answer: {result.answer}, Correct: {result.is_correct}")

    # Method 2: json_mode - requires provider support
    print("\n2. Using 'json_mode' method (if provider supports):")
    try:
        result = llm.chat(
            question,
            output_schema=SimpleAnswer,
            output_method="json_mode",
        )
        print(f"   Answer: {result.answer}, Correct: {result.is_correct}")
    except Exception as e:
        print(f"   Not supported: {e}")

    # Method 3: function_calling - uses tool/function calling
    print("\n3. Using 'function_calling' method:")
    try:
        result = llm.chat(
            question,
            output_schema=SimpleAnswer,
            output_method="function_calling",
        )
        print(f"   Answer: {result.answer}, Correct: {result.is_correct}")
    except Exception as e:
        print(f"   Not supported: {e}")

    # Method 4: json_schema - strict schema validation
    print("\n4. Using 'json_schema' method:")
    try:
        result = llm.chat(
            question,
            output_schema=SimpleAnswer,
            output_method="json_schema",
        )
        print(f"   Answer: {result.answer}, Correct: {result.is_correct}")
    except Exception as e:
        print(f"   Not supported: {e}")


async def async_structured_example():
    """Async structured output."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Async Structured Output")
    print("=" * 60)

    class Recipe(BaseModel):
        name: str
        ingredients: list[str]
        prep_time_minutes: int
        difficulty: str

    result: Recipe = await llm.achat(
        "Give me a simple recipe for scrambled eggs.",
        output_schema=Recipe,
    )

    print(f"\nRecipe: {result.name}")
    print(f"Prep Time: {result.prep_time_minutes} minutes")
    print(f"Difficulty: {result.difficulty}")
    print("Ingredients:")
    for ing in result.ingredients:
        print(f"  - {ing}")


if __name__ == "__main__":
    main()
    output_methods_example()
    asyncio.run(async_structured_example())
