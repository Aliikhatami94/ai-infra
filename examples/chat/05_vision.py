#!/usr/bin/env python
"""Vision/Multimodal Example.

This example demonstrates:
- Sending images to vision-capable models
- Image URLs vs base64/bytes
- Multiple images in one request
- Image analysis and description

Required API Keys (at least one with vision support):
- OPENAI_API_KEY (gpt-4o, gpt-4-turbo)
- ANTHROPIC_API_KEY (claude-3-opus, claude-3-sonnet, etc.)
- GOOGLE_API_KEY (gemini-pro-vision, gemini-1.5-flash, etc.)
"""

import asyncio
from pathlib import Path

from ai_infra import LLM


def main():
    llm = LLM()

    # Example with image URL
    print("=" * 60)
    print("Vision Example - Image URL")
    print("=" * 60)

    # Using a public image URL
    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    )

    response = llm.chat(
        "What do you see in this image? Describe it briefly.",
        images=[image_url],
    )

    print(f"\nImage: {image_url}")
    print(f"\nDescription: {response.content}")

    # Detailed analysis
    print("\n" + "=" * 60)
    print("Detailed Image Analysis")
    print("=" * 60)

    response = llm.chat(
        "Analyze this image in detail. Include: 1) Main subject 2) Colors 3) Mood/atmosphere 4) Any text visible",
        images=[image_url],
        system="You are an image analysis expert. Be thorough but concise.",
    )

    print(f"\nAnalysis:\n{response.content}")


def multiple_images_example():
    """Compare or analyze multiple images."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Multiple Images Comparison")
    print("=" * 60)

    # Two different images
    image1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    image2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"

    response = llm.chat(
        "Compare these two images. What are the main differences?",
        images=[image1, image2],
    )

    print(f"\nImage 1: {image1}")
    print(f"Image 2: {image2}")
    print(f"\nComparison:\n{response.content}")


def structured_vision_example():
    """Extract structured data from images."""
    from pydantic import BaseModel, Field

    llm = LLM()

    print("\n" + "=" * 60)
    print("Structured Vision Output")
    print("=" * 60)

    class ImageAnalysis(BaseModel):
        """Structured analysis of an image."""

        main_subject: str = Field(description="The primary subject of the image")
        subject_type: str = Field(description="Category: animal, person, object, landscape, etc.")
        dominant_colors: list[str] = Field(description="Top 3 dominant colors in the image")
        mood: str = Field(description="Overall mood: happy, calm, dramatic, etc.")
        contains_text: bool = Field(description="Whether the image contains any text")
        description: str = Field(description="One-sentence description")

    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    )

    result: ImageAnalysis = llm.chat(
        "Analyze this image.",
        images=[image_url],
        output_schema=ImageAnalysis,
    )

    print(f"\nImage: {image_url}")
    print("\nStructured Analysis:")
    print(f"  Main Subject: {result.main_subject}")
    print(f"  Type: {result.subject_type}")
    print(f"  Colors: {', '.join(result.dominant_colors)}")
    print(f"  Mood: {result.mood}")
    print(f"  Contains Text: {result.contains_text}")
    print(f"  Description: {result.description}")


def local_image_example():
    """Use a local image file."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Local Image File (Demo)")
    print("=" * 60)

    # Check if we have a local image to use
    local_image_path = Path("./sample_image.jpg")

    if local_image_path.exists():
        response = llm.chat(
            "What's in this image?",
            images=[local_image_path],
        )
        print(f"\nLocal image: {local_image_path}")
        print(f"Description: {response.content}")
    else:
        print("\nNo local image found.")
        print("To test with a local image, create 'sample_image.jpg' in this directory.")
        print("\nYou can use a local image like this:")
        print('  response = llm.chat("What\'s in this image?", images=[Path("./photo.jpg")])')


def image_bytes_example():
    """Use image bytes directly."""
    print("\n" + "=" * 60)
    print("Image from Bytes (Demo)")
    print("=" * 60)

    print("\nYou can also pass raw image bytes:")
    print("""
    # Read image as bytes
    with open("image.png", "rb") as f:
        image_bytes = f.read()

    # Pass bytes directly
    response = llm.chat(
        "What's in this image?",
        images=[image_bytes],
    )
    """)

    print("This is useful when you have images in memory (e.g., from PIL or downloads).")


def vision_with_specific_models():
    """Use specific vision-capable models."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Vision with Specific Models")
    print("=" * 60)

    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    )
    prompt = "What animal is this? Answer in one word."

    # Vision-capable models by provider
    vision_models = [
        ("openai", "gpt-4o"),
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("google_genai", "gemini-2.0-flash"),
        ("google_genai", "gemini-1.5-flash"),
    ]

    for provider, model in vision_models:
        if not LLM.is_provider_configured(provider):
            print(f"\n{provider}/{model}: ⏭️  Skipped (not configured)")
            continue

        try:
            response = llm.chat(
                prompt,
                images=[image_url],
                provider=provider,
                model_name=model,
            )
            print(f"\n{provider}/{model}: {response.content.strip()}")
        except Exception as e:
            print(f"\n{provider}/{model}: Error - {e}")


async def async_vision_example():
    """Async vision example."""
    llm = LLM()

    print("\n" + "=" * 60)
    print("Async Vision Example")
    print("=" * 60)

    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    )

    response = await llm.achat(
        "Describe this image in exactly 10 words.",
        images=[image_url],
    )

    print(f"\nImage: {image_url}")
    print(f"Description (10 words): {response.content}")


if __name__ == "__main__":
    main()
    multiple_images_example()
    structured_vision_example()
    local_image_example()
    image_bytes_example()
    vision_with_specific_models()
    asyncio.run(async_vision_example())
