"""Image tools for agents - analyze and generate images.

These tools allow agents to:
- Analyze images and answer questions about them
- Generate images from text descriptions

Example:
    ```python
    from ai_infra import Agent
    from ai_infra.llm.tools.custom.multimodal import analyze_image, generate_image

    agent = Agent(tools=[analyze_image, generate_image])
    result = agent.run("Describe the image at photo.jpg")
    ```
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool


@tool
def analyze_image(
    image_path: str,
    question: str = "Describe this image in detail",
) -> str:
    """Analyze an image and answer questions about it.

    Use this tool to understand what's in an image. You can ask
    specific questions about the image content.

    Args:
        image_path: Path or URL to the image to analyze.
        question: Question to answer about the image.

    Returns:
        Analysis or answer based on the image content.

    Example:
        analyze_image("photo.jpg")
        analyze_image("chart.png", question="What trend does this chart show?")
        analyze_image("https://example.com/image.jpg", question="How many people are in this image?")
    """
    from ai_infra.llm import LLM

    try:
        llm = LLM()
        response = llm.chat(
            question,
            images=[image_path],
            model_name="gpt-4o",  # Use vision-capable model
        )
        return str(response.content)
    except Exception as e:
        return f"Error analyzing image: {e}"


@tool
def generate_image(
    prompt: str,
    style: Literal["auto", "vivid", "natural"] = "auto",
    size: Literal["1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
) -> str:
    """Generate an image from a text description.

    Use this tool to create images based on text descriptions.
    Returns the path to the generated image file.

    Args:
        prompt: Text description of the image to generate.
        style: Image style - 'vivid' for dramatic, 'natural' for realistic.
        size: Image size (1024x1024, 1792x1024, or 1024x1792).

    Returns:
        Path to the generated image file, or error message.

    Example:
        generate_image("A sunset over mountains with a lake in the foreground")
        generate_image("A cute robot holding flowers", style="vivid")
    """
    from ai_infra.imagegen import ImageGen

    try:
        imagegen = ImageGen()
        results = imagegen.generate(
            prompt=prompt,
            style=style if style != "auto" else None,
            size=size,
        )
        if results:
            result = results[0]
            if result.url:
                return f"Image generated: {result.url}"
            elif result.data:
                return "Image generated: [binary data available]"
            else:
                return "Image generated but no URL or data available"
        return "Image generation returned no results"
    except Exception as e:
        return f"Error generating image: {e}"


@tool
def compare_images(
    image1_path: str,
    image2_path: str,
    comparison_type: str = "differences",
) -> str:
    """Compare two images and describe their differences or similarities.

    Use this tool to analyze and compare two images.

    Args:
        image1_path: Path or URL to the first image.
        image2_path: Path or URL to the second image.
        comparison_type: What to compare - 'differences', 'similarities', or 'both'.

    Returns:
        Comparison analysis of the two images.

    Example:
        compare_images("before.jpg", "after.jpg")
        compare_images("photo1.png", "photo2.png", comparison_type="similarities")
    """
    from ai_infra.llm import LLM

    prompts = {
        "differences": "Compare these two images and describe the key differences between them.",
        "similarities": "Compare these two images and describe what they have in common.",
        "both": "Compare these two images, describing both their similarities and differences.",
    }

    question = prompts.get(comparison_type, prompts["differences"])

    try:
        llm = LLM()
        response = llm.chat(
            question,
            images=[image1_path, image2_path],
            model_name="gpt-4o",
        )
        return str(response.content)
    except Exception as e:
        return f"Error comparing images: {e}"
