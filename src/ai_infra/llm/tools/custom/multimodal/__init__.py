"""Multimodal tools for AI agents.

These tools allow agents to work with audio and images:
- transcribe_audio: Transcribe audio files to text
- analyze_image: Analyze and describe images
- generate_image: Generate images from text descriptions

Example:
    ```python
    from ai_infra import Agent
    from ai_infra.llm.tools.custom.multimodal import (
        transcribe_audio,
        analyze_image,
        generate_image,
    )

    agent = Agent(tools=[transcribe_audio, analyze_image, generate_image])
    result = agent.run("Transcribe the audio file meeting.mp3 and summarize it")
    ```
"""

from .image_tools import analyze_image, generate_image
from .stt_tool import transcribe_audio

__all__ = [
    "analyze_image",
    "generate_image",
    "transcribe_audio",
]
