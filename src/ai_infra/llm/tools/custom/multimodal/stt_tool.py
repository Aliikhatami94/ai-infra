"""STT tool for agents - transcribe audio files.

This tool allows agents to transcribe audio files to text using
the configured STT provider.

Example:
    ```python
    from ai_infra import Agent
    from ai_infra.llm.tools.custom.multimodal import transcribe_audio

    agent = Agent(tools=[transcribe_audio])
    result = agent.run("Transcribe the audio file meeting.mp3")
    ```
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def transcribe_audio(
    file_path: str,
    language: str = "en",
) -> str:
    """Transcribe an audio file to text.

    Use this tool to convert speech in audio files to text.
    Supports MP3, WAV, M4A, and other common audio formats.

    Args:
        file_path: Path to the audio file to transcribe.
        language: Language code (e.g., 'en' for English, 'es' for Spanish).

    Returns:
        The transcribed text from the audio file.

    Example:
        transcribe_audio("meeting.mp3")
        transcribe_audio("interview.wav", language="es")
    """
    from ai_infra.llm.multimodal import STT

    try:
        stt = STT()
        result = stt.transcribe(file_path, language=language)
        return result.text
    except Exception as e:
        return f"Error transcribing audio: {e}"


@tool
def transcribe_audio_with_timestamps(
    file_path: str,
    language: str = "en",
) -> str:
    """Transcribe an audio file with timestamps.

    Use this tool to get a timestamped transcription of an audio file.
    Each segment includes start and end times.

    Args:
        file_path: Path to the audio file to transcribe.
        language: Language code (e.g., 'en' for English).

    Returns:
        Timestamped transcription with format "[start-end]: text"
    """
    from ai_infra.llm.multimodal import STT

    try:
        stt = STT()
        result = stt.transcribe(file_path, language=language, timestamps=True)

        if not result.segments:
            return result.text

        lines = []
        for seg in result.segments:
            start = f"{seg.start:.1f}s"
            end = f"{seg.end:.1f}s"
            lines.append(f"[{start}-{end}]: {seg.text}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error transcribing audio: {e}"
