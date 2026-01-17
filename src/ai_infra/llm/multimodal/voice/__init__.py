"""Voice input/output support for ai-infra.

This package provides microphone recording and audio playback for voice chat:
- Microphone: Record audio from the default input device
- AudioPlayer: Play audio through the default output device
- VoiceChat: Unified voice chat interface

Example - Voice Chat:
    ```python
    from ai_infra.llm.multimodal import VoiceChat
    from ai_infra.llm import LLM

    voice = VoiceChat()
    llm = LLM()

    while True:
        user_text = voice.listen()
        if not user_text or user_text.lower() in ["exit", "quit"]:
            break

        print(f"You: {user_text}")
        response = llm.chat(user_text)
        print(f"AI: {response}")

        voice.speak(response)
    ```

Requirements:
    Install with voice extras: `pip install ai-infra[voice]`
"""

from ai_infra.llm.multimodal.voice.chat import VoiceChat, VoiceChatConfig
from ai_infra.llm.multimodal.voice.playback import AudioPlayer
from ai_infra.llm.multimodal.voice.recording import Microphone, RecordingConfig

__all__ = [
    "Microphone",
    "RecordingConfig",
    "AudioPlayer",
    "VoiceChat",
    "VoiceChatConfig",
]
